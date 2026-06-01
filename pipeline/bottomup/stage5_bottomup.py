"""
SuperHSSG Stage 5: Bottom-up 의미 집계
========================================
같은 레벨 노드 간 정보를 교환하고 Room 노드를 생성한다.

구성:
  5-1. GAT  (intra-level 메시지 패싱)  L1 끼리, L3 끼리
  5-2. Edge MLP  (관계 패턴 인코딩)  L1-L1 엣지 feature
  5-3. PMA  (Set Transformer Pooling by Multihead Attention)  Room 노드 생성
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_l1_edges_from_sp(l1_labels_sp: np.ndarray,
                            sp_edge_src: np.ndarray,
                            sp_edge_dst: np.ndarray):
    """SP 수준 인접 그래프 → L1 수준 인접 엣지 (서로 다른 L1 사이)
    구조체(-1) SP가 포함된 엣지는 제외한다."""
    l1_s  = l1_labels_sp[sp_edge_src]
    l1_d  = l1_labels_sp[sp_edge_dst]
    cross = (l1_s != l1_d) & (l1_s >= 0) & (l1_d >= 0)
    pairs = np.stack([l1_s[cross], l1_d[cross]], axis=1)
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs[:, 0], pairs[:, 1]

D_L1    = 651
D_NODE  = 256   # GAT 후 hidden dim
D_EDGE  = 128   # Edge MLP 출력 dim
D_ROOM  = 512   # Room feature dim
D_LAYOUT = 16   # layout feature dim


# ── 5-1. GAT Layer ────────────────────────────────────────────────────────────
class GATLayer(nn.Module):
    """
    단일 Graph Attention Network 레이어.
    f_i' = f_i + Σ_j α(i,j) · W·f_j
    """
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = out_dim // n_heads
        self.W       = nn.Linear(in_dim, out_dim, bias=False)
        self.attn    = nn.Linear(2 * out_dim, n_heads, bias=False)
        self.proj    = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm    = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor,
                edge_src: torch.Tensor,
                edge_dst: torch.Tensor) -> torch.Tensor:
        """
        x        [N, in_dim]
        edge_src [E]
        edge_dst [E]
        Returns  [N, out_dim]
        """
        N = x.shape[0]
        h = self.W(x)                           # [N, out_dim]

        # attention score per edge
        e = torch.cat([h[edge_src], h[edge_dst]], dim=-1)  # [E, 2*out_dim]
        a = self.attn(e)                        # [E, n_heads]
        a = F.leaky_relu(a, 0.2)

        # scatter softmax
        agg = torch.zeros(N, self.n_heads, self.d_head, device=x.device)
        for k in range(self.n_heads):
            a_k    = a[:, k]
            # softmax per target node
            exp_a  = torch.exp(a_k - a_k.max())
            denom  = torch.zeros(N, device=x.device).scatter_add(0, edge_dst, exp_a) + 1e-9
            coeff  = (exp_a / denom[edge_dst]).unsqueeze(-1)   # [E, 1]
            msg_k  = (h[edge_src] * coeff).view(-1, self.n_heads, self.d_head)[:, k, :]
            agg[:, k].scatter_add_(0, edge_dst.unsqueeze(-1).expand(-1, self.d_head), msg_k)

        agg = agg.view(N, -1)                   # [N, out_dim]
        out = self.norm(self.proj(x) + agg)
        return F.relu(out)


# ── 5-2. Edge MLP ─────────────────────────────────────────────────────────────
class EdgeMLP(nn.Module):
    """
    L1-L1 쌍의 관계 feature 인코딩.
    입력: [f_i', f_j', f_i'-f_j', rel_pos(3), rel_orient(1), dist(1), overlap_ratio(1)]
    출력: [E, D_EDGE]
    """
    def __init__(self, node_dim: int = D_NODE,
                 extra_dim: int = 6,
                 out_dim: int = D_EDGE):
        super().__init__()
        in_d = node_dim * 3 + extra_dim
        self.net = nn.Sequential(
            nn.Linear(in_d, 256), nn.ReLU(),
            nn.Linear(256, 128),  nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, h: torch.Tensor,
                edge_src: torch.Tensor,
                edge_dst: torch.Tensor,
                rel_pos: torch.Tensor,
                rel_orient: torch.Tensor,
                dist: torch.Tensor,
                overlap: torch.Tensor) -> torch.Tensor:
        """
        h        [N, D_NODE]
        rel_pos  [E, 3]   rel_orient [E,1]   dist [E,1]   overlap [E,1]
        Returns  [E, D_EDGE]
        """
        f_i   = h[edge_src]
        f_j   = h[edge_dst]
        diff  = f_i - f_j
        extra = torch.cat([rel_pos, rel_orient, dist, overlap], dim=-1)
        return self.net(torch.cat([f_i, f_j, diff, extra], dim=-1))


# ── 5-3. PMA ─────────────────────────────────────────────────────────────────
class PMALayer(nn.Module):
    """
    Pooling by Multihead Attention (Set Transformer 핵심 모듈).
    Q = 학습 가능한 seed vector
    K = V = 노드 + 엣지 feature 시퀀스
    → Room feature 생성
    """
    def __init__(self, in_dim: int, seed_dim: int = D_ROOM, n_heads: int = 4):
        super().__init__()
        self.seed     = nn.Parameter(torch.randn(1, seed_dim))
        self.kv_proj  = nn.Linear(in_dim, seed_dim)
        self.attn     = nn.MultiheadAttention(seed_dim, n_heads, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(seed_dim),
            nn.Linear(seed_dim, seed_dim), nn.ReLU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens [M, in_dim]  (노드 + 엣지 feature 시퀀스)
        Returns [seed_dim]
        """
        kv  = self.kv_proj(tokens).unsqueeze(0)    # [1, M, seed_dim]
        q   = self.seed.unsqueeze(0)               # [1, 1, seed_dim]
        out, _ = self.attn(q, kv, kv)              # [1, 1, seed_dim]
        return self.out_proj(out.squeeze(0).squeeze(0))  # [seed_dim]


# ── 공간 배치 패턴 ─────────────────────────────────────────────────────────────
class LayoutEncoder(nn.Module):
    """
    방 안 객체들의 공간 배치 통계를 인코딩한다.
    입력: centroid 분포, 높이 분포, 공간 밀도, dominant 방향
    """
    def __init__(self, out_dim: int = D_LAYOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 32), nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, centroids: torch.Tensor) -> torch.Tensor:
        """centroids [N_l1, 3] → [D_LAYOUT]"""
        mu    = centroids.mean(0)                           # [3]
        std   = centroids.std(0).clamp(min=1e-6)            # [3]
        h_mu  = centroids[:, 2].mean().unsqueeze(0)         # [1]
        h_std = centroids[:, 2].std().clamp(min=1e-6).unsqueeze(0)  # [1]
        density  = torch.tensor([centroids.shape[0] / 100.0],
                                device=centroids.device)    # [1]
        dom_dir  = F.normalize(centroids.T @ centroids, dim=-1).mean(0)[:3]  # [3] 근사
        feat = torch.cat([mu, std, h_mu, h_std, density, dom_dir])  # [12]
        return self.net(feat)                               # [D_LAYOUT]


# ── 통합 Bottom-up 모듈 ───────────────────────────────────────────────────────
class BottomUpModule(nn.Module):
    """
    GAT (5-1) + Edge MLP (5-2) + PMA (5-3) 통합 모듈.
    L1 feature들을 받아 정제된 L1 feature + Room feature를 반환한다.
    """
    def __init__(self,
                 in_dim:   int = D_L1,
                 node_dim: int = D_NODE,
                 edge_dim: int = D_EDGE,
                 room_dim: int = D_ROOM):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, node_dim)
        self.gat1       = GATLayer(node_dim, node_dim)
        self.gat2       = GATLayer(node_dim, node_dim)
        self.edge_mlp   = EdgeMLP(node_dim, extra_dim=6, out_dim=edge_dim)
        self.layout_enc = LayoutEncoder(out_dim=D_LAYOUT)
        pma_in          = node_dim + edge_dim + D_LAYOUT
        self.pma        = PMALayer(in_dim=pma_in, seed_dim=room_dim)

    def forward(self,
                f_l1: torch.Tensor,
                edge_src: torch.Tensor,
                edge_dst: torch.Tensor,
                rel_pos: torch.Tensor,
                rel_orient: torch.Tensor,
                dist: torch.Tensor,
                overlap: torch.Tensor) -> tuple:
        """
        f_l1    [N_l1, D_L1]
        edge_*  [E]  (L1 인접 그래프)
        rel_*   [E, ?]  (관계 통계)
        Returns:
            h     [N_l1, D_NODE]  정제된 L1 feature
            f_room [D_ROOM]       Room feature
        """
        h = F.relu(self.input_proj(f_l1))   # [N_l1, D_NODE]
        h = self.gat1(h, edge_src, edge_dst)
        h = self.gat2(h, edge_src, edge_dst)

        edge_feat  = self.edge_mlp(h, edge_src, edge_dst,
                                   rel_pos, rel_orient, dist, overlap)  # [E, D_EDGE]

        # PMA 입력: 노드 feature + edge feature + layout feature 를 하나의 시퀀스로
        layout = self.layout_enc(f_l1[:, :3])   # [D_LAYOUT] (centroid 사용)
        layout_exp = layout.unsqueeze(0).expand(len(h), -1)        # [N, D_LAYOUT]
        node_tokens = torch.cat([h, layout_exp.new_zeros(len(h), edge_feat.shape[1])
                                ], dim=-1)
        # edge를 node에 집계하여 토큰 구성 (mean over edges per node)
        edge_agg = torch.zeros(len(h), edge_feat.shape[1], device=h.device)
        cnt      = torch.zeros(len(h), 1, device=h.device)
        edge_agg.scatter_add_(0, edge_dst.unsqueeze(-1).expand(-1, edge_feat.shape[1]),
                              edge_feat)
        cnt.scatter_add_(0, edge_dst.unsqueeze(-1), torch.ones(len(edge_dst), 1, device=h.device))
        edge_agg = edge_agg / (cnt + 1e-6)

        tokens = torch.cat([h, edge_agg, layout_exp], dim=-1)  # [N, D_NODE+D_EDGE+D_LAYOUT]
        f_room = self.pma(tokens)                               # [D_ROOM]

        return h, f_room


# ── 관계 통계 계산 ─────────────────────────────────────────────────────────────
def compute_edge_stats(f_l1: np.ndarray,
                       edge_src: np.ndarray,
                       edge_dst: np.ndarray) -> dict:
    """
    L1 feature 에서 엣지별 관계 통계를 계산한다.
    Returns: dict of np.ndarray
    """
    ci = f_l1[edge_src, :3]
    cj = f_l1[edge_dst, :3]
    rel_pos = ci - cj                                                     # [E, 3]
    dist    = np.linalg.norm(rel_pos, axis=-1, keepdims=True)             # [E, 1]

    # 법선 방향 (feature [3:6])
    ni = f_l1[edge_src, 3:6]
    nj = f_l1[edge_dst, 3:6]
    ni = ni / (np.linalg.norm(ni, axis=-1, keepdims=True) + 1e-6)
    nj = nj / (np.linalg.norm(nj, axis=-1, keepdims=True) + 1e-6)
    cos_n     = (ni * nj).sum(-1, keepdims=True)                          # [E, 1]
    rel_orient = cos_n

    # overlap ratio 근사: size 중첩 / min(size_i, size_j)
    si = f_l1[edge_src, 10:11]
    sj = f_l1[edge_dst, 10:11]
    overlap = np.minimum(si, sj) / (np.maximum(si + sj, 1e-6))           # [E, 1]

    return dict(rel_pos=rel_pos.astype(np.float32),
                rel_orient=rel_orient.astype(np.float32),
                dist=dist.astype(np.float32),
                overlap=overlap.astype(np.float32))


# ── GT 생성 ───────────────────────────────────────────────────────────────────
def build_gt_from_labels(f_l1_np: np.ndarray,
                          text_labels: list,
                          edge_src: np.ndarray,
                          edge_dst: np.ndarray,
                          dist_thr: float = 2.0) -> np.ndarray:
    """같은 라벨 AND 거리 < dist_thr → 1 (병합), unknown 쌍은 항상 0"""
    labels = np.array(text_labels)
    same   = (labels[edge_src] == labels[edge_dst]) & (labels[edge_src] != 'unknown')
    dist   = np.linalg.norm(f_l1_np[edge_src, :3] - f_l1_np[edge_dst, :3], axis=1)
    return (same & (dist < dist_thr)).astype(np.float32)


# ── 학습 ─────────────────────────────────────────────────────────────────────
def train_stage5(f_l1, es_t, ed_t, rp_t, ro_t, di_t, ov_t, gt_np,
                 device, n_epochs=50, lr=1e-3, save_path=None):
    """
    BottomUpModule + EdgeHead(128→1) 를 BCE loss로 학습.
    Edge MLP 출력이 "이 두 L1은 같은 그룹인가"를 잘 인코딩하도록 학습.
    학습 완료 후 BottomUpModule 가중치만 저장.
    """
    model     = BottomUpModule(D_L1, D_NODE, D_EDGE, D_ROOM).to(device)
    edge_head = nn.Linear(D_EDGE, 1).to(device)          # 학습용 head (추론 시 미사용)

    opt   = torch.optim.Adam(
        list(model.parameters()) + list(edge_head.parameters()),
        lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    gt_t  = torch.from_numpy(gt_np).to(device)
    pos_w = (gt_t == 0).float().sum() / (gt_t == 1).float().sum().clamp(min=1)

    best_loss = float('inf')
    for epoch in range(1, n_epochs + 1):
        model.train(); edge_head.train()

        h, _ = model(f_l1, es_t, ed_t, rp_t, ro_t, di_t, ov_t)

        # edge feature 재계산 (model.edge_mlp 직접 호출)
        edge_feat = model.edge_mlp(h, es_t, ed_t, rp_t, ro_t, di_t, ov_t)  # [E, D_EDGE]
        scores    = torch.sigmoid(edge_head(edge_feat).squeeze(-1))          # [E]

        w    = torch.where(gt_t == 1, pos_w.expand_as(gt_t), torch.ones_like(gt_t))
        loss = F.binary_cross_entropy(scores, gt_t, weight=w)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                pred    = (scores > 0.5).float()
                acc     = (pred == gt_t).float().mean().item()
                recall  = (pred[gt_t == 1] == 1).float().mean().item() if gt_t.sum() > 0 else 0
            print(f'  Epoch {epoch:4d}/{n_epochs} | Loss: {loss.item():.4f}'
                  f' | Acc: {acc:.3f} | Recall: {recall:.3f}')

        if loss.item() < best_loss and save_path:
            best_loss = loss.item()
            torch.save(model.state_dict(), save_path)

    print(f'  → 체크포인트 저장: {save_path}  (loss={best_loss:.4f})')
    return model


# ── 추론 ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_stage5(model, f_t, es_t, ed_t, rp_t, ro_t, di_t, ov_t,
                 l2_lab, device):
    """
    Returns:
        h      [N_l1, D_NODE]  정제된 L1 feature
        h_l2   [N_l2, D_NODE]  L2 집계 feature
        f_room [D_ROOM]        Room feature
    """
    model.eval()
    h, f_room = model(f_t, es_t, ed_t, rp_t, ro_t, di_t, ov_t)

    # L2 그룹별 attention pooling
    pooler   = nn.Linear(D_NODE, 1).to(device)
    l2_lab_t = torch.from_numpy(l2_lab.astype(np.int64)).to(device)
    n_l2     = int(l2_lab.max()) + 1
    h_l2_list = []
    for lid in range(n_l2):
        mask  = l2_lab_t == lid
        h_sub = h[mask]
        w     = torch.softmax(pooler(h_sub), dim=0)
        h_l2_list.append((w * h_sub).sum(0))
    h_l2 = torch.stack(h_l2_list)   # [N_l2, D_NODE]

    return h, h_l2, f_room


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    _ROOT       = Path(__file__).resolve().parents[1]                      # …/pipeline
    _BOTTOMUP   = Path(__file__).resolve().parent                          # …/model/bottomup
    _OPENSHAPE  = Path(__file__).resolve().parents[1] / 'openshapeCLI'    # …/model/openshapeCLI
    _SEM_MERGE  = Path(__file__).resolve().parents[1] / 'sem_merge'       # …/model/sem_merge
    _MERGE      = Path(__file__).resolve().parents[1] / 'merge'           # …/model/merge

    # ── 자동 감지: stage2_results에서 l1_*.npz ──────────────────────────────────
    _STAGE2_DIR  = _MERGE / 'stage2_results'
    _l1_cands    = sorted(_STAGE2_DIR.glob('l1_*.npz')) if _STAGE2_DIR.exists() else []
    _default_l1  = str(max(_l1_cands, key=lambda p: p.stat().st_mtime)) \
                   if _l1_cands else str(_STAGE2_DIR / 'l1_floor.npz')

    parser = argparse.ArgumentParser(description='SuperHSSG Stage 5: Bottom-up 의미 집계')
    parser.add_argument('--fl1_npy',      default=str(_OPENSHAPE  / 'stage3_results' / 'f_l1_floor.npy'))
    parser.add_argument('--l1_npz',       default=_default_l1,
                        help='Stage 2 출력 (기본: stage2_results/에서 최신 l1_*.npz 자동 감지)')
    parser.add_argument('--l2_npy',       default=str(_SEM_MERGE  / 'stage4_results' / 'floor_l2.npy'))
    parser.add_argument('--labels_json',  default=str(_OPENSHAPE  / 'stage3_results' / 'l1_labels_text.json'))
    parser.add_argument('--out_dir',      default=str(_BOTTOMUP   / 'stage5_results'))
    parser.add_argument('--ckpt',         default=str(_BOTTOMUP   / 'stage5_results' / 'stage5_model.pt'))
    parser.add_argument('--mode',         choices=['train', 'infer'],
                        default=None,
                        help='pt 파일 있으면 자동으로 infer, 없으면 train')
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--dist_thr',     type=float, default=2.0)
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir);  OUT_DIR.mkdir(exist_ok=True)
    device  = get_device()
    print(f'[Stage 5] 디바이스: {device}')

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    f_l1   = np.load(args.fl1_npy).astype(np.float32)
    l1_npz = np.load(args.l1_npz)
    sp_l1_raw = l1_npz['l1_labels'].astype(np.int64)   # -1=구조체, 0~=L1 id
    sp_es_raw = l1_npz['edge_src'].astype(np.int64)
    sp_ed_raw = l1_npz['edge_dst'].astype(np.int64)
    l2_lab = np.load(args.l2_npy)

    # ── 구조체(-1) SP 필터링 및 L1 ID 재매핑 (stage3/4와 동일) ─────────────
    valid_mask  = sp_l1_raw >= 0
    sp_l1_valid = sp_l1_raw[valid_mask]
    old_ids     = np.unique(sp_l1_valid)
    id_remap    = np.full(int(old_ids.max()) + 1, -1, dtype=np.int64)
    id_remap[old_ids] = np.arange(len(old_ids))
    sp_l1       = id_remap[sp_l1_valid]          # [N_sp_valid] 0-based 재매핑

    sp_idx_map  = np.full(len(valid_mask), -1, dtype=np.int64)
    sp_idx_map[np.where(valid_mask)[0]] = np.arange(valid_mask.sum())
    valid_edges = (sp_idx_map[sp_es_raw] >= 0) & (sp_idx_map[sp_ed_raw] >= 0)
    sp_es       = sp_idx_map[sp_es_raw[valid_edges]]
    sp_ed       = sp_idx_map[sp_ed_raw[valid_edges]]

    with open(args.labels_json) as f:
        json_data = json.load(f)
    text_labels = [o['label'] for o in sorted(json_data['objects'], key=lambda x: x['l1_id'])]

    print(f'  L1: {len(f_l1)}개  |  L2 그룹: {int(l2_lab.max())+1}개')

    # ── L1 엣지 + 관계 통계 ──────────────────────────────────────────────────
    l1_es, l1_ed = build_l1_edges_from_sp(sp_l1, sp_es, sp_ed)
    print(f'  L1 엣지: {len(l1_es)}개')

    # 구조체 제거 후 L1 ID 압축 시 엣지 인덱스가 범위를 벗어날 수 있음 → 필터
    n_l1 = len(f_l1)
    valid_edge = (l1_es < n_l1) & (l1_ed < n_l1)
    if valid_edge.sum() < len(l1_es):
        print(f'  [경고] 범위 초과 L1 엣지 {(~valid_edge).sum()}개 제거 '
              f'(n_l1={n_l1}, max_idx={max(l1_es.max(), l1_ed.max())})')
        l1_es = l1_es[valid_edge]
        l1_ed = l1_ed[valid_edge]

    stats = compute_edge_stats(f_l1, l1_es, l1_ed)

    # ── GT 생성 ───────────────────────────────────────────────────────────────
    gt = build_gt_from_labels(f_l1, text_labels, l1_es, l1_ed, args.dist_thr)
    print(f'  GT 병합 비율: {gt.mean():.1%}  ({gt.sum():.0f}/{len(gt)})')

    # ── 텐서 변환 ─────────────────────────────────────────────────────────────
    f_t  = torch.from_numpy(f_l1).to(device)
    es_t = torch.from_numpy(l1_es.astype(np.int64)).to(device)
    ed_t = torch.from_numpy(l1_ed.astype(np.int64)).to(device)
    rp_t = torch.from_numpy(stats['rel_pos']).to(device)
    ro_t = torch.from_numpy(stats['rel_orient']).to(device)
    di_t = torch.from_numpy(stats['dist']).to(device)
    ov_t = torch.from_numpy(stats['overlap']).to(device)

    # ── 모드 결정 (stage4와 동일 패턴) ───────────────────────────────────────
    ckpt_path = Path(args.ckpt)
    if args.mode is None:
        mode = 'infer' if ckpt_path.exists() else 'train'
    else:
        mode = args.mode
    print(f'[Stage 5] 모드: {mode}')

    # ── 실행 ──────────────────────────────────────────────────────────────────
    if mode == 'train':
        print(f'\n  {args.epochs}에폭 학습 시작 ...')
        ckpt_path.parent.mkdir(exist_ok=True)
        model = train_stage5(
            f_t, es_t, ed_t, rp_t, ro_t, di_t, ov_t, gt,
            device, n_epochs=args.epochs, save_path=str(ckpt_path))
    else:
        if not ckpt_path.exists():
            print(f'  [오류] 체크포인트 없음: {ckpt_path}')
            print('  → --mode train 으로 먼저 학습하세요.')
            exit(1)
        model = BottomUpModule(D_L1, D_NODE, D_EDGE, D_ROOM).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'  체크포인트 로드: {ckpt_path}')

    # ── 추론 ──────────────────────────────────────────────────────────────────
    print('  forward ...')
    h, h_l2, f_room = infer_stage5(model, f_t, es_t, ed_t, rp_t, ro_t, di_t, ov_t,
                                    l2_lab, device)

    # ── 저장 ──────────────────────────────────────────────────────────────────
    np.save(OUT_DIR / 'floor_h_l1.npy',  h.cpu().numpy())
    np.save(OUT_DIR / 'floor_h_l2.npy',  h_l2.cpu().numpy())
    np.save(OUT_DIR / 'floor_room.npy',  f_room.cpu().numpy())

    print(f'\n[Stage 5] 완료')
    print(f'  floor_h_l1.npy  {tuple(h.shape)}        정제된 L1 feature')
    print(f'  floor_h_l2.npy  {tuple(h_l2.shape)}     정제된 L2 feature')
    print(f'  floor_room.npy  {tuple(f_room.shape)}    Room feature')
