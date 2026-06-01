"""
SuperHSSG Stage 6: Top-down 문맥 정제 + HSSG 출력
===================================================
Bottom-up(Stage 5) ↔ Top-down(Stage 6)를 T=3회 반복하며 수렴시킨다.

Stage 6 역할:
  Cross-Attention: Q=f_L1', K=V=f_room → f_L1'' (방 맥락 주입)
  출력 헤드:
    - NodeClassifier  : L1 → 객체 클래스 (13),  L3 → 방 타입 (6)
    - RelationPredictor: L1-L1 intra + L1-L3 inter

학습:
  L_node    : CE (객체 분류)
  L_room    : CE (방 분류)
  L_intra   : CE (L1-L1 관계)
  L_inter   : CE (L1-L3 관계: contains / part-of)
  L_contrastive: 같은 방 타입끼리 유사하게
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse

try:
    from scipy.spatial import ConvexHull as _ConvexHull
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── 경로 설정: model/bottomup 직접 탐색 ──────────────────────────────────────
def _add_model_path():
    here = Path(__file__).resolve().parent          # pipeline/topdown/
    root = here.parent                              # pipeline/
    for p in [str(root), str(root / 'bottomup')]:
        if p not in sys.path:
            sys.path.insert(0, p)
_add_model_path()

from stage5_bottomup import (                       # noqa: E402
    BottomUpModule, EdgeMLP, compute_edge_stats,
    build_l1_edges_from_sp, get_device,
    D_L1, D_NODE, D_EDGE, D_ROOM,
)

# 분류 차원 (S3DIS 기준)
N_OBJ_CLASS  = 13    # ceiling, floor, wall, beam, column, window, door,
                     # table, chair, sofa, bookcase, board, clutter
N_ROOM_CLASS = 9     # conferenceRoom, copyRoom, hallway, lobby, office,
                     # pantry, storage, WC, openSpace
N_INTRA_REL  = 27    # 3DSSG 26개 + background
N_INTER_REL  = 3     # none, contains, part-of

T_ITERS = 3          # Top-down ↔ Bottom-up 반복 횟수

# ── scene_graph JSON 출력용 관계 이름 ─────────────────────────────────────────
INTRA_RELS = [
    'none',
    'attached to', 'lying on', 'hanging on', 'connected to',
    'leaning against', 'part of', 'belonging to', 'hanging from',
    'standing on', 'supported by', 'above', 'below', 'next to',
    'behind', 'in front of', 'to the left of', 'to the right of',
    'same as', 'inside', 'cover', 'build in',
    'standing in', 'placing on', 'under', 'on the ceiling of',
    'background',
]
INTER_RELS = ['none', 'contains', 'part-of']


# ── Top-down 모듈 ─────────────────────────────────────────────────────────────
class TopDownRefiner(nn.Module):
    """
    Cross-Attention으로 방 feature를 각 L1 객체에 주입한다.
    f_L1'' = f_L1' + CrossAttention(Q=f_L1', K=V=f_room)
    """
    def __init__(self, node_dim: int = D_NODE, room_dim: int = D_ROOM, n_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(node_dim, room_dim)
        self.attn   = nn.MultiheadAttention(room_dim, n_heads, batch_first=True)
        self.out_proj = nn.Linear(room_dim, node_dim)
        self.norm   = nn.LayerNorm(node_dim)

    def forward(self, h: torch.Tensor, f_room: torch.Tensor) -> torch.Tensor:
        """
        h      [N_l1, D_NODE]
        f_room [D_ROOM]
        Returns [N_l1, D_NODE]
        """
        q   = self.q_proj(h).unsqueeze(0)            # [1, N, D_ROOM]
        kv  = f_room.unsqueeze(0).unsqueeze(0)        # [1, 1, D_ROOM]
        out, _ = self.attn(q, kv, kv)                # [1, N, D_ROOM]
        delta  = self.out_proj(out.squeeze(0))        # [N, D_NODE]
        return self.norm(h + delta)


# ── 출력 헤드 ─────────────────────────────────────────────────────────────────
class NodeClassifier(nn.Module):
    def __init__(self, in_dim: int = D_NODE, n_obj: int = N_OBJ_CLASS,
                 n_room: int = N_ROOM_CLASS):
        super().__init__()
        self.obj_head  = nn.Linear(in_dim, n_obj)
        self.room_head = nn.Linear(D_ROOM, n_room)

    def forward(self, h: torch.Tensor, f_room: torch.Tensor):
        """h [N_l1, D_NODE], f_room [D_ROOM] → obj_logits [N_l1, 13], room_logit [1, 6]"""
        return self.obj_head(h), self.room_head(f_room.unsqueeze(0))


class RelationPredictor(nn.Module):
    """
    intra: L1-L1 관계 (edge)
    inter: L1-L3 관계 (모든 L1-방 쌍)
    """
    def __init__(self, node_dim: int = D_NODE, edge_dim: int = D_EDGE,
                 room_dim: int = D_ROOM):
        super().__init__()
        self.intra = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, 128), nn.ReLU(),
            nn.Linear(128, N_INTRA_REL),
        )
        self.inter = nn.Sequential(
            nn.Linear(node_dim + room_dim, 256), nn.ReLU(),
            nn.Linear(256, N_INTER_REL),
        )

    def forward(self, h: torch.Tensor,
                edge_feat: torch.Tensor,
                edge_src: torch.Tensor,
                edge_dst: torch.Tensor,
                f_room: torch.Tensor):
        """
        Returns:
            intra_logits [E, N_INTRA_REL]
            inter_logits [N_l1, N_INTER_REL]
        """
        intra_in = torch.cat([h[edge_src], h[edge_dst], edge_feat], dim=-1)
        intra_logits = self.intra(intra_in)

        f_room_exp   = f_room.unsqueeze(0).expand(len(h), -1)  # [N_l1, D_ROOM]
        inter_logits = self.inter(torch.cat([h, f_room_exp], dim=-1))

        return intra_logits, inter_logits


# ── 통합 HSSG 모델 ─────────────────────────────────────────────────────────────
class HSSGModel(nn.Module):
    """
    Stage 5+6를 T_ITERS 회 반복하여 최종 HSSG를 생성하는 통합 모델.
    """
    def __init__(self, t_iters: int = T_ITERS):
        super().__init__()
        self.t_iters    = t_iters
        self.bottomup   = BottomUpModule()
        self.topdown    = TopDownRefiner()
        self.classifier = NodeClassifier()
        self.relation   = RelationPredictor()
        # ── 버그 수정: forward마다 새로 생성하지 않고 __init__에서 한번만 ──
        self.edge_mlp   = EdgeMLP(D_NODE, 6, D_EDGE)

    def forward(self,
                f_l1: torch.Tensor,
                edge_src: torch.Tensor,
                edge_dst: torch.Tensor,
                rel_pos: torch.Tensor,
                rel_orient: torch.Tensor,
                dist: torch.Tensor,
                overlap: torch.Tensor):
        """
        f_l1 [N_l1, D_L1]  edge_* [E]  관계통계 [E, ?]
        Returns dict of logits for loss computation
        """
        h, f_room = None, None

        for _ in range(self.t_iters):
            h, f_room = self.bottomup(f_l1, edge_src, edge_dst,
                                      rel_pos, rel_orient, dist, overlap)
            h = self.topdown(h, f_room)

        # 최종 edge feature (학습 가능한 self.edge_mlp 사용)
        edge_feat = self.edge_mlp(h, edge_src, edge_dst,
                                  rel_pos, rel_orient, dist, overlap)

        obj_logits, room_logit = self.classifier(h, f_room)
        intra_logits, inter_logits = self.relation(
            h, edge_feat, edge_src, edge_dst, f_room)

        return {
            'h':            h,
            'f_room':       f_room,
            'obj_logits':   obj_logits,    # [N_l1, 13]
            'room_logit':   room_logit,    # [1, 9]
            'intra_logits': intra_logits,  # [E, 27]
            'inter_logits': inter_logits,  # [N_l1, 3]
        }


# ── 손실 함수 ─────────────────────────────────────────────────────────────────
class HSSGLoss(nn.Module):
    """
    L_total = λ3·L_node + λ4·L_intra + λ5·L_inter + λ6·L_room + λ7·L_contrastive
    λ 값은 튜닝 가능; 기본값은 균형을 위해 1.0
    """
    def __init__(self, lam_node=1.0, lam_room=1.0, lam_intra=0.5,
                 lam_inter=0.5, lam_contra=0.1):
        super().__init__()
        self.lam = dict(node=lam_node, room=lam_room, intra=lam_intra,
                        inter=lam_inter, contra=lam_contra)

    def forward(self, pred: dict, gt: dict) -> dict:
        """
        pred: HSSGModel 출력 dict
        gt  : {
            'obj_cls'   : [N_l1] int   (노드 클래스)
            'room_cls'  : int          (방 클래스, 없으면 -1)
            'intra_rel' : [E] int      (L1-L1 관계 라벨, 없으면 None)
            'inter_rel' : [N_l1] int   (L1-방 관계, 없으면 None)
        }
        """
        losses = {}

        # L_node: 객체 분류
        if gt.get('obj_cls') is not None:
            losses['node'] = F.cross_entropy(pred['obj_logits'], gt['obj_cls'])

        # L_room: 방 분류
        if gt.get('room_cls') is not None and gt['room_cls'] >= 0:
            room_gt = torch.tensor([gt['room_cls']], device=pred['room_logit'].device)
            losses['room'] = F.cross_entropy(pred['room_logit'], room_gt)

        # L_intra: L1-L1 관계
        if gt.get('intra_rel') is not None:
            losses['intra'] = F.cross_entropy(pred['intra_logits'], gt['intra_rel'])

        # L_inter: L1-방 관계
        if gt.get('inter_rel') is not None:
            losses['inter'] = F.cross_entropy(pred['inter_logits'], gt['inter_rel'])

        # L_contrastive: 같은 방 타입끼리 f_room 유사하게 (현재 단순 L2 정규화)
        if 'f_room' in pred:
            losses['contra'] = (1.0 - F.normalize(pred['f_room'], dim=-1).norm()) ** 2

        total = sum(self.lam[k] * v for k, v in losses.items() if k in self.lam)
        losses['total'] = total
        return losses


# ── GT 구성 헬퍼 ──────────────────────────────────────────────────────────────
S3DIS_CLASSES = ['ceiling','floor','wall','beam','column','window','door',
                 'table','chair','sofa','bookcase','board','clutter']
S3DIS_LABEL_MAP = {name: i for i, name in enumerate(S3DIS_CLASSES)}

S3DIS_ROOM_CLASSES = {
    'conferenceRoom': 0, 'copyRoom': 1, 'hallway': 2, 'lobby': 3,
    'office': 4, 'pantry': 5, 'storage': 6, 'WC': 7, 'openSpace': 8,
}

def build_gt_from_text(text_labels: list, device: torch.device,
                        room_cls: int = -1) -> dict:
    """
    text_labels: stage3 출력 텍스트 라벨 리스트
    → obj_cls  : 텍스트 → S3DIS 클래스 번호
    → inter_rel: 모든 L1은 방에 포함됨 (contains=1)
    → intra_rel: None (3DSSG 데이터 없음)
    → room_cls : 방 이름에서 추출 (-1이면 skip)
    """
    obj_cls  = np.array([S3DIS_LABEL_MAP.get(l, 12) for l in text_labels],
                        dtype=np.int64)
    inter_rel = np.ones(len(text_labels), dtype=np.int64)   # 모두 contains
    return {
        'obj_cls':   torch.from_numpy(obj_cls).long().to(device),
        'room_cls':  room_cls,
        'intra_rel': None,
        'inter_rel': torch.from_numpy(inter_rel).long().to(device),
    }


# ── 학습 ─────────────────────────────────────────────────────────────────────
def train_stage6(f_t, es_t, ed_t, rp_t, ro_t, di_t, ov_t, gt,
                 device, n_epochs=50, lr=1e-3, save_path=None):
    model     = HSSGModel().to(device)
    criterion = HSSGLoss()
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best      = float('inf')

    for epoch in range(1, n_epochs + 1):
        model.train()
        pred   = model(f_t, es_t, ed_t, rp_t, ro_t, di_t, ov_t)
        losses = criterion(pred, gt)

        opt.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if epoch % 10 == 0:
            obj_acc = (pred['obj_logits'].argmax(-1) == gt['obj_cls']).float().mean().item()
            print(f'  Epoch {epoch:4d}/{n_epochs} | Loss: {losses["total"].item():.4f}'
                  f' | ObjAcc: {obj_acc:.3f}')

        if losses['total'].item() < best and save_path:
            best = losses['total'].item()
            torch.save(model.state_dict(), save_path)

    print(f'  → 체크포인트 저장: {save_path}  (loss={best:.4f})')
    return model


# ── 추론 ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_stage56(model: HSSGModel,
                  f_l1: np.ndarray,
                  edge_src: np.ndarray,
                  edge_dst: np.ndarray,
                  edge_stats: dict) -> dict:
    """HSSG 구조 반환"""
    model.eval()
    device = next(model.parameters()).device

    f  = torch.from_numpy(f_l1).to(device)
    es = torch.from_numpy(edge_src).long().to(device)
    ed = torch.from_numpy(edge_dst).long().to(device)
    rp = torch.from_numpy(edge_stats['rel_pos']).to(device)
    ro = torch.from_numpy(edge_stats['rel_orient']).to(device)
    di = torch.from_numpy(edge_stats['dist']).to(device)
    ov = torch.from_numpy(edge_stats['overlap']).to(device)

    pred = model(f, es, ed, rp, ro, di, ov)

    obj_pred  = pred['obj_logits'].argmax(-1).cpu().numpy()
    room_pred = pred['room_logit'].argmax(-1).cpu().numpy()
    intra_pred = pred['intra_logits'].argmax(-1).cpu().numpy()
    inter_pred = pred['inter_logits'].argmax(-1).cpu().numpy()

    return {
        'obj_pred':   obj_pred,    # [N_l1] 객체 클래스
        'room_pred':  room_pred,   # [1]    방 타입
        'intra_pred': intra_pred,  # [E]    L1-L1 관계
        'inter_pred': inter_pred,  # [N_l1] L1-방 관계
        'h':          pred['h'].cpu().numpy(),
        'f_room':     pred['f_room'].cpu().numpy(),
    }


def print_hssg(hssg: dict, room_name: str,
               edge_src: np.ndarray, edge_dst: np.ndarray):
    """HSSG 구조 텍스트 출력"""
    S3DIS_CLASSES = ['ceiling','floor','wall','beam','column','window','door',
                     'table','chair','sofa','bookcase','board','clutter']
    ROOM_CLS = list(S3DIS_ROOM_CLASSES.keys())
    INTER_REL = ['none', 'contains', 'part-of']
    INTRA_REL = ['none'] + [f'rel_{i}' for i in range(1, N_INTRA_REL)]

    room_type = ROOM_CLS[int(hssg['room_pred'][0])] if int(hssg['room_pred'][0]) < len(ROOM_CLS) else '?'
    print(f"\n[{room_name}] 방 타입: {room_type}")
    print(f"  L1 객체 {len(hssg['obj_pred'])}개:")
    for i, cls_id in enumerate(hssg['obj_pred']):
        cls_name = S3DIS_CLASSES[min(cls_id, 12)]
        rel_name = INTER_REL[min(hssg['inter_pred'][i], 2)]
        if rel_name != 'none':
            print(f"    [{i:3d}] {cls_name:12s} → {rel_name} → [{room_name}]")
    print(f"  L1-L1 관계 (non-none):")
    for k, (si, di_) in enumerate(zip(edge_src, edge_dst)):
        rel = INTRA_REL[min(hssg['intra_pred'][k], N_INTRA_REL - 1)]
        if rel != 'none':
            ci = S3DIS_CLASSES[min(hssg['obj_pred'][si], 12)]
            cj = S3DIS_CLASSES[min(hssg['obj_pred'][di_], 12)]
            print(f"    {ci:12s} --{rel}--> {cj}")


# ── scene_graph_skeleton.json 형식 출력 헬퍼 ──────────────────────────────────

def _compute_zone_geometry(xy: np.ndarray, zs: np.ndarray):
    """
    L1 centroid XY, Z 배열로 ZONE geometry 계산.
    Returns (center [x,y,z], height float, coordinates [[x,y], ...])
    """
    if len(xy) == 0:
        return [0.0, 0.0, 0.0], 0.0, []

    cx = float(np.mean(xy[:, 0]))
    cy = float(np.mean(xy[:, 1]))
    cz = float(np.mean(zs))
    height = float(np.max(zs) - np.min(zs))

    if len(xy) >= 3 and _HAS_SCIPY:
        try:
            hull = _ConvexHull(xy)
            hull_pts = xy[hull.vertices]
            coords = [[round(float(x), 3), round(float(y), 3)] for x, y in hull_pts]
        except Exception:
            coords = [[round(float(x), 3), round(float(y), 3)] for x, y in xy]
    else:
        coords = [[round(float(x), 3), round(float(y), 3)] for x, y in xy]

    return [round(cx, 3), round(cy, 3), round(cz, 3)], round(height, 3), coords


def export_scene_graph_json(
    hssg: dict,
    l1_labels_sp: np.ndarray,
    sp_features: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    l2_labels: np.ndarray,
    text_labels: list,
    out_path: str,
    building_id: str = 'BLD_001',
    room_geometry: list | None = None,
) -> dict:
    """
    HSSG 추론 결과를 scene_graph_skeleton.json 구조로 저장.

    scene_graph 구조:
      nodes  : L2 방 → ZONE 노드 (geometry + assets)
      assets : 방 미배정 L1 객체 (inter_pred == none)
      edges  : L1-L1 non-none 관계

    Args:
        hssg          : infer_stage56() 반환 dict
        l1_labels_sp  : [N_sp] SP → L1 인덱스 매핑
        sp_features   : [N_sp, >=3] SP feature (첫 3열 = XYZ centroid)
        edge_src/dst  : [E] L1-L1 엣지 인덱스
        l2_labels     : [N_l1] L1 → 방(L2) 배정 (stage4 결과)
        text_labels   : [N_l1] 텍스트 라벨 리스트
        out_path      : 저장 경로
        building_id   : JSON 최상위 building_id 값
        room_geometry : stage4가 저장한 floor_l2_geometry.json 의 'rooms' 리스트.
                        제공 시 ZONE boundary를 객체 centroid convex hull 대신
                        wall_door_2d.py 기반 실제 방 경계로 대체한다.
    """
    n_l1 = len(text_labels)
    obj_pred   = hssg['obj_pred']    # [N_l1]
    intra_pred = hssg['intra_pred']  # [E]
    inter_pred = hssg['inter_pred']  # [N_l1]

    # ceiling(0), floor(1), wall(2), beam(3) — L1 객체 노드로 JSON에 포함하지 않는다.
    _STRUCTURAL_CLS = {0, 1, 2, 3}

    # ── L1 centroid 계산 (SP feature XYZ 평균) ────────────────────────────────
    centroids = np.zeros((n_l1, 3), dtype=np.float32)
    counts    = np.zeros(n_l1,      dtype=np.int32)
    np.add.at(centroids, l1_labels_sp, sp_features[:, 0:3])
    np.add.at(counts,    l1_labels_sp, 1)
    counts = np.maximum(counts, 1)
    centroids = centroids / counts[:, None]   # [N_l1, 3]

    # ── room_geometry 인덱스 빌드 (l2_id → 방 정보) ─────────────────────────
    # stage4 가 저장한 floor_l2_geometry.json 의 rooms 리스트를 l2_id 로 조회
    _room_geo_map: dict = {}
    if room_geometry:
        for rg in room_geometry:
            _room_geo_map[int(rg['l2_id'])] = rg

    # ── ZONE 노드 구성 (L2 방 1개당 ZONE 1개) ────────────────────────────────
    # room_geometry가 있으면 그 개수를 정답으로 사용(빈 방 포함 전체 방 수).
    # 없으면 l2_labels에 실제로 나타난 최대 ID+1로 폴백.
    if room_geometry:
        n_l2 = len(room_geometry)
    else:
        n_l2 = int(l2_labels.max()) + 1 if len(l2_labels) > 0 else 1
    zone_nodes = []

    for rid in range(n_l2):
        # 방 소속은 wall_door_2d.py 공간 배정(l2_labels)만으로 결정.
        # inter_pred(미학습 MLP)를 쓰면 대부분 none으로 예측돼 객체가 방에서 누락됨.
        in_room    = l2_labels == rid
        member_ids = np.where(in_room)[0]

        # ── geometry: 실제 방 경계 우선, fallback → centroid convex hull ────
        rg = _room_geo_map.get(rid)
        if rg and rg.get('boundary') and len(rg['boundary']) >= 3:
            # wall_door_2d.py 기반 실제 방 경계 사용
            boundary = rg['boundary']          # [[x,y], ...] world coord
            coords   = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in boundary]
            # center: 방 centroid (있으면 사용, 없으면 boundary 평균)
            if rg.get('centroid') and len(rg['centroid']) >= 2:
                cx_r, cy_r = float(rg['centroid'][0]), float(rg['centroid'][1])
            else:
                cx_r = float(np.mean([p[0] for p in boundary]))
                cy_r = float(np.mean([p[1] for p in boundary]))
            # z: 방 내 객체들의 평균 z (없으면 0)
            cz_r = float(np.mean(centroids[member_ids, 2])) if len(member_ids) > 0 else 0.0
            h_r  = float(np.max(centroids[member_ids, 2]) - np.min(centroids[member_ids, 2])) \
                   if len(member_ids) > 1 else 0.0
            center = [round(cx_r, 3), round(cy_r, 3), round(cz_r, 3)]
            height = round(h_r, 3)
        elif len(member_ids) > 0:
            pts    = centroids[member_ids]
            center, height, coords = _compute_zone_geometry(pts[:, :2], pts[:, 2])
        else:
            center, height, coords = [0.0, 0.0, 0.0], 0.0, []

        zone_assets = []
        for lid in member_ids:
            # text_labels(stage3 CLIP 결과)를 우선 사용.
            # 미학습 MLP obj_pred는 신뢰도가 낮으므로 참조용으로만 쓴다.
            text_lbl = text_labels[lid] if lid < len(text_labels) else 'unknown'
            cls_id   = int(obj_pred[lid]) if lid < len(obj_pred) else 12

            # 필터링: clutter·structural·unknown 제외
            if text_lbl in ('unknown', 'structural'):
                continue
            if text_lbl in ('ceiling', 'floor', 'wall', 'beam', 'column'):
                continue
            if cls_id in _STRUCTURAL_CLS and text_lbl == 'unknown':
                continue

            # 라벨: stage3 텍스트 라벨 우선, 없으면 S3DIS 클래스명 폴백
            cls_name = text_lbl if text_lbl != 'unknown' else S3DIS_CLASSES[min(cls_id, 12)]
            cx, cy, cz = centroids[lid]
            zone_assets.append({
                'id':       f'ASSET_{int(lid):03d}',
                'class':    cls_name,
                'position': [round(float(cx), 3), round(float(cy), 3), round(float(cz), 3)],
                'status':   'normal',
            })

        # 객체가 하나도 없는 ZONE (빈 방)은 scene_graph에서 제외.
        # — 시각화에서 내용 없는 폴리곤이 그려지지 않아 가독성이 높아짐.
        # — 복도/개방공간(stage4 재배정 후 0개)도 자연스럽게 제외됨.
        if len(zone_assets) == 0:
            continue

        zone_nodes.append({
            'id':   f'ZONE_{(rid + 1):03d}',
            'type': 'ZONE',
            'name': f'추출 구역 {rid + 1}',
            'geometry': {
                'type':        'Polygon',
                'center':      center,
                'height':      height,
                'coordinates': coords,
            },
            'assets': zone_assets,
        })

    # ── 글로벌 asset: l2_labels 기준으로 어느 방에도 배정되지 않은 L1 ───────────
    # (inter_pred 기준 제거 — 미학습 MLP 출력에 의존하지 않음)
    assigned_lids = set(np.where(l2_labels >= 0)[0].tolist())
    global_assets = []
    for lid in range(n_l1):
        if lid in assigned_lids:   # 이미 어떤 방에 배정됨 → 글로벌 아님
            continue
        text_lbl = text_labels[lid] if lid < len(text_labels) else 'unknown'
        cls_id   = int(obj_pred[lid]) if lid < len(obj_pred) else 12
        if text_lbl in ('unknown', 'structural', 'ceiling', 'floor', 'wall', 'beam', 'column'):
            continue
        if cls_id == 12 and text_lbl == 'unknown':
            continue
        cls_name = text_lbl if text_lbl != 'unknown' else S3DIS_CLASSES[min(cls_id, 12)]
        cx, cy, cz = centroids[lid]
        global_assets.append({
            'id':       f'ASSET_GLOBAL_{lid:03d}',
            'class':    cls_name,
            'position': [round(float(cx), 3), round(float(cy), 3), round(float(cz), 3)],
            'status':   'normal',
        })

    # ── 엣지: L1-L1 관계 중 none 제외 ────────────────────────────────────────
    edges = []
    for k, (s, d) in enumerate(zip(edge_src, edge_dst)):
        rel_id   = int(intra_pred[k]) if k < len(intra_pred) else 0
        if rel_id == 0:
            continue
        rel_name = INTRA_RELS[rel_id] if rel_id < len(INTRA_RELS) else f'rel_{rel_id}'
        edges.append({
            'src':      f'ASSET_{int(s):03d}',
            'dst':      f'ASSET_{int(d):03d}',
            'relation': rel_name,
        })

    # ── FLOOR 노드 구성 (모든 ZONE의 상위 계층) ──────────────────────────────
    # 층(floor) = 건물 내 하나의 평면. 모든 ZONE을 자식으로 묶는다.
    floor_zone_ids = [z['id'] for z in zone_nodes]

    # floor 전체 bounding box: 모든 zone 좌표 + 모든 asset 위치에서 추출
    all_xy = []
    all_z  = []
    for z in zone_nodes:
        coords = z['geometry'].get('coordinates', [])
        if coords:
            for pt in coords:
                all_xy.append(pt[:2])
        for a in z.get('assets', []):
            pos = a.get('position', [0, 0, 0])
            all_xy.append([pos[0], pos[1]])
            all_z.append(pos[2])

    if all_xy:
        xy_arr   = np.array(all_xy, dtype=np.float32)
        x_min_f  = round(float(xy_arr[:, 0].min()), 3)
        x_max_f  = round(float(xy_arr[:, 0].max()), 3)
        y_min_f  = round(float(xy_arr[:, 1].min()), 3)
        y_max_f  = round(float(xy_arr[:, 1].max()), 3)
        z_min_f  = round(float(min(all_z)), 3) if all_z else 0.0
        z_max_f  = round(float(max(all_z)), 3) if all_z else 3.0
    else:
        x_min_f = x_max_f = y_min_f = y_max_f = 0.0
        z_min_f = 0.0; z_max_f = 3.0

    floor_node = {
        'id':    'FLOOR_001',
        'type':  'FLOOR',
        'name':  '1층',
        'geometry': {
            'type':   'BoundingBox',
            'x_range': [x_min_f, x_max_f],
            'y_range': [y_min_f, y_max_f],
            'z_range': [z_min_f, z_max_f],
        },
        'zones': floor_zone_ids,   # 자식 ZONE id 목록
    }

    # nodes 리스트: FLOOR 먼저, 이후 ZONE들
    all_nodes = [floor_node] + zone_nodes

    # ── 최종 JSON 조합 ────────────────────────────────────────────────────────
    output = {
        'building_id': building_id,
        'scene_graph': {
            'nodes':  all_nodes,
            'assets': global_assets,
            'edges':  edges,
        },
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)

    print(f'  → scene_graph JSON 저장: {out_path}')
    print(f'     FLOOR 수: 1  ZONE 수: {len(zone_nodes)}  '
          f'글로벌 asset: {len(global_assets)}  '
          f'edge 수: {len(edges)}')
    return output


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    _ROOT       = Path(__file__).resolve().parents[1]                      # …/pipeline
    _TOPDOWN    = Path(__file__).resolve().parent                          # …/model/topdown
    _OPENSHAPE  = Path(__file__).resolve().parents[1] / 'openshapeCLI'    # …/model/openshapeCLI
    _SEM_MERGE  = Path(__file__).resolve().parents[1] / 'sem_merge'       # …/model/sem_merge
    _MERGE      = Path(__file__).resolve().parents[1] / 'merge'           # …/model/merge

    # ── 자동 감지: stage2_results에서 l1_*.npz, src/에서 *floor*.npz ──────────
    _STAGE2_DIR  = _MERGE / 'stage2_results'
    _l1_cands    = sorted(_STAGE2_DIR.glob('l1_*.npz')) if _STAGE2_DIR.exists() else []
    _default_l1  = str(max(_l1_cands, key=lambda p: p.stat().st_mtime)) \
                   if _l1_cands else str(_STAGE2_DIR / 'l1_floor.npz')
    # floor_npz: l1 stem에서 유도 (l1_6_floor.npz → src/6_floor.npz)
    _src_dir     = _ROOT / 'src'
    _l1_stem     = Path(_default_l1).stem[len('l1_'):]   # 'l1_6_floor' → '6_floor'
    _src_derived = _src_dir / f'{_l1_stem}.npz'
    _src_cands   = sorted(_src_dir.glob('*.npz')) if _src_dir.exists() else []
    # *_floor.npz 패턴만 (s2 중간 파일 제외)
    _floor_cands = [p for p in _src_cands if p.stem.endswith('floor')]
    _default_src = str(_src_derived) if _src_derived.exists() \
                   else (str(max(_floor_cands, key=lambda p: p.stat().st_mtime))
                         if _floor_cands else str(_src_dir / 'floor.npz'))

    parser = argparse.ArgumentParser(description='SuperHSSG Stage 6: Top-down 정제 + HSSG 출력')
    parser.add_argument('--fl1_npy',     default=str(_OPENSHAPE / 'stage3_results' / 'f_l1_floor.npy'))
    parser.add_argument('--l1_npz',      default=_default_l1,
                        help='Stage 2 출력 (기본: stage2_results/에서 최신 l1_*.npz 자동 감지)')
    parser.add_argument('--labels_json', default=str(_OPENSHAPE / 'stage3_results' / 'l1_labels_text.json'))
    parser.add_argument('--out_dir',     default=str(_TOPDOWN   / 'stage6_results'))
    parser.add_argument('--ckpt',        default=str(_TOPDOWN   / 'stage6_results' / 'stage6_model.pt'))
    parser.add_argument('--mode',        choices=['train', 'infer'], default=None,
                        help='pt 파일 있으면 자동 infer, 없으면 train')
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=1e-3)
    # ── scene_graph JSON 출력용 추가 인수 ─────────────────────────────────────
    parser.add_argument('--floor_npz',   default=_default_src,
                        help='SP feature 파일 (XYZ centroid 계산용)'
                             ' (기본: src/에서 최신 *floor*.npz 자동 감지)')
    parser.add_argument('--l2_npy',      default=str(_SEM_MERGE / 'stage4_results' / 'floor_l2.npy'),
                        help='Stage 4 방 배정 결과 [N_l1]')
    parser.add_argument('--l2_geometry', default=str(_SEM_MERGE / 'stage4_results' / 'floor_l2_geometry.json'),
                        help='stage4가 저장한 방 경계 geometry JSON (없으면 centroid convex hull 사용)')
    parser.add_argument('--building_id', default='BLD_001',
                        help='출력 JSON의 building_id')
    parser.add_argument('--no_json',     action='store_true',
                        help='scene_graph JSON 출력 건너뜀')
    parser.add_argument('--out_name',   default=None,
                        help='출력 JSON 파일명 (기본: scene_graph.json).\n'
                             '예: --out_name 6_scene_graph.json → results/6_scene_graph.json')
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir);  OUT_DIR.mkdir(exist_ok=True)
    device  = get_device()
    print(f'[Stage 6] 디바이스: {device}')

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    f_l1   = np.load(args.fl1_npy).astype(np.float32)
    l1_npz = np.load(args.l1_npz)
    sp_l1_raw = l1_npz['l1_labels'].astype(np.int64)   # -1=구조체, 0~=L1 id
    sp_es_raw = l1_npz['edge_src'].astype(np.int64)
    sp_ed_raw = l1_npz['edge_dst'].astype(np.int64)

    # ── 구조체(-1) SP 필터링 및 L1 ID 재매핑 (stage3/4/5와 동일) ────────────
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
    # stage3 에서 structural(ceiling/floor/wall/beam) 은 JSON 객체 목록에서 제외됨.
    # l1_id 에 구멍이 생기므로, n_l1 크기의 배열을 만들고 누락 ID 는 'unknown' 으로 채운다.
    n_l1_from_json = json_data.get('n_l1', len(f_l1))
    _label_map = {o['l1_id']: o['label'] for o in json_data['objects']}
    text_labels = [_label_map.get(i, 'unknown') for i in range(n_l1_from_json)]
    print(f'  L1: {len(f_l1)}개  (structural 제외 객체: {len(json_data["objects"])}개)')

    # ── L1 엣지 + 관계 통계 ──────────────────────────────────────────────────
    l1_es, l1_ed = build_l1_edges_from_sp(sp_l1, sp_es, sp_ed)
    print(f'  L1 엣지 (전체): {len(l1_es)}개')

    # 구조체 제거 후 L1 ID 압축 시 엣지 인덱스가 범위를 벗어날 수 있음 → 필터
    _n_l1 = len(f_l1)
    _valid = (l1_es < _n_l1) & (l1_ed < _n_l1)
    if _valid.sum() < len(l1_es):
        print(f'  [경고] 범위 초과 L1 엣지 {(~_valid).sum()}개 제거 '
              f'(n_l1={_n_l1}, max_idx={max(l1_es.max(), l1_ed.max())})')
        l1_es = l1_es[_valid]
        l1_ed = l1_ed[_valid]

    # ── 방별 l2_labels 로드 (wall_door_2d.py → stage4 결과) ─────────────────
    l2_npy_path = Path(args.l2_npy)
    if not l2_npy_path.exists():
        print(f'  [오류] l2_npy 없음: {l2_npy_path}'); exit(1)
    l2_labels_all = np.load(l2_npy_path).astype(np.int32)   # [N_l1]

    # ── 방 내부 엣지만 유지 (GAT가 방 경계를 넘어 메시지 전파하지 않도록) ─────
    # l1_es, l1_ed 인덱스가 l2_labels 범위 내인지 확인 후 같은 방 쌍만 남긴다
    es_safe = np.clip(l1_es, 0, len(l2_labels_all) - 1)
    ed_safe = np.clip(l1_ed, 0, len(l2_labels_all) - 1)
    intra_mask = l2_labels_all[es_safe] == l2_labels_all[ed_safe]
    l1_es_intra = l1_es[intra_mask]
    l1_ed_intra = l1_ed[intra_mask]
    print(f'  L1 엣지 (방 내부만): {len(l1_es_intra)}개  '
          f'(방 경계 교차 {intra_mask.size - intra_mask.sum()}개 제거)')

    stats       = compute_edge_stats(f_l1, l1_es_intra, l1_ed_intra)
    stats_all   = compute_edge_stats(f_l1, l1_es, l1_ed)  # scene_graph 엣지 출력용

    # ── 텐서 변환 ─────────────────────────────────────────────────────────────
    f_t  = torch.from_numpy(f_l1).to(device)
    es_t = torch.from_numpy(l1_es_intra.astype(np.int64)).to(device)
    ed_t = torch.from_numpy(l1_ed_intra.astype(np.int64)).to(device)
    rp_t = torch.from_numpy(stats['rel_pos']).to(device)
    ro_t = torch.from_numpy(stats['rel_orient']).to(device)
    di_t = torch.from_numpy(stats['dist']).to(device)
    ov_t = torch.from_numpy(stats['overlap']).to(device)

    # ── GT 구성 ───────────────────────────────────────────────────────────────
    gt = build_gt_from_text(text_labels, device, room_cls=-1)
    obj_dist = np.bincount(gt['obj_cls'].cpu().numpy(), minlength=13)
    print(f'  GT 객체 분포: ' +
          '  '.join(f'{S3DIS_CLASSES[i]}:{obj_dist[i]}' for i in range(13) if obj_dist[i] > 0))

    # ── 모드 결정 ─────────────────────────────────────────────────────────────
    ckpt_path = Path(args.ckpt)
    mode = args.mode or ('infer' if ckpt_path.exists() else 'train')
    print(f'[Stage 6] 모드: {mode}')

    # ── 실행 ──────────────────────────────────────────────────────────────────
    if mode == 'train':
        print(f'\n  {args.epochs}에폭 학습 시작 ...')
        ckpt_path.parent.mkdir(exist_ok=True)
        model = train_stage6(
            f_t, es_t, ed_t, rp_t, ro_t, di_t, ov_t, gt,
            device, n_epochs=args.epochs, lr=args.lr, save_path=str(ckpt_path))
    else:
        if not ckpt_path.exists():
            print(f'  [오류] 체크포인트 없음: {ckpt_path}')
            print('  → --mode train 으로 먼저 학습하세요.')
            exit(1)
        model = HSSGModel().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'  체크포인트 로드: {ckpt_path}')

    # ── 추론 (방 내부 엣지만 사용 — GAT가 방 경계를 넘지 않음) ─────────────────
    print('  HSSG 추론 ...')
    hssg = infer_stage56(model, f_l1, l1_es_intra, l1_ed_intra, stats)

    # ── 저장 ──────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / 'floor_hssg.npz'
    np.savez_compressed(out_path,
                        obj_pred   = hssg['obj_pred'],
                        room_pred  = hssg['room_pred'],
                        intra_pred = hssg['intra_pred'],
                        inter_pred = hssg['inter_pred'],
                        h          = hssg['h'],
                        f_room     = hssg['f_room'],
                        edge_src   = l1_es_intra,
                        edge_dst   = l1_ed_intra)

    # ── 결과 요약 출력 ────────────────────────────────────────────────────────
    print_hssg(hssg, 'floor', l1_es_intra, l1_ed_intra)

    obj_dist_pred = np.bincount(hssg['obj_pred'], minlength=13)
    print(f'\n[Stage 6] 완료')
    print(f'  저장: {out_path}')
    print(f'  예측 객체 분포:')
    for i, cnt in enumerate(obj_dist_pred):
        if cnt > 0:
            print(f'    {S3DIS_CLASSES[i]:12s}: {cnt}개')

    # ── scene_graph_skeleton.json 형식 출력 ───────────────────────────────────
    if not args.no_json:
        print('\n[Stage 6] scene_graph JSON 생성 중...')
        floor_npz_path = Path(args.floor_npz)
        l2_npy_path    = Path(args.l2_npy)

        if not floor_npz_path.exists():
            print(f'  [경고] floor_npz 없음: {floor_npz_path}  → JSON 출력 건너뜀')
        elif not l2_npy_path.exists():
            print(f'  [경고] l2_npy 없음: {l2_npy_path}  → JSON 출력 건너뜀')
        else:
            floor_data      = np.load(floor_npz_path)
            sp_features_all = floor_data['sp_features'].astype(np.float32)

            # l1_npz와 floor_npz의 SP 수가 일치해야 valid_mask를 적용할 수 있다
            if len(sp_features_all) != len(valid_mask):
                print(f'\n  [오류] SP 수 불일치 — JSON 출력 불가')
                print(f'    floor_npz : {floor_npz_path.name}  →  {len(sp_features_all):,} SP')
                print(f'    l1_npz    : {Path(args.l1_npz).name}  →  {len(valid_mask):,} SP')
                print(f'    두 파일이 같은 층 데이터인지 확인하세요.')
                print(f'    해결: python run_pipeline.py --src {floor_npz_path.name} --only 1')
                raise SystemExit(1)

            # 구조체 SP 제거 후 valid SP 기준 feature 사용 (sp_l1과 인덱스 일치)
            sp_features_valid = sp_features_all[valid_mask]
            l2_labels         = np.load(l2_npy_path).astype(np.int32)

            # ── floor_l2_geometry.json 로드 (stage4 저장 방 경계) ────────────
            room_geometry = None
            l2_geo_path   = Path(args.l2_geometry)
            if l2_geo_path.exists():
                try:
                    import json as _json
                    with open(l2_geo_path, encoding='utf-8') as _f:
                        _geo = _json.load(_f)
                    room_geometry = _geo.get('rooms', [])
                    print(f'  방 경계 geometry 로드: {len(room_geometry)}개 방 ({l2_geo_path.name})')
                except Exception as _e:
                    print(f'  [경고] l2_geometry 로드 실패: {_e}  → centroid convex hull 사용')
            else:
                print(f'  [참고] l2_geometry 없음 ({l2_geo_path.name})  → centroid convex hull 사용')

            out_fname = args.out_name if args.out_name else 'scene_graph.json'
            json_out  = _ROOT / 'results' / out_fname
            json_out.parent.mkdir(parents=True, exist_ok=True)
            export_scene_graph_json(
                hssg          = hssg,
                l1_labels_sp  = sp_l1,              # 재매핑된 0-based L1 id
                sp_features   = sp_features_valid,  # 구조체 제외 SP feature
                edge_src      = l1_es_intra,        # 방 내부 엣지만 (cross-room 제거)
                edge_dst      = l1_ed_intra,
                l2_labels     = l2_labels,
                text_labels   = text_labels,
                out_path      = str(json_out),
                building_id   = args.building_id,
                room_geometry = room_geometry,
            )
