"""
stage3_multiview.py — 2D MultiView CLIP 기반 Stage 3 대체
=========================================================
라벨 전략:
  structural (0-4)  → SPT 라벨, JSON 제외
  class 4-11        → SPT 라벨 직접 사용
  class 12 (clutter) → 2D CLIP으로 후보 비교
                        argmax 유사도 < clutter_sim_threshold(0.5)
                        → OpenShape 3D도 계산 후 더 높은 쪽 채택

sem_feat:
  모든 객체: 2D MultiView CLIP 임베딩 (기본)
  clutter + 3D 채택 시: 해당 객체만 3D 임베딩으로 교체

출력 (stage3_openshape.py 호환):
  stage3_results/f_l1_floor.npy        [N_l1, 651]
  stage3_results/l1_labels_text.json
  stage3_results/multiview_stats.json  (추가)

실행:
  python stage3_multiview.py \\
      --floor_npz Area_5_spt/floor.npz \\
      --l1_npz model/merge/stage2_results/l1_5_floor.npz
"""

import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import argparse
from collections import Counter

# ── 경로 등록 ─────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _multiview_render import render_views, VIEWS_4, VIEWS_8

# stage3_openshape.py 에서 상수·유틸만 임포트 (파일 수정 없음)
from stage3_openshape import (
    D_GEO, D_SEM, N_SAMPLE, CLUTTER_CLASS, CLIP_THRESHOLD,
    STRUCTURAL_CLASSES, S3DIS_LABELS, CLUTTER_CANDIDATES, _LVIS_CACHE,
    load_lvis_candidates, CLIPTextEncoder,
    aggregate_geo_feat, get_l1_classes,
)

# clutter argmax 유사도가 이 값 미만이면 3D도 계산해 비교
CLUTTER_SIM_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 준비
# ─────────────────────────────────────────────────────────────────────────────
def get_raw_pts_all(coord, color, sp_labels, l1_labels) -> list:
    """
    L1 객체별 원본 포인트 전체 반환 (샘플링 없음).
    각 객체 XYZ: center+scale 정규화, RGB: [0,1].
    returns: List[np.ndarray [N_i, 6]]
    """
    color_n  = color / 255.0 if color.max() > 2.0 else color
    xyz_rgb  = np.concatenate([coord, color_n], axis=1).astype(np.float32)
    point_l1 = l1_labels[sp_labels]
    n_l1     = int(l1_labels.max()) + 1
    result   = []
    for lid in range(n_l1):
        pts = xyz_rgb[point_l1 == lid].copy()
        if len(pts) > 0:
            c = pts[:, :3].mean(0);  pts[:, :3] -= c
            s = np.abs(pts[:, :3]).max() + 1e-6;  pts[:, :3] /= s
        result.append(pts)
    return result


def apply_rgb_norm(pts: np.ndarray) -> np.ndarray:
    """Per-channel min-max stretch: RGB 채널(3:6) → [0, 1]"""
    pts = pts.copy()
    for c in range(3, 6):
        mn, mx = pts[:, c].min(), pts[:, c].max()
        if mx > mn:
            pts[:, c] = (pts[:, c] - mn) / (mx - mn)
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# 2D MultiView CLIP 인코더
# ─────────────────────────────────────────────────────────────────────────────
class MultiViewCLIPEncoder:
    """open_clip ViT-B/32 → 8-뷰 평균 → L2 정규화 → [512]"""

    def __init__(self, device: str = 'cpu'):
        try:
            import open_clip
        except ImportError:
            raise ImportError("pip install open_clip_torch")
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai')
        self.model      = model.to(device).eval()
        self.preprocess = preprocess
        print("  [2D-CLIP] ViT-B/32 (openai) 로드 완료")

    @torch.no_grad()
    def encode_one(self, pts: np.ndarray, views: list,
                   rgb_norm: bool = True) -> np.ndarray:
        """pts: [N, 6] → [512]"""
        if rgb_norm:
            pts = apply_rgb_norm(pts)
        images  = render_views(pts, views, image_size=224)
        tensors = torch.stack([
            self.preprocess(Image.fromarray(img)) for img in images
        ]).to(self.device)
        embs   = self.model.encode_image(tensors)
        embs   = F.normalize(embs, dim=-1)
        return F.normalize(embs.mean(0), dim=-1).cpu().numpy().astype(np.float32)

    def encode_all(self, pcd_all: list, views: list,
                   rgb_norm: bool = True) -> np.ndarray:
        """모든 L1 객체 인코딩 → [N_l1, 512]"""
        n      = len(pcd_all)
        result = np.zeros((n, D_SEM), dtype=np.float32)
        for i, pts in enumerate(pcd_all):
            result[i] = self.encode_one(pts, views, rgb_norm)
            if (i + 1) % 100 == 0 or i == n - 1:
                print(f"    2D encode: {i+1}/{n}", end='\r', flush=True)
        print()
        return result


# ─────────────────────────────────────────────────────────────────────────────
# OpenShape 3D 인코더 (clutter fallback 전용, 지연 로드)
# ─────────────────────────────────────────────────────────────────────────────
_os_encoder = None


def _get_os_encoder(openshape_key: str, device: str):
    global _os_encoder
    if _os_encoder is None:
        print("  [3D-Fallback] OpenShape 인코더 로드 중...")
        from stage3_openshape import SemanticEncoder
        _os_encoder = SemanticEncoder(openshape_key, device).to(device).eval()
    return _os_encoder


def encode_3d_one(pts: np.ndarray, openshape_key: str,
                  device: str) -> np.ndarray:
    """단일 객체 OpenShape 3D 인코딩 → [512]"""
    model = _get_os_encoder(openshape_key, device)
    n     = len(pts)
    idx   = np.random.choice(n, N_SAMPLE, replace=n < N_SAMPLE)
    t     = torch.from_numpy(pts[idx][None]).to(device)   # [1, N_SAMPLE, 6]
    with torch.no_grad():
        out = model(t)   # [1, 512]
    return out[0].cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    _ROOT = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument('--floor_npz',
                        default=str(_ROOT / 'src' / 'floor.npz'))
    parser.add_argument('--l1_npz',
                        default=str(_ROOT / 'model' / 'merge' / 'stage2_results' / 'l1_floor.npz'))
    parser.add_argument('--out_dir',
                        default=str(_THIS_DIR / 'stage3_results'))
    parser.add_argument('--openshape', default='vitb32')
    parser.add_argument('--clutter_sim_threshold', type=float,
                        default=CLUTTER_SIM_THRESHOLD,
                        help='clutter 2D argmax 유사도가 이 값 미만이면 3D도 계산 (기본 0.5)')
    parser.add_argument('--clip_threshold', type=float, default=CLIP_THRESHOLD,
                        help=f'최종 라벨 확정 임계값 (기본 {CLIP_THRESHOLD})')
    parser.add_argument('--rgb_norm',    action='store_true', default=True)
    parser.add_argument('--no_rgb_norm', dest='rgb_norm', action='store_false')
    parser.add_argument('--n_views',     type=int, default=8, choices=[4, 8])
    parser.add_argument('--use_lvis',    action='store_true', default=False)
    parser.add_argument('--no_lvis',     dest='use_lvis', action='store_false')
    parser.add_argument('--top_k',       type=int, default=3)
    args = parser.parse_args()

    FLOOR_NPZ = Path(args.floor_npz)
    L1_NPZ    = Path(args.l1_npz)
    OUT_DIR   = Path(args.out_dir);  OUT_DIR.mkdir(exist_ok=True, parents=True)

    if torch.backends.mps.is_available():   device = 'mps'
    elif torch.cuda.is_available():          device = 'cuda'
    else:                                    device = 'cpu'

    print(f"[Stage 3 MultiView] 디바이스: {device}")
    print(f"  clutter_sim_threshold={args.clutter_sim_threshold}  "
          f"clip_threshold={args.clip_threshold}  "
          f"rgb_norm={args.rgb_norm}  n_views={args.n_views}")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    if not FLOOR_NPZ.exists(): raise FileNotFoundError(f"없음: {FLOOR_NPZ}")
    if not L1_NPZ.exists():    raise FileNotFoundError(f"없음: {L1_NPZ}")

    floor       = np.load(FLOOR_NPZ)
    sp_features = floor['sp_features'].astype(np.float32)
    sp_pred     = floor['sp_pred'].astype(np.int32)
    coord       = floor['coord'].astype(np.float32)
    color       = floor['color'].astype(np.float32)
    sp_labels   = floor['sp_labels'].astype(np.int64)
    l1_labels   = np.load(L1_NPZ)['l1_labels'].astype(np.int64)

    # 구조체(-1) 제거 + ID 재매핑 (stage3_openshape.py 동일 전처리)
    valid_mask  = l1_labels >= 0
    sp_features = sp_features[valid_mask]
    sp_pred     = sp_pred[valid_mask]
    l1_labels   = l1_labels[valid_mask]

    old_ids  = np.unique(l1_labels)
    id_remap = np.full(int(old_ids.max()) + 1, -1, dtype=np.int64)
    id_remap[old_ids] = np.arange(len(old_ids))
    l1_labels = id_remap[l1_labels]

    old_sp_idx = np.where(valid_mask)[0]
    sp_idx_map = np.full(len(valid_mask), -1, dtype=np.int64)
    sp_idx_map[old_sp_idx] = np.arange(len(old_sp_idx))
    pt_valid   = sp_idx_map[sp_labels] >= 0
    sp_labels  = sp_idx_map[sp_labels[pt_valid]]
    coord      = coord[pt_valid];  color = color[pt_valid]

    n_l1       = int(l1_labels.max()) + 1
    l1_classes = get_l1_classes(l1_labels, sp_pred)

    clutter_ids   = [i for i in range(n_l1) if l1_classes[i] == CLUTTER_CLASS]
    struct_ids    = [i for i in range(n_l1) if l1_classes[i] in STRUCTURAL_CLASSES]
    nonclutter_ids= [i for i in range(n_l1)
                     if l1_classes[i] not in STRUCTURAL_CLASSES
                     and l1_classes[i] != CLUTTER_CLASS]

    print(f"  L1: {n_l1}개  |  structural: {len(struct_ids)}  "
          f"non-clutter: {len(nonclutter_ids)}  clutter: {len(clutter_ids)}")

    # ── [1] 포인트클라우드 로드 (샘플링 없음) ────────────────────────────────
    print(f"\n[1] 포인트클라우드 로드 (원본 그대로, 샘플링 없음)...")
    pcd_all   = get_raw_pts_all(coord, color, sp_labels, l1_labels)
    pt_counts = [len(p) for p in pcd_all]
    print(f"  pt/객체: min={min(pt_counts)}  median={int(np.median(pt_counts))}  "
          f"max={max(pt_counts)}")

    # ── [2] clutter 893개: 2D MultiView CLIP 전부 먼저 ───────────────────────
    print(f"\n[2] clutter({len(clutter_ids)}개) 2D MultiView CLIP 인코딩...")
    mv_enc = MultiViewCLIPEncoder(device=device)
    views  = VIEWS_8 if args.n_views == 8 else VIEWS_4

    text_enc = CLIPTextEncoder(openshape_key='vitb32', device=device)
    if args.use_lvis:
        candidates = load_lvis_candidates()
    else:
        candidates = CLUTTER_CANDIDATES
    print(f"  후보: {len(candidates)}개")
    cand_feats = text_enc.encode_texts(candidates)   # [C, 512]

    # clutter별 2D 임베딩 + 유사도 저장
    emb_2d_map  = {}   # lid → [512]
    sims_2d_map = {}   # lid → [C]

    for i, lid in enumerate(clutter_ids):
        emb = mv_enc.encode_one(pcd_all[lid], views, rgb_norm=args.rgb_norm)
        emb_2d_map[lid]  = emb
        sims_2d_map[lid] = emb @ cand_feats.T
        if (i + 1) % 50 == 0 or i == len(clutter_ids) - 1:
            print(f"    2D encode: {i+1}/{len(clutter_ids)}", end='\r', flush=True)
    print()

    # argmax < threshold → 3D 폴백 후보
    clutter_fallback_ids = [lid for lid in clutter_ids
                             if float(sims_2d_map[lid].max()) < args.clutter_sim_threshold]
    print(f"  2D argmax < {args.clutter_sim_threshold}: {len(clutter_fallback_ids)}개 → 3D 추가 필요")

    # ── [3] OpenShape 3D: non-clutter + clutter_fallback 한 번에 배치 ─────────
    non_clutter_ids  = [i for i in range(n_l1) if l1_classes[i] != CLUTTER_CLASS]
    ids_for_3d       = non_clutter_ids + clutter_fallback_ids
    print(f"\n[3] OpenShape 3D 배치 인코딩 ({len(ids_for_3d)}개 = "
          f"non-clutter {len(non_clutter_ids)} + clutter_fallback {len(clutter_fallback_ids)})...")

    emb_3d_map = {}   # lid → [512]
    if ids_for_3d:
        BATCH    = 32
        model_3d = _get_os_encoder(args.openshape, device)
        for b_start in range(0, len(ids_for_3d), BATCH):
            batch_ids = ids_for_3d[b_start:b_start + BATCH]
            batch_pts = []
            for fid in batch_ids:
                pts = pcd_all[fid]
                n   = len(pts)
                idx = np.random.choice(n, N_SAMPLE, replace=n < N_SAMPLE)
                batch_pts.append(pts[idx])
            t = torch.from_numpy(np.stack(batch_pts)).to(device)
            with torch.no_grad():
                out = model_3d(t)
            for k, fid in enumerate(batch_ids):
                emb_3d_map[fid] = out[k].cpu().numpy().astype(np.float32)
            done = min(b_start + BATCH, len(ids_for_3d))
            print(f"    3D encode: {done}/{len(ids_for_3d)}", end='\r', flush=True)
        print()

    # ── [4] sem_feat 조립 + clutter 분류 결과 결정 ───────────────────────────
    sem_feat        = np.zeros((n_l1, D_SEM), dtype=np.float32)
    clutter_results = {}
    n_3d_used       = 0

    # non-clutter: 3D 임베딩
    for lid in non_clutter_ids:
        sem_feat[lid] = emb_3d_map[lid]

    # clutter: 2D vs 3D 비교 (fallback만), 나머지는 2D
    for lid in clutter_ids:
        emb_2d  = emb_2d_map[lid]
        sims_2d = sims_2d_map[lid]
        top_idx = np.argsort(-sims_2d)[:args.top_k]
        best_2d = float(sims_2d[top_idx[0]])
        top_2d  = [(candidates[j], float(sims_2d[j])) for j in top_idx]

        if lid not in emb_3d_map:
            # 2D 충분 (score >= threshold)
            sem_feat[lid] = emb_2d
            clutter_results[lid] = {'best_score': best_2d, 'top': top_2d, 'source': '2d_clip'}
        else:
            # 2D vs 3D 비교
            emb_3d  = emb_3d_map[lid]
            sims_3d = emb_3d @ cand_feats.T
            top_idx3 = np.argsort(-sims_3d)[:args.top_k]
            best_3d  = float(sims_3d[top_idx3[0]])
            top_3d   = [(candidates[j], float(sims_3d[j])) for j in top_idx3]

            if best_3d > best_2d:
                sem_feat[lid] = emb_3d
                clutter_results[lid] = {'best_score': best_3d, 'top': top_3d, 'source': '3d_openshape'}
                n_3d_used += 1
            else:
                sem_feat[lid] = emb_2d
                clutter_results[lid] = {'best_score': best_2d, 'top': top_2d, 'source': '2d_clip'}

    print(f"  clutter 3D 채택: {n_3d_used} / {len(clutter_ids)}개")

    # ── [4] 전체 라벨 결과 조립 ──────────────────────────────────────────────
    results = []
    for lid in range(n_l1):
        cls = int(l1_classes[lid])

        if cls in STRUCTURAL_CLASSES:
            results.append({
                'l1_id':  lid,
                'label':  S3DIS_LABELS[cls],
                'source': 'structural',
                'score':  None,
                'top3':   None,
            })
            continue

        if cls != CLUTTER_CLASS:
            results.append({
                'l1_id':  lid,
                'label':  S3DIS_LABELS[cls],
                'source': 'spt',
                'score':  None,
                'top3':   None,
            })
            continue

        # clutter
        cr    = clutter_results[lid]
        best  = cr['best_score']
        top   = cr['top']
        results.append({
            'l1_id':  lid,
            'label':  top[0][0] if best > args.clip_threshold else 'unknown',
            'source': cr['source'] if best > args.clip_threshold else 'unknown',
            'score':  best,
            'top3':   top,
        })

    # ── [5] Feature 합성 ─────────────────────────────────────────────────────
    geo_feat = aggregate_geo_feat(sp_features, l1_labels)
    f_l1     = np.concatenate([geo_feat, sem_feat], axis=1)   # [N_l1, 651]

    feat_path = OUT_DIR / 'f_l1_floor.npy'
    np.save(feat_path, f_l1)
    print(f"\n  feature 저장: {feat_path}  {f_l1.shape}")

    # ── 출력 ─────────────────────────────────────────────────────────────────
    src_counts   = Counter(r['source'] for r in results)
    n_structural = src_counts.get('structural', 0)

    print(f"\n  ── 출처 통계 ──")
    print(f"  SPT 라벨:      {src_counts.get('spt',          0):>5}개")
    print(f"  2D CLIP:       {src_counts.get('2d_clip',      0):>5}개")
    print(f"  3D OpenShape:  {src_counts.get('3d_openshape', 0):>5}개")
    print(f"  unknown:       {src_counts.get('unknown',      0):>5}개")
    print(f"  structural:    {n_structural:>5}개  ← JSON 제외")
    print(f"\n  sem_feat: 2D CLIP {n_l1-n_3d_used}개  /  3D OpenShape {n_3d_used}개")

    objects_for_json = [r for r in results if r['source'] != 'structural']
    obj_lbl_counts   = Counter(r['label'] for r in objects_for_json)
    if obj_lbl_counts:
        print(f"\n  ── 상위 라벨 분포 ──")
        for lbl, cnt in obj_lbl_counts.most_common(10):
            bar = '█' * int(cnt / max(obj_lbl_counts.values()) * 25)
            print(f"  {lbl:<30} {cnt:>4}개  {bar}")

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    json_path  = OUT_DIR / 'l1_labels_text.json'
    stats_path = OUT_DIR / 'multiview_stats.json'

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'n_l1':         n_l1,
            'n_objects':    len(objects_for_json),
            'n_structural': n_structural,
            'threshold':    args.clip_threshold,
            'candidates':   candidates,
            'objects':      objects_for_json,
        }, f, ensure_ascii=False, indent=2)

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'mode':                 'stage3_multiview',
            'n_l1':                 n_l1,
            'n_clutter':            len(clutter_ids),
            'n_clutter_2d':         len(clutter_ids) - n_3d_used,
            'n_clutter_3d':         n_3d_used,
            'clutter_sim_threshold': args.clutter_sim_threshold,
            'clip_threshold':       args.clip_threshold,
            'rgb_norm':             args.rgb_norm,
            'n_views':              args.n_views,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[Stage 3 MultiView] ✓ 완료")
    print(f"  feature : {feat_path}  {f_l1.shape}")
    print(f"  라벨    : {json_path}")
    print(f"  통계    : {stats_path}")
