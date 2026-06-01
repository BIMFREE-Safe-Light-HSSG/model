"""
stage2_s1.py
============
Stage 2: S1 → S2  Jaccard Affinity 기반 병합 (training-free)

백본 논문의 SAM affinity 개념을 3D에 적용:
  SAM affinity (2D)  →  SP 이웃 공유 Jaccard (3D)
  "같은 마스크에 자주 등장"  →  "SP 레벨 이웃 집합이 많이 겹침"

파이프라인:
  1. Stage1 → s1_labels [N_sp], s1_feat [N_s1, feat_dim]
  2. S1 KNN 그래프 생성 (centroid 기반)
  3. Jaccard affinity 계산 (SP 원본 엣지 활용, training-free)
  4. jaccard >= JACCARD_THRESH 인 엣지만 유지
  5. Connected Components → S2

클래스별 eps 하드코딩 없음. 단일 임계값 JACCARD_THRESH만 사용.

사용법:
  python stage2_s1.py --npz ../../src/6_floor.npz
  python stage2_s1.py --npz ../../src/6_floor.npz --visualize
  python stage2_s1.py --npz ../../src/6_floor.npz --thresh 0.25 --out ../../src/6_floor_s2.npz
"""

import argparse, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).parent))
from stage1_merge import merge_stage1

# ── 피처 인덱스 규약 (stage1_merge.py와 동일) ─────────────────
IDX_CENTROID = slice(0, 3)
IDX_NORMAL   = slice(3, 6)
IDX_COLOR    = slice(7, 10)
IDX_SIZE     = 10

CLASS_NAMES = {
    0:'ceiling', 1:'floor',    2:'wall',     3:'beam',
    4:'column',  5:'window',   6:'door',     7:'table',
    8:'chair',   9:'sofa',    10:'bookcase', 11:'board', 12:'clutter'
}

# ── 클래스 구분 ───────────────────────────────────────────────
AUTO_MERGE_CLASSES = {0, 1, 2, 3}           # ceiling/floor/wall/beam — Stage1에서 이미 처리
MERGE_CLASSES      = {4, 5, 6, 7, 8, 9, 10, 11, 12}  # Jaccard 병합 대상

# ── Jaccard 파라미터 ──────────────────────────────────────────
JACCARD_THRESH  = 0.005   # 이 값 이상이면 병합 (단일 임계값, 클래스 무관)
S1_KNN_K        = 8       # S1 centroid KNN 이웃 수
S1_KNN_MAX_DIST = 2.0     # KNN 최대 거리 (m) — 이 안에서 Jaccard 계산

# ── 클래스별 거리 보너스 ──────────────────────────────────────
# score = jaccard + DIST_BONUS[cls] × (1 - dist/max_dist)
# 가까울수록 보너스 ↑ → threshold 넘기 쉬워짐
CLASS_DIST_BONUS = {
    6: 0.02,   # door   — 가까운 문짝은 더 적극적으로 병합
    5: 0.05,   # window — 가까운 창문도 적극적으로 병합
}

# ── BBox 컴팩트니스 필터 ──────────────────────────────────────
# 두 S1을 병합했을 때 채움 비율 = (vol_i + vol_j) / vol_merged
# 이 값이 낮으면 병합 후 빈 공간이 많음 → 다른 객체일 가능성 높음
# 0.0 으로 설정하면 필터 비활성화
COMPACT_THRESH = 0.0    # 현재 비활성화

# ── room_grid 경로 (wall_door_2d.py 출력) ────────────────────
# npz stem 기준으로 자동 탐색: {프로젝트루트}/room_grid_{stem}.npz
_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # cnu_model/


# ══════════════════════════════════════════════════════════════
#  0-A. S1 Bounding Box 계산
# ══════════════════════════════════════════════════════════════

def compute_s1_bbox(coord:        np.ndarray,
                    sp_labels_pt: np.ndarray,
                    s1_labels:    np.ndarray,
                    N_s1:         int,
                    verbose:      bool = True) -> np.ndarray:
    """
    각 S1의 축 정렬 Bounding Box를 계산한다.

    Returns
    -------
    s1_bbox : [N_s1, 6]  float32
              columns: [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    t0 = time.time()
    s1_ids_pt = s1_labels[sp_labels_pt]   # [N_pts] — 점별 S1 ID

    # 정렬 기반 그룹 min/max (np.minimum.at 보다 빠름)
    order      = np.argsort(s1_ids_pt, kind='stable')
    sorted_ids = s1_ids_pt[order]
    sorted_xyz = coord[order]              # [N_pts, 3]

    # 각 S1 시작 인덱스
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    starts = np.concatenate([[0], boundaries])
    ends   = np.concatenate([boundaries, [len(sorted_ids)]])

    s1_bbox = np.zeros((N_s1, 6), dtype=np.float32)
    for k, (s, e) in enumerate(zip(starts, ends)):
        pts = sorted_xyz[s:e]
        sid = sorted_ids[s]
        s1_bbox[sid, 0] = pts[:, 0].min()
        s1_bbox[sid, 1] = pts[:, 0].max()
        s1_bbox[sid, 2] = pts[:, 1].min()
        s1_bbox[sid, 3] = pts[:, 1].max()
        s1_bbox[sid, 4] = pts[:, 2].min()
        s1_bbox[sid, 5] = pts[:, 2].max()

    if verbose:
        print(f"  [bbox] S1 bbox 계산 완료  ({time.time()-t0:.2f}s)")
    return s1_bbox


# ══════════════════════════════════════════════════════════════
#  0. Room Grid 로드 + S1 room ID 조회
# ══════════════════════════════════════════════════════════════

def load_room_ids(s1_feat: np.ndarray, npz_path: Path,
                  verbose: bool = True) -> np.ndarray | None:
    """
    wall_door_2d.py가 생성한 room_grid_{stem}.npz를 로드하고
    각 S1 centroid의 room ID를 반환한다.

    Returns
    -------
    room_ids : [N_s1] int32  (0 = 벽/미배정)
    None     : room_grid 파일이 없을 경우
    """
    stem      = npz_path.stem                                        # e.g. "6_floor"
    grid_path = _PROJECT_ROOT / f'room_grid_{stem}.npz'

    if not grid_path.exists():
        if verbose:
            print(f"  [room] room_grid 없음 ({grid_path.name}) → 방 제약 미적용")
        return None

    rg        = np.load(grid_path)
    x_min     = float(rg['x_min'])
    y_min     = float(rg['y_min'])
    cell_size = float(rg['cell_size'])

    centroids = s1_feat[:, IDX_CENTROID]   # [N_s1, 3]

    if 'masks' in rg:
        # 신형 포맷 (sam_room_grid.py): masks [N, H, W] uint8
        masks     = rg['masks'].astype(bool)       # [N, H, W]
        N, H, W   = masks.shape
        wx = np.clip(((centroids[:, 0] - x_min) / cell_size).astype(int), 0, W - 1)
        wy = np.clip(((centroids[:, 1] - y_min) / cell_size).astype(int), 0, H - 1)
        # 각 S1 centroid가 포함되는 마스크 중 면적 최소 → room_id (1-indexed)
        areas        = masks.sum(axis=(1, 2)).astype(np.float32)     # [N]
        centroid_hits = masks[:, wy, wx]                              # [N, N_s1]
        area_mat     = np.where(centroid_hits, areas[:, None], np.inf)
        best_mask    = area_mat.argmin(axis=0)                        # [N_s1]
        hit_any      = centroid_hits.any(axis=0)                      # [N_s1]
        room_ids     = np.where(hit_any, best_mask + 1, 0).astype(np.int32)  # 1-indexed, 0=미배정
        if verbose:
            assigned = int(hit_any.sum())
            print(f"  [room] {grid_path.name} 로드 (SAM masks)  방={N}개  "
                  f"S1 배정={assigned:,}/{len(room_ids):,}개")
    else:
        # 구형 포맷 (wall_door_2d.py): grid [H, W] int32
        grid  = rg['grid'].astype(np.int32)
        H, W  = grid.shape
        wx = np.clip(((centroids[:, 0] - x_min) / cell_size).astype(int), 0, W - 1)
        wy = np.clip(((centroids[:, 1] - y_min) / cell_size).astype(int), 0, H - 1)
        room_ids = grid[wy, wx].astype(np.int32)
        if verbose:
            n_rooms  = int(grid.max())
            assigned = int((room_ids > 0).sum())
            print(f"  [room] {grid_path.name} 로드  방={n_rooms}개  "
                  f"S1 배정={assigned:,}/{len(room_ids):,}개")

    return room_ids


# ══════════════════════════════════════════════════════════════
#  1. S1 클래스 추출 (SP majority vote)
# ══════════════════════════════════════════════════════════════

def compute_s1_pred(s1_labels: np.ndarray,
                    sp_pred:   np.ndarray) -> np.ndarray:
    """SP 클래스 majority vote → S1 대표 클래스 [N_s1]"""
    N_s1    = int(s1_labels.max()) + 1   # 소프트코딩
    s1_pred = np.zeros(N_s1, dtype=np.int32)
    for s1_id in range(N_s1):
        mask = s1_labels == s1_id
        if mask.any():
            cls, cnt = np.unique(sp_pred[mask], return_counts=True)
            s1_pred[s1_id] = cls[cnt.argmax()]
    return s1_pred


# ══════════════════════════════════════════════════════════════
#  2. S1 레벨 KNN 엣지 생성
# ══════════════════════════════════════════════════════════════

def build_s1_edges(s1_feat:       np.ndarray,
                   s1_pred:       np.ndarray,
                   target_classes: set,
                   k:             int   = S1_KNN_K,
                   max_dist:      float = S1_KNN_MAX_DIST,
                   verbose:       bool  = True):
    """
    S1 centroid 기반 KNN 엣지 (target_classes 내 same-class 쌍만).

    Returns: src [E], dst [E]
    """
    N_s1      = s1_feat.shape[0]           # 소프트코딩
    centroids = s1_feat[:, IDX_CENTROID]   # [N_s1, 3]

    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=min(k + 1, N_s1), algorithm='kd_tree')
    nn.fit(centroids)
    dists_all, nbrs_all = nn.kneighbors(centroids)

    src_list, dst_list, dist_list = [], [], []
    for i in range(N_s1):
        ci = int(s1_pred[i])
        if ci not in target_classes:
            continue
        for d, j in zip(dists_all[i, 1:], nbrs_all[i, 1:]):
            if d > max_dist:
                break
            if int(s1_pred[j]) != ci:      # 같은 클래스끼리만
                continue
            src_list.append(i)
            dst_list.append(j)
            dist_list.append(float(d))

    src   = np.array(src_list,  dtype=np.int32)
    dst   = np.array(dst_list,  dtype=np.int32)
    dists = np.array(dist_list, dtype=np.float32)

    if verbose:
        print(f"  S1 KNN 엣지: {len(src):,}개  "
              f"k={k}  max_dist={max_dist}m  ({time.time()-t0:.2f}s)")
    return src, dst, dists


# ══════════════════════════════════════════════════════════════
#  3. Jaccard Affinity 계산
# ══════════════════════════════════════════════════════════════

def compute_jaccard_affinity(s1_labels:   np.ndarray,
                              src:         np.ndarray,
                              dst:         np.ndarray,
                              sp_edge_src: np.ndarray,
                              sp_edge_dst: np.ndarray,
                              verbose:     bool = True) -> np.ndarray:
    """
    백본 논문 SAM affinity의 3D 대응:
      SP 이웃 집합 Jaccard 겹침률

    N(i) = S1 i에 속한 SP들의 이웃 SP 집합
    A(i,j) = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|

    Returns: jaccard [E]  float32, 0~1
    """
    # SP 레벨 이웃 집합 구성
    sp_nbrs: dict = defaultdict(set)
    for s, d in zip(sp_edge_src, sp_edge_dst):
        sp_nbrs[int(s)].add(int(d))
        sp_nbrs[int(d)].add(int(s))

    # S1별 소속 SP
    N_s1   = int(s1_labels.max()) + 1   # 소프트코딩
    s1_sps: dict = defaultdict(set)
    for sp_id, s1_id in enumerate(s1_labels):
        s1_sps[int(s1_id)].add(sp_id)

    # S1별 외부 이웃 SP 집합 (자신 소속 SP 제외)
    s1_nbr: dict = {}
    for s1_id, sps in s1_sps.items():
        nbrs: set = set()
        for sp in sps:
            nbrs |= sp_nbrs[sp]
        s1_nbr[s1_id] = nbrs - sps

    # 엣지별 Jaccard
    E = len(src)
    jaccard = np.zeros(E, dtype=np.float32)
    for e, (i, j) in enumerate(zip(src, dst)):
        ni = s1_nbr.get(int(i), set())
        nj = s1_nbr.get(int(j), set())
        union = len(ni | nj)
        jaccard[e] = len(ni & nj) / union if union > 0 else 0.0

    if verbose:
        print(f"  Jaccard affinity: "
              f"mean={jaccard.mean():.4f}  max={jaccard.max():.4f}  "
              f"≥{JACCARD_THRESH}: {int((jaccard >= JACCARD_THRESH).sum()):,}/{E:,}개")

    return jaccard


# ══════════════════════════════════════════════════════════════
#  4. Jaccard 병합 → S2
# ══════════════════════════════════════════════════════════════

def merge_by_jaccard(s1_feat:       np.ndarray,
                     s1_pred:       np.ndarray,
                     src:           np.ndarray,
                     dst:           np.ndarray,
                     jaccard:       np.ndarray,
                     edge_dists:    np.ndarray,
                     thresh:        float              = JACCARD_THRESH,
                     max_dist:      float              = S1_KNN_MAX_DIST,
                     room_ids:      np.ndarray | None  = None,
                     s1_bbox:       np.ndarray | None  = None,
                     compact_thresh: float             = COMPACT_THRESH,
                     verbose:       bool               = True) -> tuple[np.ndarray, np.ndarray]:
    """
    jaccard >= thresh 인 엣지로 Connected Components → S2.
    room_ids 가 주어지면 같은 방 안에서만 병합 허용.
      - room_id == 0 (벽/미배정) S1은 방 제약 면제

    Returns
    -------
    s2_labels : [N_s1]            S1별 S2 ID
    s2_feat   : [N_s2, feat_dim]  S2 집계 피처
    """
    N_s1     = s1_feat.shape[0]   # 소프트코딩
    feat_dim = s1_feat.shape[1]   # 소프트코딩

    # ── 1단계: AUTO_MERGE 클래스는 각자 독립 ID ──────────────
    s2_labels = np.arange(N_s1, dtype=np.int32)

    # ── 2단계: 클래스별 거리 보너스 적용 후 임계값 필터 ─────
    scores = jaccard.copy()
    if CLASS_DIST_BONUS:
        for e, (i, j) in enumerate(zip(src, dst)):
            cls = int(s1_pred[i])
            if cls in CLASS_DIST_BONUS:
                bonus = CLASS_DIST_BONUS[cls] * max(0.0, 1.0 - edge_dists[e] / max_dist)
                scores[e] += bonus

    merge_mask = scores >= thresh
    src_m = src[merge_mask]
    dst_m = dst[merge_mask]

    # ── 3단계: 같은 방 제약 ───────────────────────────────────
    if room_ids is not None and len(src_m) > 0:
        ri = room_ids[src_m]
        rj = room_ids[dst_m]
        # 둘 다 배정된 방(>0)이면 같은 방이어야만 병합
        # 어느 한쪽이 0(벽/미배정)이면 제약 면제
        same_room = (ri == rj) | (ri == 0) | (rj == 0)
        blocked   = int((~same_room).sum())
        src_m = src_m[same_room]
        dst_m = dst_m[same_room]
        if verbose:
            print(f"  방 제약: {blocked:,}개 크로스-룸 엣지 차단  "
                  f"→ 병합 엣지 {len(src_m):,}개")

    if verbose:
        print(f"  병합 엣지: {len(src_m):,}/{len(src):,}개  "
              f"(jaccard ≥ {thresh})")

    # ── 3.5단계: BBox 컴팩트니스 필터 ────────────────────────
    if s1_bbox is not None and compact_thresh > 0.0 and len(src_m) > 0:
        bi = s1_bbox[src_m]   # [E, 6]
        bj = s1_bbox[dst_m]   # [E, 6]

        # 각 S1 bbox 볼륨
        vol_i = (np.maximum(0.0, bi[:, 1] - bi[:, 0]) *
                 np.maximum(0.0, bi[:, 3] - bi[:, 2]) *
                 np.maximum(0.0, bi[:, 5] - bi[:, 4]))
        vol_j = (np.maximum(0.0, bj[:, 1] - bj[:, 0]) *
                 np.maximum(0.0, bj[:, 3] - bj[:, 2]) *
                 np.maximum(0.0, bj[:, 5] - bj[:, 4]))

        # 병합 bbox 볼륨
        merged_min = np.minimum(bi[:, 0::2], bj[:, 0::2])   # [E, 3] xmin,ymin,zmin
        merged_max = np.maximum(bi[:, 1::2], bj[:, 1::2])   # [E, 3] xmax,ymax,zmax
        vol_m = np.prod(np.maximum(0.0, merged_max - merged_min), axis=1)

        # compactness = 두 박스 볼륨 합 / 병합 박스 볼륨
        comp = np.where(vol_m > 1e-8, (vol_i + vol_j) / vol_m, 1.0)
        compact_mask   = comp >= compact_thresh
        blocked_compact = int((~compact_mask).sum())
        src_m = src_m[compact_mask]
        dst_m = dst_m[compact_mask]
        if verbose:
            print(f"  컴팩트니스 필터(≥{compact_thresh:.2f}): "
                  f"{blocked_compact:,}개 차단  → 최종 병합 엣지 {len(src_m):,}개")

    # ── 4단계: Connected Components ───────────────────────────
    if len(src_m) > 0:
        rows = np.concatenate([src_m, dst_m])
        cols = np.concatenate([dst_m, src_m])
        adj  = csr_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(N_s1, N_s1))
        _, cc = connected_components(adj, directed=False)

        # MERGE_CLASSES에 해당하는 S1만 CC 레이블 적용
        for i in range(N_s1):
            if int(s1_pred[i]) in MERGE_CLASSES:
                s2_labels[i] = cc[i] + N_s1   # offset으로 AUTO_MERGE와 충돌 방지

    # ── 5단계: 레이블 연속화 ─────────────────────────────────
    _, s2_labels = np.unique(s2_labels, return_inverse=True)
    s2_labels = s2_labels.astype(np.int32)
    N_s2 = int(s2_labels.max()) + 1   # 소프트코딩

    # ── 6단계: S2 피처 집계 (S1 size 가중 평균) ──────────────
    sizes   = s1_feat[:, IDX_SIZE]
    s2_feat = np.zeros((N_s2, feat_dim), dtype=np.float32)
    w_sum   = np.zeros(N_s2, dtype=np.float32)
    for s1_id in range(N_s1):
        s2_id = int(s2_labels[s1_id])
        w     = max(float(sizes[s1_id]), 1e-4)
        s2_feat[s2_id] += s1_feat[s1_id] * w
        w_sum[s2_id]   += w
    valid = w_sum > 0
    s2_feat[valid] /= w_sum[valid, None]

    if verbose:
        comp_sizes = np.bincount(s2_labels)
        singleton  = int((comp_sizes == 1).sum())
        large      = int((comp_sizes >= 5).sum())
        print(f"\n[Stage2 결과]")
        print(f"  S1 → S2:  {N_s1:,} → {N_s2:,}개  "
              f"({100*(1 - N_s2/N_s1):.1f}% 감소)")
        print(f"  singleton(1개 S1): {singleton:,}개  ({100*singleton/N_s2:.1f}%)")
        print(f"  large(≥5 S1):      {large:,}개")
        print(f"  평균 S1/S2:          {N_s1/N_s2:.2f}")

    return s2_labels, s2_feat


# ══════════════════════════════════════════════════════════════
#  5. 시각화
# ══════════════════════════════════════════════════════════════

EXCLUDE_VIZ = {0, 1, 2, 3}

def build_palette(n: int, seed: int = 99) -> np.ndarray:
    import colorsys
    rng  = np.random.default_rng(seed)
    hues = rng.permutation(np.linspace(0, 1, n, endpoint=False))
    sats = rng.uniform(0.55, 0.95, n)
    vals = rng.uniform(0.60, 0.95, n)
    return np.array([colorsys.hsv_to_rgb(h, s, v)
                     for h, s, v in zip(hues, sats, vals)], dtype=np.float32)


def view_s2(pts_xyz, pts_s2, pts_cls, N_s2):
    try:
        import open3d as o3d
    except ImportError:
        print("[오류] pip install open3d"); return

    keep    = ~np.isin(pts_cls, sorted(EXCLUDE_VIZ))
    xyz_v   = pts_xyz[keep].astype(np.float64)
    s2_v    = pts_s2[keep]
    palette = build_palette(N_s2)
    colors  = palette[s2_v]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_v)
    pcd.colors = o3d.utility.Vector3dVector(
        np.clip(colors, 0, 1).astype(np.float64))

    win = (f"[S2]  S2={N_s2:,}개  thresh={JACCARD_THRESH}  "
           f"ceiling/floor/wall/beam 제외  Q=닫기")
    print(f"\n{win}\n  마우스=회전  스크롤=확대")
    o3d.visualization.draw_geometries([pcd], window_name=win,
                                      width=1600, height=900)


# ══════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--npz',     default='../../src/6_floor.npz')
    pa.add_argument('--out',     default=None,
                    help='저장 경로 (.npz). 미지정 시 저장 안 함')
    pa.add_argument('--thresh',  type=float, default=JACCARD_THRESH,
                    help=f'Jaccard 병합 임계값 (기본={JACCARD_THRESH})')
    pa.add_argument('--k',       type=int,   default=S1_KNN_K,
                    help=f'S1 KNN 이웃 수 (기본={S1_KNN_K})')
    pa.add_argument('--max_dist',     type=float, default=S1_KNN_MAX_DIST,
                    help=f'KNN 최대 거리 m (기본={S1_KNN_MAX_DIST})')
    pa.add_argument('--compact_thresh', type=float, default=COMPACT_THRESH,
                    help=f'BBox 컴팩트니스 최소값 (기본={COMPACT_THRESH}, 0=비활성화)')
    pa.add_argument('--visualize', action='store_true')
    pa.add_argument('--max_pts',   type=int, default=2_000_000)
    args = pa.parse_args()

    # ── 데이터 로드 ──────────────────────────────────────────
    print(f"[로드] {args.npz}")
    d = np.load(args.npz)

    sp_features  = d['sp_features'].astype(np.float32)   # [N_sp, feat_dim]
    sp_pred      = d['sp_pred'].astype(np.int32)          # [N_sp]
    coord        = d['coord'].astype(np.float32)          # [N_pts, 3]
    sp_labels_pt = d['sp_labels'].astype(np.int32)        # [N_pts]

    N_sp     = sp_features.shape[0]
    feat_dim = sp_features.shape[1]
    N_pts    = coord.shape[0]
    print(f"  SP={N_sp:,}  feat_dim={feat_dim}  pts={N_pts:,}")

    # ── Stage1 ────────────────────────────────────────────────
    s1_labels, s1_feat = merge_stage1(sp_features, sp_pred, verbose=True)
    N_s1     = s1_feat.shape[0]
    feat_dim = s1_feat.shape[1]
    print(f"\n  Stage1: SP {N_sp:,} → S1 {N_s1:,}  feat_dim={feat_dim}")

    # ── S1 클래스 추출 ────────────────────────────────────────
    s1_pred = compute_s1_pred(s1_labels, sp_pred)

    # ── S1 KNN 엣지 생성 ──────────────────────────────────────
    print(f"\n[S1 KNN 엣지]  k={args.k}  max_dist={args.max_dist}m")
    src, dst, edge_dists = build_s1_edges(s1_feat, s1_pred,
                               target_classes=MERGE_CLASSES,
                               k=args.k,
                               max_dist=args.max_dist,
                               verbose=True)

    # ── Jaccard Affinity 계산 ─────────────────────────────────
    sp_edge_path = Path(args.npz).parent / (Path(args.npz).stem + '_edges.npz')
    if sp_edge_path.exists():
        print(f"\n[Jaccard] SP 엣지 로드: {sp_edge_path.name}")
        sp_edges = np.load(sp_edge_path)
        jaccard  = compute_jaccard_affinity(
                       s1_labels, src, dst,
                       sp_edges['edge_src'], sp_edges['edge_dst'],
                       verbose=True)
    else:
        print(f"\n[경고] SP 엣지 파일 없음 ({sp_edge_path.name}) → Jaccard=0으로 대체")
        jaccard = np.zeros(len(src), dtype=np.float32)

    # ── S1 BBox 계산 ─────────────────────────────────────────
    print(f"\n[BBox]")
    s1_bbox = compute_s1_bbox(coord, sp_labels_pt, s1_labels, N_s1, verbose=True)

    # ── Room Grid 로드 ────────────────────────────────────────
    print(f"\n[Room 제약]")
    room_ids = load_room_ids(s1_feat, Path(args.npz), verbose=True)

    # ── Jaccard 병합 → S2 ─────────────────────────────────────
    print(f"\n[병합]  Jaccard 임계값={args.thresh}  컴팩트니스≥{args.compact_thresh}")
    s2_labels, s2_feat = merge_by_jaccard(
                             s1_feat, s1_pred, src, dst, jaccard, edge_dists,
                             thresh=args.thresh, max_dist=args.max_dist,
                             room_ids=room_ids, s1_bbox=s1_bbox,
                             compact_thresh=args.compact_thresh,
                             verbose=True)
    N_s2 = s2_feat.shape[0]

    # SP별 S2 ID
    sp_to_s2 = s2_labels[s1_labels]   # [N_sp]

    # 클래스별 S2 통계
    s2_pred = np.zeros(N_s2, dtype=np.int32)
    for s1_id in range(N_s1):
        s2_pred[int(s2_labels[s1_id])] = int(s1_pred[s1_id])
    print(f"\n  클래스별 S2 수:")
    for cid, cnt in sorted(zip(*np.unique(s2_pred, return_counts=True))):
        print(f"    {CLASS_NAMES.get(cid, str(cid)):10s}({cid}): {cnt:,}")

    # ── 저장 ─────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else \
               Path(args.npz).parent / (Path(args.npz).stem + '_s2.npz')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        s1_labels=s1_labels,   # [N_sp]
        s1_feat=s1_feat,       # [N_s1, feat_dim]
        s2_labels=s2_labels,   # [N_s1]
        s2_feat=s2_feat,       # [N_s2, feat_dim]
        sp_to_s2=sp_to_s2,    # [N_sp]
        sp_pred=sp_pred,
    )
    print(f"\n[저장] {out_path}")

    # ── 시각화 ───────────────────────────────────────────────
    if args.visualize:
        idx = (np.random.default_rng(0).choice(N_pts, args.max_pts, replace=False)
               if N_pts > args.max_pts else np.arange(N_pts))
        pts_xyz = coord[idx].astype(np.float64)
        pts_sp  = sp_labels_pt[idx]
        pts_s2  = sp_to_s2[pts_sp]
        pts_cls = sp_pred[pts_sp]
        view_s2(pts_xyz, pts_s2, pts_cls, N_s2)
