"""
stage1_merge.py
===============
Stage 1: 기하 유사도 기반 S0→S1 병합 (MLP 없음)

목적:
  같은 클래스 + 법선 유사 + 색상 유사 + 인접한 SP들을 먼저 합쳐
  "표면/부품 단위" S1을 만든다.
  → S1이 깨끗하면 Stage2 MLP 학습 신호도 명확해진다.

사용법:
  python stage1_merge.py --npz ../../src/6_floor.npz
  python stage1_merge.py --npz ../../src/6_floor.npz --visualize
  python stage1_merge.py --npz ../../src/6_floor.npz --visualize --color_by class
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

# ── 상수 ─────────────────────────────────────────────────────
IDX_CENTROID = slice(0, 3)
IDX_NORMAL   = slice(3, 6)
IDX_COLOR    = slice(7, 10)
IDX_SIZE     = 10

CLASS_NAMES = {
    0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam',
    4: 'column',  5: 'window', 6: 'door', 7: 'table',
    8: 'chair',   9: 'sofa',  10: 'bookcase', 11: 'board', 12: 'clutter'
}

# 클래스별 병합 파라미터 (구조체는 넓게, 가구는 좁게)
CLASS_PARAMS = {
    # class_id: (max_dist, normal_thresh, color_thresh)
    'structural': (0.60, 0.88, 0.25),  # ceiling/floor/wall/beam/column
    'furniture':  (0.35, 0.80, 0.18),  # table/chair/sofa/bookcase/board
    'portal':     (0.40, 0.82, 0.20),  # window/door
    'clutter':    (0.30, 0.75, 0.22),  # clutter
}
# 같은 클래스면 무조건 하나로 묶는 클래스 (거리·법선 조건 없음)
AUTO_MERGE_CLASSES = {0, 1, 2, 3}   # ceiling, floor, wall, beam
STRUCTURAL = {0, 1, 2, 3, 4}
FURNITURE  = {7, 8, 9, 10, 11}
PORTAL     = {5, 6}
CLUTTER    = {12}


# ── 기하 유사도 병합 ─────────────────────────────────────────
def merge_stage1(sp_features: np.ndarray,
                 sp_pred:     np.ndarray,
                 verbose:     bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    S0 SP → S1 기하 유사도 병합.

    Returns
    -------
    s1_labels : [N_sp]  S1 객체 ID (0-based)
    s1_feat   : [N_s1, 139]  S1 피처 (SP 가중 평균)
    """
    N_sp      = sp_features.shape[0]
    centroids = sp_features[:, IDX_CENTROID]
    normals   = sp_features[:, IDX_NORMAL]
    colors    = sp_features[:, IDX_COLOR]
    sizes     = sp_features[:, IDX_SIZE]

    # 법선 단위 벡터 정규화
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals_n = normals / norms

    if verbose:
        print(f"\n[Stage1] S0 SP: {N_sp:,}개")

    # ── AUTO_MERGE: ceiling/floor/wall/beam → 같은 클래스 전체를 하나의 S1 ──
    # 거리·법선 조건 없이 같은 클래스면 무조건 병합
    # Union-Find 없이 label 직접 할당으로 처리
    raw_labels = np.full(N_sp, -1, dtype=np.int32)
    next_id = 0
    for cid in sorted(AUTO_MERGE_CLASSES):
        idx = np.where(sp_pred == cid)[0]
        if len(idx) == 0:
            continue
        raw_labels[idx] = next_id
        next_id += 1
        if verbose:
            print(f"  [auto-merge] {CLASS_NAMES[cid]:8s}({cid}): "
                  f"{len(idx):,} SP → S1 ID {next_id-1} (하나로 통합)")

    # ── 클래스별로 엣지 후보 생성 → 조건 필터 ────────────────
    all_src, all_dst = [], []
    t0 = time.time()

    for group_name, class_set, params in [
        ('structural', STRUCTURAL - AUTO_MERGE_CLASSES, CLASS_PARAMS['structural']),
        ('furniture',  FURNITURE,  CLASS_PARAMS['furniture']),
        ('portal',     PORTAL,     CLASS_PARAMS['portal']),
        ('clutter',    CLUTTER,    CLASS_PARAMS['clutter']),
    ]:
        max_dist, normal_thresh, color_thresh = params
        mask = np.isin(sp_pred, list(class_set))
        idx  = np.where(mask)[0]
        if len(idx) < 2:
            continue

        c_sub = centroids[idx]
        k     = min(12, len(idx) - 1)
        nn    = NearestNeighbors(n_neighbors=k + 1,
                                 algorithm='kd_tree').fit(c_sub)
        dists, nbrs = nn.kneighbors(c_sub)   # [M, k+1]

        for i_local, (ds, ns) in enumerate(zip(dists, nbrs)):
            i_global = idx[i_local]
            for d, j_local in zip(ds[1:], ns[1:]):
                if d > max_dist:
                    continue
                j_global = idx[j_local]

                # 같은 클래스 확인
                if sp_pred[i_global] != sp_pred[j_global]:
                    continue

                # 법선 유사도
                cos_n = float(np.dot(normals_n[i_global],
                                     normals_n[j_global]))
                if cos_n < normal_thresh:
                    continue

                # 색상 차이
                c_diff = float(np.linalg.norm(
                    colors[i_global] - colors[j_global]))
                if c_diff > color_thresh:
                    continue

                all_src.append(i_global)
                all_dst.append(j_global)

        if verbose:
            print(f"  [{group_name:10s}] SP {len(idx):6,}개  "
                  f"max_dist={max_dist}m  normal≥{normal_thresh}  "
                  f"color≤{color_thresh}")

    if verbose:
        print(f"  엣지 후보: {len(all_src):,}개  ({time.time()-t0:.1f}s)")

    # ── Connected components → 나머지 클래스 S1 레이블 ────────
    # auto-merge 안 된 SP들(-1)만 connected_components로 처리
    remaining = np.where(raw_labels == -1)[0]

    if len(remaining) > 0 and all_src:
        # remaining SP에 한정된 로컬 인덱스로 변환
        global_to_local = np.full(N_sp, -1, dtype=np.int32)
        global_to_local[remaining] = np.arange(len(remaining), dtype=np.int32)

        loc_src, loc_dst = [], []
        for gs, gd in zip(all_src, all_dst):
            ls, ld = global_to_local[gs], global_to_local[gd]
            if ls >= 0 and ld >= 0:
                loc_src.append(ls); loc_dst.append(ld)

        if loc_src:
            rows = np.concatenate([loc_src, loc_dst])
            cols = np.concatenate([loc_dst, loc_src])
            adj  = csr_matrix(
                (np.ones(len(rows)), (rows, cols)),
                shape=(len(remaining), len(remaining)))
            _, loc_labels = connected_components(adj, directed=False)
        else:
            loc_labels = np.arange(len(remaining), dtype=np.int32)

        # 로컬 레이블을 글로벌 레이블 공간으로 변환 (next_id 이후 ID 부여)
        raw_labels[remaining] = loc_labels + next_id

    elif len(remaining) > 0:
        raw_labels[remaining] = np.arange(len(remaining), dtype=np.int32) + next_id

    # 레이블 연속화
    _, raw_labels = np.unique(raw_labels, return_inverse=True)
    raw_labels = raw_labels.astype(np.int32)
    n_s1 = int(raw_labels.max()) + 1

    # ── S1 피처 집계 (size 가중 평균) ────────────────────────
    s1_feat = np.zeros((n_s1, sp_features.shape[1]), dtype=np.float32)
    w_sum   = np.zeros(n_s1, dtype=np.float32)

    for sp_id in range(N_sp):
        s1_id = raw_labels[sp_id]
        w     = max(float(sizes[sp_id]), 1e-4)
        s1_feat[s1_id] += sp_features[sp_id] * w
        w_sum[s1_id]   += w

    valid = w_sum > 0
    s1_feat[valid] /= w_sum[valid, None]

    # ── Singleton 흡수: 단독 SP를 같은 클래스 최근접 S1에 병합 ──
    comp_sizes = np.bincount(raw_labels)
    singleton_mask = comp_sizes[raw_labels] == 1   # [N_sp] singleton 여부

    n_singletons = singleton_mask.sum()
    if n_singletons > 0:
        sing_idx  = np.where(singleton_mask)[0]
        non_sing  = np.where(~singleton_mask)[0]

        if len(non_sing) > 0:
            # 클래스별로 singleton → 최근접 non-singleton S1 흡수
            absorbed = 0
            nn_global = NearestNeighbors(
                n_neighbors=1, algorithm='kd_tree').fit(centroids[non_sing])
            dists_s, nbrs_s = nn_global.kneighbors(centroids[sing_idx])

            for i_loc, (d_arr, n_arr) in enumerate(zip(dists_s, nbrs_s)):
                sp_i = sing_idx[i_loc]
                sp_j = non_sing[n_arr[0]]
                dist_ij = d_arr[0]

                # 같은 클래스 + 거리 1.5m 이내만 흡수
                if (sp_pred[sp_i] == sp_pred[sp_j] and dist_ij < 1.5):
                    raw_labels[sp_i] = raw_labels[sp_j]
                    absorbed += 1

            # label 재정리 (연속 ID로)
            _, raw_labels = np.unique(raw_labels, return_inverse=True)
            raw_labels = raw_labels.astype(np.int32)
            n_s1 = int(raw_labels.max()) + 1

            if verbose:
                print(f"  singleton 흡수: {absorbed:,}/{n_singletons:,}개 "
                      f"→ S1 {n_s1:,}개")

            # S1 피처 재집계
            s1_feat = np.zeros((n_s1, sp_features.shape[1]), dtype=np.float32)
            w_sum   = np.zeros(n_s1, dtype=np.float32)
            for sp_id in range(N_sp):
                s1_id = raw_labels[sp_id]
                w     = max(float(sizes[sp_id]), 1e-4)
                s1_feat[s1_id] += sp_features[sp_id] * w
                w_sum[s1_id]   += w
            valid = w_sum > 0
            s1_feat[valid] /= w_sum[valid, None]

    # ── 통계 출력 ────────────────────────────────────────────
    if verbose:
        comp_sizes = np.bincount(raw_labels)
        singleton  = int((comp_sizes == 1).sum())
        large      = int((comp_sizes >= 10).sum())
        print(f"\n[Stage1 결과]")
        print(f"  S0 → S1:  {N_sp:,} → {n_s1:,}개  "
              f"({100*(1-n_s1/N_sp):.1f}% 감소)")
        print(f"  singleton(1개 SP): {singleton:,}개  "
              f"({100*singleton/n_s1:.1f}%)")
        print(f"  large(≥10 SP):     {large:,}개")
        print(f"  평균 SP/S1:         {N_sp/n_s1:.1f}")

        # 클래스별 S1 수
        s1_pred = np.zeros(n_s1, dtype=np.int32)
        for sp_id in range(N_sp):
            s1_pred[raw_labels[sp_id]] = sp_pred[sp_id]
        print(f"\n  클래스별 S1 수:")
        for cid, cnt in sorted(zip(*np.unique(s1_pred, return_counts=True))):
            print(f"    {CLASS_NAMES.get(cid, str(cid)):10s}({cid}): {cnt:,}")

    return raw_labels.astype(np.int32), s1_feat


# ── Open3D 시각화 ────────────────────────────────────────────
def visualize_s1(sp_features: np.ndarray,
                 sp_pred:     np.ndarray,
                 s1_labels:   np.ndarray,
                 coord:       np.ndarray,
                 sp_labels_pt:np.ndarray,
                 color_by:    str = 's1',
                 max_pts:     int = 800_000):
    """
    color_by:
      's1'    - S1 객체마다 랜덤 색 (병합 결과 확인)
      'class' - 시맨틱 클래스 색 (분류 확인)
      'both'  - 두 창 동시 표시
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[오류] pip install open3d"); return

    CLASS_COLORS = {
        0: [0.5, 0.5, 0.5],   # ceiling  회색
        1: [0.8, 0.7, 0.5],   # floor    베이지
        2: [0.4, 0.6, 0.8],   # wall     파랑
        3: [0.3, 0.3, 0.6],   # beam     남색
        4: [0.2, 0.5, 0.3],   # column   녹색
        5: [0.9, 0.9, 0.4],   # window   노랑
        6: [0.8, 0.5, 0.2],   # door     주황
        7: [0.9, 0.2, 0.2],   # table    빨강
        8: [0.9, 0.5, 0.9],   # chair    보라
        9: [0.2, 0.8, 0.8],   # sofa     청록
        10:[0.6, 0.3, 0.1],   # bookcase 갈색
        11:[0.5, 0.9, 0.3],   # board    연두
        12:[0.7, 0.7, 0.7],   # clutter  밝은회
    }

    # 포인트 다운샘플
    N = len(coord)
    if N > max_pts:
        idx = np.random.choice(N, max_pts, replace=False)
    else:
        idx = np.arange(N)

    pts_xyz  = coord[idx].astype(np.float64)
    pts_sp   = sp_labels_pt[idx]               # 각 포인트의 S0 SP ID
    pts_s1   = s1_labels[pts_sp]               # 각 포인트의 S1 ID
    pts_cls  = sp_pred[pts_sp]                 # 각 포인트의 클래스

    def make_pcd(colors_arr):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_xyz)
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(colors_arr, 0, 1).astype(np.float64))
        return pcd

    def s1_colors():
        """S1마다 랜덤 색"""
        rng = np.random.default_rng(42)
        n_s1 = int(s1_labels.max()) + 1
        palette = rng.random((n_s1, 3))
        return palette[pts_s1]

    def class_colors():
        col = np.zeros((len(idx), 3))
        for cid, rgb in CLASS_COLORS.items():
            col[pts_cls == cid] = rgb
        return col

    modes = ['s1', 'class'] if color_by == 'both' else [color_by]

    for mode in modes:
        col  = s1_colors() if mode == 's1' else class_colors()
        pcd  = make_pcd(col)
        n_s1 = int(s1_labels.max()) + 1
        title = (f"Stage1 결과 — S1 색상  "
                 f"(S0={len(sp_features):,} → S1={n_s1:,})"
                 if mode == 's1' else
                 f"Semantic Class — {len(sp_features):,} SP")
        print(f"\n[뷰어] {title}  (Q=닫기, 마우스=회전)")
        o3d.visualization.draw_geometries([pcd], window_name=title,
                                          width=1600, height=900)


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--npz',       default='../../src/6_floor.npz')
    pa.add_argument('--out',       default=None,
                    help='S1 결과 저장 경로 (미지정 시 저장 안 함)')
    pa.add_argument('--visualize', action='store_true')
    pa.add_argument('--color_by',  default='both',
                    choices=['s1', 'class', 'both'])
    pa.add_argument('--max_pts',   type=int, default=800_000)
    args = pa.parse_args()

    npz_path = Path(args.npz)
    print(f"[로드] {npz_path}")
    d = np.load(npz_path)

    sp_features  = d['sp_features'].astype(np.float32)   # [N_sp, 139]
    sp_pred      = d['sp_pred'].astype(np.int32)          # [N_sp]
    coord        = d['coord'].astype(np.float32)          # [N_pts, 3]
    sp_labels_pt = d['sp_labels'].astype(np.int32)        # [N_pts]

    print(f"  SP: {sp_features.shape[0]:,}  "
          f"포인트: {coord.shape[0]:,}")

    # ── Stage1 병합 ──────────────────────────────────────────
    s1_labels, s1_feat = merge_stage1(sp_features, sp_pred)

    # ── 저장 ────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path,
                            s1_labels=s1_labels,
                            s1_feat=s1_feat,
                            sp_pred=sp_pred)
        print(f"\n[저장] {out_path}")

    # ── 시각화 ──────────────────────────────────────────────
    if args.visualize:
        visualize_s1(sp_features, sp_pred, s1_labels,
                     coord, sp_labels_pt,
                     color_by=args.color_by,
                     max_pts=args.max_pts)
