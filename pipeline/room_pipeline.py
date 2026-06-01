"""
room_pipeline.py
================
벽 분리 + 방 검출 파이프라인

  1. wall_pct_low=50  → 넓은 벽 후보 생성
  2. wall_pct_high=85 → 확실한 벽 seed 생성
  3. low 후보 중 seed와 겹치거나 길고 직선적인 component만 유지
  4. HoughLinesP로 수평/수직 선분 추출
  5. 같은 row/column 근처 선분 gap 연결
  6. flood fill로 닫힌 방 영역 추출
  7. 2×3 시각화 PNG 저장

실행:
    python room_pipeline.py
    python room_pipeline.py --wall_pct_low 50 --wall_pct_high 85 --max_gap 20
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cv2
import argparse
from pathlib import Path
from scipy.ndimage import label as scipy_label
from scipy.ndimage import convolve

# ── 한글 폰트 ─────────────────────────────────────────────────────────────────
_FONT_CANDIDATES = [
    '/System/Library/Fonts/AppleSDGothicNeo.ttc',
    '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
    '/Library/Fonts/Arial Unicode.ttf',
]
for _fp in _FONT_CANDIDATES:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
        _prop = fm.FontProperties(fname=_fp)
        matplotlib.rcParams['font.family'] = _prop.get_name()
        break

BASE      = Path(__file__).resolve().parent
FLOOR_NPZ = BASE / 'Area_6_spt' / 'floor.npz'
OUT_PNG   = BASE / 'room_pipeline_result.png'

CELL_SIZE      = 0.07084175512849847
CEILING_MARGIN = 0.30942782733905927


# ── 함수 ──────────────────────────────────────────────────────────────────────
def compute_density(coord, sp_labels_pt, sp_centroids, cell_size, wall_pct, ceiling_margin):
    """forced class → 밀도맵 반환"""
    N_sp   = sp_centroids.shape[0]
    forced = np.full(N_sp, -1, dtype=np.int32)

    ceiling_z = np.percentile(coord[:, 2], 95)
    forced[sp_centroids[:, 2] > ceiling_z - ceiling_margin] = 0

    x_vals, y_vals = coord[:, 0], coord[:, 1]
    x_min, x_max   = x_vals.min(), x_vals.max()
    y_min, y_max   = y_vals.min(), y_vals.max()

    xi = np.clip(((x_vals - x_min) / cell_size).astype(int), 0, int((x_max - x_min) / cell_size))
    yi = np.clip(((y_vals - y_min) / cell_size).astype(int), 0, int((y_max - y_min) / cell_size))
    dg = np.zeros((xi.max() + 1, yi.max() + 1), dtype=np.int32)
    np.add.at(dg, (xi, yi), 1)

    thr   = np.percentile(dg[dg > 0], wall_pct)
    sp_xi = np.clip(((sp_centroids[:, 0] - x_min) / cell_size).astype(int), 0, dg.shape[0] - 1)
    sp_yi = np.clip(((sp_centroids[:, 1] - y_min) / cell_size).astype(int), 0, dg.shape[1] - 1)
    forced[(dg[sp_xi, sp_yi] >= thr) & (forced == -1)] = 2

    # density map (wall 포인트만)
    W = int((x_max - x_min) / cell_size) + 2
    H = int((y_max - y_min) / cell_size) + 2
    density = np.zeros((H, W), dtype=np.float32)
    wall_sp = forced == 2
    pt_wall = wall_sp[sp_labels_pt]
    wx = np.clip(((coord[pt_wall, 0] - x_min) / cell_size).astype(int), 0, W - 1)
    wy = np.clip(((coord[pt_wall, 1] - y_min) / cell_size).astype(int), 0, H - 1)
    np.add.at(density, (wy, wx), 1)

    return density, x_min, y_min, W, H


def build_wall_mask(density, wall_pct, min_px=0):
    """density percentile 기준 이진 마스크 + 노이즈 제거"""
    nonzero = density[density > 0]
    if len(nonzero) == 0:
        return np.zeros_like(density, dtype=bool)
    thr    = np.percentile(nonzero, wall_pct)
    mask   = density >= thr
    if min_px > 0:
        labeled, n = scipy_label(mask)
        for cid in range(1, n + 1):
            if (labeled == cid).sum() < min_px:
                mask[labeled == cid] = False
    return mask


def component_pca(ys, xs):
    """PCA로 길쭉함(ratio) + 길이(px) 반환"""
    if len(xs) < 2:
        return 0.0, 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    ev  = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ratio  = float(ev[0] / ev[1]) if ev[1] > 1e-6 else float('inf')
    length = float(np.sqrt(max(ev[0], 0)) * 4.0)
    return ratio, length


def filter_components(candidate, seed, min_overlap=1, min_overlap_ratio=0.05,
                      min_px=10, min_length=15, min_pca_ratio=3.0):
    """
    low 후보 중 아래 조건 중 하나를 만족하는 component 유지:
      - seed와 겹치는 픽셀 수 ≥ min_overlap  OR  겹침 비율 ≥ min_overlap_ratio
      - 충분히 길고(length ≥ min_length) 직선적(pca_ratio ≥ min_pca_ratio)
    """
    labeled, n = scipy_label(candidate)
    kept = np.zeros_like(candidate, dtype=bool)
    for cid in range(1, n + 1):
        comp    = labeled == cid
        area    = int(comp.sum())
        if area < min_px:
            continue
        overlap = int((comp & seed).sum())
        ys, xs  = np.where(comp)
        ratio, length = component_pca(ys, xs)
        overlap_ok = (overlap >= min_overlap) or (overlap / area >= min_overlap_ratio)
        shape_ok   = (length >= min_length) and (ratio >= min_pca_ratio)
        if overlap_ok or shape_ok:
            kept[comp] = True
    return kept


def hough_lines(wall_mask, thr=20, min_len=15, max_gap=10, angle_tol_deg=10):
    """
    HoughLinesP 적용 후 수평/수직에 가까운 선분만 추출.
    반환: (lines_mask, lines_list)
    """
    img_u8 = wall_mask.astype(np.uint8) * 255
    lines  = cv2.HoughLinesP(img_u8, 1, np.pi / 180,
                              threshold=thr,
                              minLineLength=min_len,
                              maxLineGap=max_gap)
    out  = np.zeros_like(wall_mask, dtype=np.uint8)
    kept = []
    tol  = np.deg2rad(angle_tol_deg)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.arctan2(y2 - y1, x2 - x1))
            # 수평(≈0) 또는 수직(≈π/2) 만 허용
            is_h = angle <= tol or angle >= (np.pi - tol)
            is_v = abs(angle - np.pi / 2) <= tol
            if is_h or is_v:
                cv2.line(out, (x1, y1), (x2, y2), 255, 1)
                kept.append((x1, y1, x2, y2, 'H' if is_h else 'V'))
    return out.astype(bool), kept


def connect_gaps(hough_mask, lines_list, max_gap=20):
    """
    같은 row(수평) 또는 같은 col(수직) 위치의 선분 끝점들 사이
    gap ≤ max_gap 이면 직선으로 연결.
    """
    out = hough_mask.copy().astype(np.uint8) * 255
    H, W = out.shape

    h_lines = [(x1, y1, x2, y2) for x1, y1, x2, y2, t in lines_list if t == 'H']
    v_lines = [(x1, y1, x2, y2) for x1, y1, x2, y2, t in lines_list if t == 'V']

    # 수평 선분: y 기준으로 그룹화
    h_by_row = {}
    for x1, y1, x2, y2 in h_lines:
        row = round((y1 + y2) / 2)
        h_by_row.setdefault(row, []).append((min(x1, x2), max(x1, x2)))

    for row, segs in h_by_row.items():
        segs_sorted = sorted(segs)
        for i in range(len(segs_sorted) - 1):
            _, xe = segs_sorted[i]
            xs, _ = segs_sorted[i + 1]
            if 0 < (xs - xe) <= max_gap:
                cv2.line(out, (xe, row), (xs, row), 255, 1)

    # 수직 선분: x 기준으로 그룹화
    v_by_col = {}
    for x1, y1, x2, y2 in v_lines:
        col = round((x1 + x2) / 2)
        v_by_col.setdefault(col, []).append((min(y1, y2), max(y1, y2)))

    for col, segs in v_by_col.items():
        segs_sorted = sorted(segs)
        for i in range(len(segs_sorted) - 1):
            _, ye = segs_sorted[i]
            ys, _ = segs_sorted[i + 1]
            if 0 < (ys - ye) <= max_gap:
                cv2.line(out, (col, ye), (col, ys), 255, 1)

    return out.astype(bool)


def flood_fill_rooms(wall_mask, min_room_px=10):
    """
    벽 마스크에서 flood fill → 방 레이블 grid 반환
    """
    free = ~wall_mask
    labeled, _ = scipy_label(free)
    for rid in range(1, labeled.max() + 1):
        if (labeled == rid).sum() < min_room_px:
            labeled[labeled == rid] = 0
    unique = np.unique(labeled[labeled > 0])
    remap  = np.zeros(labeled.max() + 1, dtype=np.int32)
    for new_id, old_id in enumerate(unique, start=1):
        remap[old_id] = new_id
    return remap[labeled]


def label_rooms(ax, room_grid, fontsize=6):
    for rid in range(1, int(room_grid.max()) + 1):
        ys, xs = np.where(room_grid == rid)
        if len(ys) == 0:
            continue
        ax.text(xs.mean(), ys.mean(), str(rid),
                color='white', fontsize=fontsize,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.4, linewidth=0))


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main(args):
    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    print("데이터 로드 중...")
    d            = np.load(FLOOR_NPZ)
    coord        = d['coord'].astype(np.float32)
    sp_features  = d['sp_features'].astype(np.float32)
    sp_labels_pt = d['sp_labels'].astype(np.int64)
    sp_centroids = sp_features[:, 0:3]

    # ── Step 1: low 후보 마스크 ───────────────────────────────────────────────
    print(f"[1] wall_pct_low={args.wall_pct_low} → 넓은 벽 후보...")
    density, x_min, y_min, W, H = compute_density(
        coord, sp_labels_pt, sp_centroids,
        args.cell_size, args.wall_pct_low, CEILING_MARGIN)
    mask_low = build_wall_mask(density, args.wall_pct_low, min_px=args.min_px_low)
    print(f"    {mask_low.sum():,} px")

    # ── Step 2: high seed 마스크 ──────────────────────────────────────────────
    print(f"[2] wall_pct_high={args.wall_pct_high} → 확실한 벽 seed...")
    density_h, *_ = compute_density(
        coord, sp_labels_pt, sp_centroids,
        args.cell_size, args.wall_pct_high, CEILING_MARGIN)
    mask_high = build_wall_mask(density_h, args.wall_pct_high, min_px=args.min_px_high)
    print(f"    {mask_high.sum():,} px")

    # ── Step 3: component 필터링 ──────────────────────────────────────────────
    print("[3] seed와 겹치거나 길고 직선적인 component 유지...")
    mask_filtered = filter_components(
        mask_low, mask_high,
        min_overlap=args.min_overlap,
        min_overlap_ratio=args.min_overlap_ratio,
        min_px=args.min_px_low,
        min_length=args.min_length,
        min_pca_ratio=args.min_pca_ratio,
    )
    print(f"    {mask_filtered.sum():,} px")

    # ── Step 4: HoughLinesP (수평/수직) ──────────────────────────────────────
    print(f"[4] HoughLinesP (thr={args.hough_thr}, min_len={args.hough_min_len}, "
          f"max_gap={args.hough_max_gap})...")
    hough_mask, lines_list = hough_lines(
        mask_filtered,
        thr=args.hough_thr,
        min_len=args.hough_min_len,
        max_gap=args.hough_max_gap,
        angle_tol_deg=args.angle_tol,
    )
    wall_with_hough = mask_filtered | hough_mask
    print(f"    선분 {len(lines_list)}개 추출")

    # ── Step 5: gap 연결 ──────────────────────────────────────────────────────
    print(f"[5] 같은 row/col 선분 gap 연결 (max_gap={args.max_gap})...")
    wall_connected = connect_gaps(wall_with_hough, lines_list, max_gap=args.max_gap)
    print(f"    {wall_connected.sum():,} px")

    # ── Step 6: flood fill 방 검출 ────────────────────────────────────────────
    print(f"[6] flood fill 방 검출 (min_room_px={args.min_room_px})...")
    room_grid = flood_fill_rooms(wall_connected, min_room_px=args.min_room_px)
    n_rooms   = int(room_grid.max())
    print(f"    {n_rooms}개 방 검출")

    # ── Step 7: 2×3 시각화 ───────────────────────────────────────────────────
    print("[7] 시각화 저장...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a1a')
    BG = '#1a1a1a'

    def show(ax, img, title, cmap='gray'):
        ax.imshow(img, origin='lower', aspect='equal',
                  interpolation='nearest', cmap=cmap)
        ax.set_title(title, color='white', fontsize=12, pad=6)
        ax.set_facecolor(BG)
        ax.axis('off')

    # ① low 후보
    show(axes[0, 0], mask_low,
         f'① low 후보 (wall_pct={args.wall_pct_low}, {mask_low.sum():,}px)')

    # ② high seed
    show(axes[0, 1], mask_high,
         f'② high seed (wall_pct={args.wall_pct_high}, {mask_high.sum():,}px)')

    # ③ 필터링 결과
    show(axes[0, 2], mask_filtered,
         f'③ 필터링 (seed겹침+직선형, {mask_filtered.sum():,}px)')

    # ④ Hough 적용
    show(axes[1, 0], wall_with_hough,
         f'④ Hough 수평/수직 ({len(lines_list)}개 선분)')

    # ⑤ gap 연결
    show(axes[1, 1], wall_connected,
         f'⑤ gap 연결 (max_gap={args.max_gap}, {wall_connected.sum():,}px)')

    # ⑥ 방 검출
    cmap_r = matplotlib.colormaps.get_cmap('hsv').resampled(max(n_rooms, 1))
    room_vis = np.zeros((*room_grid.shape, 3))
    for rid in range(1, n_rooms + 1):
        room_vis[room_grid == rid] = cmap_r(rid - 1)[:3]
    room_vis[wall_connected] = [1.0, 1.0, 1.0]

    axes[1, 2].imshow(room_vis, origin='lower', aspect='equal', interpolation='nearest')
    label_rooms(axes[1, 2], room_grid, fontsize=6)
    axes[1, 2].set_title(f'⑥ 방 검출 ({n_rooms}개)', color='white', fontsize=12, pad=6)
    axes[1, 2].set_facecolor(BG)
    axes[1, 2].axis('off')

    fig.suptitle(
        f'room_pipeline  |  low={args.wall_pct_low}  high={args.wall_pct_high}  '
        f'hough_thr={args.hough_thr}  max_gap={args.max_gap}  rooms={n_rooms}',
        color='white', fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\n  저장: {OUT_PNG}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_size',         type=float, default=CELL_SIZE)
    parser.add_argument('--wall_pct_low',       type=float, default=40.0)
    parser.add_argument('--wall_pct_high',      type=float, default=75.0)
    parser.add_argument('--min_px_low',         type=int,   default=10,
                        help='low 마스크 노이즈 제거 최소 픽셀')
    parser.add_argument('--min_px_high',        type=int,   default=10,
                        help='high 마스크 노이즈 제거 최소 픽셀')
    parser.add_argument('--min_overlap',        type=int,   default=1,
                        help='seed와 겹치는 최소 픽셀 수')
    parser.add_argument('--min_overlap_ratio',  type=float, default=0.05,
                        help='seed와 겹치는 최소 비율')
    parser.add_argument('--min_length',         type=float, default=15.0,
                        help='직선형 component 최소 길이(px)')
    parser.add_argument('--min_pca_ratio',      type=float, default=3.0,
                        help='직선형 판단 PCA 비율')
    parser.add_argument('--hough_thr',          type=int,   default=20,
                        help='HoughLinesP threshold')
    parser.add_argument('--hough_min_len',      type=int,   default=15,
                        help='HoughLinesP minLineLength')
    parser.add_argument('--hough_max_gap',      type=int,   default=10,
                        help='HoughLinesP maxLineGap')
    parser.add_argument('--angle_tol',          type=float, default=10.0,
                        help='수평/수직 허용 각도 오차(도)')
    parser.add_argument('--max_gap',            type=int,   default=20,
                        help='같은 row/col 선분 gap 연결 최대 거리(px)')
    parser.add_argument('--min_room_px',        type=int,   default=50,
                        help='방으로 인정할 최소 픽셀 수')
    args = parser.parse_args()
    main(args)
