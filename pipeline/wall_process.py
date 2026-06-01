"""
wall_process.py — wall_mask + clean_wall_mask 통합 파이프라인
=============================================================
파이프라인:
  원본 마스크 (original)
    ① [wall_mask]    skeleton → prune → endpoint projection → gap 연결
                     → gap_connected
    ② [clean_mask]   gap_connected → 모폴로지 클로징 → 노이즈 제거
                     → gap_connected와 교집합 → cleaned
    ③ [최종]         cleaned ∩ original
                     (wall_mask가 그린 gap선이라도 원본에 없으면 제거)

사용법:
  python wall_process.py --input results/sam_out/sam_baseline_6_floor.png
  python wall_process.py --input results/sam_out/sam_baseline_6_floor.png \\
      --max_dist 50 --min_px 300 --morph_kernel 7
  python wall_process.py --npz src/6_floor.npz
"""

import argparse
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

# ── 한글 폰트 ─────────────────────────────────────────────────────────────────
for _fp in [
    '/System/Library/Fonts/AppleSDGothicNeo.ttc',
    '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
    '/Library/Fonts/Arial Unicode.ttf',
]:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
        _prop = fm.FontProperties(fname=_fp)
        matplotlib.rcParams['font.family'] = _prop.get_name()
        break

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — wall_mask: gap 연결
# ═══════════════════════════════════════════════════════════════════════════════

def skeleton_endpoints(skel: np.ndarray) -> np.ndarray:
    """스켈레톤에서 끝점 좌표 반환 → (N, 2) float32 (x, y)"""
    kern   = np.ones((3, 3), dtype=np.uint8)
    cnt    = cv2.filter2D(skel.astype(np.uint8), -1, kern)
    ep_mask = skel & (cnt == 2)
    ys, xs  = np.where(ep_mask)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack([xs, ys]).astype(np.float32)


def _trace_dir(skel: np.ndarray, px: int, py: int, trace_len: int = 5) -> np.ndarray:
    """끝점에서 스켈레톤을 따라 trace_len px 이동 → 방향 벡터"""
    H, W    = skel.shape
    visited = {(px, py)}
    cx, cy  = px, py
    for _ in range(max(trace_len, 1)):
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < W and 0 <= ny < H
                        and skel[ny, nx] and (nx, ny) not in visited):
                    nxt = (nx, ny)
                    break
            if nxt:
                break
        if nxt is None:
            break
        visited.add(nxt)
        cx, cy = nxt
    vx, vy = cx - px, cy - py
    n = np.hypot(vx, vy)
    return np.array([vx / n, vy / n], dtype=np.float32) if n > 1e-6 \
           else np.zeros(2, dtype=np.float32)


def prune_skeleton(skel: np.ndarray, min_branch_len: int = 8) -> np.ndarray:
    """짧은 dangling branch 제거 (min_branch_len px 미만)."""
    result = skel.copy().astype(bool)
    H, W   = result.shape
    changed = True
    while changed:
        changed = False
        kern    = np.ones((3, 3), dtype=np.uint8)
        cnt     = cv2.filter2D(result.astype(np.uint8), -1, kern)
        ep_ys, ep_xs = np.where(result & (cnt == 2))
        for py, px in zip(ep_ys.tolist(), ep_xs.tolist()):
            if not result[py, px]:
                continue
            branch  = [(px, py)]
            visited = {(px, py)}
            cx, cy  = px, py
            while len(branch) < min_branch_len:
                nbrs = []
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < W and 0 <= ny < H
                                and result[ny, nx]
                                and (nx, ny) not in visited):
                            nbrs.append((nx, ny))
                if len(nbrs) == 0:
                    break
                if len(nbrs) >= 2:
                    break
                nxt = nbrs[0]
                branch.append(nxt)
                visited.add(nxt)
                cx, cy = nxt
            if len(branch) < min_branch_len:
                for bx, by in branch:
                    result[by, bx] = False
                changed = True
    return result


def project_to_boundary(mask: np.ndarray, skel: np.ndarray,
                         px: int, py: int, trace_len: int = 5) -> tuple:
    """skeleton endpoint를 벽 경계(mask 가장자리)로 이동."""
    inward  = _trace_dir(skel, px, py, trace_len)
    out_dx  = -float(inward[0])
    out_dy  = -float(inward[1])
    if abs(out_dx) < 1e-6 and abs(out_dy) < 1e-6:
        return px, py
    H, W   = mask.shape
    cx, cy = float(px), float(py)
    last_x, last_y = px, py
    for _ in range(30):
        nx = int(round(cx + out_dx))
        ny = int(round(cy + out_dy))
        if not (0 <= nx < W and 0 <= ny < H) or not mask[ny, nx]:
            break
        last_x, last_y = nx, ny
        cx += out_dx
        cy += out_dy
    return last_x, last_y


def add_gap_connections(mask: np.ndarray,
                        max_dist: float = 30.0,
                        min_cos: float = -1.0,
                        line_width: int = 1,
                        trace_len: int = 5,
                        min_branch_len: int = 8) -> tuple:
    """두꺼운 벽 마스크의 gap을 메운 마스크 반환.

    Returns
    -------
    (result, gap_only, n_connected, proj_pts, thin_pruned)
    result: bool ndarray (gap 메워진 마스크)
    """
    thin        = skeletonize(mask.astype(bool))
    thin_pruned = prune_skeleton(thin, min_branch_len)
    raw_pts     = skeleton_endpoints(thin_pruned)

    proj_pts = []
    for p in raw_pts:
        bx, by = project_to_boundary(mask, thin_pruned,
                                      int(p[0]), int(p[1]), trace_len)
        proj_pts.append([bx, by])
    proj_pts = np.array(proj_pts, dtype=np.float32) if proj_pts \
               else np.empty((0, 2), dtype=np.float32)

    gap_canvas  = np.zeros(mask.shape, dtype=np.uint8)
    n_connected = 0

    if len(proj_pts) >= 2:
        D    = cdist(proj_pts, proj_pts)
        dirs = np.array([_trace_dir(thin_pruned, int(p[0]), int(p[1]), trace_len)
                         for p in raw_pts])
        for i in range(len(proj_pts)):
            for j in range(i + 1, len(proj_pts)):
                if D[i, j] > max_dist:
                    continue
                if min_cos > -1.0:
                    conn = proj_pts[j] - proj_pts[i]
                    cn   = np.hypot(conn[0], conn[1])
                    if cn > 1e-6:
                        conn /= cn
                        if (np.dot(dirs[i],  conn) < min_cos or
                                np.dot(dirs[j], -conn) < min_cos):
                            continue
                cv2.line(gap_canvas,
                         (int(proj_pts[i, 0]), int(proj_pts[i, 1])),
                         (int(proj_pts[j, 0]), int(proj_pts[j, 1])),
                         1, line_width)
                n_connected += 1

    gap_only = gap_canvas.astype(bool)
    result   = mask | gap_only
    return result, gap_only, n_connected, proj_pts, thin_pruned


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — clean_wall_mask: 모폴로지 + 노이즈 제거
# ═══════════════════════════════════════════════════════════════════════════════

def apply_morphology(mask_u8: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """모폴로지 클로징: 벽 사이 작은 틈을 메워 컴포넌트 연결을 강화한다."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)


def remove_isolated_noise(mask_u8: np.ndarray,
                           min_component_px: int = 500) -> np.ndarray:
    """min_component_px 미만인 고립 컴포넌트 제거."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8)
    result  = np.zeros_like(mask_u8)
    kept    = removed = 0
    for lid in range(1, n_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area >= min_component_px:
            result[labels == lid] = 255
            kept += 1
        else:
            removed += 1
    print(f'  컴포넌트 전체: {n_labels-1}개  유지: {kept}개  제거: {removed}개')
    return result


def clean_mask(mask_u8: np.ndarray,
               morph_kernel: int = 5,
               min_px: int = 500) -> np.ndarray:
    """gap_connected(u8)에 모폴로지+노이즈제거 적용 후 gap_connected와 교집합.

    Returns u8 마스크.
    """
    if morph_kernel > 0:
        morph = apply_morphology(mask_u8, morph_kernel)
    else:
        morph = mask_u8.copy()

    morph_cleaned = remove_isolated_noise(morph, min_px)
    # 교집합: morph 팽창으로 생긴 여분 픽셀 제거
    cleaned = cv2.bitwise_and(morph_cleaned, mask_u8)
    return cleaned


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_from_png(png_path: Path) -> np.ndarray:
    """PNG → bool ndarray"""
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'이미지 로드 실패: {png_path}')
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bw > 0


def load_from_npz(npz_path: Path) -> tuple:
    WALL_CLASS     = 2
    DOOR_CLASS     = 6
    CEILING_MARGIN = 0.30942782733905927

    from room_pipeline import compute_density, build_wall_mask
    from scipy.ndimage import label as scipy_label

    d           = np.load(npz_path)
    coord       = d['coord'].astype(np.float32)
    sp_labels   = d['sp_labels'].astype(np.int64)
    sp_pred     = d['sp_pred'].astype(np.int64)
    sp_features = d['sp_features'].astype(np.float32)
    sp_cents    = sp_features[:, 0:3]

    all_xy = coord[:, :2]
    x_min, y_min = all_xy[:, 0].min(), all_xy[:, 1].min()
    x_max, y_max = all_xy[:, 0].max(), all_xy[:, 1].max()
    cs = 0.07084175512849847
    W  = int((x_max - x_min) / cs) + 2
    H  = int((y_max - y_min) / cs) + 2

    def _pts_to_density(xy):
        density = np.zeros((H, W), dtype=np.int32)
        wx = np.clip(((xy[:, 0] - x_min) / cs).astype(int), 0, W - 1)
        wy = np.clip(((xy[:, 1] - y_min) / cs).astype(int), 0, H - 1)
        np.add.at(density, (wy, wx), 1)
        return density

    wall_xy = coord[np.isin(sp_labels, np.where(sp_pred == WALL_CLASS)[0]), :2]
    door_xy = coord[np.isin(sp_labels, np.where(sp_pred == DOOR_CLASS)[0]), :2]
    combined = (_pts_to_density(wall_xy) >= 1) | (_pts_to_density(door_xy) >= 1)

    density_w75, *_ = compute_density(coord, sp_labels, sp_cents, cs, 75.0, CEILING_MARGIN)
    wall75 = build_wall_mask(density_w75, 60.0, min_px=64)

    combined75_raw = combined | wall75
    labeled, n = scipy_label(combined75_raw)
    result = combined75_raw.copy()
    for cid in range(1, n + 1):
        if (labeled == cid).sum() <= 64:
            result[labeled == cid] = False

    return result, x_min, y_min, cs


# ═══════════════════════════════════════════════════════════════════════════════
#  시각화 (6-panel)
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_all(original, gap_connected, cleaned, final,
                  thin_pruned, raw_pts, proj_pts, gap_only,
                  n_connected, out_path: Path, stem: str,
                  max_dist: float, min_branch: int,
                  morph_kernel: int, min_px: int):
    BG   = '#111'
    fig, axes = plt.subplots(1, 6, figsize=(42, 7))
    fig.patch.set_facecolor(BG)

    # ① 원본
    axes[0].imshow(original, cmap='gray', interpolation='nearest')
    axes[0].set_title(f'① 원본\n({original.sum():,}px)', color='white', fontsize=10)
    axes[0].axis('off')

    # ② pruned skeleton + endpoints
    skel_rgb = np.zeros((*thin_pruned.shape, 3), dtype=np.uint8)
    skel_rgb[thin_pruned] = [180, 180, 180]
    for p in raw_pts:
        cv2.circle(skel_rgb, (int(p[0]), int(p[1])), 3, (255, 60, 60), -1)
    for p in proj_pts:
        cv2.circle(skel_rgb, (int(p[0]), int(p[1])), 2, (255, 230, 0), -1)
    axes[1].imshow(skel_rgb, interpolation='nearest')
    axes[1].set_title(
        f'② pruned skeleton\n빨강=EP({len(raw_pts)}개)  노랑=proj',
        color='white', fontsize=10)
    axes[1].axis('off')

    # ③ gap 연결 후 (Stage 1 결과)
    gap_rgb = np.zeros((*original.shape, 3), dtype=np.uint8)
    gap_rgb[original]           = [90, 90, 90]
    gap_rgb[gap_only & ~original] = [60, 210, 60]
    axes[2].imshow(gap_rgb, interpolation='nearest')
    axes[2].set_title(
        f'③ gap 연결 후\n(초록={int((gap_only & ~original).sum()):,}px  쌍:{n_connected})',
        color='white', fontsize=10)
    axes[2].axis('off')

    # ④ 모폴로지+노이즈제거 후 (clean 적용)
    axes[3].imshow(cleaned, cmap='gray', interpolation='nearest')
    axes[3].set_title(
        f'④ clean 적용 후\n(morph={morph_kernel}, min_px={min_px})\n'
        f'{int((cleaned > 0).sum()):,}px',
        color='white', fontsize=10)
    axes[3].axis('off')

    # ⑤ 최종 (cleaned ∩ original)
    axes[4].imshow(final, cmap='gray', interpolation='nearest')
    axes[4].set_title(
        f'⑤ 최종 (∩ original)\n{int((final > 0).sum()):,}px',
        color='white', fontsize=10)
    axes[4].axis('off')

    # ⑥ 원본 대비 변화 (빨강=제거, 초록=추가)
    final_bool = final.astype(bool)
    orig_bool  = original.astype(bool)
    added   = final_bool & ~orig_bool
    removed = orig_bool & ~final_bool
    diff_rgb = np.zeros((*original.shape, 3), dtype=np.uint8)
    diff_rgb[orig_bool & final_bool] = [90, 90, 90]   # 유지
    diff_rgb[added]                  = [60, 210, 60]   # 초록=추가
    diff_rgb[removed]                = [230, 60, 60]   # 빨강=제거
    axes[5].imshow(diff_rgb, interpolation='nearest')
    axes[5].set_title(
        f'⑥ 원본 대비 변화\n초록+{added.sum():,}px  빨강-{removed.sum():,}px',
        color='white', fontsize=10)
    axes[5].axis('off')

    fig.suptitle(
        f'Wall Process — {stem}  |  '
        f'max_dist={max_dist}  min_branch={min_branch}  '
        f'morph={morph_kernel}  min_px={min_px}',
        color='white', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  비교 이미지: {out_path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap  = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--input', help='입력 벽 마스크 PNG 경로')
    src.add_argument('--npz',   help='NPZ 파일 경로 (벽 마스크 자동 생성)')

    # Stage 1 (wall_mask) 파라미터
    ap.add_argument('--max_dist',   type=float, default=100.0,
                    help='endpoint 간 최대 연결 거리 (px). 기본: 100')
    ap.add_argument('--min_cos',    type=float, default=0.9,
                    help='방향 코사인 최솟값 (-1.0=방향 무시). 기본: 0.9')
    ap.add_argument('--line_width', type=int,   default=1,
                    help='연결선 두께 (px). 기본: 1')
    ap.add_argument('--trace_len',  type=int,   default=5,
                    help='방향 추적 길이 (px). 기본: 5')
    ap.add_argument('--min_branch', type=int,   default=16,
                    help='dangling branch pruning 최솟값 (px). 기본: 16')

    # Stage 2 (clean_wall_mask) 파라미터
    ap.add_argument('--morph_kernel', type=int, default=5,
                    help='모폴로지 클로징 커널 크기. 0이면 생략. 기본: 5')
    ap.add_argument('--min_px',       type=int, default=500,
                    help='최소 컴포넌트 픽셀 수. 기본: 500')

    ap.add_argument('--out_dir', default=None,
                    help='출력 디렉토리. 기본: 입력 파일과 같은 폴더')
    args = ap.parse_args()

    # ── 로드 ──────────────────────────────────────────────────────────────────
    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            sys.exit(f'파일 없음: {in_path}')
        print(f'[입력] PNG: {in_path.name}')
        original_bool = load_from_png(in_path)
        stem    = in_path.stem
        out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    else:
        npz_path = Path(args.npz)
        if not npz_path.exists():
            alt = BASE / 'src' / npz_path.name
            if alt.exists():
                npz_path = alt
        if not npz_path.exists():
            sys.exit(f'NPZ 없음: {args.npz}')
        print(f'[입력] NPZ: {npz_path.name}')
        original_bool, *_ = load_from_npz(npz_path)
        stem    = npz_path.stem
        out_dir = Path(args.out_dir) if args.out_dir else BASE / 'results' / 'wall_out'

    out_dir.mkdir(parents=True, exist_ok=True)
    original_u8 = (original_bool.astype(np.uint8) * 255)
    print(f'  원본 벽 픽셀: {original_bool.sum():,}px  크기: {original_bool.shape}')

    # ── Stage 1: gap 연결 ──────────────────────────────────────────────────────
    print(f'\n[Stage 1] gap 연결  '
          f'max_dist={args.max_dist}  min_cos={args.min_cos}  '
          f'min_branch={args.min_branch}  trace_len={args.trace_len}')

    gap_connected_bool, gap_only, n_connected, proj_pts, thin_pruned = add_gap_connections(
        original_bool,
        max_dist=args.max_dist,
        min_cos=args.min_cos,
        line_width=args.line_width,
        trace_len=args.trace_len,
        min_branch_len=args.min_branch,
    )
    raw_pts           = skeleton_endpoints(thin_pruned)
    gap_connected_u8  = (gap_connected_bool.astype(np.uint8) * 255)
    new_gap_px        = int((gap_only & ~original_bool).sum())
    print(f'  탐지 endpoint: {len(raw_pts)}개  연결 쌍: {n_connected}개  '
          f'추가 픽셀: {new_gap_px:,}px  → {gap_connected_bool.sum():,}px')

    # ── Stage 2: 모폴로지 + 노이즈 제거 ───────────────────────────────────────
    print(f'\n[Stage 2] clean  morph_kernel={args.morph_kernel}  min_px={args.min_px}')
    cleaned_u8 = clean_mask(gap_connected_u8, args.morph_kernel, args.min_px)
    print(f'  clean 후: {int((cleaned_u8 > 0).sum()):,}px')

    # ── Stage 3: 원본과 교집합 ─────────────────────────────────────────────────
    # 최종 = cleaned ∩ original
    # (gap선이 original에 없는 픽셀이면 제거 → 원본 범위 내 정제)
    final_u8 = cv2.bitwise_and(cleaned_u8, original_u8)

    orig_px  = int(original_bool.sum())
    final_px = int((final_u8 > 0).sum())
    print(f'\n[최종]  원본: {orig_px:,}px → 최종: {final_px:,}px  '
          f'(Δ {final_px - orig_px:+,}px)')

    # ── 저장 ──────────────────────────────────────────────────────────────────
    out_mask = out_dir / f'{stem}_processed.png'
    cv2.imwrite(str(out_mask), final_u8)
    print(f'\n  마스크 저장: {out_mask}')

    out_vis = out_dir / f'{stem}_processed_compare.png'
    visualize_all(
        original   = original_bool,
        gap_connected = gap_connected_bool,
        cleaned    = cleaned_u8,
        final      = final_u8,
        thin_pruned= thin_pruned,
        raw_pts    = raw_pts,
        proj_pts   = proj_pts,
        gap_only   = gap_only,
        n_connected= n_connected,
        out_path   = out_vis,
        stem       = stem,
        max_dist   = args.max_dist,
        min_branch = args.min_branch,
        morph_kernel = args.morph_kernel,
        min_px     = args.min_px,
    )
    print('\n완료!')


if __name__ == '__main__':
    main()
