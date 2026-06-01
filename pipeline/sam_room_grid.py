"""
sam_room_grid.py — SAM 기반 방 그리드 생성 (wall_door_2d.py 대체)
================================================================
wall_door_2d.py 의 flood fill 방 검출을
wall_process + SAM 방 인식으로 대체한다.

파이프라인:
  NPZ
    → [Step 1] build_baseline_mask  → original [H,W]
    → [Step 2] wall_process         → processed [H,W]  (gap 연결 + 노이즈 제거)
    → [Step 3] SAM 추론             → room masks
    → [Step 4] 마스크 필터링         → 방 마스크 목록
    → [Step 5] masks → room_grid    → grid [H,W] (0=벽, 1~N=방ID)
    → room_grid_{stem}.npz          ← stage4_sem_merge.py 입력 포맷

출력 포맷 (wall_door_2d.py 와 동일):
  grid       [H, W] int32  — 0=벽/미배정, 1~N=방 ID
  x_min, y_min, cell_size, W, H

사용법:
  python sam_room_grid.py --npz src/6_floor.npz
  python sam_room_grid.py --npz src/6_floor.npz --out_dir .
"""

import argparse
import pickle
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from scipy.ndimage import label as scipy_label
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
from room_pipeline import compute_density, build_wall_mask

# ── 상수 ──────────────────────────────────────────────────────────────────────
CEILING_MARGIN = 0.30942782733905927
WALL_CLASS     = 2
DOOR_CLASS     = 6
DEFAULT_CS     = 0.07084175512849847
SAM_CKPT       = BASE / 'sam_vit_h_4b8939.pth'
SAM_MODEL      = 'vit_h'


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1 — 베이스라인 마스크
# ═══════════════════════════════════════════════════════════════════════════════

def _load_npz(npz_path: Path):
    d           = np.load(npz_path)
    coord       = d['coord'].astype(np.float32)
    sp_labels   = d['sp_labels'].astype(np.int64)
    sp_pred     = d['sp_pred'].astype(np.int64)
    sp_features = d['sp_features'].astype(np.float32)
    sp_cents    = sp_features[:, 0:3]
    all_xy      = coord[:, :2]
    x_min, y_min = all_xy[:, 0].min(), all_xy[:, 1].min()
    x_max, y_max = all_xy[:, 0].max(), all_xy[:, 1].max()
    return coord, sp_labels, sp_pred, sp_cents, x_min, y_min, x_max, y_max


def _pts_to_density(xy, cs, x_min, y_min, W, H):
    density = np.zeros((H, W), dtype=np.int32)
    wx = np.clip(((xy[:, 0] - x_min) / cs).astype(int), 0, W - 1)
    wy = np.clip(((xy[:, 1] - y_min) / cs).astype(int), 0, H - 1)
    np.add.at(density, (wy, wx), 1)
    return density


def _remove_noise(mask, min_px):
    labeled, n = scipy_label(mask)
    result = mask.copy()
    for cid in range(1, n + 1):
        if (labeled == cid).sum() <= min_px:
            result[labeled == cid] = False
    return result


def build_baseline_mask(npz_path: Path, cs: float = DEFAULT_CS,
                        min_wall_px: int = 64):
    """NPZ → 벽+문 바이너리 마스크 (bool [H,W]) + 좌표계"""
    coord, sp_labels, sp_pred, sp_cents, x_min, y_min, x_max, y_max = _load_npz(npz_path)
    W = int((x_max - x_min) / cs) + 2
    H = int((y_max - y_min) / cs) + 2

    wall_xy = coord[np.isin(sp_labels, np.where(sp_pred == WALL_CLASS)[0]), :2]
    door_xy = coord[np.isin(sp_labels, np.where(sp_pred == DOOR_CLASS)[0]), :2]
    combined = (_pts_to_density(wall_xy, cs, x_min, y_min, W, H) >= 1) | \
               (_pts_to_density(door_xy, cs, x_min, y_min, W, H) >= 1)

    density_w75, *_ = compute_density(coord, sp_labels, sp_cents,
                                      cs, 75.0, CEILING_MARGIN)
    wall75 = build_wall_mask(density_w75, 60.0, min_px=min_wall_px)

    result = _remove_noise(combined | wall75, min_wall_px)
    print(f'  베이스라인: {H}×{W}  벽 픽셀: {result.sum():,}px')
    return result, x_min, y_min, cs, W, H


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2 — wall_process (gap 연결 + 노이즈 제거)
# ═══════════════════════════════════════════════════════════════════════════════

def _trace_dir(skel, px, py, trace_len=5):
    H, W = skel.shape
    visited = {(px, py)}
    cx, cy = px, py
    for _ in range(max(trace_len, 1)):
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H and skel[ny, nx] and (nx, ny) not in visited:
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
    return np.array([vx/n, vy/n], dtype=np.float32) if n > 1e-6 else np.zeros(2, dtype=np.float32)


def _skeleton_endpoints(skel):
    kern = np.ones((3, 3), dtype=np.uint8)
    cnt  = cv2.filter2D(skel.astype(np.uint8), -1, kern)
    ep   = skel & (cnt == 2)
    ys, xs = np.where(ep)
    return np.column_stack([xs, ys]).astype(np.float32) if len(ys) else np.empty((0,2), np.float32)


def _prune_skeleton(skel, min_branch_len=16):
    result = skel.copy().astype(bool)
    H, W   = result.shape
    changed = True
    while changed:
        changed = False
        kern = np.ones((3,3), dtype=np.uint8)
        cnt  = cv2.filter2D(result.astype(np.uint8), -1, kern)
        ep_ys, ep_xs = np.where(result & (cnt == 2))
        for py, px in zip(ep_ys.tolist(), ep_xs.tolist()):
            if not result[py, px]:
                continue
            branch, visited, cx, cy = [(px,py)], {(px,py)}, px, py
            while len(branch) < min_branch_len:
                nbrs = [(cx+dx, cy+dy)
                        for dy in (-1,0,1) for dx in (-1,0,1)
                        if not (dx==0 and dy==0)
                        and 0<=cx+dx<W and 0<=cy+dy<H
                        and result[cy+dy,cx+dx] and (cx+dx,cy+dy) not in visited]
                if not nbrs: break
                if len(nbrs) >= 2: break
                branch.append(nbrs[0]); visited.add(nbrs[0]); cx, cy = nbrs[0]
            if len(branch) < min_branch_len:
                for bx, by in branch: result[by,bx] = False
                changed = True
    return result


def _project_boundary(mask, skel, px, py, trace_len=5):
    inward = _trace_dir(skel, px, py, trace_len)
    out_dx, out_dy = -float(inward[0]), -float(inward[1])
    if abs(out_dx) < 1e-6 and abs(out_dy) < 1e-6:
        return px, py
    H, W = mask.shape
    cx, cy = float(px), float(py)
    lx, ly = px, py
    for _ in range(30):
        nx, ny = int(round(cx+out_dx)), int(round(cy+out_dy))
        if not (0<=nx<W and 0<=ny<H) or not mask[ny,nx]: break
        lx, ly = nx, ny; cx += out_dx; cy += out_dy
    return lx, ly


def apply_wall_process(original_bool: np.ndarray,
                       max_dist: float = 100.0,
                       min_cos: float = 0.9,
                       line_width: int = 1,
                       trace_len: int = 5,
                       min_branch: int = 16,
                       morph_kernel: int = 5,
                       min_px: int = 500) -> np.ndarray:
    """원본 벽 마스크 → gap 연결 + 노이즈 제거 → processed u8 마스크"""
    # ① gap 연결
    thin        = skeletonize(original_bool)
    thin_pruned = _prune_skeleton(thin, min_branch)
    raw_pts     = _skeleton_endpoints(thin_pruned)

    proj_pts = np.array([[*_project_boundary(original_bool, thin_pruned, int(p[0]), int(p[1]), trace_len)]
                          for p in raw_pts], dtype=np.float32) if len(raw_pts) else np.empty((0,2), np.float32)

    gap_canvas = np.zeros(original_bool.shape, dtype=np.uint8)
    if len(proj_pts) >= 2:
        D    = cdist(proj_pts, proj_pts)
        dirs = np.array([_trace_dir(thin_pruned, int(p[0]), int(p[1]), trace_len) for p in raw_pts])
        for i in range(len(proj_pts)):
            for j in range(i+1, len(proj_pts)):
                if D[i,j] > max_dist: continue
                if min_cos > -1.0:
                    conn = proj_pts[j] - proj_pts[i]
                    cn   = np.hypot(conn[0], conn[1])
                    if cn > 1e-6:
                        conn /= cn
                        if np.dot(dirs[i], conn) < min_cos or np.dot(dirs[j], -conn) < min_cos:
                            continue
                cv2.line(gap_canvas,
                         (int(proj_pts[i,0]), int(proj_pts[i,1])),
                         (int(proj_pts[j,0]), int(proj_pts[j,1])), 1, line_width)

    gap_connected_bool = original_bool | gap_canvas.astype(bool)
    gap_connected_u8   = gap_connected_bool.astype(np.uint8) * 255
    original_u8        = original_bool.astype(np.uint8) * 255

    # ② 모폴로지 + 노이즈 제거 → original과 교집합 (wall_process.py 동일 로직)
    if morph_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
        morph  = cv2.morphologyEx(gap_connected_u8, cv2.MORPH_CLOSE, kernel)
    else:
        morph = gap_connected_u8.copy()

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
    cleaned = np.zeros_like(morph)
    for lid in range(1, n_labels):
        if stats[lid, cv2.CC_STAT_AREA] >= min_px:
            cleaned[labels == lid] = 255

    final_u8 = cv2.bitwise_and(cleaned, original_u8)
    n_conn = int((gap_canvas.astype(bool) & ~original_bool).sum())
    print(f'  wall_process: gap+{n_conn:,}px  →  최종 {int((final_u8>0).sum()):,}px')
    return final_u8


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3 — SAM 추론
# ═══════════════════════════════════════════════════════════════════════════════

def mask_to_rgb(mask_bool: np.ndarray, invert: bool = True) -> np.ndarray:
    gray = (~mask_bool if invert else mask_bool).astype(np.uint8) * 255
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def run_sam(rgb_img: np.ndarray,
            points_per_side: int = 64,
            pred_iou_thresh: float = 0.88,
            stability_score_thresh: float = 0.95,
            min_mask_region_area: int = 0) -> list:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import torch

    print(f'  SAM 모델 로드: {SAM_CKPT.name}')
    sam = sam_model_registry[SAM_MODEL](checkpoint=str(SAM_CKPT))
    sam.eval()
    sam.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  디바이스: {"CUDA" if torch.cuda.is_available() else "CPU"}')

    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    print(f'  SAM 추론 중... ({rgb_img.shape[1]}×{rgb_img.shape[0]})')
    masks = gen.generate(rgb_img)
    print(f'  생성 마스크: {len(masks)}개')
    return masks


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4 — 마스크 필터링
# ═══════════════════════════════════════════════════════════════════════════════

def filter_room_masks(masks, min_area_px=0, containment_thr=0.7,
                      residual_thr_px=0, wall_mask=None, wall_ratio_thr=0.8):
    # ① 면적
    masks = [m for m in masks if m['area'] >= min_area_px]

    # ② 벽 비율
    if wall_mask is not None:
        wb = wall_mask.astype(bool)
        masks = [m for m in masks
                 if (m['segmentation'].astype(bool) & wb).sum() / max(m['area'],1) < wall_ratio_thr]
    print(f'  필터 후 마스크: {len(masks)}개')

    # ③ 잔여 면적 기반 중복 제거
    masks_sorted = sorted(masks, key=lambda m: m['area'], reverse=True)
    result = []
    for i, mi in enumerate(masks_sorted):
        seg_i = mi['segmentation'].astype(bool)
        cu = np.zeros_like(seg_i)
        has_child = False
        for j in range(i+1, len(masks_sorted)):
            seg_j = masks_sorted[j]['segmentation'].astype(bool)
            if (seg_i & seg_j).sum() / max(masks_sorted[j]['area'],1) >= containment_thr:
                cu |= seg_j; has_child = True
        if not has_child:
            result.append(mi)
        else:
            res_seg  = seg_i & ~cu
            res_area = int(res_seg.sum())
            if res_area >= residual_thr_px:
                nm = dict(mi); nm['segmentation'] = res_seg; nm['area'] = res_area
                result.append(nm)
    print(f'  최종 방 마스크: {len(result)}개')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 5 — masks → room_grid
# ═══════════════════════════════════════════════════════════════════════════════

def masks_to_room_grid(room_masks: list,
                       wall_mask_bool: np.ndarray,
                       H: int, W: int) -> tuple:
    """SAM room masks → room_grid [H,W] int32

    - 큰 방부터 ID 할당, 이미 할당된 픽셀은 덮어쓰지 않음
    - 미배정 픽셀(0)은 stage4_sem_merge.py가 nearest L1로 처리

    Returns
    -------
    grid   [H,W] int32  — 0=미배정, 1~N=방 ID
    n_rooms int
    """
    grid = np.zeros((H, W), dtype=np.int32)

    # 큰 방부터 할당 (작은 방이 겹치면 덮어쓰지 않음)
    sorted_masks = sorted(room_masks, key=lambda m: m['area'], reverse=True)
    for room_id, m in enumerate(sorted_masks, start=1):
        seg = m['segmentation'].astype(bool)
        assignable = seg & ~wall_mask_bool & (grid == 0)
        grid[assignable] = room_id
    n_rooms = len(sorted_masks)

    assigned_px   = int((grid > 0).sum())
    total_nonwall = int((~wall_mask_bool).sum())
    print(f'  room_grid: {n_rooms}개 방  '
          f'할당 {assigned_px:,}/{total_nonwall:,}px '
          f'({100*assigned_px/max(total_nonwall,1):.1f}%)')
    return grid, n_rooms


# ═══════════════════════════════════════════════════════════════════════════════
#  시각화
# ═══════════════════════════════════════════════════════════════════════════════

def visualize(original_bool, processed_u8, all_masks, out_path, stem):
    BG  = '#111'
    rng = np.random.default_rng(42)
    H, W = original_bool.shape

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    fig.patch.set_facecolor(BG)

    # ① 베이스라인
    axes[0].imshow(original_bool, cmap='gray', interpolation='nearest')
    axes[0].set_title(f'① 베이스라인\n({original_bool.sum():,}px)', color='white', fontsize=11)
    axes[0].axis('off')

    # ② wall_process 결과
    axes[1].imshow(processed_u8, cmap='gray', interpolation='nearest')
    axes[1].set_title(f'② wall_process\n({int((processed_u8>0).sum()):,}px)', color='white', fontsize=11)
    axes[1].axis('off')

    # ③ SAM 마스크 오버레이 (wall_process 배경)
    bg_rgb = cv2.cvtColor(processed_u8, cv2.COLOR_GRAY2RGB)
    axes[2].imshow(bg_rgb, interpolation='nearest')
    for m in all_masks:
        col = rng.random(3)
        overlay = np.ones((*m['segmentation'].shape, 4))
        overlay[:,:,:3] = col
        overlay[:,:,3]  = 0.5 * m['segmentation'].astype(float)
        axes[2].imshow(overlay)
    axes[2].set_title(f'③ SAM 마스크 전체\n({len(all_masks)}개)', color='white', fontsize=11)
    axes[2].axis('off')

    # ④ 마스크 커버리지 (픽셀이 몇 개 마스크에 포함되는지)
    coverage = np.zeros((H, W), dtype=np.int32)
    for m in all_masks:
        coverage += m['segmentation'].astype(np.int32)
    axes[3].imshow(coverage, cmap='hot', interpolation='nearest')
    covered_px = int((coverage > 0).sum())
    total_px   = H * W
    axes[3].set_title(f'④ 마스크 커버리지\n({covered_px:,}/{total_px:,}px, '
                      f'{100*covered_px/total_px:.1f}%)', color='white', fontsize=11)
    axes[3].axis('off')

    fig.suptitle(f'SAM Room Grid — {stem}', color='white', fontsize=13)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  시각화: {out_path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='wall_door_2d.py 대체 — SAM 기반 room_grid 생성')
    ap.add_argument('--npz',          required=True, help='입력 NPZ 파일')
    ap.add_argument('--out_dir',      default=None,  help='출력 디렉토리. 기본: 프로젝트 루트')
    ap.add_argument('--cell_size',    type=float, default=DEFAULT_CS)
    ap.add_argument('--min_wall_px',  type=int,   default=64,   help='노이즈 제거 최소 픽셀')

    # wall_process 파라미터
    ap.add_argument('--max_dist',     type=float, default=100.0)
    ap.add_argument('--min_cos',      type=float, default=0.9)
    ap.add_argument('--min_branch',   type=int,   default=16)
    ap.add_argument('--morph_kernel', type=int,   default=5)
    ap.add_argument('--min_px',       type=int,   default=500)

    # SAM 파라미터
    ap.add_argument('--points_per_side', type=int,   default=64)
    ap.add_argument('--pred_iou',        type=float, default=0.88)
    ap.add_argument('--stability',       type=float, default=0.95)

    # 방 필터 파라미터
    ap.add_argument('--min_room_m2',  type=float, default=2.0)
    ap.add_argument('--residual_m2',  type=float, default=3.0)
    ap.add_argument('--wall_ratio',   type=float, default=0.8)
    args = ap.parse_args()

    # ── 경로 결정 ──────────────────────────────────────────────────────────────
    npz_path = Path(args.npz)
    if not npz_path.exists():
        alt = BASE / 'src' / npz_path.name
        npz_path = alt if alt.exists() else npz_path
    if not npz_path.exists():
        sys.exit(f'NPZ 없음: {args.npz}')
    if not SAM_CKPT.exists():
        sys.exit(f'SAM 가중치 없음: {SAM_CKPT}')

    stem    = npz_path.stem
    out_dir = Path(args.out_dir) if args.out_dir else BASE
    out_dir.mkdir(parents=True, exist_ok=True)
    sam_cache_dir = BASE / 'results' / 'sam_out'
    sam_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  SAM Room Grid — {stem}')
    print(f'{"="*60}')

    # ── Step 1: 베이스라인 ────────────────────────────────────────────────────
    print(f'\n[Step 1] 베이스라인 마스크 생성')
    original_bool, x_min, y_min, cs, W, H = build_baseline_mask(
        npz_path, args.cell_size, args.min_wall_px)

    # ── Step 2: wall_process ──────────────────────────────────────────────────
    print(f'\n[Step 2] wall_process  '
          f'(max_dist={args.max_dist}, min_cos={args.min_cos}, '
          f'min_branch={args.min_branch}, morph={args.morph_kernel})')
    processed_u8 = apply_wall_process(
        original_bool,
        max_dist=args.max_dist, min_cos=args.min_cos,
        min_branch=args.min_branch, morph_kernel=args.morph_kernel,
        min_px=args.min_px,
    )
    processed_bool = processed_u8 > 0

    # ── Step 3: SAM 추론 (캐시) ───────────────────────────────────────────────
    cache_path = sam_cache_dir / f'sam_cache_{stem}_room_grid.pkl'
    print(f'\n[Step 3] SAM 추론  (points_per_side={args.points_per_side})')
    if cache_path.exists():
        print(f'  캐시 로드: {cache_path.name}')
        with open(cache_path, 'rb') as f:
            all_masks = pickle.load(f)
        print(f'  캐시 마스크: {len(all_masks)}개')
    else:
        rgb_img   = mask_to_rgb(processed_bool, invert=True)
        all_masks = run_sam(rgb_img,
                            points_per_side=args.points_per_side,
                            pred_iou_thresh=args.pred_iou,
                            stability_score_thresh=args.stability)
        with open(cache_path, 'wb') as f:
            pickle.dump(all_masks, f)
        print(f'  캐시 저장: {cache_path.name}')

    # ── Step 4: 마스크 필터링 ─────────────────────────────────────────────────
    print(f'\n[Step 4] 방 마스크 필터링')
    min_area_px    = int(args.min_room_m2 / (cs ** 2))
    residual_thr   = int(args.residual_m2 / (cs ** 2))
    print(f'  min_area={args.min_room_m2}m²→{min_area_px}px  '
          f'residual={args.residual_m2}m²→{residual_thr}px  '
          f'wall_ratio<{args.wall_ratio:.0%}')
    room_masks = filter_room_masks(
        all_masks,
        min_area_px=min_area_px,
        containment_thr=0.7,
        residual_thr_px=residual_thr,
        wall_mask=original_bool,
        wall_ratio_thr=args.wall_ratio,
    )

    # ── Step 5: SAM 마스크 스택 저장 (masks_to_room_grid 사용 안 함) ──────────
    print(f'\n[Step 5] SAM 마스크 저장')
    # 큰 마스크부터 정렬 (stage4 assign 시 작은 마스크 우선 배정을 위해)
    all_masks_sorted = sorted(all_masks, key=lambda m: m['area'], reverse=True)
    mask_stack = np.stack(
        [m['segmentation'].astype(np.uint8) for m in all_masks_sorted], axis=0
    )   # [N, H, W]  uint8 (0/1)
    print(f'  마스크 스택: {mask_stack.shape}  ({len(all_masks_sorted)}개 마스크)')

    # ── 저장 ──────────────────────────────────────────────────────────────────
    out_npz = out_dir / f'room_grid_{stem}.npz'
    np.savez_compressed(
        out_npz,
        masks     = mask_stack,
        x_min     = np.float32(x_min),
        y_min     = np.float32(y_min),
        cell_size = np.float32(cs),
        W         = np.int32(W),
        H         = np.int32(H),
    )
    print(f'\n  마스크 저장: {out_npz}  ({len(all_masks_sorted)}개 마스크)')

    # ── 시각화 ────────────────────────────────────────────────────────────────
    vis_path = sam_cache_dir / f'sam_room_grid_{stem}.png'
    visualize(original_bool, processed_u8, all_masks, vis_path, stem)

    print(f'\n완료! → {out_npz}')


if __name__ == '__main__':
    main()
