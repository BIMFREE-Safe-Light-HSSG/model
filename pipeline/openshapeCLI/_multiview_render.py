"""
_multiview_render.py — numpy 기반 포인트클라우드 멀티뷰 렌더러
=============================================================
Open3D / CUDA 없이 순수 numpy로 구현.
orthographic projection + zbuffer 방식.

사용:
  from _multiview_render import render_views
  images = render_views(pts_norm)   # List[np.ndarray [H,W,3] uint8]
"""

import numpy as np

# ── 카메라 8방향 (azimuth°, elevation°) ────────────────────────────────────
VIEWS_8 = [
    (  0,  25),   # 정면
    ( 90,  25),   # 오른쪽
    (180,  25),   # 뒷면
    (270,  25),   # 왼쪽
    ( 45,  40),   # 앞-오른쪽 대각
    (225,  40),   # 뒤-왼쪽 대각
    (  0,  70),   # 위에서 내려다봄
    (  0, -15),   # 약간 아래에서
]

VIEWS_4 = [
    (  0,  25),
    ( 90,  25),
    (180,  25),
    (  0,  65),
]


def _rot_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """azimuth(수평 회전) + elevation(仰角) 회전 행렬 [3,3]"""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    # Z축 회전 (azimuth)
    Rz = np.array([
        [ np.cos(az), -np.sin(az), 0],
        [ np.sin(az),  np.cos(az), 0],
        [ 0,           0,          1],
    ], dtype=np.float32)

    # X축 회전 (elevation)
    Rx = np.array([
        [1, 0,           0          ],
        [0, np.cos(el), -np.sin(el) ],
        [0, np.sin(el),  np.cos(el) ],
    ], dtype=np.float32)

    return Rx @ Rz


def render_single_view(
    xyz: np.ndarray,          # [N, 3]  normalized to [-1, 1]
    rgb_u8: np.ndarray,       # [N, 3]  uint8
    azimuth: float,
    elevation: float,
    image_size: int = 224,
    bg_color: int = 220,      # 연한 회색 배경
) -> np.ndarray:
    """단일 시점 렌더링 → [H, W, 3] uint8"""
    R   = _rot_matrix(azimuth, elevation)
    pts = (R @ xyz.T).T                       # [N, 3] 회전된 좌표

    # orthographic projection
    #   x_cam → 이미지 x (horizontal)
    #   y_cam → depth (앞뒤)
    #   z_cam → 이미지 y (vertical, 위가 +)
    x_proj  = pts[:, 0]
    y_depth = pts[:, 1]
    z_proj  = pts[:, 2]

    margin = 0.05
    scale  = (1.0 - 2 * margin) * image_size / 2.0
    cx = cy = image_size // 2

    px = (x_proj *  scale + cx).astype(np.int32)
    py = (z_proj * -scale + cy).astype(np.int32)   # z+ → 이미지 위쪽

    # 범위 필터
    valid = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.full((image_size, image_size, 3), bg_color, dtype=np.uint8)

    # 포인트 크기 — 밀도에 따라 adaptive
    area_per_pt = image_size * image_size / n_valid
    r = max(1, int(np.sqrt(area_per_pt * 0.35)))

    # depth 순 정렬 (먼 것 먼저 → 가까운 것 나중에 덮음)
    v_idx  = np.where(valid)[0]
    order  = np.argsort(-y_depth[v_idx])   # 내림차순 depth

    img = np.full((image_size, image_size, 3), bg_color, dtype=np.uint8)

    for i in order:
        idx = v_idx[i]
        cx_pt, cy_pt = px[idx], py[idx]
        y1 = max(0, cy_pt - r);  y2 = min(image_size, cy_pt + r + 1)
        x1 = max(0, cx_pt - r);  x2 = min(image_size, cx_pt + r + 1)
        img[y1:y2, x1:x2] = rgb_u8[idx]

    return img


def render_views(
    pts_norm: np.ndarray,     # [N, 6]  XYZ normalized + RGB [0,1]
    views: list = None,       # None → VIEWS_8
    image_size: int = 224,
) -> list:
    """
    pts_norm : [N, 6]  ← normalize_pts + apply_rgb_norm 적용 후 입력 권장
    returns  : List of [H, W, 3] uint8  (len = len(views))
    """
    if views is None:
        views = VIEWS_8

    xyz    = pts_norm[:, :3]
    rgb    = pts_norm[:, 3:6]
    rgb_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    images = []
    for az, el in views:
        img = render_single_view(xyz, rgb_u8, az, el, image_size)
        images.append(img)
    return images
