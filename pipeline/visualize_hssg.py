"""
visualize_hssg.py
=================
stage6_results/scene_graph.json 을 읽어 Open3D 로 3D 시각화한다.
단층 및 다층(BUILDING → FLOOR → ZONE → ASSET) 모두 지원한다.

표시 요소:
  - ZONE 폴리곤  : 방 경계를 바닥에 LineSets 으로 표시 (층별 Z 오프셋 적용)
  - ZONE 면      : 반투명 Mesh 로 방 영역 표시
  - Asset 구체   : 클래스별 색상 구체 + 텍스트 레이블
  - Edge 선      : L1-L1 관계선 (relation 별 색상)
  - FLOOR bbox   : 각 층 경계 박스 표시

실행:
    python visualize_hssg.py
    python visualize_hssg.py --json results/scene_graph.json
    python visualize_hssg.py --no_edges          # 관계선 숨김
    python visualize_hssg.py --no_zones          # 방 면 숨김
    python visualize_hssg.py --no_assets         # asset 구체 숨김
    python visualize_hssg.py --class chair       # 특정 클래스만 표시
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d 미설치.  pip install open3d")

# ── 경로 기본값 ───────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent
_JSON_DEFAULT = _ROOT / 'results' / 'scene_graph.json'

# ── 클래스별 색상 (S3DIS) ──────────────────────────────────────────────────────
CLASS_COLORS: dict[str, list] = {
    'ceiling':  [0.80, 0.80, 0.80],
    'floor':    [0.60, 0.55, 0.45],
    'wall':     [1.00, 0.73, 0.47],
    'beam':     [0.55, 0.40, 0.25],
    'column':   [0.30, 0.75, 0.93],
    'window':   [0.50, 0.85, 0.70],
    'door':     [0.95, 0.70, 0.30],
    'chair':    [0.20, 0.60, 1.00],
    'table':    [0.95, 0.45, 0.45],
    'bookcase': [0.60, 0.40, 0.80],
    'sofa':     [0.95, 0.65, 0.80],
    'board':    [0.40, 0.80, 0.40],
    'clutter':  [0.65, 0.65, 0.65],
}
DEFAULT_COLOR = [0.70, 0.70, 0.70]

# ZONE 팔레트 (방별 색상, 8가지 순환)
_ZONE_PALETTE = [
    [0.49, 0.72, 1.00],
    [0.43, 0.91, 0.72],
    [0.98, 0.75, 0.14],
    [0.77, 0.71, 0.98],
    [0.98, 0.66, 0.83],
    [0.40, 0.91, 0.97],
    [0.99, 0.73, 0.45],
    [0.64, 0.71, 0.98],
]

# 관계선 색상
_RELATION_COLORS: dict[str, list] = {
    'none':            [0.30, 0.30, 0.30],
    'lying on':        [0.95, 0.60, 0.20],
    'hanging from':    [0.20, 0.80, 0.90],
    'standing on':     [0.40, 0.90, 0.40],
    'attached to':     [0.90, 0.40, 0.40],
    'part of':         [0.80, 0.40, 0.80],
    'above':           [0.90, 0.90, 0.20],
    'below':           [0.20, 0.60, 0.90],
    'next to':         [0.70, 0.70, 0.70],
    'connected to':    [0.40, 0.70, 0.40],
    'supported by':    [0.60, 0.40, 0.20],
}
_DEFAULT_REL_COLOR = [0.50, 0.50, 0.50]


# ── Ear-clipping 삼각분할 (외부 라이브러리 없이 오목 폴리곤 처리) ──────────────
def _earcut(polygon: list) -> list:
    """
    2D 다각형 [[x,y], ...] → 삼각형 인덱스 리스트 [[i,j,k], ...]
    Ear-clipping 알고리즘 (O(n²), 단순 다각형에 적합).
    """
    pts  = [list(p) for p in polygon]
    idx  = list(range(len(pts)))
    tris = []

    def _cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def _in_triangle(p, a, b, c):
        d1 = _cross(a, b, p)
        d2 = _cross(b, c, p)
        d3 = _cross(c, a, p)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    # 반시계방향 보장
    area = sum(_cross(pts[idx[i]], pts[idx[(i+1) % len(idx)]],
                      pts[idx[(i+2) % len(idx)]])
               for i in range(len(idx)))
    if area < 0:
        idx.reverse()

    max_iter = len(idx) * len(idx) + 10
    it = 0
    while len(idx) > 3 and it < max_iter:
        it += 1
        n = len(idx)
        found = False
        for i in range(n):
            prev_i = idx[(i - 1) % n]
            curr_i = idx[i]
            next_i = idx[(i + 1) % n]
            a, b, c = pts[prev_i], pts[curr_i], pts[next_i]
            if _cross(a, b, c) <= 0:
                continue   # 오목 꼭짓점
            # 내부에 다른 꼭짓점 있는지 확인
            ear = True
            for j in range(n):
                ji = idx[j]
                if ji in (prev_i, curr_i, next_i):
                    continue
                if _in_triangle(pts[ji], a, b, c):
                    ear = False
                    break
            if ear:
                tris.append([prev_i, curr_i, next_i])
                idx.pop(i)
                found = True
                break
        if not found:
            break   # 퇴화 폴리곤 → 중단

    if len(idx) == 3:
        tris.append(idx[:])
    return tris


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def _sphere(center, radius: float, color: list) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
    s.translate(center)
    s.paint_uniform_color(color)
    s.compute_vertex_normals()
    return s


def _lineset(points: list, lines: list, color: list) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines,  dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(color, (len(lines), 1)).astype(np.float64))
    return ls


def _thick_line(p0: list, p1: list, color: list,
                radius: float = 0.08) -> o3d.geometry.TriangleMesh:
    """두 점을 잇는 원기둥 메시 — Open3D LineSet은 굵기를 지원하지 않으므로
    두꺼운 선이 필요할 때 이 함수를 사용한다."""
    p0 = np.array(p0, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return _sphere(p0.tolist(), radius, color)

    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=12, split=1)
    cyl.paint_uniform_color(color)
    cyl.compute_vertex_normals()

    # 기본 원기둥은 Z축 방향 → 목표 방향으로 회전
    z_axis = np.array([0.0, 0.0, 1.0])
    direction = vec / length
    axis = np.cross(z_axis, direction)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        # 이미 Z축과 같거나 반대
        if direction[2] < 0:
            cyl.rotate(cyl.get_rotation_matrix_from_axis_angle(
                np.array([1.0, 0.0, 0.0]) * np.pi))
    else:
        angle = float(np.arcsin(np.clip(axis_len, -1, 1)))
        if np.dot(z_axis, direction) < 0:
            angle = np.pi - angle
        cyl.rotate(cyl.get_rotation_matrix_from_axis_angle(
            axis / axis_len * angle))

    # 중심을 p0~p1 중간으로 이동
    mid = (p0 + p1) / 2
    cyl.translate(mid)
    return cyl


def _zone_color(index: int) -> list:
    return _ZONE_PALETTE[index % len(_ZONE_PALETTE)]


# ── 로드 ─────────────────────────────────────────────────────────────────────
def load_scene_graph(json_path: str) -> dict:
    with open(json_path, encoding='utf-8') as f:
        raw = json.load(f)

    # 최상위가 {building_id, scene_graph} 구조인 경우 대응
    if 'scene_graph' in raw:
        sg = raw['scene_graph']
    else:
        sg = raw

    zones  = [n for n in sg.get('nodes', []) if n.get('type') == 'ZONE']
    floors = [n for n in sg.get('nodes', []) if n.get('type') == 'FLOOR']
    assets_global = sg.get('assets', [])
    edges  = sg.get('edges', [])

    # asset_id → (position, class, zone_id)
    asset_map: dict = {}
    for zi, zone in enumerate(zones):
        for a in zone.get('assets', []):
            asset_map[a['id']] = {
                'pos':     a['position'],
                'cls':     a.get('class', '기타'),
                'status':  a.get('status', 'normal'),
                'zone_id': zi,
            }
    for a in assets_global:
        asset_map[a['id']] = {
            'pos':     a['position'],
            'cls':     a.get('class', '기타'),
            'status':  a.get('status', 'normal'),
            'zone_id': -1,
        }

    return {
        'floors':       floors,
        'zones':        zones,
        'asset_map':    asset_map,
        'edges':        edges,
        'building_id':  raw.get('building_id', 'BLD'),
    }


# ── geometry 생성 ──────────────────────────────────────────────────────────────
def _zone_floor_z(zone: dict) -> float:
    """ZONE의 바닥 Z 좌표를 결정한다.
    center.z → assets min z → 0.0 순으로 fallback.
    """
    geo = zone.get('geometry', {})
    center = geo.get('center', None)
    if center and len(center) >= 3:
        # center.z는 대략 방 중간이므로 바닥은 약간 낮음 — 그대로 사용 (시각적으로 충분)
        return float(center[2]) - 1.0   # center에서 -1m 내려서 바닥 근사

    # assets 중 최솟값
    zs = [a['position'][2] for a in zone.get('assets', [])
          if len(a.get('position', [])) >= 3]
    if zs:
        return float(min(zs))
    return 0.0


def make_zone_geometries(zones: list,
                          show_fill: bool = True,
                          show_outline: bool = True,
                          floor_z: float | None = None,
                          room_height: float = 2.5) -> list:
    """방 경계 폴리곤 + 바닥 면 생성.

    floor_z    : 고정 바닥 z 좌표. None 이면 각 zone 의 center/asset Z 에서 자동 결정.
                 멀티플로어 scene_graph 에서는 None 을 사용해야 층별 Z 가 올바르게 적용된다.
    room_height: 3D 박스 높이 (기본 2.5m)
    """
    geoms = []

    for zi, zone in enumerate(zones):
        geo    = zone.get('geometry', {})
        coords = geo.get('coordinates', [])   # [[x, y], ...]

        if len(coords) < 3:
            continue

        color  = _zone_color(zi)
        fz     = floor_z if floor_z is not None else _zone_floor_z(zone)
        top_z  = fz + room_height

        # ── 바닥 외곽선 (2D 루프, 바닥면) ────────────────────────────────
        if show_outline:
            pts_floor = [[float(c[0]), float(c[1]), fz]    for c in coords]
            pts_top   = [[float(c[0]), float(c[1]), top_z] for c in coords]
            pts_all   = pts_floor + pts_top
            n = len(coords)
            lines = []
            for i in range(n):                   # 바닥 테두리
                lines.append([i, (i + 1) % n])
            for i in range(n):                   # 천장 테두리
                lines.append([n + i, n + (i + 1) % n])
            for i in range(n):                   # 수직 기둥
                lines.append([i, n + i])
            geoms.append(_lineset(pts_all, lines, color))

        # ── 바닥 채움 면 (Ear-clipping 삼각분할) ─────────────────────────
        if show_fill and len(coords) >= 3:
            poly2d = [[float(c[0]), float(c[1])] for c in coords]
            tris   = _earcut(poly2d)
            if tris:
                verts = np.array(
                    [[float(c[0]), float(c[1]), fz + 0.01] for c in coords],
                    dtype=np.float64)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices  = o3d.utility.Vector3dVector(verts)
                mesh.triangles = o3d.utility.Vector3iVector(
                    np.array(tris, dtype=np.int32))
                mesh.paint_uniform_color(color)
                mesh.compute_vertex_normals()
                geoms.append(mesh)

    return geoms


def make_zone_membership_lines(zones: list, asset_map: dict) -> list:
    """각 Zone 중심 → 소속 Asset을 잇는 멤버십 선으로 계층 관계를 표현한다.

    Zone center는 geometry.center → asset centroid 순으로 fallback.
    zone_id=-1인 global asset은 건너뜀.

    최적화: 모든 선을 하나의 LineSet으로 배치 (개별 LineSet 생성 시 22k+ 오브젝트 → OOM).
    """
    all_pts    = []
    all_lines  = []
    all_colors = []

    for zi, zone in enumerate(zones):
        geo    = zone.get('geometry', {})
        center = geo.get('center', None)
        if center and len(center) >= 3:
            zone_center = [float(center[0]), float(center[1]), float(center[2])]
        else:
            positions = [asset_map[a['id']]['pos']
                         for a in zone.get('assets', [])
                         if a['id'] in asset_map]
            if not positions:
                continue
            zone_center = [
                sum(p[0] for p in positions) / len(positions),
                sum(p[1] for p in positions) / len(positions),
                sum(p[2] for p in positions) / len(positions),
            ]

        color = [c * 0.6 for c in _zone_color(zi)]   # 선은 약간 어둡게
        for a in zone.get('assets', []):
            if a['id'] not in asset_map:
                continue
            if asset_map[a['id']]['cls'] == 'clutter':
                continue
            pos = asset_map[a['id']]['pos']
            base = len(all_pts)
            all_pts.append(zone_center)
            all_pts.append([float(pos[0]), float(pos[1]), float(pos[2])])
            all_lines.append([base, base + 1])
            all_colors.append(color)

    if not all_pts:
        return []

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(all_pts,    dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(all_lines,  dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.array(all_colors, dtype=np.float64))
    return [ls]


def make_asset_geometries(asset_map: dict,
                           filter_cls: str | None = None,
                           radius: float = 0.18,
                           color_by: str = 'class') -> list:
    """asset 구체 생성.

    color_by:
      'class' (기본) — 객체 클래스별 색상
      'zone'         — 소속 Zone별 색상 (같은 방의 객체는 같은 색)
      'floor'        — 층별 색상 (5F=파랑, 6F=주황)

    최적화: 모든 구체를 하나의 merged TriangleMesh로 배치
    (개별 구체 6k+개 → 22k+ draw call → OOM 방지).
    구체 해상도 8→6으로 낮춰 폴리곤 수 추가 절감.
    """
    _FLOOR_COLORS = [
        [0.20, 0.60, 1.00],   # 0번째 층 — 파랑
        [1.00, 0.55, 0.15],   # 1번째 층 — 주황
        [0.20, 0.85, 0.50],   # 2번째 층 — 초록
        [0.90, 0.20, 0.70],   # 3번째 층 — 분홍
    ]

    # 템플릿 구체 (resolution=6: vertex 50개, triangle 96개)
    tmpl   = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
    t_verts = np.asarray(tmpl.vertices,   dtype=np.float64)   # [V, 3]
    t_tris  = np.asarray(tmpl.triangles,  dtype=np.int32)     # [F, 3]
    n_v     = len(t_verts)

    all_verts  = []
    all_tris   = []
    all_colors = []
    v_offset   = 0

    for aid, info in asset_map.items():
        cls = info['cls']
        if cls == 'clutter':
            continue
        if filter_cls and cls != filter_cls:
            continue
        pos  = np.array(info['pos'], dtype=np.float64)
        zid  = info.get('zone_id', -1)

        if color_by == 'zone':
            color = _zone_color(zid) if zid >= 0 else DEFAULT_COLOR
        elif color_by == 'floor':
            color = _FLOOR_COLORS[zid % len(_FLOOR_COLORS)] if zid >= 0 else DEFAULT_COLOR
        else:
            color = CLASS_COLORS.get(cls, DEFAULT_COLOR)

        all_verts.append(t_verts + pos)
        all_tris.append(t_tris + v_offset)
        all_colors.append(np.tile(color, (n_v, 1)))
        v_offset += n_v

    if not all_verts:
        return []

    merged = o3d.geometry.TriangleMesh()
    merged.vertices      = o3d.utility.Vector3dVector(np.concatenate(all_verts))
    merged.triangles     = o3d.utility.Vector3iVector(np.concatenate(all_tris))
    merged.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(all_colors))
    merged.compute_vertex_normals()
    return [merged]


def make_edge_geometries(edges: list,
                          asset_map: dict,
                          filter_cls: str | None = None) -> list:
    """asset 간 관계선 생성.

    최적화: 모든 관계선을 하나의 LineSet으로 배치
    (개별 LineSet 8k+개 → 8k+ draw call → OOM 방지).
    """
    all_pts    = []
    all_lines  = []
    all_colors = []

    for e in edges:
        src_id = e.get('src') or e.get('source')
        dst_id = e.get('dst') or e.get('target')
        rel    = e.get('relation', 'none')

        if src_id not in asset_map or dst_id not in asset_map:
            continue
        src_cls = asset_map[src_id]['cls']
        dst_cls = asset_map[dst_id]['cls']
        if src_cls == 'clutter' or dst_cls == 'clutter':
            continue
        if filter_cls:
            if src_cls != filter_cls and dst_cls != filter_cls:
                continue

        sp    = asset_map[src_id]['pos']
        dp    = asset_map[dst_id]['pos']
        color = _RELATION_COLORS.get(rel, _DEFAULT_REL_COLOR)
        base  = len(all_pts)
        all_pts.append([float(sp[0]), float(sp[1]), float(sp[2])])
        all_pts.append([float(dp[0]), float(dp[1]), float(dp[2])])
        all_lines.append([base, base + 1])
        all_colors.append(color)

    if not all_pts:
        return []

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(all_pts,    dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(all_lines,  dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.array(all_colors, dtype=np.float64))
    return [ls]


def make_floor_bbox_geometries(floors: list) -> list:
    """각 FLOOR 노드의 bounding box를 와이어프레임으로 시각화한다."""
    geoms = []
    floor_colors = [
        [0.20, 0.60, 1.00],   # 5F — 파랑 계열
        [1.00, 0.50, 0.15],   # 6F — 주황 계열
        [0.20, 0.85, 0.50],   # 7F — 초록 계열
        [0.90, 0.20, 0.70],   # 8F — 분홍 계열
    ]
    for fi, fl in enumerate(floors):
        geo = fl.get('geometry', {})
        xr  = geo.get('x_range', None)
        yr  = geo.get('y_range', None)
        zr  = geo.get('z_range', None)
        if not (xr and yr and zr):
            continue
        x0, x1 = float(xr[0]), float(xr[1])
        y0, y1 = float(yr[0]), float(yr[1])
        z0, z1 = float(zr[0]), float(zr[1])

        # 8꼭짓점 BBox
        pts = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
        lines = [
            [0,1],[1,2],[2,3],[3,0],   # 바닥
            [4,5],[5,6],[6,7],[7,4],   # 천장
            [0,4],[1,5],[2,6],[3,7],   # 기둥
        ]
        color = floor_colors[fi % len(floor_colors)]
        geoms.append(_lineset(pts, lines, color))

    return geoms


def make_zone_center_markers(zones: list, radius: float = 0.30) -> list:
    """방 중심 구체 (방 구분용)."""
    geoms = []
    for zi, zone in enumerate(zones):
        geo    = zone.get('geometry', {})
        center = geo.get('center', None)
        if center is None:
            continue
        color = _zone_color(zi)
        # 조금 더 진하게
        dark  = [c * 0.7 for c in color]
        geoms.append(_sphere(
            [float(center[0]), float(center[1]), float(center[2])],
            radius, dark))
    return geoms


def _floor_center(fl: dict) -> list | None:
    """FLOOR 노드의 기하 중심 [x, y, z]를 반환. geometry 없으면 None."""
    geo = fl.get('geometry', {})
    xr = geo.get('x_range')
    yr = geo.get('y_range')
    zr = geo.get('z_range')
    if not (xr and yr and zr):
        return None
    return [
        (float(xr[0]) + float(xr[1])) / 2,
        (float(yr[0]) + float(yr[1])) / 2,
        (float(zr[0]) + float(zr[1])) / 2,
    ]


def make_floor_links(floors: list) -> list:
    """층끼리 연결선 — 인접 FLOOR 중심을 굵은 선으로 잇는다.

    멀티플로어 씬에서 "이 두 층이 같은 건물에 속한다"는 계층 관계를 표현.
    색상: 두 층 색의 혼합 (흰색 계열 굵은 선).
    """
    _FLOOR_COLORS = [
        [0.20, 0.60, 1.00],
        [1.00, 0.50, 0.15],
        [0.20, 0.85, 0.50],
        [0.90, 0.20, 0.70],
    ]
    geoms = []
    centers = []
    for fi, fl in enumerate(floors):
        c = _floor_center(fl)
        if c:
            centers.append((fi, c))

    # 인접 층끼리 연결 (순서대로) — 두꺼운 원기둥으로 표현
    for i in range(len(centers) - 1):
        fi, c0 = centers[i]
        fj, c1 = centers[i + 1]
        col0 = _FLOOR_COLORS[fi % len(_FLOOR_COLORS)]
        col1 = _FLOOR_COLORS[fj % len(_FLOOR_COLORS)]
        color = [(col0[k] + col1[k]) / 2 for k in range(3)]
        geoms.append(_thick_line(c0, c1, color, radius=0.15))
        # 양 끝에 구체로 마감
        geoms.append(_sphere(c0, 0.20, col0))
        geoms.append(_sphere(c1, 0.20, col1))

    return geoms


def make_zone_links(floors: list, zones: list) -> list:
    """층 안에서 방끼리 연결선 — 같은 FLOOR에 속한 ZONE 중심을 허브-스포크로 잇는다.

    FLOOR 중심 → 각 ZONE 중심 선으로 "이 방들이 이 층에 묶여 있다"를 표현.
    각 층의 색상으로 선을 그린다.
    """
    _FLOOR_COLORS = [
        [0.20, 0.60, 1.00],
        [1.00, 0.50, 0.15],
        [0.20, 0.85, 0.50],
        [0.90, 0.20, 0.70],
    ]
    # zone_id → zone 객체 맵
    zone_by_id = {z['id']: z for z in zones}

    geoms = []
    for fi, fl in enumerate(floors):
        floor_center = _floor_center(fl)
        if not floor_center:
            continue
        color = [c * 0.8 for c in _FLOOR_COLORS[fi % len(_FLOOR_COLORS)]]

        for zone_id in fl.get('zones', []):
            zone = zone_by_id.get(zone_id)
            if not zone:
                continue
            geo    = zone.get('geometry', {})
            center = geo.get('center')
            if not center or len(center) < 3:
                continue
            zc = [float(center[0]), float(center[1]), float(center[2])]
            # 두꺼운 원기둥으로 Floor→Zone 연결 표현
            geoms.append(_thick_line(floor_center, zc, color, radius=0.06))

    return geoms


# ── 통계 출력 ─────────────────────────────────────────────────────────────────
def print_stats(data: dict) -> None:
    floors    = data.get('floors', [])
    zones     = data['zones']
    asset_map = data['asset_map']
    edges     = data['edges']

    from collections import Counter
    cls_cnt = Counter(v['cls'] for v in asset_map.values())
    rel_cnt = Counter(e.get('relation', 'none') for e in edges)

    print(f"\n{'='*50}")
    print(f"  building_id : {data['building_id']}")
    print(f"  FLOOR (층)  : {len(floors)}개")
    for fl in floors:
        geo = fl.get('geometry', {})
        xr  = geo.get('x_range', [0, 0])
        yr  = geo.get('y_range', [0, 0])
        zr  = geo.get('z_range', [0, 0])
        print(f"    {fl['id']} | {fl.get('name','층')} | "
              f"x=[{xr[0]:.1f},{xr[1]:.1f}] y=[{yr[0]:.1f},{yr[1]:.1f}] "
              f"z=[{zr[0]:.1f},{zr[1]:.1f}] | zones={len(fl.get('zones',[]))}")
    print(f"  ZONE (방)   : {len(zones)}개")
    print(f"  Asset       : {len(asset_map)}개")
    print(f"  Edge (관계) : {len(edges)}개")
    print(f"\n  ── 클래스 분포 ──")
    for cls, n in cls_cnt.most_common():
        bar = '█' * int(n / max(cls_cnt.values()) * 20)
        print(f"  {cls:<12} {n:>4}개  {bar}")
    print(f"\n  ── 관계 분포 (상위 8) ──")
    for rel, n in rel_cnt.most_common(8):
        print(f"  {rel:<20} {n:>4}개")
    print(f"{'='*50}\n")


# ── 레전드 출력 ───────────────────────────────────────────────────────────────
def _ansi_swatch(r: int, g: int, b: int, label: str) -> str:
    """터미널 트루컬러 블록 ██ + 라벨 문자열 반환."""
    block = f"\033[38;2;{r};{g};{b}m██\033[0m"
    return f"  {block}  {label}"


def print_legend(color_by: str = 'class') -> None:
    print("\n[범례]")
    if color_by == 'class':
        print("  구체 색상 = 객체 클래스")
        for cls, col in CLASS_COLORS.items():
            r, g, b = [int(c * 255) for c in col]
            print(_ansi_swatch(r, g, b, cls))
    elif color_by == 'zone':
        print("  구체 색상 = Zone(방) 소속  →  같은 색 = 같은 방")
        for i, col in enumerate(_ZONE_PALETTE):
            r, g, b = [int(c * 255) for c in col]
            print(_ansi_swatch(r, g, b, f"Zone 팔레트 {i}"))
    elif color_by == 'floor':
        print("  구체 색상 = Floor(층) 소속")

    print("\n  계층 멤버십 선  →  Zone 중심으로 모이는 가는 선 (같은 색 = 같은 방)")
    print("\n  관계선 색상 (semantic edges)")
    for rel, col in list(_RELATION_COLORS.items())[:6]:
        r, g, b = [int(c * 255) for c in col]
        print(_ansi_swatch(r, g, b, rel))
    print()


# ── 뷰어 실행 ─────────────────────────────────────────────────────────────────
def show(geoms: list, title: str = 'HSSG Viewer') -> None:
    o3d.visualization.draw_geometries(
        geoms,
        window_name=title,
        width=1400, height=900,
        point_show_normal=False,
    )


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='HSSG scene_graph.json 3D 뷰어')
    parser.add_argument('--json',       default=None,
                        help='scene_graph.json 경로 직접 지정')
    parser.add_argument('--src',        action='store_true',
                        help='results/ 폴더 안의 JSON 파일 목록에서 선택')
    parser.add_argument('--no_zones',   action='store_true',
                        help='방 면(반투명) 숨김')
    parser.add_argument('--no_outline', action='store_true',
                        help='방 외곽선 숨김')
    parser.add_argument('--no_assets',  action='store_true',
                        help='asset 구체 숨김')
    parser.add_argument('--no_edges',   action='store_true',
                        help='관계선 숨김')
    parser.add_argument('--no_centers',   action='store_true',
                        help='방 중심 마커 숨김')
    parser.add_argument('--no_floor_bbox', action='store_true',
                        help='층 bounding box 숨김 (단층 파일에서 유용)')
    parser.add_argument('--floor_z',      type=float, default=None,
                        help='단층 파일용 고정 바닥 Z (미지정 시 각 zone에서 자동)')
    parser.add_argument('--class',        dest='filter_cls', default=None,
                        help='특정 클래스만 표시 (예: chair)')
    parser.add_argument('--radius',       type=float, default=0.18,
                        help='asset 구체 반지름 (기본 0.18m)')
    # ── 계층 구조 시각화 ──────────────────────────────────────────────────────
    parser.add_argument('--no_hierarchy', action='store_true',
                        help='Zone→Asset 계층 멤버십 선 숨김 (기본은 표시)')
    parser.add_argument('--floor_links',  action='store_true',
                        help='층끼리 연결선 표시 — 인접 FLOOR 중심을 잇는 선 (기본 숨김)')
    parser.add_argument('--zone_links',   action='store_true',
                        help='층 안에서 방끼리 연결선 표시 — FLOOR 중심 → ZONE 중심 허브-스포크 (기본 숨김)')
    parser.add_argument('--color_by',     choices=['class', 'zone', 'floor'],
                        default='class',
                        help='asset 구체 색상 기준: class(기본)/zone/floor. '
                             'zone은 같은 방 객체들을 동일 색으로 표시하여 '
                             '계층 소속 관계를 직관적으로 표현한다.')
    args = parser.parse_args()

    if args.src:
        results_dir = _ROOT / 'results'
        jsons = sorted(results_dir.glob('*.json'))
        if not jsons:
            print(f"[오류] results/ 폴더에 JSON 파일 없음: {results_dir}")
            return
        if len(jsons) == 1:
            json_path = jsons[0]
            print(f"  [자동 선택] {json_path.name}")
        else:
            print(f"\n📂 results/ 폴더의 JSON 파일 목록:")
            for i, f in enumerate(jsons):
                size_kb = f.stat().st_size / 1024
                print(f"  [{i+1}] {f.name}  ({size_kb:.0f} KB)")
            while True:
                try:
                    choice = int(input(f"\n번호 선택 (1~{len(jsons)}): ").strip())
                    if 1 <= choice <= len(jsons):
                        json_path = jsons[choice - 1]
                        break
                except (ValueError, KeyboardInterrupt):
                    pass
                print(f"  1~{len(jsons)} 사이의 번호를 입력하세요.")
    else:
        json_path = Path(args.json) if args.json else _JSON_DEFAULT

    if not json_path.exists():
        print(f"[오류] JSON 파일 없음: {json_path}")
        return

    print(f"[로드] {json_path}")
    data = load_scene_graph(str(json_path))
    print_stats(data)
    print_legend(color_by=args.color_by)

    geoms = []
    is_multifloor = len(data['floors']) > 1

    # 층 bounding box (멀티플로어에서 층 구분 시각화)
    if data['floors'] and not args.no_floor_bbox:
        geoms += make_floor_bbox_geometries(data['floors'])
        if is_multifloor:
            print(f"  → {len(data['floors'])}개 층 BBox 생성됨 (5F=파랑, 6F=주황)")

    # 방 geometry (멀티플로어는 floor_z=None으로 자동 결정)
    if not args.no_zones or not args.no_outline:
        # 단층 구 파일: args.floor_z 가 지정된 경우 사용, 미지정 시 None(자동)
        fz = args.floor_z  # None → 각 zone 에서 center/asset Z 자동 결정
        geoms += make_zone_geometries(
            data['zones'],
            show_fill    = not args.no_zones,
            show_outline = not args.no_outline,
            floor_z      = fz,
        )

    # 방 중심 마커
    if not args.no_centers:
        geoms += make_zone_center_markers(data['zones'])

    # 층끼리 연결선 (Floor↔Floor — 같은 건물 소속 표현)
    if args.floor_links and len(data['floors']) > 1:
        fl_links = make_floor_links(data['floors'])
        geoms += fl_links
        print(f"  → 층 연결선: {len(fl_links)}개 (--floor_links)")

    # 층 안에서 방끼리 연결선 (Floor→Zone 허브-스포크)
    if args.zone_links and data['floors']:
        zn_links = make_zone_links(data['floors'], data['zones'])
        geoms += zn_links
        print(f"  → 층 내 Zone 연결선: {len(zn_links)}개 (--zone_links)")

    # Zone→Asset 계층 멤버십 선 (핵심: 어떤 객체가 어느 방에 속하는지 표현)
    if not args.no_hierarchy:
        membership_lines = make_zone_membership_lines(data['zones'], data['asset_map'])
        geoms += membership_lines
        print(f"  → 계층 멤버십 선: {len(membership_lines)}개 "
              f"(Zone→Asset 포함 관계, 숨기려면 --no_hierarchy)")

    # asset 구체
    if not args.no_assets:
        geoms += make_asset_geometries(
            data['asset_map'],
            filter_cls = args.filter_cls,
            radius     = args.radius,
            color_by   = args.color_by,
        )
        if args.color_by != 'class':
            print(f"  → asset 색상 기준: {args.color_by} "
                  f"(같은 {'Zone' if args.color_by=='zone' else '층'}은 동일 색)")

    # 관계선
    if not args.no_edges:
        geoms += make_edge_geometries(
            data['edges'],
            data['asset_map'],
            filter_cls = args.filter_cls,
        )

    if not geoms:
        print("[경고] 표시할 geometry가 없음")
        return

    floors_label = f"FLOOR {len(data['floors'])}개  " if data['floors'] else ""
    print(f"[뷰어] geometry {len(geoms)}개 생성 완료")
    print("  창이 열리면: 마우스 드래그=회전, 스크롤=줌, Q=종료\n")

    title = (f"HSSG Viewer — {data['building_id']}  |  "
             f"{floors_label}ZONE {len(data['zones'])}개  Asset {len(data['asset_map'])}개")
    show(geoms, title)


if __name__ == '__main__':
    main()
