"""
SuperHSSG Stage 4: L1 → L2 → L3 병합 (의미 MLP)
=================================================
L1 쌍 (i, j)에 대해 semantic + geometric 특징으로 병합 스코어를 예측한다.

입력 feature 차원:
  f_L1_i (651) + f_L1_j (651) + diff (651) + cos_sem (1) + dist (1) + rel_pos (3)
  = 1958 차원

레벨:
  L1 → XY flood fill (벽 기반 방 검출) → L2 그룹  [room_id 불필요]
  L2 → (attention pooling) → L3 방   [전체 floor]

실행:
  # 기본 실행 (flood fill 방 검출)
  python stage4_sem_merge.py

  # MLP 학습
  python stage4_sem_merge.py --mode train

  # MLP 추론
  python stage4_sem_merge.py --mode infer --ckpt ./stage4_results/stage4_model.pt
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import label as scipy_label, binary_closing
from pathlib import Path
import argparse
import cv2

D_L1      = 651
PAIR_DIM  = D_L1 * 3 + 5    # 1958
IDX_CENT  = slice(0, 3)       # centroid in f_L1
IDX_SEM   = slice(139, 651)   # semantic part in f_L1


# ── 디바이스 ──────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── 모듈 ─────────────────────────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    """자식 노드 feature들을 attention 가중합으로 부모 노드 feature로 집계"""
    def __init__(self, in_dim: int = D_L1):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [K, D] → [D]"""
        w = torch.softmax(self.attn(feats), dim=0)  # [K, 1]
        return (w * feats).sum(0)                    # [D]


class SemMergeScoreMLP(nn.Module):
    """
    L1 쌍의 병합 여부를 0~1 스코어로 예측
    입력 [E, 1958] → 출력 [E]
    """
    def __init__(self, input_dim: int = PAIR_DIM, hidden_dims=(1024, 512, 256)):
        super().__init__()
        layers, in_d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, pair_feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(pair_feat).squeeze(-1))


# ── 특징 구성 ─────────────────────────────────────────────────────────────────
def build_l1_pair_features(f_l1: torch.Tensor,
                            edge_src: torch.Tensor,
                            edge_dst: torch.Tensor) -> torch.Tensor:
    """f_l1 [N_l1, 651] + 인접 쌍 → pair_feat [E, 1958]"""
    f_i   = f_l1[edge_src]                                         # [E, 651]
    f_j   = f_l1[edge_dst]
    diff  = f_i - f_j                                              # [E, 651]

    dist    = torch.norm(f_i[:, IDX_CENT] - f_j[:, IDX_CENT],
                         dim=-1, keepdim=True)                     # [E, 1]
    sem_i   = F.normalize(f_i[:, IDX_SEM], dim=-1)
    sem_j   = F.normalize(f_j[:, IDX_SEM], dim=-1)
    cos_sem = (sem_i * sem_j).sum(-1, keepdim=True)                # [E, 1]
    rel_pos = f_i[:, IDX_CENT] - f_j[:, IDX_CENT]                 # [E, 3]

    return torch.cat([f_i, f_j, diff, cos_sem, dist, rel_pos], dim=-1)  # [E, 1958]


# ── GT 생성 ───────────────────────────────────────────────────────────────────
def build_gt_from_labels(f_l1_np: np.ndarray,
                          text_labels: list,
                          edge_src: np.ndarray,
                          edge_dst: np.ndarray,
                          dist_threshold: float = 2.0) -> np.ndarray:
    """
    l1_labels_text.json의 텍스트 라벨 기반 GT:
      같은 라벨 AND centroid 거리 < dist_threshold → 1 (병합)
      'unknown' 라벨 쌍은 항상 0 (병합 안 함)
    """
    labels = np.array(text_labels)
    i_lab  = labels[edge_src]
    j_lab  = labels[edge_dst]
    same   = (i_lab == j_lab) & (i_lab != 'unknown')
    dist   = np.linalg.norm(
        f_l1_np[edge_src, :3] - f_l1_np[edge_dst, :3], axis=1)
    return (same & (dist < dist_threshold)).astype(np.float32)


# ── L1 인접 엣지 구성 ─────────────────────────────────────────────────────────
def build_l1_edges_from_sp(l1_labels_sp: np.ndarray,
                            sp_edge_src: np.ndarray,
                            sp_edge_dst: np.ndarray):
    """
    SP 수준 인접 그래프 → L1 수준 인접 엣지 (서로 다른 L1 객체 사이)
    구조체(-1) SP가 포함된 엣지는 제외한다.
    Returns: (l1_edge_src, l1_edge_dst) np.ndarray
    """
    l1_s = l1_labels_sp[sp_edge_src]
    l1_d = l1_labels_sp[sp_edge_dst]
    # 구조체(-1) 제외 + 서로 다른 L1 쌍만
    cross = (l1_s != l1_d) & (l1_s >= 0) & (l1_d >= 0)
    pairs = np.stack([l1_s[cross], l1_d[cross]], axis=1)
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs[:, 0], pairs[:, 1]


# ── 병합 → 상위 레벨 feature ──────────────────────────────────────────────────
def merge_level(f_low: torch.Tensor,
                edge_src: torch.Tensor,
                edge_dst: torch.Tensor,
                scores: torch.Tensor,
                threshold: float,
                device: torch.device):
    """
    score > threshold 엣지로 connected components → 상위 레벨 레이블 + feature
    Returns: labels [N_low], f_high [N_high, D_L1]
    """
    N = f_low.shape[0]
    mask   = (scores > threshold).cpu().numpy()
    src_np = edge_src.cpu().numpy()[mask]
    dst_np = edge_dst.cpu().numpy()[mask]

    if len(src_np) > 0:
        rows = np.concatenate([src_np, dst_np])
        cols = np.concatenate([dst_np, src_np])
        adj  = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
        n_high, labels = connected_components(adj, directed=False)
    else:
        n_high = N
        labels = np.arange(N)

    labels_t = torch.from_numpy(labels).long()
    # AttentionPooling은 cpu/mps에서도 동작
    pooler   = AttentionPooling(f_low.shape[1]).to(device)
    f_high_list = []
    for lid in range(n_high):
        mask_l = labels_t == lid
        f_high_list.append(pooler(f_low[mask_l]))
    f_high = torch.stack(f_high_list)
    return labels_t, f_high


# ── 방 기하 정보 추출 ────────────────────────────────────────────────────────
def extract_room_geometry_from_grid(
    grid: np.ndarray,        # [H, W] 0=wall, 1~N=room (1-indexed)
    x_min: float,
    y_min: float,
    cell_size: float,
    l2_labels: np.ndarray,   # [N_l1] 0-indexed L2 ID
    f_l1: np.ndarray,        # [N_l1, D] (앞 3열 = centroid XYZ)
) -> list:
    """
    각 L2(방) 노드의 기하학적 정보를 추출한다.
    Returns list of dicts (l2_id순):
      l2_id    : int
      centroid : [x, y, z]   방 중심 (xy=폴리곤 중심, z=L1 오브젝트 평균)
      area_m2  : float
      boundary : [[x,y], ...]  방 경계 폴리곤 (world coord, 닫힌 루프)
    """
    n_l2   = int(grid.max())          # grid에 존재하는 실제 방 수 (빈 방 포함)
    l1_xyz = f_l1[:, :3]
    rooms  = []

    for l2_id in range(n_l2):
        room_mask = (grid == l2_id + 1).astype(np.uint8)   # 1-indexed grid

        # L1 오브젝트 Z 평균 → 방의 z 높이
        in_room = l2_labels == l2_id
        z_val   = float(l1_xyz[in_room, 2].mean()) if in_room.any() else 0.0

        boundary = []
        area_m2  = 0.0

        if room_mask.any():
            contours, _ = cv2.findContours(
                room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt     = max(contours, key=cv2.contourArea)
                area_m2 = float(cv2.contourArea(cnt)) * cell_size ** 2
                # 폴리곤 단순화 (너무 많은 꼭짓점 줄이기)
                eps    = max(1.0, 0.015 * cv2.arcLength(cnt, True))
                approx = cv2.approxPolyDP(cnt, eps, True)
                for pt in approx.reshape(-1, 2):
                    col, row = int(pt[0]), int(pt[1])
                    boundary.append([
                        round(x_min + col * cell_size, 3),
                        round(y_min + row * cell_size, 3),
                    ])

        # 중심: 폴리곤 bbox 중심 우선, 없으면 L1 centroid 평균
        if boundary:
            bnd = np.array(boundary)
            cx, cy = float(bnd[:, 0].mean()), float(bnd[:, 1].mean())
        elif in_room.any():
            cx = float(l1_xyz[in_room, 0].mean())
            cy = float(l1_xyz[in_room, 1].mean())
        else:
            cx, cy = 0.0, 0.0

        rooms.append({
            'l2_id':    l2_id,
            'centroid': [round(cx, 3), round(cy, 3), round(z_val, 3)],
            'area_m2':  round(area_m2, 2),
            'boundary': boundary,
        })

    return rooms


# ── room_grid 기반 방 할당 ───────────────────────────────────────────────────
def assign_l2_by_room_grid(
    f_l1: np.ndarray,            # [N_l1, D]  L1 feature (앞 3차원 = centroid XYZ)
    room_grid_path: Path,
    device: torch.device,
):
    """
    wall_door_2d.py 출력 room_grid_*.npz를 사용해 L1 → 방(L2) 할당.
    forced_class 불필요. L1 centroid XY를 격자에 투영해 방 ID를 직접 읽는다.
    """
    d = np.load(room_grid_path)
    grid      = d['grid'].astype(np.int32)   # [H, W] 0=벽/미배정, 1~N=방
    x_min     = float(d['x_min'])
    y_min     = float(d['y_min'])
    cell_size = float(d['cell_size'])
    H, W = grid.shape
    n_rooms = int(grid.max())

    n_l1    = f_l1.shape[0]
    l1_cent = f_l1[:, :3]    # [N_l1, 3]  centroid XYZ

    px = np.clip(((l1_cent[:, 0] - x_min) / cell_size).astype(np.int32), 0, W - 1)
    py = np.clip(((l1_cent[:, 1] - y_min) / cell_size).astype(np.int32), 0, H - 1)
    room_ids  = grid[py, px]                        # [N_l1]  0=미배정, 1~N
    l2_labels = (room_ids - 1).astype(np.int32)     # 0-indexed, -1=미배정

    # 미배정 L1 → 가장 가까운 배정 L1의 방으로 보정
    unassigned = l2_labels < 0
    if unassigned.any() and (~unassigned).any():
        assigned_idx = np.where(~unassigned)[0]
        for idx in np.where(unassigned)[0]:
            dists = np.linalg.norm(l1_cent[assigned_idx] - l1_cent[idx], axis=1)
            l2_labels[idx] = l2_labels[assigned_idx[dists.argmin()]]
    l2_labels = np.maximum(l2_labels, 0)

    # ── 복도/개방 공간 감지 및 재배정 ────────────────────────────────────────────
    # 전체 유효 픽셀(방 영역) 대비 20% 이상을 차지하는 방은 복도/개방 공간으로 간주.
    # 해당 방의 L1 객체들을 가장 가까운 정상 방으로 재배정한다.
    CORRIDOR_RATIO_THRESH = 0.20
    total_room_pixels = int((grid > 0).sum())
    corridor_l2ids = set()
    for l2_id in range(n_rooms):
        room_id = l2_id + 1                     # grid는 1-indexed
        pix_cnt = int((grid == room_id).sum())
        ratio   = pix_cnt / max(total_room_pixels, 1)
        if ratio > CORRIDOR_RATIO_THRESH:
            corridor_l2ids.add(l2_id)
            print(f"  [복도 감지] l2_id={l2_id} (room_id={room_id}): "
                  f"픽셀 {pix_cnt:,}/{total_room_pixels:,} ({100*ratio:.1f}%) → 복도로 분류")

    if corridor_l2ids:
        # 정상 방에 배정된 객체 인덱스
        normal_mask = np.array([l2_labels[i] not in corridor_l2ids
                                for i in range(n_l1)], dtype=bool)
        if normal_mask.any():
            normal_idx = np.where(normal_mask)[0]
            # 복도 객체 → 가장 가까운 정상 방 객체의 l2_id로 재배정
            corridor_mask = ~normal_mask
            for idx in np.where(corridor_mask)[0]:
                dists = np.linalg.norm(l1_cent[normal_idx] - l1_cent[idx], axis=1)
                l2_labels[idx] = l2_labels[normal_idx[dists.argmin()]]
            n_reassigned = int(corridor_mask.sum())
            print(f"  [복도 재배정] {n_reassigned}개 L1 객체 → 가장 가까운 정상 방으로 이동")
        else:
            print(f"  [경고] 모든 방이 복도로 분류됨 — 재배정 생략 (임계값 완화 필요)")

    n_l2 = n_rooms                    # grid.max() — 빈 방(L1 없음)도 포함한 전체 방 수
    print(f"  room_grid 방 배정: 그리드 {n_rooms}개 방 → L1 {n_l1}개 → L2 {n_l2}개")

    # L2 feature (attention pooling)
    f_t    = torch.from_numpy(f_l1).to(device)
    lab_t  = torch.from_numpy(l2_labels).long().to(device)
    pooler = AttentionPooling(f_l1.shape[1]).to(device)
    f_l2_list = []
    for lid in range(n_l2):
        mask = lab_t == lid
        f_l2_list.append(pooler(f_t[mask]) if mask.any()
                         else torch.zeros(f_l1.shape[1], device=device))
    f_l2 = torch.stack(f_l2_list).detach().cpu().numpy()   # [N_l2, D]

    l3_labels = np.zeros(n_l2, dtype=np.int32)
    f_l3      = f_l2.mean(axis=0, keepdims=True)

    # 방 기하 정보 추출
    geometry = extract_room_geometry_from_grid(
        grid, x_min, y_min, cell_size, l2_labels, f_l1)
    return l2_labels, f_l2, l3_labels, f_l3, geometry


# ── SAM 마스크 직접 방 할당 ──────────────────────────────────────────────────
def assign_l2_by_sam_masks(
    f_l1: np.ndarray,            # [N_l1, D]  L1 feature (앞 3차원 = centroid XYZ)
    room_grid_path: Path,
    device: torch.device,
):
    """
    sam_room_grid.py 가 저장한 masks [N, H, W] 를 사용해 L1 → 방(L2) 할당.
    grid 변환 없이 SAM 마스크 segmentation 배열을 직접 사용한다.

    할당 규칙: 각 L1 centroid 픽셀을 포함하는 마스크 중 면적이 가장 작은
               (= 가장 구체적인) 마스크를 방으로 배정한다.
    """
    d         = np.load(room_grid_path)
    masks     = d['masks'].astype(bool)      # [N, H, W]
    x_min     = float(d['x_min'])
    y_min     = float(d['y_min'])
    cell_size = float(d['cell_size'])
    N, H, W   = masks.shape

    n_l1    = f_l1.shape[0]
    l1_cent = f_l1[:, :3]

    px = np.clip(((l1_cent[:, 0] - x_min) / cell_size).astype(np.int32), 0, W - 1)
    py = np.clip(((l1_cent[:, 1] - y_min) / cell_size).astype(np.int32), 0, H - 1)

    # masks[:, py, px] → [N, N_l1] bool : 각 마스크가 각 L1 centroid를 포함하는지
    centroid_hits = masks[:, py, px]                       # [N, N_l1]
    areas = masks.sum(axis=(1, 2)).astype(np.float32)      # [N]

    # 포함하지 않는 마스크 → 면적을 inf 로 설정 후 argmin
    area_mat  = np.where(centroid_hits, areas[:, None], np.inf)  # [N, N_l1]
    best_mask = area_mat.argmin(axis=0)                          # [N_l1]
    hit_any   = centroid_hits.any(axis=0)                        # [N_l1]
    l2_labels = np.where(hit_any, best_mask, -1).astype(np.int32)

    # 미배정 L1 → 가장 가까운 배정 L1의 방으로 nearest-neighbor 보정
    unassigned = l2_labels < 0
    if unassigned.any() and (~unassigned).any():
        assigned_idx = np.where(~unassigned)[0]
        for idx in np.where(unassigned)[0]:
            dists = np.linalg.norm(l1_cent[assigned_idx] - l1_cent[idx], axis=1)
            l2_labels[idx] = l2_labels[assigned_idx[dists.argmin()]]
    l2_labels = np.maximum(l2_labels, 0)

    n_l2 = N
    n_assigned = int(hit_any.sum())
    print(f"  SAM masks 방 배정: {N}개 마스크 → "
          f"L1 {n_l1}개 (직접배정 {n_assigned}, NN보정 {n_l1-n_assigned}) → L2 {n_l2}개")

    # 기하 추출용 grid: 마스크를 래스터라이즈 (큰 마스크 먼저, 작은 마스크 우선 유지)
    grid = np.zeros((H, W), dtype=np.int32)
    for i in range(N - 1, -1, -1):          # 역순(작은→큰), 작은 마스크가 최종 승
        grid[masks[i]] = i + 1              # 1-indexed

    # L2 feature (attention pooling)
    f_t    = torch.from_numpy(f_l1).to(device)
    lab_t  = torch.from_numpy(l2_labels).long().to(device)
    pooler = AttentionPooling(f_l1.shape[1]).to(device)
    f_l2_list = []
    for lid in range(n_l2):
        mask_l = lab_t == lid
        f_l2_list.append(pooler(f_t[mask_l]) if mask_l.any()
                         else torch.zeros(f_l1.shape[1], device=device))
    f_l2 = torch.stack(f_l2_list).detach().cpu().numpy()   # [N_l2, D]

    l3_labels = np.zeros(n_l2, dtype=np.int32)
    f_l3      = f_l2.mean(axis=0, keepdims=True)

    geometry = extract_room_geometry_from_grid(
        grid, x_min, y_min, cell_size, l2_labels, f_l1)
    return l2_labels, f_l2, l3_labels, f_l3, geometry


# ── 벽 밀도맵 + gap closing 기반 방 검출 (폴백) ─────────────────────────────
def assign_l2_by_flood_fill(
    sp_centroids: np.ndarray,          # [N_sp_valid, 3]  구조체 제외 SP centroid XYZ
    sp_pred: np.ndarray,               # [N_sp_valid]     구조체 제외 SP 예측 클래스
    l1_labels_sp: np.ndarray,          # [N_sp_valid]     SP → L1 id (remapped)
    f_l1: np.ndarray,                  # [N_l1, D]        L1 feature
    device: torch.device,
    # ▼ 벽 검출용: 구조체 SP 포함 원본 데이터 (wall SP가 필요하므로 반드시 전달)
    sp_centroids_all: np.ndarray = None,   # [N_sp_all, 3] 필터링 전 전체 SP
    sp_pred_all: np.ndarray = None,        # [N_sp_all]    필터링 전 전체 예측
    cell_size: float = 0.1,
    wall_pct: float = 60,
    gap_close_m: float = 1.2,
    min_room_m2: float = 1.0,
):
    """
    1. 벽 SP(class==2)를 XY 격자에 투영 → 밀도맵 → 벽 마스크
       ※ 구조체 필터링 후엔 wall SP가 제거되므로, 원본(sp_centroids_all, sp_pred_all)을
          벽 검출에 사용한다. 미제공 시 sp_centroids/sp_pred로 폴백.
    2. binary_closing으로 문 틈새(gap_close_m 이하)를 이어 다각형 완성
    3. flood fill → 방 영역 검출 → L1 배정

    --room_grid 미지정 시 폴백으로 동작.
    """
    WALL_CLASS = 2
    n_l1 = f_l1.shape[0]

    # ── L1 centroid: f_l1 앞 3열 직접 사용 (SP 경유 불필요) ──────────────────
    l1_cent = f_l1[:, :3].copy()   # [N_l1, 3]

    # ── 벽 검출용 SP: 원본(전체) 우선, 없으면 필터링된 것으로 폴백 ──────────
    wall_sp_centroids = sp_centroids_all if sp_centroids_all is not None else sp_centroids
    wall_sp_pred      = sp_pred_all      if sp_pred_all      is not None else sp_pred
    if sp_centroids_all is None:
        print("  [주의] sp_centroids_all 미제공 — 구조체 필터링된 SP로 벽 검출 시도"
              " (wall SP 부족으로 방 분리가 제대로 안 될 수 있음)")

    x = wall_sp_centroids[:, 0]
    y = wall_sp_centroids[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    W = int((x_max - x_min) / cell_size) + 2
    H = int((y_max - y_min) / cell_size) + 2

    # ── Step 1: 벽 밀도맵 → 벽 마스크 ──────────────────────────────────────
    wall_density = np.zeros((H, W), dtype=np.float32)
    wall_sp = wall_sp_pred == WALL_CLASS
    print(f"  벽 SP: {wall_sp.sum():,}개 / 전체 SP: {len(wall_sp_pred):,}개")
    if wall_sp.any():
        xi = np.clip(((x[wall_sp] - x_min) / cell_size).astype(int), 0, W-1)
        yi = np.clip(((y[wall_sp] - y_min) / cell_size).astype(int), 0, H-1)
        np.add.at(wall_density, (yi, xi), 1)

    nonzero = wall_density[wall_density > 0]
    if len(nonzero) == 0:
        print("  [경고] 벽 SP 없음 — 전체를 방 1개로 처리")
        l2_labels = np.zeros(n_l1, dtype=np.int32)
        f_t = torch.from_numpy(f_l1).to(device)
        pooler = AttentionPooling(f_l1.shape[1]).to(device)
        f_l2 = pooler(f_t).unsqueeze(0).detach().cpu().numpy()
        return l2_labels, f_l2, np.zeros(1, dtype=np.int32), f_l2.mean(0, keepdims=True)

    thr = np.percentile(nonzero, wall_pct)
    wall_mask = wall_density >= thr   # [H, W] bool — 전개도 벽 위치

    # ── Step 2: binary_closing — 문 틈새(gap_close_m) 이어서 다각형 완성 ─────
    gap_px = max(1, int(gap_close_m / cell_size))
    struct = np.ones((gap_px, gap_px), dtype=bool)   # 정사각형 구조 요소
    wall_closed = binary_closing(wall_mask, structure=struct)
    # closing = dilation(gap_px) → erosion(gap_px)
    # 효과: gap_close_m 이하 틈새(문)를 자동으로 이어 닫힌 다각형 형성
    # 벽 자체 두께는 closing 후 원래대로 복원됨

    n_wall_px   = int(wall_mask.sum())
    n_closed_px = int(wall_closed.sum())
    print(f"  벽 마스크: {n_wall_px:,} 셀  →  gap closing 후: {n_closed_px:,} 셀"
          f"  (추가 {n_closed_px - n_wall_px:,} 셀로 틈새 연결)")

    # ── Step 3: flood fill → 방 영역 ─────────────────────────────────────────
    free_mask = ~wall_closed
    labeled, n_found = scipy_label(free_mask)

    min_room_px = max(1, int(min_room_m2 / (cell_size ** 2)))
    for rid in range(1, n_found + 1):
        if (labeled == rid).sum() < min_room_px:
            labeled[labeled == rid] = 0

    unique_ids = np.unique(labeled[labeled > 0])
    if len(unique_ids) == 0:
        print("  [경고] 방 검출 실패 — 전체를 방 1개로 처리")
        l2_labels = np.zeros(n_l1, dtype=np.int32)
        f_t = torch.from_numpy(f_l1).to(device)
        pooler = AttentionPooling(f_l1.shape[1]).to(device)
        f_l2 = pooler(f_t).unsqueeze(0).detach().cpu().numpy()
        return l2_labels, f_l2, np.zeros(1, dtype=np.int32), f_l2.mean(0, keepdims=True)

    remap = np.zeros(labeled.max() + 1, dtype=np.int32)
    for new_id, old_id in enumerate(unique_ids, start=1):
        remap[old_id] = new_id
    room_grid = remap[labeled]   # [H, W]  0=벽, 1~N=방
    n_rooms = int(room_grid.max())

    print(f"  Flood fill 방 검출: {n_rooms}개  (격자 {H}×{W}, cell={cell_size}m, gap={gap_close_m}m)")

    # ── Step 4: L1 → 방 할당 ─────────────────────────────────────────────────
    lx = np.clip(((l1_cent[:, 0] - x_min) / cell_size).astype(int), 0, W-1)
    ly = np.clip(((l1_cent[:, 1] - y_min) / cell_size).astype(int), 0, H-1)
    l2_labels = (room_grid[ly, lx] - 1).astype(np.int32)   # 0-indexed, -1=미할당

    # 미할당 L1 → 가장 가까운 할당 L1의 방으로 보정
    unassigned = l2_labels < 0
    if unassigned.any() and (~unassigned).any():
        assigned_idx = np.where(~unassigned)[0]
        for idx in np.where(unassigned)[0]:
            dists = np.linalg.norm(l1_cent[assigned_idx] - l1_cent[idx], axis=1)
            l2_labels[idx] = l2_labels[assigned_idx[dists.argmin()]]
    l2_labels = np.maximum(l2_labels, 0)

    n_l2 = int(l2_labels.max()) + 1
    print(f"  L1 {n_l1}개 → L2(방) {n_l2}개 배정 완료")

    # ── L2 feature ────────────────────────────────────────────────────────────
    f_t   = torch.from_numpy(f_l1).to(device)
    lab_t = torch.from_numpy(l2_labels).long().to(device)
    pooler = AttentionPooling(f_l1.shape[1]).to(device)
    f_l2_list = []
    for lid in range(n_l2):
        mask = lab_t == lid
        f_l2_list.append(pooler(f_t[mask]) if mask.any()
                         else torch.zeros(f_l1.shape[1], device=device))
    f_l2 = torch.stack(f_l2_list).detach().cpu().numpy()   # [N_l2, D]

    l3_labels = np.zeros(n_l2, dtype=np.int32)
    f_l3      = f_l2.mean(axis=0, keepdims=True)

    # 방 기하 정보 추출 (flood fill이 만든 room_grid 사용)
    geometry = extract_room_geometry_from_grid(
        room_grid, x_min, y_min, cell_size, l2_labels, f_l1)
    return l2_labels, f_l2, l3_labels, f_l3, geometry


# ── 규칙 기반 병합 ─────────────────────────────────────────────────────────────
@torch.no_grad()
def rule_based_merge(f_l1: np.ndarray,
                     text_labels: list,
                     edge_src: np.ndarray,
                     edge_dst: np.ndarray,
                     dist_threshold: float = 2.0,
                     device: torch.device = None):
    """
    MLP 없이 규칙만으로 L1→L2 병합:
      같은 라벨 AND centroid 거리 < dist_threshold → 병합
    """
    if device is None:
        device = get_device()

    gt = build_gt_from_labels(f_l1, text_labels, edge_src, edge_dst, dist_threshold)
    scores = torch.from_numpy(gt).to(device)
    f_t    = torch.from_numpy(f_l1).to(device)
    es_t   = torch.from_numpy(edge_src.astype(np.int64)).to(device)
    ed_t   = torch.from_numpy(edge_dst.astype(np.int64)).to(device)

    l2_labels_t, f_l2 = merge_level(f_t, es_t, ed_t, scores, 0.5, device)
    n_l2 = f_l2.shape[0]

    # L2 → L3 (floor 전체를 하나의 방으로)
    pooler = AttentionPooling(D_L1).to(device)
    f_l3   = pooler(f_l2).unsqueeze(0)   # [1, 651]
    l3_labels = np.zeros(n_l2, dtype=np.int64)

    merge_r = gt.mean()
    print(f"  L1 {len(f_l1)}개 → L2 {n_l2}개 → L3 1개  (병합률 {merge_r:.1%})")
    return (l2_labels_t.numpy(), f_l2.cpu().numpy(),
            l3_labels, f_l3.cpu().numpy())


# ── 학습 ─────────────────────────────────────────────────────────────────────
def train_stage4(data_list, device, n_epochs: int = 50, lr: float = 1e-3,
                 threshold: float = 0.5, save_path: str = None):
    """
    data_list 항목:
        f_l1     : [N_l1, 651] np.ndarray
        edge_src : [E] np.ndarray  (L1 수준 엣지)
        edge_dst : [E] np.ndarray
        gt       : [E] float32  (1=병합, 0=경계)
    """
    model = SemMergeScoreMLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best  = float('inf')

    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0.0
        for s in data_list:
            f   = torch.from_numpy(s['f_l1']).to(device)
            es  = torch.from_numpy(s['edge_src']).long().to(device)
            ed  = torch.from_numpy(s['edge_dst']).long().to(device)
            gt  = torch.from_numpy(s['gt']).to(device)

            pf     = build_l1_pair_features(f, es, ed)
            scores = model(pf)
            pos_w  = (gt == 0).float().sum() / (gt == 1).float().sum().clamp(min=1)
            w      = torch.where(gt == 1, pos_w.expand_as(gt), torch.ones_like(gt))
            loss   = F.binary_cross_entropy(scores, gt, weight=w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        sched.step()
        if epoch % 10 == 0:
            n = max(len(data_list), 1)
            print(f"  Epoch {epoch:4d}/{n_epochs} | Loss: {total/n:.4f}")
        if total < best and save_path:
            best = total
            torch.save(model.state_dict(), save_path)
            print(f"  → 체크포인트 저장: {save_path}  (loss={best:.4f})")
    return model


# ── MLP 추론 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_stage4(model: SemMergeScoreMLP,
                 f_l1: np.ndarray,
                 edge_src: np.ndarray,
                 edge_dst: np.ndarray,
                 threshold: float = 0.5):
    """
    Returns:
        l2_labels [N_l1],  f_l2 [N_l2, 651]
        l3_labels [N_l2],  f_l3 [1, 651]    (L3 = floor 전체)
    """
    model.eval()
    device = next(model.parameters()).device

    f   = torch.from_numpy(f_l1).to(device)
    es  = torch.from_numpy(edge_src.astype(np.int64)).to(device)
    ed  = torch.from_numpy(edge_dst.astype(np.int64)).to(device)

    pf     = build_l1_pair_features(f, es, ed)
    scores = model(pf)

    # L1 → L2
    l2_labels, f_l2 = merge_level(f, es, ed, scores, threshold, device)
    n_l2 = f_l2.shape[0]

    # L2 → L3
    l3_labels = np.zeros(n_l2, dtype=np.int64)
    pooler    = AttentionPooling(D_L1).to(device)
    f_l3      = pooler(f_l2).unsqueeze(0)   # [1, 651]

    merge_r = (scores > threshold).float().mean().item()
    print(f"  L1 {len(f_l1)}개 → L2 {n_l2}개 → L3 1개  (병합률 {merge_r:.1%})")
    return (l2_labels.numpy(), f_l2.cpu().numpy(),
            l3_labels, f_l3.cpu().numpy())


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    _ROOT        = Path(__file__).resolve().parents[1]   # …/pipeline
    _OPENSHAPE   = Path(__file__).resolve().parents[1] / 'openshapeCLI'   # …/model/openshapeCLI
    _MERGE       = Path(__file__).resolve().parent                         # …/model/sem_merge

    # ── 자동 감지: stage2_results에서 l1_*.npz, 프로젝트 루트에서 room_grid_*.npz ──
    _STAGE2_DIR  = _ROOT / 'merge' / 'stage2_results'
    _l1_cands    = sorted(_STAGE2_DIR.glob('l1_*.npz')) if _STAGE2_DIR.exists() else []
    _default_l1  = str(max(_l1_cands, key=lambda p: p.stat().st_mtime)) \
                   if _l1_cands else str(_STAGE2_DIR / 'l1_floor.npz')
    # room_grid는 l1 파일의 stem에서 유도 (l1_6_floor.npz → room_grid_6_floor.npz)
    _l1_stem     = Path(_default_l1).stem[len('l1_'):]   # 'l1_6_floor' → '6_floor'
    _rg_derived  = _ROOT / f'room_grid_{_l1_stem}.npz'
    _rg_cands    = sorted(_ROOT.glob('room_grid_*.npz'))
    _default_rg  = str(_rg_derived) if _rg_derived.exists() \
                   else (str(max(_rg_cands, key=lambda p: p.stat().st_mtime))
                         if _rg_cands else str(_ROOT / 'room_grid_floor.npz'))

    parser = argparse.ArgumentParser(description='SuperHSSG Stage 4: L1→L2→L3 병합')
    parser.add_argument('--fl1_npy',     default=str(_OPENSHAPE / 'stage3_results' / 'f_l1_floor.npy'),
                        help='Stage 3 출력: f_L1 feature [N_l1, 651]')
    parser.add_argument('--l1_npz',      default=_default_l1,
                        help='Stage 2 출력: l1_labels, sp_features, edge_src, edge_dst'
                             ' (기본: stage2_results/에서 최신 l1_*.npz 자동 감지)')
    parser.add_argument('--labels_json', default=str(_OPENSHAPE / 'stage3_results' / 'l1_labels_text.json'),
                        help='Stage 3 출력: 텍스트 라벨 JSON')
    parser.add_argument('--out_dir',     default=str(_MERGE / 'stage4_results'))
    parser.add_argument('--ckpt',        default=str(_MERGE / 'stage4_results' / 'stage4_model.pt'))
    parser.add_argument('--mode',        choices=['train', 'infer'],
                        default=None,
                        help='train: MLP 학습 후 저장 | infer: 저장된 MLP 추론'
                             ' (기본: 체크포인트 없으면 train, 있으면 infer)')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--threshold',   type=float, default=0.5)
    parser.add_argument('--dist_thr',    type=float, default=2.0,
                        help='L1→L2 병합 거리 임계값 (m)')
    parser.add_argument('--cell_size',    type=float, default=0.1,
                        help='XY 격자 크기 (m)')
    parser.add_argument('--wall_pct',    type=float, default=60,
                        help='벽 밀도 판정 백분위 (낮을수록 더 많은 셀을 벽으로)')
    parser.add_argument('--gap_close_m', type=float, default=1.2,
                        help='문 틈새 연결 최대 폭 (m) — 이 이하 간격을 이어서 다각형 완성')
    parser.add_argument('--min_room_m2', type=float, default=1.0,
                        help='방으로 인정하는 최소 면적 (m²)')
    parser.add_argument('--room_grid',
                        default=_default_rg,
                        help='wall_door_2d.py 출력 room_grid_*.npz 경로 (방별 L2 할당)'
                             ' (기본: 프로젝트 루트에서 최신 room_grid_*.npz 자동 감지)')
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True)
    device  = get_device()
    print(f"[Stage 4] 디바이스: {device}")

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    fl1_path    = Path(args.fl1_npy)
    l1_npz_path = Path(args.l1_npz)
    json_path   = Path(args.labels_json)

    for p in [fl1_path, l1_npz_path, json_path]:
        if not p.exists():
            print(f"  [오류] 파일 없음: {p}")
            exit(1)

    f_l1   = np.load(fl1_path).astype(np.float32)   # [N_l1, 651]
    l1_npz = np.load(l1_npz_path)
    sp_l1_raw = l1_npz['l1_labels'].astype(np.int64)  # [N_sp] -1=구조체, 0~=L1 id
    sp_es     = l1_npz['edge_src'].astype(np.int64)
    sp_ed     = l1_npz['edge_dst'].astype(np.int64)

    # sp_features: l1_floor.npz에 있으면 사용, 없으면 floor.npz에서 로드
    if 'sp_features' in l1_npz.files:
        sp_features_raw = l1_npz['sp_features'].astype(np.float32)
    else:
        floor_path = _ROOT / 'src' / 'floor.npz'
        if not floor_path.exists():
            print(f"  [오류] sp_features 없음, floor.npz도 없음: {floor_path}")
            exit(1)
        print(f"  [정보] sp_features를 floor.npz에서 로드: {floor_path}")
        sp_features_raw = np.load(floor_path)['sp_features'].astype(np.float32)

    # sp_pred 로드 (flood fill 폴백용 — wall 클래스 판별)
    if 'sp_pred' in l1_npz.files:
        sp_pred_raw = l1_npz['sp_pred'].astype(np.int32)
    else:
        floor_path = _ROOT / 'src' / 'floor.npz'
        sp_pred_raw = np.load(floor_path)['sp_pred'].astype(np.int32)

    # ── 구조체(-1) SP 필터링 및 L1 ID 재매핑 (stage3와 동일) ─────────────────
    valid_mask   = sp_l1_raw >= 0
    sp_l1_valid  = sp_l1_raw[valid_mask]
    sp_feat_valid = sp_features_raw[valid_mask]
    sp_pred_valid = sp_pred_raw[valid_mask]

    old_ids      = np.unique(sp_l1_valid)
    id_remap     = np.full(int(old_ids.max()) + 1, -1, dtype=np.int64)
    id_remap[old_ids] = np.arange(len(old_ids))
    sp_l1        = id_remap[sp_l1_valid]      # [N_sp_valid] 0-based 재매핑

    # SP 엣지도 valid SP 기준으로 재인덱싱
    sp_idx_map   = np.full(len(valid_mask), -1, dtype=np.int64)
    sp_idx_map[np.where(valid_mask)[0]] = np.arange(valid_mask.sum())
    valid_edges  = (sp_idx_map[sp_es] >= 0) & (sp_idx_map[sp_ed] >= 0)
    sp_es        = sp_idx_map[sp_es[valid_edges]]
    sp_ed        = sp_idx_map[sp_ed[valid_edges]]

    sp_centroids = sp_feat_valid[:, 0:3]

    with open(json_path) as f:
        json_data = json.load(f)
    objects     = sorted(json_data['objects'], key=lambda x: x['l1_id'])
    text_labels = [o['label'] for o in objects]      # [N_l1]

    n_l1 = len(f_l1)
    assert len(text_labels) == n_l1, \
        f"라벨 수({len(text_labels)}) ≠ f_l1 행 수({n_l1})"

    print(f"  L1 객체: {n_l1}개  |  SP 수: {len(sp_l1)}  |  SP 엣지: {len(sp_es)}")

    # ── L1 수준 인접 엣지 구성 ────────────────────────────────────────────────
    l1_es, l1_ed = build_l1_edges_from_sp(sp_l1, sp_es, sp_ed)
    print(f"  L1 엣지: {len(l1_es)}개")

    # 구조체 제거 후 L1 ID가 재압축되면서 엣지 인덱스가 범위를 벗어날 수 있음 → 필터
    n_l1 = len(f_l1)
    valid_edge = (l1_es < n_l1) & (l1_ed < n_l1)
    if valid_edge.sum() < len(l1_es):
        print(f"  [경고] 범위 초과 L1 엣지 {(~valid_edge).sum()}개 제거 "
              f"(n_l1={n_l1}, max_idx={max(l1_es.max(), l1_ed.max())})")
        l1_es = l1_es[valid_edge]
        l1_ed = l1_ed[valid_edge]

    # ── GT 생성 (train 용) ────────────────────────────────────────────────────
    gt = build_gt_from_labels(f_l1, text_labels, l1_es, l1_ed, args.dist_thr)
    pos_ratio = gt.mean()
    print(f"  GT 병합 비율: {pos_ratio:.1%}  ({gt.sum():.0f}/{len(gt)})")

    # ── L2 방 할당: room_grid 우선, 없으면 flood fill 폴백 ───────────────────
    # room_grid_*.npz 포맷 자동 감지:
    #   'masks' 키 존재 → sam_room_grid.py 신형 (SAM 마스크 직접 사용)
    #   'grid'  키 존재 → wall_door_2d.py 구형 (래스터 그리드)
    room_grid_path = Path(args.room_grid) if args.room_grid else None
    if room_grid_path and room_grid_path.exists():
        _keys = list(np.load(room_grid_path).keys())
        if 'masks' in _keys:
            print(f"\n[Stage 4] SAM 마스크 직접 방 할당: {room_grid_path.name}")
            l2_lab, f_l2, l3_lab, f_l3, room_geometry = assign_l2_by_sam_masks(
                f_l1, room_grid_path, device,
            )
        else:
            print(f"\n[Stage 4] room_grid 기반 L2 방 할당: {room_grid_path.name}")
            l2_lab, f_l2, l3_lab, f_l3, room_geometry = assign_l2_by_room_grid(
                f_l1, room_grid_path, device,
            )
    else:
        if args.room_grid:
            print(f"  [경고] room_grid 파일 없음: {args.room_grid}  → flood fill 폴백")
        print("\n[Stage 4] XY flood fill 방 검출 중 (폴백)...")
        l2_lab, f_l2, l3_lab, f_l3, room_geometry = assign_l2_by_flood_fill(
            sp_centroids, sp_pred_valid, sp_l1, f_l1, device,
            sp_centroids_all = sp_features_raw[:, 0:3],
            sp_pred_all      = sp_pred_raw,
            cell_size=args.cell_size,
            wall_pct=args.wall_pct,
            gap_close_m=args.gap_close_m,
            min_room_m2=args.min_room_m2,
        )

    # ── 모드 결정: 체크포인트 없으면 train, 있으면 infer ─────────────────────
    ckpt_path = Path(args.ckpt)
    if args.mode is None:
        mode = 'infer' if ckpt_path.exists() else 'train'
    else:
        mode = args.mode
    print(f"[Stage 4] 모드: {mode}  (ckpt: {ckpt_path.name})")

    # ── 실행 ──────────────────────────────────────────────────────────────────
    if mode == 'train':
        sample = {
            'f_l1':     f_l1,
            'edge_src': l1_es.astype(np.int64),
            'edge_dst': l1_ed.astype(np.int64),
            'gt':       gt,
        }
        print(f"\n  [train] {args.epochs}에폭 학습 시작 (단일 floor)")
        ckpt_path.parent.mkdir(exist_ok=True)
        model = train_stage4(
            [sample], device,
            n_epochs=args.epochs,
            threshold=args.threshold,
            save_path=str(ckpt_path),
        )
        print(f"\n  [train] 학습 완료 → 즉시 infer 실행")
        l2_lab, f_l2, l3_lab, f_l3 = infer_stage4(
            model, f_l1, l1_es.astype(np.int64), l1_ed.astype(np.int64),
            threshold=args.threshold)

    else:  # infer
        print(f"  체크포인트 로드: {ckpt_path}")
        model = SemMergeScoreMLP().to(device)
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=True))
        l2_lab, f_l2, l3_lab, f_l3 = infer_stage4(
            model, f_l1, l1_es.astype(np.int64), l1_ed.astype(np.int64),
            threshold=args.threshold)

    # ── 저장 ──────────────────────────────────────────────────────────────────
    prefix = OUT_DIR / 'floor'
    np.save(f"{prefix}_l2.npy",  l2_lab)
    np.save(f"{prefix}_fl2.npy", f_l2)
    np.save(f"{prefix}_l3.npy",  l3_lab)
    np.save(f"{prefix}_fl3.npy", f_l3)

    # 방 기하 정보 JSON 저장
    geo_path = OUT_DIR / 'floor_l2_geometry.json'
    with open(geo_path, 'w', encoding='utf-8') as fp:
        json.dump({'n_rooms': len(room_geometry), 'rooms': room_geometry},
                  fp, ensure_ascii=False, indent=2)
    print(f"  방 기하 JSON 저장: {geo_path}")

    # L2 라벨 통계
    n_l2 = f_l2.shape[0]
    l2_sizes = np.bincount(l2_lab)
    print(f"\n[Stage 4] 완료")
    print(f"  L2 그룹: {n_l2}개  (최대 크기: {l2_sizes.max()}, 평균: {l2_sizes.mean():.1f})")
    print(f"  저장:")
    print(f"    {prefix}_l2.npy   [{len(l2_lab)}]   SP별 L2 그룹 id")
    print(f"    {prefix}_fl2.npy  {f_l2.shape}  L2 feature")
    print(f"    {prefix}_l3.npy   [{len(l3_lab)}]   L2별 L3 id (전부 0)")
    print(f"    {prefix}_fl3.npy  {f_l3.shape}  L3 feature (floor 전체)")
