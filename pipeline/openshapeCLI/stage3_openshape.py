"""
SuperHSSG Stage 3: OpenShape + CLIP 분류
=========================================
모든 L1 객체 → OpenShape → [512-dim 벡터] (GNN 입력용)

라벨 부여 전략:
  class 0~11 → SPT 분류 결과를 라벨로 직접 사용
  class 12   → CLIP 유사도 비교
               유사도 > 0.25 → CLIP 라벨
               유사도 ≤ 0.25 → "unknown"

OpenShape 설치:
  1) pip install huggingface_hub open_clip_torch
  2) git clone https://github.com/Colin97/OpenShape_code
  3) export PYTHONPATH=/path/to/OpenShape_code/src

실행:
  python stage3_openshape.py \\
      --floor_npz ./Area_6_spt/floor.npz \\
      --l1_npz    ./stage2_results/l1_floor.npz \\
      --openshape vitb32

출력:
  stage3_results/f_l1_floor.npy       [N_l1, 651]  GNN 입력용
  stage3_results/l1_labels_text.json               텍스트 라벨
"""

import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse

# ── OpenShape_code/src 자동 경로 등록 ────────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
_OPENSHAPE_SRC = _THIS_DIR.parents[1] / 'OpenShape_code' / 'src'
if _OPENSHAPE_SRC.exists() and str(_OPENSHAPE_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENSHAPE_SRC))

D_GEO          = 139
D_SEM          = 512        # OpenShape vitb32 출력 차원
D_L1           = D_GEO + D_SEM   # 651
N_SAMPLE       = 1024
N_CLASSES      = 13
CLUTTER_CLASS     = 12
CLIP_THRESHOLD    = 0.25
# ceiling(0), floor(1), wall(2), beam(3), column(4) — L1 객체 노드로 JSON에 넣지 않는다.
# stage2_colab.py와 일치: column(4)도 구조체로 분류.
# stage4 flood fill 은 sp_pred 배열을 직접 읽으므로 이 필터와 무관하다.
STRUCTURAL_CLASSES = {0, 1, 2, 3, 4}

S3DIS_LABELS = [
    'ceiling',   # 0
    'floor',     # 1
    'wall',      # 2
    'beam',      # 3
    'column',    # 4
    'window',    # 5
    'door',      # 6
    'chair',     # 7
    'table',     # 8
    'bookcase',  # 9
    'sofa',      # 10
    'board',     # 11
    'clutter',   # 12  → CLIP으로 세분화
]

_HF_REPOS = {
    'vitb32': ('OpenShape/openshape-pointbert-vitb32-rgb', 512),
    'vitl14': ('OpenShape/openshape-pointbert-vitl14-rgb', 768),
    'vitg14': ('OpenShape/openshape-pointbert-vitg14-rgb', 1280),
}

CLUTTER_CANDIDATES = [
    "window", "ceiling lights", "lights",
    "coffee machine", "printer", "monitor", "computer screen",
    "fire extinguisher", "trash can", "waste bin", "recycling bin",
    "plant", "potted plant", "indoor tree",
    "lamp", "desk lamp", "light fixture",
    "bag", "backpack", "luggage", "box", "cardboard box",
    "picture frame", "painting", "clock",
    "microwave", "refrigerator", "kettle",
    "pillow", "blanket",
    "keyboard", "mouse", "speaker",
    "bicycle", "stroller",
    "vending machine", "water dispenser", "air conditioner",
    "pipe", "duct", "cable",
    "sign", "notice board",
]

# LVIS 카테고리 캐시 경로
_LVIS_CACHE = Path(__file__).resolve().parent / "lvis_categories.json"


def load_lvis_candidates(cache_path: Path = _LVIS_CACHE) -> list:
    """
    LVIS v1 카테고리 1203개를 반환한다.
    캐시(lvis_categories.json)가 있으면 로드, 없으면 mmdetection GitHub에서 파싱 후 저장.
    """
    if cache_path.exists():
        with open(cache_path) as f:
            names = json.load(f)
        print(f"  [LVIS] 캐시 로드: {len(names)}개  ({cache_path.name})")
        return names

    print("  [LVIS] 카테고리 다운로드 중 (최초 1회)...")
    import requests, re
    url = ("https://raw.githubusercontent.com/open-mmlab/mmdetection"
           "/main/mmdet/datasets/lvis.py")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    text = r.text

    idx_start = text.find("LVISV1Dataset(LVISDataset)")
    idx_end   = text.find("\nclass ", idx_start + 10)
    chunk     = text[idx_start:idx_end]

    m = re.search(r"'classes':\s*\((.+?)(?:\n\s*\}|\n\s*'palette')",
                  chunk, re.DOTALL)
    if not m:
        raise RuntimeError("LVIS 카테고리 파싱 실패")

    raw   = m.group(1)
    names = [n.replace('_', ' ')
             for n in re.findall(r"'([^']+)'", raw)]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(names, f, ensure_ascii=False, indent=2)
    print(f"  [LVIS] {len(names)}개 저장 → {cache_path.name}")
    return names


# ── OpenShape 로더 ──────────────────────────────────────────────────────────
def _download_openshape(model_key: str) -> tuple:
    """HuggingFace에서 가중치 다운로드. (ckpt_path, out_dim) 반환"""
    if model_key not in _HF_REPOS:
        raise ValueError(f"알 수 없는 모델: {model_key}. 선택: {list(_HF_REPOS)}")
    repo_id, out_dim = _HF_REPOS[model_key]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub 미설치.\n"
            "  pip install huggingface_hub"
        )
    print(f"  [OpenShape] HuggingFace 다운로드: {repo_id}")
    path = hf_hub_download(repo_id=repo_id, filename="model.pt")
    print(f"  [OpenShape] 다운로드 완료: {path}")
    return path, out_dim


def load_openshape(source: str, device: str = 'cpu') -> tuple:
    """
    OpenShape 인코더 로드.

    source:
      'vitb32' | 'vitl14' | 'vitg14'  → HuggingFace 자동 다운로드
      '/path/to/model.pt'              → 로컬 파일

    Returns:
      (encoder: nn.Module, out_dim: int)

    필수 조건:
      공식 레포 클론 후 PYTHONPATH 설정
        git clone https://github.com/Colin97/OpenShape_code
        export PYTHONPATH=/path/to/OpenShape_code/src
    """
    # 로컬 파일 vs HuggingFace
    src_path = Path(source)
    if src_path.exists():
        ckpt_path = str(src_path)
        # 로컬 파일은 out_dim을 vitb32 기본값으로 설정
        out_dim = D_SEM
    else:
        ckpt_path, out_dim = _download_openshape(source)

    # 체크포인트 로드
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))

    # ── ppat.py / pointnet_util.py 직접 로드 ────────────────────────
    # dgl.geometry.farthest_point_sampler → 순수 PyTorch FPS로 대체
    import importlib.util, sys, os, types

    def _fps_torch(xyz, npoint):
        """dgl FPS 대체: [B, N, 3] → [B, npoint] 인덱스"""
        B, N, _ = xyz.shape
        device  = xyz.device
        idx     = torch.zeros(B, npoint, dtype=torch.long, device=device)
        dist    = torch.full((B, N), 1e10, device=device)
        farthest = torch.randint(0, N, (B,), device=device)
        for i in range(npoint):
            idx[:, i] = farthest
            c    = xyz[torch.arange(B), farthest].unsqueeze(1)
            d    = ((xyz - c) ** 2).sum(-1)
            dist = torch.minimum(dist, d)
            farthest = dist.argmax(-1)
        return idx

    # 가짜 dgl.geometry 모듈 주입
    fake_dgl_geo = types.ModuleType('dgl.geometry')
    fake_dgl_geo.farthest_point_sampler = _fps_torch
    fake_dgl = types.ModuleType('dgl')
    fake_dgl.geometry = fake_dgl_geo
    sys.modules.setdefault('dgl',          fake_dgl)
    sys.modules.setdefault('dgl.geometry', fake_dgl_geo)

    # OpenShape 레포 src 경로 탐색
    # 상대 임포트(from .pointnet_util import ...) 해결을 위해
    # 가짜 'models' 패키지를 sys.modules에 등록 후 로드
    PointPatchTransformer = None
    for base in sys.path:
        pu_file   = os.path.join(base, 'models', 'pointnet_util.py')
        ppat_file = os.path.join(base, 'models', 'ppat.py')
        if os.path.exists(pu_file) and os.path.exists(ppat_file):
            # 가짜 models 패키지 생성
            models_pkg = types.ModuleType('models')
            models_pkg.__path__ = [os.path.join(base, 'models')]
            models_pkg.__package__ = 'models'
            sys.modules['models'] = models_pkg

            # pointnet_util 로드
            spec_pu = importlib.util.spec_from_file_location(
                'models.pointnet_util', pu_file,
                submodule_search_locations=[]
            )
            mod_pu = importlib.util.module_from_spec(spec_pu)
            mod_pu.__package__ = 'models'
            sys.modules['models.pointnet_util'] = mod_pu
            models_pkg.pointnet_util = mod_pu
            spec_pu.loader.exec_module(mod_pu)

            # ppat 로드
            spec_pp = importlib.util.spec_from_file_location(
                'models.ppat', ppat_file,
                submodule_search_locations=[]
            )
            mod_pp = importlib.util.module_from_spec(spec_pp)
            mod_pp.__package__ = 'models'
            sys.modules['models.ppat'] = mod_pp
            models_pkg.ppat = mod_pp
            spec_pp.loader.exec_module(mod_pp)

            PointPatchTransformer = mod_pp.PointPatchTransformer
            print(f"  [OpenShape] ppat.py 로드 완료: {ppat_file}")
            break

    if PointPatchTransformer is None:
        raise ImportError(
            "\n"
            "OpenShape 공식 레포가 PYTHONPATH에 없습니다.\n"
            "\n"
            "설치 방법:\n"
            "  git clone https://github.com/Colin97/OpenShape_code\n"
            "  export PYTHONPATH=/path/to/OpenShape_code/src\n"
            "\n"
            "Colab:\n"
            "  !git clone https://github.com/Colin97/OpenShape_code\n"
            "  import sys; sys.path.insert(0, '/content/OpenShape_code/src')\n"
        )

    # PointPatchTransformer 생성 (vitb32 하이퍼파라미터)
    encoder = PointPatchTransformer(
        dim=512, depth=12, heads=8, mlp_dim=1024,
        sa_dim=128, patches=512, prad=0.2, nsamp=64,
        in_dim=6, dim_head=64,
    )

    # 키 prefix 제거 (pc_encoder. → '')
    new_state = {k.replace('pc_encoder.', ''): v
                 for k, v in state.items()
                 if k.startswith('pc_encoder.')}

    missing, unexpected = encoder.load_state_dict(new_state, strict=False)
    if missing:
        print(f"  [OpenShape] 누락 키: {len(missing)}개")

    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    print(f"  [OpenShape] ✓ PointPatchTransformer 로드 완료 (dim={out_dim})")
    return encoder, out_dim


# ── SemanticEncoder ─────────────────────────────────────────────────────────
class SemanticEncoder(nn.Module):
    """
    OpenShape 기반 3D 인코더.
    pretrained_path: 'vitb32' | 'vitl14' | '/path/to/model.pt'
    """
    def __init__(self, pretrained_path: str, device: str = 'cpu'):
        super().__init__()
        encoder, out_dim = load_openshape(pretrained_path, device)
        self.encoder = encoder
        self.out_dim = out_dim

    def forward(self, xyz_rgb: torch.Tensor) -> torch.Tensor:
        """[B, N, 6] → [B, out_dim]  L2 정규화
        PointNetSetAbstraction은 [B, C, N] 포맷 입력 필요
        → xyz [B,3,N], features [B,6,N] 으로 transpose 후 전달
        """
        xyz      = xyz_rgb[:, :, :3].transpose(1, 2).contiguous()  # [B, 3, N]
        features = xyz_rgb.transpose(1, 2).contiguous()             # [B, 6, N]
        out = self.encoder(xyz, features)
        return F.normalize(out, dim=-1)


# ── CLIP 텍스트 인코더 ───────────────────────────────────────────────────────
class CLIPTextEncoder:
    """open_clip 기반 텍스트 → 벡터 변환"""

    # OpenShape 모델별 대응 CLIP 백본
    _CLIP_BACKBONE = {
        'vitb32': ('ViT-B-32',    'openai'),
        'vitl14': ('ViT-L-14',    'openai'),
        'vitg14': ('ViT-bigG-14', 'laion2b_s39b_b160k'),
    }

    def __init__(self, openshape_key: str = 'vitb32', device: str = 'cpu'):
        try:
            import open_clip
        except ImportError:
            raise ImportError("pip install open_clip_torch")

        model_name, pretrained = self._CLIP_BACKBONE.get(
            openshape_key, ('ViT-B-32', 'openai')
        )
        self.device    = device
        model, _, _    = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model     = model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print(f"  [CLIP] {model_name} ({pretrained}) 로드 완료")

    @torch.no_grad()
    def encode_texts(self, texts: list) -> np.ndarray:
        """texts → [N, dim]  L2 정규화"""
        tokens = self.tokenizer(texts).to(self.device)
        feats  = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1).cpu().numpy().astype(np.float32)


# ── 라벨 분류 ────────────────────────────────────────────────────────────────
def classify_l1_objects(sem_feat: np.ndarray,
                         l1_classes: np.ndarray,
                         clip_enc: CLIPTextEncoder,
                         candidates: list,
                         confidence_threshold: float = CLIP_THRESHOLD,
                         top_k: int = 3) -> list:
    """
    class 0,1,2,3 (structural) → 'structural' 소스로 표시, JSON 출력 시 제외
    class 4~11               → SPT 라벨 직접 사용
    class 12 (clutter)       → CLIP 유사도 비교
                               score > threshold → CLIP 라벨
                               score ≤ threshold → "unknown"
    """
    n_l1 = len(sem_feat)

    # clutter 인덱스만 모아서 CLIP 일괄 처리
    clutter_ids = [i for i in range(n_l1) if l1_classes[i] == CLUTTER_CLASS]
    clip_sim    = None
    if clutter_ids:
        text_feats    = clip_enc.encode_texts(candidates)
        clutter_feats = sem_feat[clutter_ids]
        clip_sim      = clutter_feats @ text_feats.T   # [N_clutter, N_cand]

    results     = []
    clutter_ptr = 0

    for lid in range(n_l1):
        cls = int(l1_classes[lid])

        # ── structural class (ceiling/floor/wall/beam): L1 객체 노드에서 제외 ──
        if cls in STRUCTURAL_CLASSES:
            results.append({
                'l1_id':  lid,
                'label':  S3DIS_LABELS[cls],   # 참조용으로만 보존
                'source': 'structural',         # 이 소스는 JSON 저장 시 필터링됨
                'score':  None,
                'top3':   None,
            })
            continue

        # ── class 4~11: SPT 라벨 직접 사용 ──────────────────────
        if cls != CLUTTER_CLASS:
            results.append({
                'l1_id':  lid,
                'label':  S3DIS_LABELS[cls],
                'source': 'spt',
                'score':  None,
                'top3':   None,
            })
            continue

        # ── class 12: CLIP 비교 ──────────────────────────────────
        scores  = clip_sim[clutter_ptr]
        top_idx = np.argsort(-scores)[:top_k]
        top     = [(candidates[j], float(scores[j])) for j in top_idx]
        best    = top[0][1]
        clutter_ptr += 1

        results.append({
            'l1_id':  lid,
            'label':  top[0][0] if best > confidence_threshold else 'unknown',
            'source': 'clip'    if best > confidence_threshold else 'unknown',
            'score':  best,
            'top3':   top,
        })

    return results


# ── 데이터 준비 ──────────────────────────────────────────────────────────────
def aggregate_geo_feat(sp_feat: np.ndarray,
                        l1_labels: np.ndarray) -> np.ndarray:
    n_l1 = int(l1_labels.max()) + 1
    geo  = np.zeros((n_l1, D_GEO), dtype=np.float32)
    cnt  = np.zeros(n_l1, dtype=np.float32)
    np.add.at(geo, l1_labels, sp_feat)
    np.add.at(cnt, l1_labels, 1)
    return geo / np.maximum(cnt[:, None], 1)


def sample_l1_point_clouds(coord: np.ndarray,
                            color: np.ndarray,
                            sp_labels: np.ndarray,
                            l1_labels: np.ndarray,
                            n_sample: int = N_SAMPLE) -> np.ndarray:
    n_l1 = int(l1_labels.max()) + 1
    if color.max() > 2.0:
        color = color / 255.0
    xyz_rgb  = np.concatenate([coord, color], axis=1).astype(np.float32)
    point_l1 = l1_labels[sp_labels]
    result   = np.zeros((n_l1, n_sample, 6), dtype=np.float32)
    for lid in range(n_l1):
        pts = xyz_rgb[point_l1 == lid]
        if len(pts) == 0:
            continue
        idx = np.random.choice(len(pts), n_sample, replace=len(pts) < n_sample)
        pts = pts[idx].copy()
        c = pts[:, :3].mean(0); pts[:, :3] -= c
        s = np.abs(pts[:, :3]).max() + 1e-6; pts[:, :3] /= s
        result[lid] = pts
    return result


def get_l1_classes(l1_labels: np.ndarray,
                    sp_pred: np.ndarray) -> np.ndarray:
    n_l1   = int(l1_labels.max()) + 1
    l1_cls = np.zeros(n_l1, dtype=np.int32)
    for lid in range(n_l1):
        mask = l1_labels == lid
        if mask.any():
            cls, cnt    = np.unique(sp_pred[mask], return_counts=True)
            l1_cls[lid] = cls[cnt.argmax()]
    return l1_cls


# ── 추론 ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_stage3(model: SemanticEncoder,
                 sp_feat: np.ndarray,
                 l1_labels: np.ndarray,
                 coord: np.ndarray,
                 color: np.ndarray,
                 sp_labels: np.ndarray,
                 batch_size: int = 32) -> tuple:
    """
    Returns:
      f_l1     [N_l1, 651]  geo(139) + sem(512)
      sem_feat [N_l1, 512]  CLIP 비교용
    """
    model.eval()
    device   = next(model.parameters()).device
    geo_feat = aggregate_geo_feat(sp_feat, l1_labels)
    pcd_all  = sample_l1_point_clouds(coord, color, sp_labels, l1_labels)

    parts = []
    for i in range(0, len(pcd_all), batch_size):
        b = torch.from_numpy(pcd_all[i:i+batch_size]).to(device)
        parts.append(model(b).cpu().numpy())

    sem_feat = np.concatenate(parts, axis=0)
    f_l1     = np.concatenate([geo_feat, sem_feat], axis=1)
    print(f"  L1 {len(pcd_all)}개 → f_L1 {f_l1.shape}")
    return f_l1, sem_feat


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ── 기본 경로: 이 파일 기준 cnu_model/ 루트 ───────────────────
    _ROOT = Path(__file__).resolve().parents[1]   # …/pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument('--floor_npz',  default=str(_ROOT / 'src' / 'floor.npz'))
    parser.add_argument('--l1_npz',     default=str(_ROOT / 'merge' / 'stage2_results' / 'l1_floor.npz'))
    parser.add_argument('--out_dir',    default=str(Path(__file__).resolve().parent / 'stage3_results'))
    parser.add_argument('--openshape',  default='vitb32',
                        help="'vitb32'(기본) | 'vitl14' | 'vitg14' | '/path/model.pt'")
    parser.add_argument('--candidates', default=None,
                        help='clutter 후보 텍스트 (쉼표 구분, 기본: indoor_labels.json)')
    parser.add_argument('--use_lvis',   action='store_true', default=False,
                        help='LVIS v1 1203개 카테고리 사용 (기본: 꺼짐)')
    parser.add_argument('--threshold',  type=float, default=CLIP_THRESHOLD,
                        help=f'CLIP 신뢰도 임계값 (기본 {CLIP_THRESHOLD})')
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--top_k',      type=int,   default=3)
    args = parser.parse_args()

    FLOOR_NPZ = Path(args.floor_npz)
    L1_NPZ    = Path(args.l1_npz)
    OUT_DIR   = Path(args.out_dir);  OUT_DIR.mkdir(exist_ok=True, parents=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[Stage 3] 디바이스: {device}")

    # ── 데이터 로드 ──────────────────────────────────────────────
    if not FLOOR_NPZ.exists():
        raise FileNotFoundError(f"floor.npz 없음: {FLOOR_NPZ}")
    if not L1_NPZ.exists():
        raise FileNotFoundError(f"l1_floor.npz 없음: {L1_NPZ}")

    floor       = np.load(FLOOR_NPZ)
    sp_features = floor['sp_features'].astype(np.float32)
    sp_pred     = floor['sp_pred'].astype(np.int32)
    coord       = floor['coord'].astype(np.float32)
    color       = floor['color'].astype(np.float32)
    sp_labels   = floor['sp_labels'].astype(np.int64)

    l1_data   = np.load(L1_NPZ)
    l1_labels = l1_data['l1_labels'].astype(np.int64)

    # ── 구조체(-1) SP 제거 ────────────────────────────────────────────
    valid_mask  = l1_labels >= 0          # [N_sp] 구조체 제외
    sp_features = sp_features[valid_mask]
    sp_pred     = sp_pred[valid_mask]
    l1_labels   = l1_labels[valid_mask]

    # l1 ID를 0-based 연속 정수로 재매핑
    old_ids   = np.unique(l1_labels)
    id_remap  = np.full(int(old_ids.max()) + 1, -1, dtype=np.int64)
    id_remap[old_ids] = np.arange(len(old_ids))
    l1_labels = id_remap[l1_labels]

    # sp_labels도 valid_mask에 맞게 재인덱싱
    old_sp_idx = np.where(valid_mask)[0]         # 원래 SP 인덱스
    sp_idx_map = np.full(len(valid_mask), -1, dtype=np.int64)
    sp_idx_map[old_sp_idx] = np.arange(len(old_sp_idx))
    pt_valid   = sp_idx_map[sp_labels] >= 0      # 유효 SP에 속한 포인트만
    sp_labels  = sp_idx_map[sp_labels[pt_valid]]
    coord      = coord[pt_valid]
    color      = color[pt_valid]

    n_sp = len(sp_features)
    n_l1 = int(l1_labels.max()) + 1
    print(f"  SP: {n_sp:,}  (구조체 제외)  |  L1 객체: {n_l1:,}")

    l1_classes = get_l1_classes(l1_labels, sp_pred)
    n_clutter  = int((l1_classes == CLUTTER_CLASS).sum())
    print(f"  class 0~11: {n_l1 - n_clutter}개  |  clutter(12): {n_clutter}개")

    # ── OpenShape 로드 ────────────────────────────────────────────
    model = SemanticEncoder(pretrained_path=args.openshape,
                            device=str(device)).to(device)

    # ── OpenShape 추론 (전체 L1) ──────────────────────────────────
    f_l1, sem_feat = infer_stage3(
        model, sp_features, l1_labels,
        coord, color, sp_labels,
        batch_size=args.batch_size,
    )

    # ── GNN feature 저장 ──────────────────────────────────────────
    feat_path = OUT_DIR / 'f_l1_floor.npy'
    np.save(feat_path, f_l1)
    print(f"  feature 저장: {feat_path}  shape={f_l1.shape}")

    # ── 라벨 분류 ─────────────────────────────────────────────────
    print(f"\n[라벨] 분류 시작  (임계값={args.threshold})")
    clip_enc   = CLIPTextEncoder(
        openshape_key=args.openshape if args.openshape in _HF_REPOS else 'vitb32',
        device=str(device),
    )
    _indoor_json = Path(__file__).resolve().parent / 'indoor_labels.json'
    if args.candidates:
        candidates = [c.strip() for c in args.candidates.split(',')]
    elif args.use_lvis:
        candidates = load_lvis_candidates()
    elif _indoor_json.exists():
        candidates = json.load(open(_indoor_json, encoding='utf-8'))
    else:
        candidates = CLUTTER_CANDIDATES
    print(f"  후보 텍스트: {len(candidates)}개"
          f"  ({'LVIS v1' if args.use_lvis else '내장 목록'})")

    results = classify_l1_objects(
        sem_feat, l1_classes, clip_enc, candidates,
        confidence_threshold=args.threshold,
        top_k=args.top_k,
    )

    # ── 결과 출력 ─────────────────────────────────────────────────
    print(f"\n  ── L1 객체 라벨 미리보기 (상위 20개) ──")
    print(f"  {'ID':>4}  {'출처':<8}  {'라벨':<30}  {'점수':>6}")
    print(f"  {'─'*55}")
    for r in results[:20]:
        score_str = f"{r['score']:.3f}" if r['score'] is not None else "  -  "
        print(f"  {r['l1_id']:>4}  {r['source']:<8}  {r['label']:<30}  {score_str}")
    if n_l1 > 20:
        print(f"  ... ({n_l1 - 20}개 생략)")

    from collections import Counter
    src_counts = Counter(r['source'] for r in results)
    lbl_counts = Counter(r['label']  for r in results)
    n_structural = src_counts.get('structural', 0)
    print(f"\n  ── 출처 통계 ──")
    print(f"  SPT 라벨:    {src_counts.get('spt',        0):>4}개")
    print(f"  CLIP 라벨:   {src_counts.get('clip',       0):>4}개")
    print(f"  unknown:     {src_counts.get('unknown',    0):>4}개")
    print(f"  structural:  {n_structural:>4}개  ← JSON에서 제외됨 (ceiling/floor/wall/beam)")
    print(f"\n  ── 상위 라벨 분포 (structural 제외) ──")
    obj_lbl_counts = Counter(r['label'] for r in results if r['source'] != 'structural')
    if obj_lbl_counts:
        for lbl, cnt in obj_lbl_counts.most_common(10):
            bar = '█' * int(cnt / max(obj_lbl_counts.values()) * 25)
            print(f"  {lbl:<30} {cnt:>4}개  {bar}")

    # structural(ceiling/floor/wall/beam) 은 JSON 객체 목록에서 제외한다.
    # stage4 flood fill 은 sp_pred 배열을 직접 사용하므로 영향 없음.
    objects_for_json = [r for r in results if r['source'] != 'structural']

    # ── JSON 저장 ─────────────────────────────────────────────────
    json_path = OUT_DIR / 'l1_labels_text.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'n_l1':            n_l1,
            'n_objects':       len(objects_for_json),
            'n_structural':    n_structural,
            'threshold':       args.threshold,
            'candidates':      candidates,
            'objects':         objects_for_json,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  → 저장된 객체 수: {len(objects_for_json)}개  (전체 L1 {n_l1}개 중 structural {n_structural}개 제외)")

    print(f"\n[Stage 3] 완료")
    print(f"  feature : {feat_path}  {f_l1.shape}")
    print(f"  라벨    : {json_path}")
