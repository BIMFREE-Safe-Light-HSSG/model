# Model Server

SuperSafeTwin 백엔드와 연동되는 FastAPI 기반 모델 서버입니다.

---

## 빠른 시작 (파이프라인 단독 실행)

### 1. 클론

```bash
git clone https://github.com/BIMFREE-Safe-Light-HSSG/model.git
cd model
```

### 2. Python 패키지 설치

PyTorch는 CUDA 버전에 맞게 먼저 설치합니다.

```bash
# CUDA 12.1 (RTX 30xx/40xx)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

나머지 패키지:

```bash
pip install -r pipeline/requirements.txt
```

### 3. SAM 체크포인트 다운로드 (2.5 GB, 1회만)

```bash
# macOS
curl -L -o pipeline/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Linux
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P pipeline/
```

### 4. 샘플 입력 파일 다운로드

| 파일 | 링크 |
|------|------|
| `6_floor.npz` | [Google Drive](https://drive.google.com/file/d/1JPGTxzs2gOvZeZf3dUqG28zyIxZdRu3r/view?usp=sharing) |
| `5_floor.npz` | [Google Drive](https://drive.google.com/file/d/1tB0PbRT_Zm3DzXQAOzAqNEVzSOIW6s99/view?usp=sharing) |

다운로드한 파일을 `pipeline/src/`에 저장합니다.

```bash
mkdir -p pipeline/src
# 다운로드한 파일을 pipeline/src/ 로 이동
mv 6_floor.npz pipeline/src/
mv 5_floor.npz pipeline/src/
```

### 5. 파이프라인 실행

```bash
cd pipeline/
python run_pipeline.py --src src/6_floor.npz
```

실행이 완료되면 결과물이 `pipeline/results/6_scene_graph.json`에 생성됩니다.

> **자동 처리 항목**
> - OpenShape_code 레포: 첫 실행 시 자동 `git clone` → `pipeline/OpenShape_code/`
> - OpenShape 모델(vitb32): 첫 실행 시 HuggingFace에서 자동 다운로드
> - 중간 산출물 디렉토리: 각 스테이지가 자동 생성
> - Stage 3~5 체크포인트: 없으면 자동 학습 후 저장, 있으면 바로 추론

### 6. 결과 시각화

파이프라인 실행 후 생성된 `results/*.json`을 Open3D로 3D 시각화합니다.

```bash
cd pipeline/

# 기본 실행 (JSON 경로 직접 지정)
python visualize_hssg.py --json results/6_scene_graph.json

# results/ 폴더의 JSON 목록에서 선택
python visualize_hssg.py --src
```

**주요 옵션:**

| 옵션 | 설명 |
|------|------|
| `--json <path>` | 시각화할 scene_graph.json 경로 |
| `--src` | results/ 폴더의 JSON 목록에서 선택 |
| `--color_by zone` | 같은 방(Zone) 소속 객체를 동일 색으로 표시 |
| `--color_by floor` | 층별 색상으로 표시 |
| `--no_hierarchy` | Zone→Asset 계층 연결선 숨김 |
| `--no_zones` | 방 반투명 면 숨김 |
| `--no_assets` | asset 구체 숨김 |
| `--no_edges` | 관계선 숨김 |
| `--class chair` | 특정 클래스만 표시 |
| `--floor_links` | 층 간 연결선 표시 |
| `--zone_links` | 방 간 연결선 표시 |

**사용 예시:**

```bash
# Zone별 색상으로 계층 구조 확인
python visualize_hssg.py --json results/6_scene_graph.json --color_by zone

# 의자만 표시
python visualize_hssg.py --json results/6_scene_graph.json --class chair

# 관계선·계층선 없이 깔끔하게
python visualize_hssg.py --json results/6_scene_graph.json --no_edges --no_hierarchy
```

> Open3D가 설치되어 있어야 합니다: `pip install open3d`

---

백엔드가 `/transform`으로 변환 작업을 submit하면 모델 서버는 즉시 `202 Accepted`를 반환하고, background task에서 SuperHSSG 파이프라인을 실행한 뒤 백엔드 callback API로 상태를 전달합니다.

## Tech Stack

- Python 3.12
- FastAPI
- uv
- Docker

## Project Structure

```text
app/
  api/
    transform.py
  schemas/
    callback.py
    transform.py
  services/
    callback_service.py
    transform_service.py
  model/
    pipeline.py          # 파이프라인 오케스트레이터
    stages.py            # InputStage / PipelineStage / OutputStage
  config.py
  main.py
pipeline/                # SuperHSSG ML 파이프라인
  merge/
  openshapeCLI/
  sem_merge/
  bottomup/
  topdown/
  sam_room_grid.py
  wall_process.py
  room_pipeline.py
  run_pipeline.py
```

## Environment

환경변수 예시는 `.env.example`에 있습니다.

```env
MODEL_CALLBACK_SECRET=change-this-model-callback-secret
MODEL_CALLBACK_TIMEOUT_SECONDS=30

MINIO_ENDPOINT=http://minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_USE_SSL=false

PIPELINE_PYTHON=python3
```

실제 값은 `.env`에 직접 작성합니다.

## Install

```bash
uv sync
```

## Run

개발 서버는 8002번 포트를 사용합니다.

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8002
```

## API

### POST `/transform`

Request:

```json
{
  "task_id": "task-uuid",
  "building_id": "building-uuid",
  "scan_file_path": "s3://scan-files/data-transform/task-uuid/scan.zip",
  "bucket_name": "scan-files",
  "object_key": "data-transform/task-uuid/scan.zip",
  "callback_url": "https://api.example.com/internal/model/data-transform-status"
}
```

Response: `202 Accepted`  `{ "status": "ACCEPTED" }`

## Processing Flow

```
POST /transform
    → 202 Accepted
    → PROCESSING 10%  callback

    → MinIO에서 scan.zip 다운로드
    → zip 해제 → pipeline/src/{task_id}.npz

    → run_pipeline.py subprocess 실행
      Stage 0 완료 → PROCESSING 20%
      Stage 1 완료 → PROCESSING 40%
      Stage 2 완료 → PROCESSING 60%
      Stage 3 완료 → PROCESSING 75%
      Stage 4 완료 → PROCESSING 85%
      Stage 5 완료 → PROCESSING 95%

    → pipeline/results/{stem}_scene_graph.json → GraphData 변환
    → COMPLETED 100% + graph_data  callback
```

## Docker

Build:

```bash
docker build -t model-server .
```

Run:

```bash
docker run --env-file .env -p 8002:8002 model-server
```

## Integration Contract

백엔드와의 상세 연동 계약은 `MODEL.md`를 참고합니다.

---

# SuperHSSG 파이프라인

포인트클라우드(`.npz`) → 계층적 씬 그래프(`.json`) 생성 파이프라인.  
파이프라인 파일은 `pipeline/` 디렉토리에 위치한다.

---

## 전체 흐름

```
pipeline/src/{stem}.npz
    │
    ▼
[Stage 0]  sam_room_grid.py          → room_grid_{stem}.npz
    │          (SAM 기반 방 마스크 생성)
    ▼
[Stage 1]  merge/stage2_s1.py        → src/{stem}_s2.npz
           merge/s2_to_l1.py         → merge/stage2_results/l1_{stem}.npz
    │          (Jaccard Affinity 슈퍼포인트 병합)
    ▼
[Stage 2]  openshapeCLI/stage3_multiview.py   → openshapeCLI/stage3_results/f_l1_floor.npy
           (또는 stage3_openshape.py)           → openshapeCLI/stage3_results/l1_labels_text.json
    │          (OpenShape 3D + MultiView CLIP 의미 임베딩)
    ▼
[Stage 3]  sem_merge/stage4_sem_merge.py      → sem_merge/stage4_results/floor_l2.npy
    │                                           → sem_merge/stage4_results/floor_l2_geometry.json
    │          (L1 → L2 의미 병합, SAM 마스크 직접 방 할당)
    ▼
[Stage 4]  bottomup/stage5_bottomup.py        → bottomup/stage5_results/floor_h_l1.npy
    │                                           → bottomup/stage5_results/floor_h_l2.npy
    │                                           → bottomup/stage5_results/floor_room.npy
    │          (Bottom-up GAT 의미 집계)
    ▼
[Stage 5]  topdown/stage6_topdown.py          → results/{stem}_scene_graph.json
               (Top-down Cross-Attention 정제 + HSSG 출력)
```

---

## 파이프라인 직접 실행

```bash
cd pipeline/

# 전체 실행 (기본: MultiView CLIP)
python run_pipeline.py --src src/6_floor.npz

# OpenShape 3D only (MultiView 없이)
python run_pipeline.py --src src/6_floor.npz --no_multiview

# 특정 스테이지만
python run_pipeline.py --src src/6_floor.npz --only 3

# 특정 스테이지부터
python run_pipeline.py --src src/6_floor.npz --from_stage 2

# 특정 스테이지 건너뜀
python run_pipeline.py --src src/6_floor.npz --skip 0
```

### src 파일명 → 출력 파일명 규칙
| 입력 | 출력 |
|------|------|
| `src/floor.npz` | `results/scene_graph.json` |
| `src/6_floor.npz` | `results/6_scene_graph.json` |
| `src/5_floor.npz` | `results/5_scene_graph.json` |

---

## 스테이지별 상세

### Stage 0 — SAM 기반 방 마스크 생성 (`sam_room_grid.py`)

wall_door_2d.py의 flood fill 방 검출을 SAM으로 대체한다.

| 단계 | 내용 |
|------|------|
| Step 1 | NPZ → sp_pred 기반 벽+문 바이너리 마스크 (베이스라인) |
| Step 2 | wall_process: skeleton endpoint gap 연결 + 모폴로지 노이즈 제거 (∩ 원본) |
| Step 3 | SAM(vit_h) 추론 → 전체 마스크 목록 `all_masks` (캐시: `results/sam_out/sam_cache_{stem}_room_grid.pkl`) |
| Step 4 | 마스크 필터링 — 현재 미사용 (코드 보존) |
| Step 5 | `all_masks` 면적 내림차순 정렬 → `masks [N, H, W]` 스택으로 NPZ 저장 |

**출력:** `room_grid_{stem}.npz`
```
masks      [N, H, W] uint8   SAM 마스크 스택 (N개 방, 면적 내림차순)
x_min, y_min, cell_size      좌표계
W, H                         그리드 크기
```

> **포맷 호환성**: Stage 0 출력 NPZ를 읽는 모든 하위 스테이지(Stage 1, Stage 3)는
> `masks` 키 존재 여부로 신형/구형을 자동 감지한다.
> - `masks` 키 → 면적 최소 마스크 룩업 (신형, sam_room_grid.py)
> - `grid` 키 → 래스터 그리드 룩업 (구형, wall_door_2d.py)

---

### Stage 1 — 슈퍼포인트 병합 (`merge/stage2_s1.py` + `merge/s2_to_l1.py`)

SP(슈퍼포인트) → L1 클러스터로 Jaccard Affinity 기반 병합.  
학습 파라미터 없음 (training-free).

- `room_grid_{stem}.npz` 의 `masks` 를 로드해 동일 방 내에서만 병합 허용 (Room 제약)
- S1 centroid를 각 마스크의 `segmentation[py, px]` 에 직접 조회 → 면적 최소 마스크의 방 ID 사용

**출력:** `merge/stage2_results/l1_{stem}.npz`

---

### Stage 2 — 의미 임베딩 (`openshapeCLI/stage3_multiview.py`)

각 L1 객체에 OpenShape 3D 임베딩 + CLIP 라벨을 부여한다.

- **class 0~11 (비 clutter)**: SPT 예측 라벨 그대로 사용
- **class 12 (clutter)**: MultiView CLIP으로 세부 라벨 결정
  - CLIP 유사도 < `clutter_sim_threshold(0.5)` → OpenShape 3D 병행 후 max 채택

**출력:**
```
openshapeCLI/stage3_results/f_l1_floor.npy        [N_l1, 651]  GNN 입력 feature
openshapeCLI/stage3_results/l1_labels_text.json              텍스트 라벨
```

---

### Stage 3 — L2 방 병합 (`sem_merge/stage4_sem_merge.py`)

L1 쌍의 semantic + geometric feature로 병합 스코어를 MLP로 예측, L2(방 단위)로 병합.

**방 할당 우선순위:**
1. `room_grid_{stem}.npz` (`masks` 키) → SAM 마스크 직접 룩업  
   각 L1 centroid를 포함하는 마스크 중 **면적 최소** 마스크가 해당 방
2. `room_grid_{stem}.npz` (`grid` 키) → 래스터 그리드 룩업 (구형)
3. 없으면 → XY flood fill 폴백

**모드 결정 (자동):**
| 조건 | 동작 |
|------|------|
| `stage4_model.pt` 없음 | MLP 학습 → 체크포인트 저장 → 즉시 infer |
| `stage4_model.pt` 있음 | 체크포인트 로드 → infer |

**출력:**
```
sem_merge/stage4_results/floor_l2.npy            [N_l1] L2 라벨
sem_merge/stage4_results/floor_l2_geometry.json  방 경계 기하 정보
sem_merge/stage4_results/stage4_model.pt         체크포인트 (학습 후 자동 저장)
```

---

### Stage 4 — Bottom-up 집계 (`bottomup/stage5_bottomup.py`)

L1 간 GAT 메시지 패싱 + PMA(Set Transformer)로 Room 노드 feature 생성.

**출력:**
```
bottomup/stage5_results/floor_h_l1.npy   [N_l1, D]  L1 refined feature
bottomup/stage5_results/floor_h_l2.npy   [N_l2, D]  L2 feature
bottomup/stage5_results/floor_room.npy   [N_room, D] Room feature
```

---

### Stage 5 — Top-down 정제 + HSSG 출력 (`topdown/stage6_topdown.py`)

Cross-Attention으로 방 맥락을 L1에 주입, Bottom-up ↔ Top-down을 T=3회 반복.  
최종 씬 그래프(BUILDING→FLOOR→ZONE→ASSET)를 JSON으로 출력.

**출력:** `results/{stem}_scene_graph.json`

---

## 필요한 체크포인트 파일

| 파일 | 위치 | 용도 | 비고 |
|------|------|------|------|
| `sam_vit_h_4b8939.pth` | `pipeline/` | Stage 0 SAM 추론 | **필수** — [Meta AI 공식 배포](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| OpenShape `vitb32` | HuggingFace 자동 다운로드 | Stage 2 3D 임베딩 | 첫 실행 시 자동 다운로드 (`~/.cache/huggingface/`) |
| `stage4_results/stage4_model.pt` | `pipeline/sem_merge/` | Stage 3 MLP 병합 스코어 | 없으면 **자동 학습** 후 저장, 있으면 infer |
| `stage5_results/stage5_model.pt` | `pipeline/bottomup/` | Stage 4 GAT+PMA | 없으면 **자동 학습** 후 저장 |
| `stage6_results/stage6_model.pt` | `pipeline/topdown/` | Stage 5 Cross-Attention | 없으면 **자동 학습** 후 저장 |

> **모든 모델 체크포인트**: 파일이 없으면 해당 스테이지에서 자동으로 학습(train)한 뒤 저장하고,  
> 이후 실행부터는 저장된 체크포인트로 추론(infer)한다.

### SAM 가중치 다운로드

```bash
# vit_h (현재 사용, 636M) — pipeline/ 디렉토리에 저장
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P pipeline/
```

---

## 중간 산출물 위치 요약

```
pipeline/
├── src/
│   ├── {stem}.npz                          입력 포인트클라우드
│   └── {stem}_s2.npz                       Stage 1 중간 산출물
├── room_grid_{stem}.npz                    Stage 0 출력 (masks [N,H,W] 포맷)
├── results/
│   ├── sam_out/
│   │   ├── sam_cache_{stem}_room_grid.pkl  SAM 추론 캐시
│   │   └── sam_room_grid_{stem}.png        Stage 0 시각화 (4패널)
│   └── {stem}_scene_graph.json             최종 출력
├── merge/stage2_results/
│   └── l1_{stem}.npz                       Stage 1 출력
├── openshapeCLI/stage3_results/
│   ├── f_l1_floor.npy                      Stage 2 임베딩
│   └── l1_labels_text.json                 Stage 2 라벨
├── sem_merge/stage4_results/
│   ├── floor_l2.npy                        Stage 3 L2 라벨
│   ├── floor_l2_geometry.json              Stage 3 방 기하
│   └── stage4_model.pt                     체크포인트 (자동 학습 후 저장)
├── bottomup/stage5_results/
│   ├── floor_h_l1.npy                      Stage 4 L1 feature
│   ├── floor_h_l2.npy                      Stage 4 L2 feature
│   ├── floor_room.npy                      Stage 4 Room feature
│   └── stage5_model.pt                     체크포인트
└── topdown/stage6_results/
    ├── floor_hssg.npz                      Stage 5 중간
    └── stage6_model.pt                     체크포인트
```
