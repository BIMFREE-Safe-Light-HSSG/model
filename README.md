# Model Server

SuperSafeTwin 백엔드와 연동되는 FastAPI 기반 모델 서버입니다.

현재 구현은 실제 모델 파이프라인을 넣기 전 단계의 기본 뼈대입니다. 백엔드가 `/transform`으로 변환 작업을 submit하면 모델 서버는 즉시 `202 Accepted`를 반환하고, background task에서 임시 파이프라인을 실행한 뒤 백엔드 callback API로 상태를 전달합니다.

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
    pipeline.py
    stages.py
  config.py
  main.py
```

## Environment

환경변수 예시는 `.env.example`에 있습니다.

```env
MODEL_CALLBACK_SECRET=change-this-model-callback-secret
MODEL_CALLBACK_TIMEOUT_SECONDS=30
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

백엔드가 업로드 완료 후 모델 서버에 변환 작업을 요청하는 endpoint입니다.

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

Response:

```json
{
  "status": "ACCEPTED"
}
```

HTTP status는 `202 Accepted`입니다.

## Processing Flow

1. `/transform` 요청을 받습니다.
2. 즉시 `202 Accepted`를 반환합니다.
3. background task에서 `PROCESSING` callback을 전송합니다.
4. `app/model`의 임시 파이프라인을 실행합니다.
5. 성공하면 `COMPLETED` callback과 `graph_data`를 전송합니다.
6. 실패하면 `FAILED` callback과 `error_message`를 전송합니다.

## Model Pipeline

모델 파이프라인은 `app/model` 아래에 있습니다.

- `pipeline.py`: stage 실행 순서를 조립합니다.
- `stages.py`: 입력 stage, 임시 처리 stage, 출력 stage를 정의합니다.

현재 `StageOne`은 실제 모델 처리를 하지 않고, 이후 구현을 위한 자리만 제공합니다.

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
