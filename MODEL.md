# Model Server Integration Guide

SuperSafeTwin 백엔드와 모델 서버 사이의 연동 계약입니다. 현재 연동은 비동기 callback 방식입니다. 백엔드는 모델 서버에 변환 작업 시작만 요청하고, 모델 서버는 상태가 바뀔 때마다 백엔드 내부 callback API를 호출합니다.

## Base Rules

- 백엔드가 모델 서버에 변환 작업을 submit합니다.
- 모델 서버는 submit 요청을 받으면 긴 변환 결과를 기다리지 않고 `2xx`로 응답합니다. 권장 상태 코드는 `202 Accepted`입니다.
- 모델 서버는 변환 진행 중 `PROCESSING`, `COMPLETED`, `FAILED` 상태를 백엔드 callback API로 전달합니다.
- 완료 callback에는 `graph_data`를 함께 넣습니다. 백엔드는 해당 JSON을 `graph_data.graph_json`에 저장합니다.
- callback API는 `Authorization: Bearer <MODEL_CALLBACK_SECRET>` 인증을 사용합니다.
- 요청/응답은 JSON입니다.

## Environment

백엔드에서 사용하는 모델 서버 관련 환경변수입니다.

```env
MODEL_SERVER_URL=http://example.com/transform
MODEL_SERVER_TIMEOUT_SECONDS=30
MODEL_CALLBACK_URL=https://api.example.com/internal/model/data-transform-status
MODEL_CALLBACK_SECRET=change-this-model-callback-secret
```

| Name | Default | Description |
| --- | --- | --- |
| `MODEL_SERVER_URL` | `http://localhost:8001/transform` | 백엔드가 작업 시작 요청을 보낼 모델 서버 endpoint |
| `MODEL_SERVER_TIMEOUT_SECONDS` | `30` | submit 요청 timeout. 0보다 큰 숫자여야 함 |
| `MODEL_CALLBACK_URL` | `http://localhost:8000/internal/model/data-transform-status` | 모델 서버가 상태 업데이트를 보낼 백엔드 callback URL |
| `MODEL_CALLBACK_SECRET` | 없음 | callback API 인증용 bearer secret. 반드시 설정해야 함 |

모델 서버도 같은 `MODEL_CALLBACK_SECRET` 값을 알고 있어야 합니다. 실제 secret은 `.env`에만 작성합니다.

## Backend Flow

```text
1. 프론트엔드가 POST /data-transforms/upload 호출
2. 백엔드가 data_transform task 생성
3. 백엔드가 MinIO presigned PUT URL 반환
4. 프론트엔드가 presigned URL로 스캔 파일 업로드
5. 프론트엔드가 POST /data-transforms/{task_id}/complete-upload 호출
6. 백엔드가 task 상태를 PROCESSING, progress_percent=10으로 변경
7. 백엔드 background task가 모델 서버에 transform 작업을 submit
8. 모델 서버가 변환 상태 변경 시 백엔드 callback API 호출
9. COMPLETED callback 수신 시 백엔드가 graph_data.graph_json 저장
10. 프론트엔드는 GET /data-transforms/{task_id} polling으로 상태 확인
```

## Model Server Submit Endpoint

### POST `MODEL_SERVER_URL`

백엔드가 업로드 완료 후 모델 서버에 보내는 작업 시작 요청입니다.

Request body:

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

Fields:

| Field | Type | Nullable | Description |
| --- | --- | --- | --- |
| `task_id` | string UUID | no | `data_transform.id` |
| `building_id` | string UUID | yes | 변환 대상 건물 ID. 현재 정상 task에서는 값이 있음 |
| `scan_file_path` | string | no | `s3://{bucket_name}/{object_key}` 형식의 스캔 파일 위치 |
| `bucket_name` | string | no | MinIO bucket 이름 |
| `object_key` | string | no | MinIO object key |
| `callback_url` | string URL | no | 모델 서버가 상태 업데이트를 보낼 백엔드 callback URL |

권장 응답:

```json
{
  "status": "ACCEPTED"
}
```

백엔드는 submit 응답 body를 사용하지 않습니다. HTTP `2xx`이면 submit 성공으로 봅니다.

## Backend Callback Endpoint

### POST `/internal/model/data-transform-status`

모델 서버가 백엔드로 상태 업데이트를 보낼 때 호출합니다.

Headers:

```http
Authorization: Bearer <MODEL_CALLBACK_SECRET>
Content-Type: application/json
```

### PROCESSING

```json
{
  "task_id": "task-uuid",
  "status": "PROCESSING",
  "progress_percent": 45,
  "error_message": null
}
```

### COMPLETED

```json
{
  "task_id": "task-uuid",
  "status": "COMPLETED",
  "progress_percent": 100,
  "error_message": null,
  "graph_data": {
    "version": "1.0",
    "nodes": [],
    "edges": [],
    "assets": {}
  }
}
```

`COMPLETED` 상태에서는 `graph_data`가 필수입니다. 백엔드는 이 값을 그대로 `graph_data.graph_json`에 저장합니다.

### FAILED

```json
{
  "task_id": "task-uuid",
  "status": "FAILED",
  "progress_percent": 60,
  "error_message": "failed to parse scan file"
}
```

`FAILED` 상태에서는 `error_message`가 필수입니다.

Callback response:

```json
{
  "task_id": "task-uuid",
  "building_id": "building-uuid",
  "status": "PROCESSING",
  "progress_percent": 45,
  "error_message": null
}
```

## Status Rules

허용 상태는 아래 세 가지입니다.

```text
PROCESSING
COMPLETED
FAILED
```

백엔드는 terminal status를 보호합니다. `COMPLETED` 또는 `FAILED`가 된 task에 다른 상태 callback이 오면 `409 Conflict`를 반환합니다. 같은 terminal status callback이 재전송되면 현재 task 상태를 그대로 반환하여 중복 callback을 안전하게 처리합니다.

진행률은 `0`부터 `100` 사이 정수여야 합니다. `PROCESSING` 또는 `FAILED` callback에서 기존 progress보다 낮은 값이 오면 백엔드는 기존 progress를 유지합니다.

## Storage Contract

모델 서버는 submit 요청의 `bucket_name`과 `object_key`를 사용해 업로드된 스캔 파일을 읽습니다.

```text
bucket_name: scan-files
object_key: data-transform/0f2f5f6a-2cc8-4db6-a6ed-2c092dff4676/scan.zip
scan_file_path: s3://scan-files/data-transform/0f2f5f6a-2cc8-4db6-a6ed-2c092dff4676/scan.zip
```

백엔드는 모델 서버에 presigned GET URL을 전달하지 않습니다. 모델 서버는 자체 MinIO/S3 client 설정으로 객체를 읽어야 합니다. Docker Compose 내부 네트워크에서 접근한다면 MinIO endpoint는 보통 `http://minio:9000`입니다.

## Backend Persistence

성공 시 백엔드는 callback의 `graph_data`를 아래 테이블에 저장합니다.

```text
graph_data.building_id = data_transform.building_id
graph_data.data_transform_id = task_id
graph_data.graph_json = callback.graph_data
```

이후 프론트엔드는 최종 결과를 아래 API로 조회합니다.

```text
GET /buildings/{building_id}/scene-graph
```

## Error Handling

모델 서버 submit 요청이 실패하면 백엔드는 task를 `FAILED`로 변경합니다.

Callback API의 주요 에러는 아래와 같습니다.

```text
400 Bad Request        payload가 계약과 맞지 않음
401 Unauthorized       MODEL_CALLBACK_SECRET 불일치
404 Not Found          task 또는 building 없음
409 Conflict           terminal status 이후 다른 상태 callback 수신
422 Unprocessable      request body validation 실패
500 Internal Error     callback secret 미설정 등 서버 설정 오류
```

## Minimal Model Server Example

```python
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_CALLBACK_SECRET = "change-this-model-callback-secret"

app = FastAPI()


class TransformRequest(BaseModel):
    task_id: str
    building_id: str | None = None
    scan_file_path: str
    bucket_name: str
    object_key: str
    callback_url: str


@app.post("/transform", status_code=202)
async def transform_scan(payload: TransformRequest) -> dict[str, str]:
    # 실제 구현에서는 background worker/queue로 넘깁니다.
    return {"status": "ACCEPTED"}


async def notify_completed(callback_url: str, task_id: str, graph_data: dict[str, Any]) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(
            callback_url,
            headers={"Authorization": f"Bearer {MODEL_CALLBACK_SECRET}"},
            json={
                "task_id": task_id,
                "status": "COMPLETED",
                "progress_percent": 100,
                "error_message": None,
                "graph_data": graph_data,
            },
        )
```

## Compatibility Notes

- 프론트엔드의 `POST /data-transforms/{task_id}/complete-upload`와 polling 흐름은 유지됩니다.
- 모델 서버는 상태가 바뀔 때마다 callback을 보내되, 네트워크 실패에 대비해 재시도할 수 있어야 합니다.
- `graph_data` 안에 binary 또는 매우 큰 asset 본문을 직접 넣지 않습니다.
- 영상, point cloud, mesh, glb 같은 무거운 asset은 MinIO에 저장하고 scene graph에는 asset id, object key, URL 같은 참조만 넣는 방향을 권장합니다.
