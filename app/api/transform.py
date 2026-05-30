from fastapi import APIRouter, BackgroundTasks, status

from app.schemas.transform import TransformAcceptedResponse, TransformRequest
from app.services.transform_service import TransformService


router = APIRouter()


@router.post(
    "/transform",
    response_model=TransformAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def transform_scan(
    payload: TransformRequest,
    background_tasks: BackgroundTasks,
) -> TransformAcceptedResponse:
    service = TransformService()
    background_tasks.add_task(service.process, payload)
    return TransformAcceptedResponse(status="ACCEPTED")
