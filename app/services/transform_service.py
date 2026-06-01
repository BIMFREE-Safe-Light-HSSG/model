from pydantic import HttpUrl

from app.model.pipeline import ModelPipeline
from app.schemas.callback import GraphData, TransformCallbackPayload
from app.schemas.transform import TransformRequest
from app.services.callback_service import CallbackService


class TransformService:
    def __init__(self) -> None:
        self.callback_service = CallbackService()
        self.pipeline = ModelPipeline()

    async def process(self, payload: TransformRequest) -> None:
        callback_url = self._callback_url_to_string(payload.callback_url)

        try:
            await self._send_processing(callback_url, payload.task_id, 10)

            # 스테이지별 진행률을 PROCESSING callback으로 실시간 전송
            async def on_progress(pct: int) -> None:
                await self._send_processing(callback_url, payload.task_id, pct)

            graph_data = await self.pipeline.run(payload, progress_cb=on_progress)
            await self._send_completed(callback_url, payload.task_id, graph_data)

        except Exception as exc:
            await self._send_failed(callback_url, payload.task_id, exc)

    async def _send_processing(
        self,
        callback_url: str,
        task_id: str,
        progress_percent: int,
    ) -> None:
        await self.callback_service.send(
            callback_url,
            TransformCallbackPayload(
                task_id=task_id,
                status="PROCESSING",
                progress_percent=progress_percent,
            ),
        )

    async def _send_completed(
        self,
        callback_url: str,
        task_id: str,
        graph_data: GraphData,
    ) -> None:
        await self.callback_service.send(
            callback_url,
            TransformCallbackPayload(
                task_id=task_id,
                status="COMPLETED",
                progress_percent=100,
                graph_data=graph_data,
            ),
        )

    async def _send_failed(
        self,
        callback_url: str,
        task_id: str,
        exc: Exception,
    ) -> None:
        await self.callback_service.send(
            callback_url,
            TransformCallbackPayload(
                task_id=task_id,
                status="FAILED",
                progress_percent=100,
                error_message=str(exc),
            ),
        )

    def _callback_url_to_string(self, callback_url: HttpUrl) -> str:
        return str(callback_url)
