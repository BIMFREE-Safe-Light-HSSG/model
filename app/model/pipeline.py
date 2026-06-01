from __future__ import annotations

from typing import Awaitable, Callable

from app.model.stages import InputStage, OutputStage, PipelineStage
from app.schemas.callback import GraphData
from app.schemas.transform import TransformRequest

ProgressCallback = Callable[[int], Awaitable[None]]


class ModelPipeline:
    def __init__(self) -> None:
        self.input_stage    = InputStage()
        self.pipeline_stage = PipelineStage()
        self.output_stage   = OutputStage()

    async def run(
        self,
        payload: TransformRequest,
        progress_cb: ProgressCallback | None = None,
    ) -> GraphData:
        # 1. MinIO 다운로드 + NPZ 추출
        model_input = await self.input_stage.run(payload)

        # 2. 파이프라인 실행 (진행률 콜백 포함)
        await self.pipeline_stage.run(model_input, progress_cb)

        # 3. scene_graph.json → GraphData
        return await self.output_stage.run(model_input)
