from dataclasses import dataclass

from app.schemas.callback import GraphData
from app.schemas.transform import TransformRequest


@dataclass(frozen=True)
class ModelInput:
    task_id: str
    building_id: str | None
    scan_file_path: str
    bucket_name: str
    object_key: str


@dataclass(frozen=True)
class StageResult:
    task_id: str
    building_id: str | None


class InputStage:
    async def run(self, payload: TransformRequest) -> ModelInput:
        return ModelInput(
            task_id=payload.task_id,
            building_id=payload.building_id,
            scan_file_path=payload.scan_file_path,
            bucket_name=payload.bucket_name,
            object_key=payload.object_key,
        )


class StageOne:
    async def run(self, model_input: ModelInput) -> StageResult:
        return StageResult(
            task_id=model_input.task_id,
            building_id=model_input.building_id,
        )


class OutputStage:
    async def run(self, stage_result: StageResult) -> GraphData:
        return GraphData(
            version="1.0",
            nodes=[],
            edges=[],
            assets={
                "task_id": stage_result.task_id,
                "building_id": stage_result.building_id,
            },
        )
