from app.model.stages import InputStage, OutputStage, StageOne
from app.schemas.callback import GraphData
from app.schemas.transform import TransformRequest


class ModelPipeline:
    def __init__(self) -> None:
        self.input_stage = InputStage()
        self.stage_one = StageOne()
        self.output_stage = OutputStage()

    async def run(self, payload: TransformRequest) -> GraphData:
        model_input = await self.input_stage.run(payload)
        stage_result = await self.stage_one.run(model_input)
        return await self.output_stage.run(stage_result)
