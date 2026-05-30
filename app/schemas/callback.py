from typing import Any, Literal

from pydantic import BaseModel, Field


TransformStatus = Literal["PROCESSING", "COMPLETED", "FAILED"]


class GraphData(BaseModel):
    version: str = "1.0"
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    assets: dict[str, Any] = Field(default_factory=dict)


class TransformCallbackPayload(BaseModel):
    task_id: str
    status: TransformStatus
    progress_percent: int = Field(ge=0, le=100)
    error_message: str | None = None
    graph_data: GraphData | None = None
