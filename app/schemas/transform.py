from pydantic import BaseModel, HttpUrl


class TransformRequest(BaseModel):
    task_id: str
    building_id: str | None = None
    scan_file_path: str
    bucket_name: str
    object_key: str
    callback_url: HttpUrl


class TransformAcceptedResponse(BaseModel):
    status: str
