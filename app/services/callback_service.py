import httpx

from app.config import get_settings
from app.schemas.callback import TransformCallbackPayload


class CallbackService:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def send(
        self,
        callback_url: str,
        payload: TransformCallbackPayload,
    ) -> None:
        headers = self._build_headers()
        timeout = self.settings.model_callback_timeout_seconds

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                callback_url,
                json=payload.model_dump(mode="json", exclude_none=True),
                headers=headers,
            )
            response.raise_for_status()

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}

        if self.settings.model_callback_secret:
            headers["Authorization"] = (
                f"Bearer {self.settings.model_callback_secret}"
            )

        return headers
