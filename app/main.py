from fastapi import FastAPI

from app.api.transform import router as transform_router


def create_app() -> FastAPI:
    app = FastAPI(title="Model Server")
    app.include_router(transform_router)
    return app


app = create_app()
