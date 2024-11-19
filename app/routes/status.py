from fastapi import APIRouter

from app.managers import FileEmbeddingStatusManager

StatusRouter = APIRouter(
    responses={404: {"description": "Not found"}},
)


@StatusRouter.get("")
async def fetch_status_from_resource_id(resource_id: str):
    return await FileEmbeddingStatusManager.fetch_status_from_resource_id(resource_id)
