from fastapi import APIRouter, Request

from app.config import AppConfig
from app.managers import VectorEmbeddingManager
from fastapi import BackgroundTasks

config = AppConfig()

AIRouter = APIRouter()


@AIRouter.post("/embeddings")
async def create_embedding(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    file_data = body.get("file_data")
    response = await VectorEmbeddingManager.embed_document(file_data, background_tasks)
    return response


@AIRouter.get("/embeddings/search")
async def search_embedding(request: Request):
    query = request.query_params.get("query")
    response = await VectorEmbeddingManager.query_and_generate(
        query=query,
        response_type="default",
        num_chunks=4,
        similarity_threshold=0.5,
        temperature=0.7,
    )

    await VectorEmbeddingManager.clear_expired_cache()
    return response
