
from fastapi import APIRouter, Request

from app.config import AppConfig
from app.managers import VectorEmbeddingManager

config = AppConfig()

AIRouter = APIRouter(prefix="/ai", tags=["ai"])


@AIRouter.post("/embeddings")
async def create_embedding(request: Request):
    body = await request.json()
    file_data = body.get("file_data")
    meta_data = body.get("meta_data")
    ## Clean Up Data + Enrich Textual Content
    response = await VectorEmbeddingManager.process_document(file_data=file_data)
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
