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
    file_id = request.query_params.get("file_id", None)
    workspace_id = request.query_params.get("workspace_id", None)
    response = await VectorEmbeddingManager.query_and_generate(
        query=query,
        filters={
            "source_id": file_id,
            "workspace_id": workspace_id,
        },
    )
    return response


@AIRouter.get("/embeddings/chat")
async def start_chat(request: Request):
    query = request.query_params.get("query")
    file_id = request.query_params.get("file_id", None)
    workspace_id = request.query_params.get("workspace_id", None)
    thread_id = request.query_params.get("thread_id", None)
    response = await VectorEmbeddingManager.init_chat(
        query=query,
        filters={
            "source_id": file_id,
            "workspace_id": workspace_id,
        },
        thread_id=thread_id,
    )
    return response


@AIRouter.get("/embeddings/summarize")
async def summarize_document(request: Request):
    file_id = request.query_params.get("file_id")
    response = await VectorEmbeddingManager.generate_document_summary(
        source_id=file_id, summary_type="default"
    )
    return response


@AIRouter.post("/generate_mermaid")
async def generate_mermaid_diagram(request: Request):
    file_data = await request.json()
    response = await VectorEmbeddingManager.generate_mermaid_diagram(
        file_data=file_data
    )
    return response
