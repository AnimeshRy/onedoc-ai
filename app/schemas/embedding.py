from uuid import UUID
from pydantic import BaseModel
from typing import List, Optional
from app.models.status import FileStatusEnum


class FileEmbeddingStatusSchema(BaseModel):
    id: str
    resource_id: str
    status: FileStatusEnum
    document_ids: List[str]
    root_id: str | None
    meta_info: dict | None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class FileEmbeddingStatusCreateSchema(BaseModel):
    id: UUID
    resource_id: str
    status: FileStatusEnum
    document_ids: Optional[List[str]] = None
    root_id: Optional[str] = None
    meta_info: Optional[dict] = None


class FileEmbeddingStatusUpdateSchema(BaseModel):
    resource_id: Optional[str] = None
    status: Optional[FileStatusEnum] = None
    document_ids: Optional[List[str]] = None
    root_id: Optional[str] = None
    meta_info: Optional[dict] = None
