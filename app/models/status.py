from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP, ARRAY
from sqlalchemy.sql import func
from enum import Enum

from . import Base


class FileStatusEnum(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FileEmbeddingStatus(Base):
    __tablename__ = "file_embedding_status"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True)
    resource_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # Use String instead of ENUM
    document_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String)
    )  # PostgreSQL array of strings
    root_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # Optional root_id
    meta_info: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True
    )  # JSONB for metadata
    created_at: Mapped[str] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[str] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def to_dict(self):
        return {
            "id": str(self.id),
            "resource_id": self.resource_id,
            "status": self.status,
            "document_ids": self.document_ids,
            "root_id": self.root_id,
            "meta_info": self.meta_info,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"FileEmbeddingStatus("
            f"id={self.id!r}, resource_id={self.resource_id!r}, "
            f"status={self.status!r}, document_ids={self.document_ids!r}, "
            f"root_id={self.root_id!r}, meta_info={self.meta_info!r}, "
            f"created_at={self.created_at!r}, updated_at={self.updated_at!r})"
        )
