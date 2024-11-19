from fastapi import HTTPException
from sqlalchemy import select, update
from app.models.status import FileEmbeddingStatus, FileStatusEnum
from app.schemas import (
    FileEmbeddingStatusSchema,
    FileEmbeddingStatusCreateSchema,
)
from app.database import get_db_session


class FileEmbeddingStatusManager:
    @classmethod
    async def fetch_status_from_resource_id(
        cls, resource_id: str
    ) -> FileEmbeddingStatusSchema:
        """Fetch embedding status from the resource_id."""
        query = select(FileEmbeddingStatus).where(
            FileEmbeddingStatus.resource_id == resource_id
        )
        async for session in get_db_session():
            db_session = session
        result = await db_session.execute(query)
        status = result.scalars().first()
        if not status:
            raise HTTPException(status_code=404, detail="Resource Not Found")
        return status.to_dict()

    @classmethod
    async def update_status(cls, resource_id: str, status: str):
        """Update the status and return the updated object."""
        query = (
            update(FileEmbeddingStatus)
            .where(FileEmbeddingStatus.resource_id == resource_id)
            .values(status=status)
            .returning(FileEmbeddingStatus)  # Return the updated object
        )
        async for session in get_db_session():
            db_session = session
        await db_session.execute(query)
        return await db_session.commit()

    @classmethod
    async def create_embedding_status(
        cls, data: FileEmbeddingStatusCreateSchema
    ) -> FileEmbeddingStatusSchema:
        """Create a new embedding status or update the existing one if resource_id already exists."""

        # Check if resource_id already exists
        query = select(FileEmbeddingStatus).where(
            FileEmbeddingStatus.resource_id == data.resource_id
        )
        async for session in get_db_session():
            db_session = session
        result = await db_session.execute(query)
        existing_status = result.scalars().first()

        if existing_status:
            # If resource_id exists, update the record with the new data
            query = (
                update(FileEmbeddingStatus)
                .where(FileEmbeddingStatus.resource_id == data.resource_id)
                .values(
                    status=data.status,
                    document_ids=data.document_ids,
                    root_id=data.root_id,
                    meta_info=data.meta_info,
                )
                .returning(FileEmbeddingStatus)  # Return the updated object
            )
            await db_session.execute(query)
            await db_session.commit()

            # Fetch and return the updated status
            updated_status = await db_session.execute(
                select(FileEmbeddingStatus).where(
                    FileEmbeddingStatus.resource_id == data.resource_id
                )
            )
            updated_status = updated_status.scalars().first()
            return updated_status.to_dict()

        else:
            # If resource_id does not exist, create a new record
            status = FileEmbeddingStatus(**data.model_dump())
            db_session.add(status)
            await db_session.commit()
            await db_session.refresh(status)  # Refresh to get the latest data
            return status.to_dict()

    @classmethod
    async def update_document_ids(cls, resource_id: str, document_ids: list[str]):
        """Update document IDs and mark status as COMPLETED."""
        query = (
            update(FileEmbeddingStatus)
            .where(FileEmbeddingStatus.id == resource_id)
            .values(document_ids=document_ids, status=FileStatusEnum.COMPLETED.value)
            .returning(FileEmbeddingStatus)  # Return the updated object
        )
        async for session in get_db_session():
            db_session = session
        await db_session.execute(query)
        return await db_session.commit()
