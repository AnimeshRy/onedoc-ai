import asyncpg
from app.config import AppConfig
from app.schemas.langchain_embedding import DocumentEmbeddingModel, LangChainDocument

config = AppConfig.get_config()


class AsyncPGManager:
    @classmethod
    async def get_embeddings_by_source_id(
        cls, source_id
    ) -> list[DocumentEmbeddingModel]:
        formatted_connection_string = config.db_connection_string.split("+asyncpg")
        connection_string = (
            formatted_connection_string[0] + formatted_connection_string[1]
        )
        conn = await asyncpg.connect(connection_string)
        try:
            rows = await conn.fetch(
                """
                SELECT *
                FROM langchain_pg_embedding
                WHERE cmetadata->>'source_id' = $1
            """,
                source_id,
            )
            data = []
            for row in rows:
                data.append(DocumentEmbeddingModel(**row))
        finally:
            await conn.close()
        return data

    @classmethod
    async def get_embeddings_by_source_id_to_document(
        cls, source_id
    ) -> list[LangChainDocument]:
        formatted_connection_string = config.db_connection_string.split("+asyncpg")
        connection_string = (
            formatted_connection_string[0] + formatted_connection_string[1]
        )
        conn = await asyncpg.connect(connection_string)
        try:
            rows = await conn.fetch(
                """
                SELECT *
                FROM langchain_pg_embedding
                WHERE cmetadata->>'source_id' = $1
            """,
                source_id,
            )
            data = []
            for row in rows:
                d = {"page_content": row["document"], "cmetadata": row["cmetadata"]}
                data.append(LangChainDocument(**d))
        finally:
            await conn.close()
        return data
