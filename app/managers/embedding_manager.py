from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple
import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from fastapi import BackgroundTasks
from app.config import AppConfig
from app.constants import EMBEDDING_PROMPT_TEMPLATES
from app.managers.status_manager import FileEmbeddingStatusManager
from app.schemas.embedding import FileEmbeddingStatusCreateSchema
from app.models import FileStatusEnum

config = AppConfig.get_config()


class VectorEmbeddingManager:
    """
    A comprehensive utility class for managing the chunking, embedding, retrieval, and generation of vector embeddings from documents
    """

    _embeddings_model: Optional[OpenAIEmbeddings] = None
    _vector_store: Optional[PGVector] = None
    _llm: Optional[ChatOpenAI] = None
    EMBEDDING_FILE_COLLCECTION_NAME = "document_embeddings"
    EMBEDDING_LENGTH = 1536

    # Cache for storing retrieved chunks with TTL
    _chunk_cache: Dict[str, Tuple[List[str], datetime]] = {}
    CACHE_TTL_HOURS = 24

    @classmethod
    def _get_embeddings_model(
        cls,
    ) -> OpenAIEmbeddings:
        """
        Create and cache an instance of the OpenAIEmbeddings model.

        Returns:
            An instance of OpenAIEmbeddings initialized with the specified model and API key.

        Raises:
            ValueError: If the OpenAI API key or embeddings model is missing in configuration.
        """
        if cls._embeddings_model is None:
            openai_api_key = config.openai_api_key
            model = config.embeddings_model

            if not openai_api_key or not model:
                raise ValueError(
                    "Missing OpenAI API key or embeddings model in configuration."
                )

            cls._embeddings_model = OpenAIEmbeddings(
                model=model, api_key=openai_api_key
            )
        return cls._embeddings_model

    @classmethod
    def _initialize_vector_store(cls) -> PGVector:
        """Initialize and return the vector store instance."""
        if cls._vector_store is None:
            connection_string = config.langchain_db_connection_string
            embedding_model = cls._get_embeddings_model()
            cls._vector_store = PGVector(
                embeddings=embedding_model,
                embedding_length=cls.EMBEDDING_LENGTH,
                collection_name=cls.EMBEDDING_FILE_COLLCECTION_NAME,
                connection=connection_string,
                use_jsonb=True,
                async_mode=True,
                create_extension=True,
            )
        return cls._vector_store

    @classmethod
    def create_recursive_text_splitter(
        cls, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> RecursiveCharacterTextSplitter:
        """
        Create a text splitter instance.

        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Number of characters to overlap between chunks

        Returns:
            RecursiveCharacterTextSplitter instance
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    @classmethod
    def create_markdown_text_splitter(
        cls, chunk_size: int = 1000, chunk_overlap: int = 200
    ):
        """Create a markdown-specific text splitter."""
        return MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    @classmethod
    def create_markdown_header_text_splitter(cls):
        """Create a markdown header-aware text splitter."""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

    @classmethod
    async def process_document(
        cls,
        file_data: dict,
        chunk_size: int = 1200,
        chunk_overlap: int = 50,
    ) -> List[str]:
        """
        Process a document by splitting it into chunks and storing it in a vector store.

        Args:
            file_data (dict): Dictionary containing file information and markdown content.
            chunk_size (int): Maximum size of text chunks.
            chunk_overlap (int): Number of characters to overlap between chunks.

        Returns:
            List[str]: List of document IDs added to the vector store.
        """

        ## TODO: Enrich Document before consumption
        ## TODO: Add Workspace and File MetaData Info
        text_splitter = cls.create_recursive_text_splitter(chunk_size, chunk_overlap)
        vector_store = cls._initialize_vector_store()
        document_id = file_data.get("id")
        markdown_content = file_data.get("markdown_data") or file_data.get("data")
        document_chunks = text_splitter.split_text(markdown_content)

        # Create Document objects with unique IDs
        documents = [
            Document(page_content=chunk, metadata={"source_id": f"{document_id}_{i}"})
            for i, chunk in enumerate(document_chunks)
        ]
        document_ids = await vector_store.aadd_documents(documents=documents)
        return document_ids

    @classmethod
    async def process_document_and_update_db(
        cls,
        file_data: dict,
        status_id: str,
        chunk_size: int = 1200,
        chunk_overlap: int = 50,
    ) -> None:
        try:
            document_ids = await cls.process_document(
                file_data, chunk_size, chunk_overlap
            )
            await FileEmbeddingStatusManager.update_document_ids(
                status_id, document_ids
            )
        except Exception as _e:
            await FileEmbeddingStatusManager.update_status(
                status_id, FileStatusEnum.FAILED.value
            )
        return None

    @classmethod
    async def embed_document(
        cls, file_data: dict, background_tasks: BackgroundTasks
    ) -> str:
        """
        Handle document embedding creation in the database.

        This method creates the embedding status in the database and initiates
        the embedding process (in the background, if needed).

        Returns:
            FileEmbeddingStatusSchema: The created embedding status object.
        """
        file_id = file_data.get("id")
        root_id = file_data.get("workspace_id")
        meta_info = file_data.get("meta_info", {})

        id = uuid.uuid4()
        data = FileEmbeddingStatusCreateSchema(
            id=id,
            resource_id=file_id,
            status=FileStatusEnum.IN_PROGRESS.value,
            document_ids=[],
            root_id=root_id,
            meta_info=meta_info,
        )
        response = await FileEmbeddingStatusManager.create_embedding_status(data)
        status_id = response["id"]
        background_tasks.add_task(
            cls.process_document_and_update_db, file_data=file_data, status_id=status_id
        )
        return response

    @classmethod
    def _get_llm(cls, model_name: str) -> ChatOpenAI:
        """
        Create and cache an instance of the ChatOpenAI language model.

        Returns:
            An instance of OpenAI LLM initialized with the specified model and API key.

        Raises:
            ValueError: If the OpenAI API key or model is missing in configuration.
        """
        if cls._llm is None:
            openai_api_key = config.openai_api_key

            if not openai_api_key or not model_name:
                raise ValueError(
                    "Missing OpenAI API key or chat model in configuration."
                )

            cls._llm = ChatOpenAI(
                model=model_name, api_key=openai_api_key, temperature=0.7
            )
        return cls._llm

    @classmethod
    def _get_cache_key(cls, query: str, retrieval_type: str) -> str:
        """Generate a unique cache key."""
        return f"{query}_{retrieval_type}"

    @classmethod
    def _is_cache_valid(cls, timestamp: datetime) -> bool:
        """Check if cached data is still valid based on TTL."""
        return datetime.now() - timestamp < timedelta(hours=cls.CACHE_TTL_HOURS)

    @classmethod
    @lru_cache(maxsize=1000)
    def _get_embedding_cache(cls, text: str) -> List[float]:
        """Cache embeddings for frequently accessed text."""
        embedding_model = cls._get_embeddings_model()
        return embedding_model.embed_query(text)

    @classmethod
    async def retrieve_relevant_chunks(
        cls,
        query: str,
        retrieval_type: Literal["similarity", "mmr", "hybrid"] = "similarity",
        num_chunks: int = 4,
        similarity_threshold: float = 0.5,
    ) -> List[str]:
        """
        Enhanced retrieval with multiple strategies and caching.

        Args:
            query: The search query string
            retrieval_type: Type of retrieval strategy to use
            num_chunks: Number of relevant chunks to retrieve
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of relevant document chunks
        """
        cache_key = cls._get_cache_key(query, retrieval_type)

        # Check cache first
        if cache_key in cls._chunk_cache:
            chunks, timestamp = cls._chunk_cache[cache_key]
            if cls._is_cache_valid(timestamp):
                return chunks

        vector_store = cls._initialize_vector_store()

        async def get_similarity_chunks():
            """
            Get chunks using the cosine similarity retrieval strategy.

            The cosine similarity is calculated between the query vector and the document vectors.
            The results are filtered by the similarity threshold, and the top K chunks are returned.

            Args:
                query: The search query string
                num_chunks: Number of relevant chunks to retrieve
                similarity_threshold: Minimum similarity score threshold

            Returns:
                List of relevant document chunks
            """
            results = await vector_store.asimilarity_search_with_relevance_scores(
                query=query, k=num_chunks
            )
            print("result", results)
            return [
                doc.page_content
                for doc, score in results
                if score >= similarity_threshold
            ]

        async def get_mmr_chunks():
            """
            Get chunks using the Maximal Marginal Relevance (MMR) retrieval strategy.

            MMR is a retrieval strategy that uses a combination of similarity and relevance
            scores to rank documents. The similarity score is the cosine similarity between the
            query and the document, and the relevance score is the dot product of the query and
            document vectors.

            Args:
                query: The search query string
                num_chunks: Number of relevant chunks to retrieve

            Returns:
                List of relevant document chunks
            """
            results = await vector_store.amax_marginal_relevance_search(
                query=query, k=num_chunks, fetch_k=num_chunks * 2, lambda_mult=0.7
            )
            return [doc.page_content for doc in results]

        if retrieval_type == "similarity":
            chunks = await get_similarity_chunks()
        elif retrieval_type == "mmr":
            chunks = await get_mmr_chunks()
        else:  # hybrid approach
            # Get chunks using both methods and combine them
            similarity_chunks = await get_similarity_chunks()
            mmr_chunks = await get_mmr_chunks()

            # Combine and deduplicate chunks while preserving order
            seen = set()
            chunks = []
            for chunk in similarity_chunks + mmr_chunks:
                if chunk not in seen:
                    seen.add(chunk)
                    chunks.append(chunk)
                    if len(chunks) >= num_chunks:
                        break

        # Update cache
        cls._chunk_cache[cache_key] = (chunks, datetime.now())
        return chunks

    @classmethod
    async def generate_response(
        cls,
        query: str,
        context_chunks: List[str],
        response_type: Literal[
            "default", "technical", "summary", "analytical"
        ] = "default",
        temperature: float = 0.7,
    ) -> str:
        """
        Enhanced generation with multiple prompt templates and response types.

        Args:
            query: The user's question
            context_chunks: List of relevant document chunks
            response_type: Type of response template to use
            temperature: Temperature for response generation

        Returns:
            Generated response string
        """
        llm = cls._get_llm(config.chat_model)
        llm.temperature = temperature

        # Select appropriate prompt template
        prompt = EMBEDDING_PROMPT_TEMPLATES.get(
            response_type, EMBEDDING_PROMPT_TEMPLATES["default"]
        )

        formatted = prompt.format_messages(
            context="\n\n".join(context_chunks), question=query
        )
        print("PROMPT")
        print(formatted)

        # Create the generation chain
        chain = (
            RunnableMap(
                {
                    "context": RunnablePassthrough(),  # Pass the context unchanged
                    "question": RunnablePassthrough(),  # Pass the question unchanged
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        # Generate response
        inputs = {"context": "\n\n".join(context_chunks), "question": query}
        response = await chain.ainvoke(inputs)
        return response

    @classmethod
    async def query_and_generate(
        cls,
        query: str,
        retrieval_type: Literal["similarity", "mmr", "hybrid"] = "similarity",
        response_type: Literal[
            "default", "technical", "summary", "analytical"
        ] = "default",
        num_chunks: int = 4,
        similarity_threshold: float = 0.7,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Enhanced combined method with multiple retrieval and generation strategies.

        Args:
            query: The user's question
            retrieval_type: Type of retrieval strategy to use
            response_type: Type of response template to use
            num_chunks: Number of relevant chunks to retrieve
            similarity_threshold: Minimum similarity score threshold
            temperature: Temperature for response generation

            Terminal -> temrinal
            Movie ->

        Returns:
            Dictionary containing generated response and metadata
        """
        # Start time for performance tracking
        start_time = datetime.now()

        # Retrieve relevant chunks
        relevant_chunks = await cls.retrieve_relevant_chunks(
            query=query,
            retrieval_type=retrieval_type,
            num_chunks=num_chunks,
            similarity_threshold=similarity_threshold,
        )

        if not relevant_chunks:
            return {
                "response": "I couldn't find any relevant information to answer your question.",
                "chunks_found": 0,
                "retrieval_type": retrieval_type,
                "response_type": response_type,
                "processing_time": (datetime.now() - start_time).total_seconds(),
            }

        # Generate response using retrieved chunks
        response = await cls.generate_response(
            query=query,
            context_chunks=relevant_chunks,
            response_type=response_type,
            temperature=temperature,
        )

        # Return response with metadata
        return {
            "response": response,
            "chunks_found": len(relevant_chunks),
            "retrieval_type": retrieval_type,
            "response_type": response_type,
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "cached": cls._get_cache_key(query, retrieval_type) in cls._chunk_cache,
        }

    @classmethod
    async def clear_expired_cache(cls):
        """Clear expired entries from the chunk cache."""
        expired_keys = [
            key
            for key, (_, timestamp) in cls._chunk_cache.items()
            if not cls._is_cache_valid(timestamp)
        ]
        for key in expired_keys:
            del cls._chunk_cache[key]

    @classmethod
    async def generate_document_summary(
        cls,
        document_id: str,
        query: str,
        response_type: Literal[
            "default", "technical", "summary", "analytical"
        ] = "default",
        num_chunks: int = 4,
        similarity_threshold: float = 0.7,
        temperature: float = 0.7,
    ):
        """
        Generate a summary of a document based on the provided query.

        Args:
            document_id: The ID of the document to summarize
            query: The user's question
            response_type: Type of response template to use
            num_chunks: Number of relevant chunks to retrieve
            similarity_threshold: Minimum similarity score threshold
            temperature: Temperature for response generation

        Returns:
            Generated summary string
        """
        return await cls.query_and_generate(
            query=query,
            retrieval_type="similarity",
            response_type=response_type,
            num_chunks=num_chunks,
            similarity_threshold=similarity_threshold,
            temperature=temperature,
        )
