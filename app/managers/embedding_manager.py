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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from app.constants.prompt_templates import (
    CHAT_EMBEDDING_PROMPT_TEMPLATES,
    CONTEXT_SYSTEM_PROMPT,
    SUMMARY_PROMPT_TEMPLATES,
)
from app.managers.asyncpg_manager import AsyncPGManager
from app.managers.status_manager import FileEmbeddingStatusManager
from app.schemas.embedding import FileEmbeddingStatusCreateSchema
from app.models import FileStatusEnum
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate

config = AppConfig.get_config()


### Statefully manage chat history ###
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


class CustomPGVector(PGVector):
    pass


class VectorEmbeddingManager:
    """
    A comprehensive utility class for managing the chunking, embedding, retrieval, and generation of vector embeddings from documents
    """

    _embeddings_model: Optional[OpenAIEmbeddings] = None
    _llm: Optional[ChatOpenAI] = None
    EMBEDDING_FILE_COLLECTION_NAME = "document_embeddings"
    EMBEDDING_LENGTH = 1536

    # Cache for storing retrieved chunks with TTL
    _chunk_cache: Dict[str, Tuple[List[str], datetime]] = {}
    CACHE_TTL_HOURS = 24

    _session_ids = []

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
    def _initialize_vector_store(cls) -> CustomPGVector:
        """Initialize and return the vector store instance."""
        connection_string = config.langchain_db_connection_string
        embedding_model = cls._get_embeddings_model()
        cls._vector_store = CustomPGVector(
            embeddings=embedding_model,
            embedding_length=cls.EMBEDDING_LENGTH,
            collection_name=cls.EMBEDDING_FILE_COLLECTION_NAME,
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
    async def fetch_meta_data_from_document(cls, file_data: dict) -> Tuple[dict, str]:
        """
        Extract metadata and file ID from file_data.

        Args:
            file_data (dict): Dictionary containing file information.

        Returns:
            Tuple[dict, str]: A tuple containing metadata dictionary and file ID.
        """
        file_id = file_data.get("id", "")

        meta_data = {
            "file_title": file_data.get("title", ""),
            "file_owner_email": file_data.get("file_owner", {}).get("email", ""),
            "file_owner_name": file_data.get("file_owner", {}).get("name", ""),
            "file_status": file_data.get("status", ""),
            "workspace_id": file_data.get("workspace_id", ""),
            "workspace_title": file_data.get("workspace_info", {}).get(
                "workspace_title", ""
            ),
        }
        return meta_data, file_id

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
        # Split the text into chunks
        text_splitter = cls.create_recursive_text_splitter(chunk_size, chunk_overlap)
        markdown_content = file_data.get("markdown_data") or file_data.get("data", "")

        if not markdown_content:
            raise ValueError("No markdown content available in file_data.")

        document_chunks = text_splitter.split_text(markdown_content)
        meta_data, document_id = await cls.fetch_meta_data_from_document(file_data)

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    **meta_data,
                    "chunk_index": i,
                    "source_id": document_id,
                },
            )
            for i, chunk in enumerate(document_chunks)
        ]

        vector_store = cls._initialize_vector_store()
        ids = [f"{document_id}_chunk_{i}" for i in range(len(documents))]
        print("ddd", ids)
        document_ids = await vector_store.aadd_documents(
            documents=documents,
            ids=ids,
        )

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
                model=model_name, api_key=openai_api_key, temperature=0.4
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
        retrieval_type: Literal[
            "similarity", "mmr", "similarity_score_threshold"
        ] = "similarity_score_threshold",
        num_chunks: int = 4,
        similarity_threshold: float = 0.5,
        filters: Dict[str, Any] = {},
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

        if filters.get("source_id"):
            filters = {
                "source_id": filters.get("source_id"),
            }
        elif filters.get("workspace_id"):
            filters = {
                "workspace_id": filters.get("workspace_id"),
            }

        retriever = vector_store.as_retriever(
            search_type=retrieval_type,
            search_kwargs={
                "score_threshold": similarity_threshold,
                "k": num_chunks,
                "filter": filters,
            },
        )
        return await retriever.ainvoke(query)

    @classmethod
    async def init_chat(
        cls,
        query: str,
        retrieval_type: Literal[
            "similarity", "mmr", "similarity_score_threshold"
        ] = "similarity_score_threshold",
        response_type: Literal[
            "default", "technical", "summary", "analytical"
        ] = "default",
        num_chunks: int = 4,
        similarity_threshold: float = 0.5,
        temperature: float = 0.6,
        filters: Dict[str, Any] = {},
        thread_id: str = None,
    ):
        ## TODO: Figure out a way to store thread_id, work with langgraph persistance
        ## TODO: Stream response

        llm = cls._get_llm(config.chat_model)
        llm.temperature = temperature

        vector_store = cls._initialize_vector_store()

        if filters.get("source_id"):
            filters = {
                "source_id": filters.get("source_id"),
            }
        elif filters.get("workspace_id"):
            filters = {
                "workspace_id": filters.get("workspace_id"),
            }

        retriever = vector_store.as_retriever(
            search_type=retrieval_type,
            search_kwargs={
                "score_threshold": similarity_threshold,
                "k": num_chunks,
                "filter": filters,
            },
        )

        prompt = CHAT_EMBEDDING_PROMPT_TEMPLATES.get(
            response_type, CHAT_EMBEDDING_PROMPT_TEMPLATES["default"]
        )

        contextualize_q_prompt = CONTEXT_SYSTEM_PROMPT

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        if thread_id is None or thread_id not in cls._session_ids:
            thread_id = str(uuid.uuid4())
            cls._session_ids.append(thread_id)

        async def call_model(state: State):
            response = await rag_chain.ainvoke(state)
            return {
                "chat_history": [
                    HumanMessage(state["input"]),
                    AIMessage(response["answer"]),
                ],
                "context": response["context"],
                "answer": response["answer"],
            }

        workflow = StateGraph(state_schema=State)
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)

        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)

        pointer_config = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke(
            {"input": query},
            config=pointer_config,
        )

        return {**result, "thread_id": thread_id}

    @classmethod
    async def generate_response(
        cls,
        query: str,
        context_chunks: List[Document],
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
        prompt = EMBEDDING_PROMPT_TEMPLATES.get(
            response_type, EMBEDDING_PROMPT_TEMPLATES["default"]
        )
        context = "\n\n".join(chunk.page_content for chunk in context_chunks)

        chain = (
            RunnableMap(
                {
                    "context": lambda _: context,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        chat_response = await chain.ainvoke({"question": query})
        return chat_response

    @classmethod
    async def query_and_generate(
        cls,
        query: str,
        retrieval_type: Literal[
            "similarity", "mmr", "similarity_score_threshold"
        ] = "similarity_score_threshold",
        response_type: Literal[
            "default", "technical", "summary", "analytical"
        ] = "default",
        num_chunks: int = 4,
        similarity_threshold: float = 0.5,
        temperature: float = 0.6,
        filters: Dict[str, Any] = {},
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
            filters: Additional filters to apply

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
            filters=filters,
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
            "chunks": relevant_chunks,
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
        source_id: str,
        summary_type: Literal[
            "default", "executive", "detailed", "comparative"
        ] = "default",
    ):
        """
        This uses MapReduce
        """
        document_data = await AsyncPGManager.get_embeddings_by_source_id_to_document(
            source_id=source_id
        )
        if len(document_data):
            return {
                "documents": document_data,
                "file_id": source_id,
                "error": "No Embedding Found",
            }

        llm = cls._get_llm(config.chat_model)
        map_template = SUMMARY_PROMPT_TEMPLATES.get(summary_type).get("map_prompt", {})
        reduce_template = SUMMARY_PROMPT_TEMPLATES.get(summary_type).get(
            "reduce_prompt", {}
        )
        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            verbose=True,
            map_prompt=PromptTemplate(template=map_template, input_variables=["text"]),
            combine_prompt=PromptTemplate(
                template=reduce_template, input_variables=["text"]
            ),
        )
        response = await chain.arun(document_data)
        return {"documents": document_data, "file_id": source_id, "summary": response}
