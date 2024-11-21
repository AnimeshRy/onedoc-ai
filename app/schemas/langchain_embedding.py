import json
from uuid import UUID
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from langchain.schema import Document


class DocumentEmbeddingModel(BaseModel):
    cmetadata: Optional[Dict[str, Union[str, int]]] = Field(None)
    document: str
    embedding: Union[List[float], str]
    collection_id: str
    id: str

    @field_validator("embedding", mode="before")
    def convert_embedding(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)  # Parse the string into a list
            except (json.JSONDecodeError, TypeError):
                raise ValueError(
                    "Invalid format for embedding; expected a JSON list of floats."
                )
        return value

    @field_validator("cmetadata", mode="before")
    def parse_cmetadata(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for cmetadata.")
        return value

    @field_validator("collection_id", mode="before")
    def convert_uuid_to_str(cls, value):
        if isinstance(value, UUID):
            return str(value)  # Convert UUID to string
        if not isinstance(value, str):
            raise ValueError("collection_id must be a string or a UUID.")
        return value

    class Config:
        json_schema_extra = {
            "example": {
                "cmetadata": '{"source_id": "123"}',
                "document_id": "doc123",
                "document": "Some text content",
                "embedding": "[0.1, 0.2, 0.3]",
                "collection_id": "77732a9c-b9f5-4743-9feb-df99fd9a2b25",
                "id": "123",
            }
        }


class LangChainDocument(Document):
    cmetadata: Optional[Dict[str, Union[str, int]]]
    page_content: str

    @field_validator("cmetadata", mode="before")
    def parse_cmetadata(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for cmetadata.")
        return value
