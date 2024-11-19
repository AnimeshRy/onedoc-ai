
from pydantic import BaseModel


class DocumentQuery(BaseModel):
    question: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str


class DocumentAnalysisResponse(BaseModel):
    document_id: str
    query: str
    response: str
