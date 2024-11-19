from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import ai

app = FastAPI(
    title="OneDoc AI",
    description="AI-powered document processing and analysis API",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ai.AIRouter, prefix="/api/v1", tags=["documents"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

