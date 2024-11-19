from fastapi import FastAPI
import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from app.config import AppConfig
from app.database import sessionmanager

from app.routes import AIRouter, StatusRouter

config = AppConfig.get_config()


logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG if config.debug_logs else logging.INFO
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Function that handles startup and shutdown events.
    To understand more, read https://fastapi.tiangolo.com/advanced/events/
    """
    yield
    if sessionmanager._engine is not None:
        # Close the DB connection
        await sessionmanager.close()


app = FastAPI(lifespan=lifespan, title="OneDoc AI", docs_url="/api/docs")

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Include routers
app.include_router(AIRouter, prefix="/api/v1/ai", tags=["ai"])
app.include_router(StatusRouter, prefix="/api/v1/status", tags=["embedding_status"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port or 8000, reload=True)
