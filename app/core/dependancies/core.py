"""
This file contains a type alias for a dependency injection that provides a SQLAlchemy
AsyncSession. It's used to inject a database session into a FastAPI endpoint.

The `Annotated` type is used to add a dependency to the type hint. The `Depends` class is
used to indicate that the dependency should be injected by FastAPI. The `get_db_session`
function is a dependency that provides a database session.

So, when you use `DBSessionDep` as a type hint in a FastAPI endpoint, FastAPI will
inject a database session into the endpoint.
"""

from typing import Annotated
from app.database import get_db_session
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

DBSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
