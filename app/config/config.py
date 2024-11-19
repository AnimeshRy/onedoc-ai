from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    openai_api_key: None | SecretStr = None
    db_connection_string: str = ""
    embeddings_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-3.5"
    priority_chat_model: str = "gpt-4o"

    model_config = SettingsConfigDict(env_file=".env")

    @lru_cache
    def get_config():
        return AppConfig()
