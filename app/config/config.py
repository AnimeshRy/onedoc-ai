from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    openai_api_key: None | SecretStr = None
    db_connection_string: str = ""
    langchain_db_connection_string: str = ""
    embeddings_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-3.5"
    priority_chat_model: str = "gpt-4o"
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: SecretStr | None = None
    langchain_project: str = "OneDoc-Dev"
    debug_logs: bool = False

    # Specifiy Env
    model_config = SettingsConfigDict(env_file=".env")

    @lru_cache
    def get_config():
        """
        Get the configuration for the application.

        The configuration is loaded from the environment and defaults.
        The configuration is cached, so it's only loaded once.

        Returns:
            AppConfig: The configuration for the application.
        """
        return AppConfig()
