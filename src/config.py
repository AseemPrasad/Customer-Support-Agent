from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    LLM_PROVIDER: str = "local"
    LOCAL_BASE_URL: str = "http://localhost:11434/v1"
    LOCAL_MODEL_NAME: str = "llama3.1:8b"
    REMOTE_BASE_URL: str = "https://api.openai.com/v1"
    REMOTE_API_KEY: str = ""
    REMOTE_MODEL_NAME: str = "gpt-4o-mini"


settings = Settings()
