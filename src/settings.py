from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):
    BASE_API: str
    MODEL: str
    API_KEY: str
    
    model_config = SettingsConfigDict(
        env_prefix="VLLM_LLM_", 
        env_file=".env",
        extra='ignore'
    )

class EmbeddingsSettings(BaseSettings):
    BASE_API: str
    MODEL: str
    
    model_config = SettingsConfigDict(
        env_prefix="VLLM_EMB_",
        env_file=".env",
        extra='ignore'
    )

class LangfuseSettings(BaseSettings):
    PUBLIC_KEY: str
    SECRET_KEY: str
    HOST: str
    
    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_",
        env_file=".env",
        extra='ignore'
    )

class QdrantSettings(BaseSettings):
    URL: str
    COLLECTION_NAME: str
    TOP_N: int
    
    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file=".env",
        extra='ignore'
    )

class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    langfuse: LangfuseSettings = LangfuseSettings()
    qdrant: QdrantSettings = QdrantSettings()

settings = Settings()
