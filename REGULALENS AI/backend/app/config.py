from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "RegulaLens AI"
    database_url: str = "sqlite:///./regulalens.db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "regulatory_chunks"
    embedding_dimensions: int = 384
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    reranker_model: str = ""
    retrieval_default: str = "hybrid_rerank"
    risk_model: str = "selected"
    clause_model: str = "tfidf"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    local_slm_url: str = ""
    local_slm_model: str = ""
    auth_username: str = ""
    auth_password: str = ""
    auth_password_hash: str = ""
    auth_role: str = "admin"
    auth_secret: str = ""
    upload_dir: str = "./data/uploads"
    cors_origins: str = "http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174"
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

settings = Settings()
