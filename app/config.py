from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_PATH: str = "ml/saved_model/model.joblib"
    MODEL_METADATA_PATH: str = "ml/saved_model/model_metadata.json"
    MODEL_VERSION: str = "1.0"
    LOG_LEVEL: str = "INFO"
    MAX_BATCH_SIZE: int = 100
    API_TITLE: str = "Iris ML API"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()