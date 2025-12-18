"""
Configuration settings for Advanced Log Analyzer
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    """Application settings"""

    # App Info
    APP_NAME: str = "Advanced Log Analyzer"
    VERSION: str = "2.0.0"
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MODELS_DIR: Path = BASE_DIR.parent / "models"
    DATA_DIR: Path = BASE_DIR.parent / "data"

    # File Upload
    MAX_UPLOAD_SIZE: int = 2 * 1024 * 1024 * 1024  # 2GB
    ALLOWED_EXTENSIONS: List[str] = [".log", ".txt", ".csv", ".tsv"]

    # VAE Model Configuration
    VAE_LATENT_DIM: int = 16
    VAE_HIDDEN_DIM: int = 64
    VAE_BATCH_SIZE: int = 512
    VAE_EPOCHS: int = 20
    VAE_LEARNING_RATE: float = 1e-3
    VAE_PATIENCE: int = 5

    # Anomaly Detection
    ANOMALY_THRESHOLD_PCT: float = 0.02  # Top 2% as anomalies
    MIN_SAMPLES_FOR_TRAINING: int = 1000

    # HDBSCAN Clustering
    HDBSCAN_MIN_CLUSTER_SIZE: int = 50
    HDBSCAN_MIN_SAMPLES: int = 20

    # Explainability
    TOP_N_SHAP_EXPLANATIONS: int = 100
    N_FEATURES_TO_EXPLAIN: int = 5

    # Forecasting
    FORECAST_HORIZONS: List[int] = [24, 168]  # 24h and 7d (168h)
    FORECAST_SEASONALITY_MODES: List[str] = ["additive"]

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Initialize settings
settings = Settings()

# Create directories if they don't exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
