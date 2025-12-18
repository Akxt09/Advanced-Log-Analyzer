"""
API routes for log analysis
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import polars as pl
import numpy as np
from pathlib import Path
import time
import logging
from typing import Optional

from ..core.data_processor import LogDataProcessor
from ..core.feature_engineer import LogFeatureEngineer
from ..core.anomaly_detector import LogAnomalyDetectionPipeline
from ..core.explainer import AnomalyExplainer
from ..core.forecaster import LogTrafficForecaster
from ..config import settings
from .schemas import (
    AnalysisConfig,
    AnalysisResults,
    BasicStats,
    HealthCheck,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== Helper Functions ====================
async def save_upload_file(upload_file: UploadFile) -> Path:
    """Save uploaded file to disk"""
    # Generate unique filename
    timestamp = int(time.time())
    filename = f"{timestamp}_{upload_file.filename}"
    file_path = settings.UPLOAD_DIR / filename

    # Save file
    contents = await upload_file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    logger.info(f"File saved: {file_path}")
    return file_path


def cleanup_file(file_path: Path):
    """Delete temporary file"""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup {file_path}: {e}")


# ==================== API Endpoints ====================
@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    from datetime import datetime

    return HealthCheck(
        status="healthy", timestamp=datetime.now().isoformat(), version=settings.VERSION
    )


@router.post("/analyze", response_model=AnalysisResults)
async def analyze_log_file(
    file: UploadFile = File(...),
    config: Optional[str] = None,  # JSON string of AnalysisConfig
):
    """
    Main endpoint for log file analysis

    Args:
        file: Log file (tab-separated)
        config: Analysis configuration (JSON)

    Returns:
        Complete analysis results
    """
    start_time = time.time()
    file_path = None

    try:
        # Parse config
        if config:
            import json

            config_dict = json.loads(config)
            analysis_config = AnalysisConfig(**config_dict)
        else:
            analysis_config = AnalysisConfig()

        logger.info(f"Starting analysis for file: {file.filename}")

        # Save uploaded file
        file_path = await save_upload_file(file)

        # ==================== Phase 1: Data Loading ====================
        logger.info("Phase 1: Loading and cleaning data...")
        processor = LogDataProcessor()

        df_lazy = processor.load_log_file(file_path, lazy=True)
        df_lazy = processor.clean_and_parse(df_lazy)

        # Compute basic stats
        basic_stats_dict = processor.compute_basic_stats(df_lazy)

        # Collect for feature engineering
        df = df_lazy.collect()

        logger.info(f"Loaded {len(df)} log entries")

        # Check minimum sample size
        if len(df) < settings.MIN_SAMPLES_FOR_TRAINING:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data. Need at least {settings.MIN_SAMPLES_FOR_TRAINING} samples, got {len(df)}",
            )

        # ==================== Phase 2: Feature Engineering ====================
        logger.info("Phase 2: Engineering features...")
        engineer = LogFeatureEngineer()

        # Convert to lazy for feature engineering
        df_lazy = df.lazy()
        df_lazy = engineer.engineer_features(df_lazy)

        # Collect and add IP aggregated features
        df_features = df_lazy.collect()
        df_features_lazy = df_features.lazy()
        df_features_lazy = engineer.add_ip_aggregated_features(
            df_features_lazy
        )
        df_features = df_features_lazy.collect()

        # Parse user agents (expensive)
        df_features = engineer.parse_user_agents_batch(df_features)

        # Get feature lists
        numerical_features, categorical_features = engineer.get_feature_lists(
            df_features
        )

        logger.info(
            f"Engineered {len(numerical_features)} numerical + {len(categorical_features)} categorical features"
        )

        # Encode categoricals (simple one-hot for now)
        if categorical_features:
            df_encoded = df_features.to_dummies(columns=categorical_features)
        else:
            df_encoded = df_features

        # Extract feature matrix
        feature_cols = [
            col
            for col in df_encoded.columns
            if col
            not in [
                "datetime",
                "cs_ip",
                "cs_uri",
                "cs_referer",
                "cs_user_agent",
                "cs_method",
            ]
        ]

        X = df_encoded.select(feature_cols).to_numpy()
        feature_names = feature_cols

        logger.info(f"Feature matrix shape: {X.shape}")

        # Initialize results
        results = {
            "success": True,
            "message": "Analysis completed successfully",
            "basic_stats": basic_stats_dict,
        }

        # ==================== Phase 3: Anomaly Detection ====================
        if analysis_config.perform_anomaly_detection:
            logger.info("Phase 3: Anomaly detection with VAE...")

            detector = LogAnomalyDetectionPipeline(
                anomaly_threshold_pct=analysis_config.anomaly_threshold_pct
            )

            detection_results = detector.run_full_pipeline(
                X=X,
                df=df_features,
                feature_names=feature_names,
                train_vae=True,
                perform_clustering=analysis_config.perform_clustering,
            )

            # Explainability
            logger.info("Generating explanations...")
            explainer = AnomalyExplainer(
                top_n_shap=analysis_config.top_n_shap
            )

            explanations = explainer.create_hybrid_explanations(
                vae_model=detector.vae,
                X=X,
                feature_names=feature_names,
                anomaly_scores=detection_results["anomaly_scores"],
                is_anomaly=detection_results["is_anomaly"],
                feature_reconstruction_errors=detection_results[
                    "feature_reconstruction_errors"
                ],
            )

            # Root cause analysis
            root_cause = explainer.generate_root_cause_analysis(
                explanations, feature_names
            )

            # Add to results
            results["anomaly_detection"] = {
                "metadata": detection_results["anomaly_metadata"],
                "num_anomalies": int(detection_results["is_anomaly"].sum()),
                "top_anomalies": self._get_top_anomalies(
                    df_features,
                    detection_results["anomaly_scores"],
                    detection_results["is_anomaly"],
                    explanations,
                    top_n=50,
                ),
            }

            if analysis_config.perform_clustering:
                results["clustering"] = {
                    "metadata": detection_results["cluster_metadata"],
                    "cluster_analysis": detection_results["cluster_analysis"],
                }

            results["explanations"] = {
                "count": len(explanations),
                "sample": list(explanations.values())[:10],  # Sample only
            }

            results["root_cause"] = root_cause

        # ==================== Phase 4: Forecasting ====================
        if analysis_config.perform_forecasting:
            logger.info("Phase 4: Traffic forecasting with NeuralProphet...")

            forecaster = LogTrafficForecaster(
                horizons=analysis_config.forecast_horizons
            )

            forecast_results = forecaster.run_full_forecast_pipeline(
                df_features.lazy(), train_model=True
            )

            results["forecasting"] = forecast_results

        # Calculate processing time
        processing_time = time.time() - start_time
        results["processing_time_seconds"] = processing_time

        logger.info(f"Analysis completed in {processing_time:.2f}s")

        # Cleanup
        if file_path:
            cleanup_file(file_path)

        return AnalysisResults(**results)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)

        # Cleanup
        if file_path:
            cleanup_file(file_path)

        raise HTTPException(status_code=500, detail=str(e))


def _get_top_anomalies(
    df: pl.DataFrame,
    anomaly_scores: np.ndarray,
    is_anomaly: np.ndarray,
    explanations: dict,
    top_n: int = 50,
) -> list:
    """Extract top N anomalies with details"""
    anomaly_indices = np.where(is_anomaly)[0]

    if len(anomaly_indices) == 0:
        return []

    # Get top N by score
    top_indices = anomaly_indices[
        np.argsort(anomaly_scores[anomaly_indices])[-top_n:][::-1]
    ]

    top_anomalies = []

    for idx in top_indices:
        row = df[int(idx)]

        anomaly_data = {
            "index": int(idx),
            "anomaly_score": float(anomaly_scores[idx]),
            "timestamp": str(row["datetime"][0])
            if "datetime" in row.columns
            else None,
            "ip": str(row["cs_ip"][0]) if "cs_ip" in row.columns else None,
            "url": str(row["cs_uri"][0]) if "cs_uri" in row.columns else None,
            "status": int(row["sc_status"][0])
            if "sc_status" in row.columns
            else None,
            "explanation": explanations.get(int(idx), {}),
        }

        top_anomalies.append(anomaly_data)

    return top_anomalies


@router.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    return {
        "upload_dir": str(settings.UPLOAD_DIR),
        "models_dir": str(settings.MODELS_DIR),
        "version": settings.VERSION,
    }
