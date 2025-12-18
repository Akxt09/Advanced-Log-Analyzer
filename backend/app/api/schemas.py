"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


# ==================== Request Schemas ====================
class AnalysisConfig(BaseModel):
    """Configuration for log analysis"""

    perform_anomaly_detection: bool = True
    perform_forecasting: bool = True
    perform_clustering: bool = True
    anomaly_threshold_pct: float = Field(default=0.02, ge=0.001, le=0.1)
    forecast_horizons: List[int] = [24, 168]
    top_n_shap: int = Field(default=100, ge=10, le=500)


# ==================== Response Schemas ====================
class BasicStats(BaseModel):
    """Basic log statistics"""

    total_requests: int
    unique_ips: int
    start_date: Optional[str]
    end_date: Optional[str]
    duration_hours: float
    error_rate: float
    total_bytes: int
    avg_response_time: float


class AnomalyMetadata(BaseModel):
    """Anomaly detection metadata"""

    total_samples: int
    num_anomalies: int
    anomaly_rate: float
    threshold: float
    mean_score: float
    std_score: float


class ClusterMetadata(BaseModel):
    """Clustering metadata"""

    n_clusters: int
    n_noise_points: int
    noise_rate: float
    cluster_sizes: Dict[int, int]


class FeatureExplanation(BaseModel):
    """Explanation for a single feature"""

    feature: str
    zscore: Optional[float] = None
    value: Optional[float] = None
    mean: Optional[float] = None
    deviation_pct: Optional[float] = None
    reconstruction_error: Optional[float] = None
    shap_value: Optional[float] = None
    importance: Optional[float] = None


class AnomalyExplanation(BaseModel):
    """Complete explanation for an anomaly"""

    index: int
    anomaly_score: float
    has_shap: bool
    zscore_summary: Optional[str] = None
    reconstruction_summary: Optional[str] = None
    shap_summary: Optional[str] = None
    top_features: List[FeatureExplanation]


class ForecastPoint(BaseModel):
    """Single forecast data point"""

    timestamp: str
    predicted_requests: float
    lower_bound: float
    upper_bound: float


class ForecastResult(BaseModel):
    """Forecast results for a specific horizon"""

    horizon_hours: int
    forecast: List[ForecastPoint]
    metadata: Dict[str, Any]
    anomalies: List[Dict[str, Any]] = []


class RootCauseAnalysis(BaseModel):
    """Root cause analysis results"""

    total_anomalies: int
    top_contributing_features: List[Dict[str, Any]]
    summary: str


class AnalysisResults(BaseModel):
    """Complete analysis results"""

    success: bool
    message: str
    basic_stats: BasicStats
    anomaly_detection: Optional[Dict[str, Any]] = None
    clustering: Optional[Dict[str, Any]] = None
    forecasting: Optional[Dict[str, Any]] = None
    explanations: Optional[Dict[str, Any]] = None
    root_cause: Optional[RootCauseAnalysis] = None
    processing_time_seconds: float


class HealthCheck(BaseModel):
    """Health check response"""

    status: str
    timestamp: str
    version: str


class ErrorResponse(BaseModel):
    """Error response"""

    success: bool = False
    error: str
    detail: Optional[str] = None
