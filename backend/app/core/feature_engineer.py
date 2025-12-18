"""
Feature Engineering for Web Server Logs
Extracts 50+ features for VAE anomaly detection
"""
import polars as pl
import numpy as np
from typing import Tuple, List
import logging
from user_agents import parse
import re

logger = logging.getLogger(__name__)


class LogFeatureEngineer:
    """Extract comprehensive features from web server logs"""

    def __init__(self):
        self.categorical_features = []
        self.numerical_features = []

    def engineer_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Extract all features from log data

        Args:
            df: Cleaned log data

        Returns:
            LazyFrame with engineered features
        """
        logger.info("Engineering features from log data...")

        # Chain all feature engineering steps
        df = (
            df.pipe(self._add_temporal_features)
            .pipe(self._add_url_features)
            .pipe(self._add_status_features)
            .pipe(self._add_performance_features)
            .pipe(self._add_method_features)
            .pipe(self._add_referer_features)
        )

        # User agent parsing (expensive - do after filter if needed)
        # We'll parse user agents in a separate optimized step

        logger.info("Feature engineering completed")
        return df

    def _add_temporal_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add time-based features with cyclical encoding"""
        return df.with_columns(
            [
                # Basic temporal
                pl.col("datetime").dt.hour().alias("hour"),
                pl.col("datetime").dt.day().alias("day"),
                pl.col("datetime").dt.weekday().alias("day_of_week"),
                pl.col("datetime").dt.month().alias("month"),
                pl.col("datetime").dt.quarter().alias("quarter"),
                # Binary flags
                (pl.col("datetime").dt.weekday() >= 5)
                .cast(pl.Int8)
                .alias("is_weekend"),
                (
                    (pl.col("datetime").dt.hour() >= 9)
                    & (pl.col("datetime").dt.hour() <= 17)
                )
                .cast(pl.Int8)
                .alias("is_business_hours"),
                (
                    (pl.col("datetime").dt.hour() >= 22)
                    | (pl.col("datetime").dt.hour() <= 6)
                )
                .cast(pl.Int8)
                .alias("is_night"),
                # Cyclical encoding (for neural networks)
                (2 * np.pi * pl.col("datetime").dt.hour() / 24)
                .sin()
                .alias("hour_sin"),
                (2 * np.pi * pl.col("datetime").dt.hour() / 24)
                .cos()
                .alias("hour_cos"),
                (2 * np.pi * pl.col("datetime").dt.weekday() / 7)
                .sin()
                .alias("day_sin"),
                (2 * np.pi * pl.col("datetime").dt.weekday() / 7)
                .cos()
                .alias("day_cos"),
            ]
        )

    def _add_url_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Extract URL pattern features"""
        return df.with_columns(
            [
                # URL length and complexity
                pl.col("cs_uri").str.len_chars().alias("url_length"),
                pl.col("cs_uri").str.count_matches("/").alias("url_depth"),
                pl.col("cs_uri")
                .str.contains(r"\?")
                .cast(pl.Int8)
                .alias("has_query_params"),
                # Count query parameters
                pl.col("cs_uri")
                .str.extract_all(r"[?&]")
                .list.len()
                .alias("query_param_count"),
                # Extract file extension
                pl.col("cs_uri")
                .str.extract(r"\.([a-zA-Z0-9]+)(?:\?|$)", group_index=1)
                .fill_null("none")
                .alias("file_extension"),
                # File type categories
                pl.col("cs_uri")
                .str.contains(r"\.(css|js|png|jpg|jpeg|gif|svg|ico|woff|ttf)")
                .cast(pl.Int8)
                .alias("is_static_resource"),
                pl.col("cs_uri")
                .str.contains(r"\.(html|htm|php|asp|jsp)")
                .cast(pl.Int8)
                .alias("is_dynamic_page"),
                pl.col("cs_uri")
                .str.contains(r"\.(pdf|doc|docx|xls|xlsx|zip)")
                .cast(pl.Int8)
                .alias("is_download"),
                # Security patterns
                pl.col("cs_uri")
                .str.to_lowercase()
                .str.contains(r"(union|select|insert|delete|drop|script|javascript)")
                .cast(pl.Int8)
                .alias("has_suspicious_pattern"),
                pl.col("cs_uri")
                .str.contains(r"\.\./")
                .cast(pl.Int8)
                .alias("has_directory_traversal"),
                # Special characters count (potential attack indicator)
                pl.col("cs_uri")
                .str.count_matches(r"[<>\"'`]")
                .alias("special_chars_count"),
            ]
        )

    def _add_status_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add HTTP status code features"""
        return df.with_columns(
            [
                # Status categories
                (pl.col("sc_status") // 100)
                .cast(pl.Int8)
                .alias("status_category"),
                (pl.col("sc_status") == 200).cast(pl.Int8).alias("is_success"),
                (pl.col("sc_status").is_in([301, 302, 303, 307, 308]))
                .cast(pl.Int8)
                .alias("is_redirect"),
                (pl.col("sc_status") >= 400).cast(pl.Int8).alias("is_error"),
                (pl.col("sc_status").is_in([400, 401, 403, 404]))
                .cast(pl.Int8)
                .alias("is_client_error"),
                (pl.col("sc_status") >= 500).cast(pl.Int8).alias("is_server_error"),
                # Specific errors
                (pl.col("sc_status") == 404).cast(pl.Int8).alias("is_not_found"),
                (pl.col("sc_status") == 403).cast(pl.Int8).alias("is_forbidden"),
                (pl.col("sc_status").is_in([500, 502, 503, 504]))
                .cast(pl.Int8)
                .alias("is_server_failure"),
            ]
        )

    def _add_performance_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add performance-related features"""
        return df.with_columns(
            [
                # Response time categories
                (pl.col("time_taken") > 1000)
                .cast(pl.Int8)
                .alias("is_slow_response"),
                (pl.col("time_taken") > 5000)
                .cast(pl.Int8)
                .alias("is_very_slow"),
                # Bandwidth
                (pl.col("sc_bytes") / (pl.col("time_taken") + 1))
                .alias("bytes_per_ms"),
                # Log transformations for skewed distributions
                (pl.col("sc_bytes") + 1).log1p().alias("log_bytes"),
                (pl.col("time_taken") + 1).log1p().alias("log_time_taken"),
                # Size categories
                (pl.col("sc_bytes") > 100000)
                .cast(pl.Int8)
                .alias("is_large_response"),
                (pl.col("sc_bytes") == 0).cast(pl.Int8).alias("is_empty_response"),
            ]
        )

    def _add_method_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add HTTP method features"""
        return df.with_columns(
            [
                (pl.col("cs_method") == "GET").cast(pl.Int8).alias("is_get"),
                (pl.col("cs_method") == "POST").cast(pl.Int8).alias("is_post"),
                (pl.col("cs_method") == "HEAD").cast(pl.Int8).alias("is_head"),
                (pl.col("cs_method").is_in(["OPTIONS", "PUT", "DELETE", "PATCH"]))
                .cast(pl.Int8)
                .alias("is_other_method"),
            ]
        )

    def _add_referer_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add referer-based features"""
        return df.with_columns(
            [
                (pl.col("cs_referer") == "direct")
                .cast(pl.Int8)
                .alias("is_direct_traffic"),
                pl.col("cs_referer")
                .str.contains(r"google|bing|yahoo|duckduckgo")
                .cast(pl.Int8)
                .alias("is_search_engine"),
                pl.col("cs_referer")
                .str.contains(r"facebook|twitter|linkedin|instagram")
                .cast(pl.Int8)
                .alias("is_social_media"),
                pl.col("cs_referer")
                .str.contains("isro.gov.in")
                .cast(pl.Int8)
                .alias("is_internal_referer"),
            ]
        )

    def add_ip_aggregated_features(
        self, df: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Add IP-based aggregated features (requires collect for window functions)
        These features capture user behavior patterns
        """
        logger.info("Adding IP aggregated features...")

        # Compute per-IP statistics
        ip_stats = (
            df.group_by("cs_ip")
            .agg(
                [
                    pl.count().alias("ip_request_count"),
                    pl.col("cs_uri").n_unique().alias("ip_unique_urls"),
                    pl.col("is_error").mean().alias("ip_error_rate"),
                    pl.col("time_taken").mean().alias("ip_avg_response_time"),
                    pl.col("sc_bytes").sum().alias("ip_total_bytes"),
                    pl.col("is_post").sum().alias("ip_post_count"),
                    pl.col("has_suspicious_pattern")
                    .sum()
                    .alias("ip_suspicious_count"),
                ]
            )
        )

        # Join back to main dataframe
        df = df.join(ip_stats, on="cs_ip", how="left")

        return df

    def parse_user_agents_batch(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Parse user agents in batch (expensive operation)
        Only call on collected DataFrame

        Args:
            df: Collected DataFrame

        Returns:
            DataFrame with user agent features
        """
        logger.info("Parsing user agents (this may take a moment)...")

        def parse_ua(ua_string: str) -> dict:
            """Parse single user agent"""
            if not ua_string or ua_string == "unknown":
                return {
                    "browser": "unknown",
                    "os": "unknown",
                    "device": "unknown",
                    "is_mobile": 0,
                    "is_bot": 0,
                }
            try:
                ua = parse(ua_string)
                return {
                    "browser": ua.browser.family,
                    "os": ua.os.family,
                    "device": ua.device.family,
                    "is_mobile": int(ua.is_mobile),
                    "is_bot": int(ua.is_bot),
                }
            except Exception:
                return {
                    "browser": "unknown",
                    "os": "unknown",
                    "device": "unknown",
                    "is_mobile": 0,
                    "is_bot": 0,
                }

        # Parse all user agents
        ua_data = df["cs_user_agent"].to_list()
        parsed = [parse_ua(ua) for ua in ua_data]

        # Add to dataframe
        df = df.with_columns(
            [
                pl.Series("browser", [p["browser"] for p in parsed]),
                pl.Series("os", [p["os"] for p in parsed]),
                pl.Series("device", [p["device"] for p in parsed]),
                pl.Series("is_mobile", [p["is_mobile"] for p in parsed]),
                pl.Series("is_bot", [p["is_bot"] for p in parsed]),
            ]
        )

        logger.info("User agent parsing completed")
        return df

    def get_feature_lists(self, df: pl.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Get lists of numerical and categorical features

        Args:
            df: DataFrame with all features

        Returns:
            Tuple of (numerical_features, categorical_features)
        """
        # Exclude metadata columns
        exclude_cols = {
            "datetime",
            "cs_ip",
            "cs_uri",
            "cs_referer",
            "cs_user_agent",
            "cs_method",
        }

        # Get numerical features
        numerical = [
            col
            for col in df.columns
            if col not in exclude_cols
            and df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]
        ]

        # Get categorical features
        categorical = [
            col
            for col in df.columns
            if col not in exclude_cols and df[col].dtype in [pl.Utf8, pl.Categorical]
        ]

        logger.info(
            f"Found {len(numerical)} numerical and {len(categorical)} categorical features"
        )

        return numerical, categorical
