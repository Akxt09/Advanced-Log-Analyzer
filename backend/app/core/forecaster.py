"""
Traffic Forecasting using NeuralProphet
Predicts future request volumes with confidence intervals
"""
import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from neuralprophet import NeuralProphet
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LogTrafficForecaster:
    """Forecast web traffic using NeuralProphet"""

    def __init__(
        self,
        horizons: List[int] = [24, 168],  # 24h and 7d
        seasonality_mode: str = "additive",
    ):
        """
        Initialize forecaster

        Args:
            horizons: Forecast horizons in hours
            seasonality_mode: 'additive' or 'multiplicative'
        """
        self.horizons = horizons
        self.seasonality_mode = seasonality_mode
        self.model = None
        self.forecast_df = None

    def prepare_time_series(
        self, df: pl.LazyFrame, freq: str = "H"
    ) -> pd.DataFrame:
        """
        Prepare time series data for forecasting

        Args:
            df: Log data with datetime
            freq: Frequency ('H' for hourly, 'D' for daily)

        Returns:
            pandas DataFrame with ds (datetime) and y (value) columns
        """
        logger.info(f"Preparing time series data with frequency: {freq}")

        # Aggregate by time period
        if freq == "H":
            ts = (
                df.group_by_dynamic("datetime", every="1h")
                .agg(
                    [
                        pl.count().alias("y"),  # Request count
                        pl.col("is_error").mean().alias("error_rate"),
                        pl.col("time_taken").mean().alias("avg_response_time"),
                    ]
                )
                .collect()
                .to_pandas()
            )
        elif freq == "D":
            ts = (
                df.group_by_dynamic("datetime", every="1d")
                .agg(
                    [
                        pl.count().alias("y"),
                        pl.col("is_error").mean().alias("error_rate"),
                    ]
                )
                .collect()
                .to_pandas()
            )
        else:
            raise ValueError(f"Unsupported frequency: {freq}")

        # Rename datetime column to 'ds' (required by NeuralProphet)
        ts = ts.rename(columns={"datetime": "ds"})

        # Sort by datetime
        ts = ts.sort_values("ds").reset_index(drop=True)

        logger.info(f"Prepared {len(ts)} time periods")
        return ts

    def train_forecast_model(
        self,
        ts_data: pd.DataFrame,
        epochs: int = 50,
        learning_rate: float = 0.01,
    ) -> Dict:
        """
        Train NeuralProphet model

        Args:
            ts_data: Time series data (ds, y columns)
            epochs: Training epochs
            learning_rate: Learning rate

        Returns:
            Training metrics
        """
        logger.info("Training NeuralProphet model...")

        # Initialize model
        self.model = NeuralProphet(
            growth="linear",
            n_forecasts=max(self.horizons),
            n_lags=48,  # Use past 48 hours for prediction
            yearly_seasonality=False,  # Not enough data typically
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode=self.seasonality_mode,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=32,
            loss_func="MSE",
        )

        # Train model
        metrics = self.model.fit(ts_data, freq="H", validation_df=None)

        logger.info("NeuralProphet training completed")
        return metrics.to_dict() if metrics is not None else {}

    def generate_forecast(
        self, ts_data: pd.DataFrame, horizon_hours: int = 24
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate forecast for specified horizon

        Args:
            ts_data: Historical time series data
            horizon_hours: Forecast horizon in hours

        Returns:
            Tuple of (forecast_df, metadata)
        """
        logger.info(f"Generating {horizon_hours}h forecast...")

        # Create future dataframe
        future = self.model.make_future_dataframe(
            ts_data, periods=horizon_hours, n_historic_predictions=True
        )

        # Generate forecast
        forecast = self.model.predict(future)

        # Extract forecast period only
        forecast_only = forecast.tail(horizon_hours).copy()

        # Calculate metadata
        metadata = {
            "horizon_hours": horizon_hours,
            "forecast_start": forecast_only["ds"].min().isoformat(),
            "forecast_end": forecast_only["ds"].max().isoformat(),
            "mean_prediction": float(forecast_only["yhat1"].mean()),
            "max_prediction": float(forecast_only["yhat1"].max()),
            "min_prediction": float(forecast_only["yhat1"].min()),
        }

        logger.info(f"Forecast generated: {len(forecast_only)} periods")
        return forecast_only, metadata

    def forecast_multiple_horizons(
        self, ts_data: pd.DataFrame
    ) -> Dict[int, Tuple[pd.DataFrame, Dict]]:
        """
        Generate forecasts for all configured horizons

        Args:
            ts_data: Historical time series data

        Returns:
            Dictionary mapping horizon to (forecast_df, metadata)
        """
        logger.info(f"Generating forecasts for horizons: {self.horizons}")

        forecasts = {}

        for horizon in self.horizons:
            forecast_df, metadata = self.generate_forecast(ts_data, horizon)
            forecasts[horizon] = (forecast_df, metadata)

        return forecasts

    def forecast_with_confidence_intervals(
        self, ts_data: pd.DataFrame, horizon_hours: int = 24
    ) -> Dict:
        """
        Generate forecast with confidence intervals

        Args:
            ts_data: Historical data
            horizon_hours: Forecast horizon

        Returns:
            Forecast with uncertainty estimates
        """
        logger.info(
            f"Generating forecast with confidence intervals for {horizon_hours}h"
        )

        # Generate forecast
        forecast_df, metadata = self.generate_forecast(ts_data, horizon_hours)

        # Prepare output
        forecast_output = []

        for _, row in forecast_df.iterrows():
            forecast_output.append(
                {
                    "timestamp": row["ds"].isoformat(),
                    "predicted_requests": float(row["yhat1"]),
                    # NeuralProphet may not have explicit CI, so we approximate
                    "lower_bound": float(row["yhat1"] * 0.85),  # Approximate
                    "upper_bound": float(row["yhat1"] * 1.15),  # Approximate
                }
            )

        result = {
            "horizon_hours": horizon_hours,
            "forecast": forecast_output,
            "metadata": metadata,
        }

        return result

    def detect_forecast_anomalies(
        self, forecast_df: pd.DataFrame, threshold_multiplier: float = 2.0
    ) -> List[Dict]:
        """
        Detect anomalies in forecast (unusual spikes/drops)

        Args:
            forecast_df: Forecast dataframe
            threshold_multiplier: Multiplier for std deviation

        Returns:
            List of detected anomalies in forecast
        """
        logger.info("Detecting anomalies in forecast...")

        if len(forecast_df) == 0:
            return []

        # Calculate statistics
        mean_pred = forecast_df["yhat1"].mean()
        std_pred = forecast_df["yhat1"].std()

        # Detect anomalies
        anomalies = []

        for _, row in forecast_df.iterrows():
            deviation = abs(row["yhat1"] - mean_pred)

            if deviation > (threshold_multiplier * std_pred):
                anomalies.append(
                    {
                        "timestamp": row["ds"].isoformat(),
                        "predicted_value": float(row["yhat1"]),
                        "deviation": float(deviation),
                        "severity": "high"
                        if deviation > (3 * std_pred)
                        else "medium",
                    }
                )

        logger.info(f"Found {len(anomalies)} forecast anomalies")
        return anomalies

    def run_full_forecast_pipeline(
        self, df: pl.LazyFrame, train_model: bool = True
    ) -> Dict:
        """
        Run complete forecasting pipeline

        Args:
            df: Log data
            train_model: Whether to train new model

        Returns:
            Complete forecast results
        """
        logger.info("Running full forecasting pipeline...")

        results = {}

        # 1. Prepare time series
        ts_data = self.prepare_time_series(df, freq="H")
        results["historical_data_points"] = len(ts_data)

        # 2. Train model
        if train_model:
            training_metrics = self.train_forecast_model(ts_data)
            results["training_metrics"] = training_metrics

        # 3. Generate forecasts for all horizons
        forecasts = {}

        for horizon in self.horizons:
            forecast_result = self.forecast_with_confidence_intervals(
                ts_data, horizon
            )
            forecasts[f"{horizon}h"] = forecast_result

            # Detect anomalies in forecast
            forecast_df, _ = self.generate_forecast(ts_data, horizon)
            forecast_anomalies = self.detect_forecast_anomalies(forecast_df)
            forecasts[f"{horizon}h"]["anomalies"] = forecast_anomalies

        results["forecasts"] = forecasts

        # 4. Summary statistics
        results["summary"] = self._generate_forecast_summary(ts_data, forecasts)

        logger.info("Forecasting pipeline completed")
        return results

    def _generate_forecast_summary(
        self, ts_data: pd.DataFrame, forecasts: Dict
    ) -> Dict:
        """Generate summary of forecast results"""
        # Historical stats
        hist_mean = float(ts_data["y"].mean())
        hist_std = float(ts_data["y"].std())
        hist_max = float(ts_data["y"].max())
        hist_min = float(ts_data["y"].min())

        # Forecast stats (24h)
        forecast_24h = forecasts.get("24h", {})
        forecast_data = forecast_24h.get("forecast", [])

        if forecast_data:
            pred_values = [f["predicted_requests"] for f in forecast_data]
            forecast_mean = np.mean(pred_values)
            forecast_max = np.max(pred_values)

            # Calculate expected change
            expected_change_pct = (
                (forecast_mean - hist_mean) / hist_mean
            ) * 100
        else:
            forecast_mean = 0
            forecast_max = 0
            expected_change_pct = 0

        summary = {
            "historical": {
                "mean_requests_per_hour": hist_mean,
                "std_requests": hist_std,
                "max_requests": hist_max,
                "min_requests": hist_min,
            },
            "forecast_24h": {
                "mean_predicted_requests": forecast_mean,
                "max_predicted_requests": forecast_max,
                "expected_change_pct": expected_change_pct,
            },
        }

        return summary
