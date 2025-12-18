"""
Polars-based data processor for web server logs
Handles 474K+ rows efficiently with lazy evaluation
"""
import polars as pl
from pathlib import Path
from typing import Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LogDataProcessor:
    """High-performance log processor using Polars"""

    def __init__(self):
        self.column_names = [
            "date",
            "time",
            "cs_ip",
            "cs_method",
            "cs_uri",
            "sc_status",
            "sc_bytes",
            "time_taken",
            "cs_referer",
            "cs_user_agent",
        ]

    def load_log_file(
        self, file_path: Path, lazy: bool = True
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Load log file with Polars (10x faster than pandas)

        Args:
            file_path: Path to log file
            lazy: Use lazy evaluation for better performance

        Returns:
            Polars LazyFrame or DataFrame
        """
        logger.info(f"Loading log file: {file_path}")

        try:
            # Read with lazy evaluation for better performance
            if lazy:
                df = pl.scan_csv(
                    file_path,
                    separator="\t",
                    has_header=False,
                    new_columns=self.column_names,
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                )
            else:
                df = pl.read_csv(
                    file_path,
                    separator="\t",
                    has_header=False,
                    new_columns=self.column_names,
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                )

            logger.info("Log file loaded successfully")
            return df

        except Exception as e:
            logger.error(f"Error loading log file: {e}")
            raise

    def clean_and_parse(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Clean and parse log data with optimized Polars operations

        Args:
            df: Raw log data (LazyFrame)

        Returns:
            Cleaned LazyFrame
        """
        logger.info("Cleaning and parsing log data...")

        df = (
            df
            # Parse datetime
            .with_columns(
                [
                    pl.concat_str(
                        [pl.col("date"), pl.lit(" "), pl.col("time")],
                    )
                    .str.strptime(pl.Datetime, format="%d-%m-%Y %H:%M:%S", strict=False)
                    .alias("datetime")
                ]
            )
            # Clean numeric columns
            .with_columns(
                [
                    pl.col("sc_status").cast(pl.Int32, strict=False),
                    pl.col("sc_bytes").cast(pl.Int64, strict=False),
                    pl.col("time_taken").cast(pl.Int32, strict=False),
                ]
            )
            # Fill nulls
            .with_columns(
                [
                    pl.col("sc_bytes").fill_null(0),
                    pl.col("time_taken").fill_null(0),
                    pl.col("cs_referer").fill_null("direct"),
                    pl.col("cs_user_agent").fill_null("unknown"),
                ]
            )
            # Remove rows with null datetime
            .filter(pl.col("datetime").is_not_null())
            # Drop original date/time columns
            .drop(["date", "time"])
        )

        logger.info("Data cleaning completed")
        return df

    def compute_basic_stats(self, df: pl.LazyFrame) -> dict:
        """
        Compute basic statistics with Polars (very fast)

        Args:
            df: Cleaned log data

        Returns:
            Dictionary of statistics
        """
        logger.info("Computing basic statistics...")

        # Collect stats efficiently
        stats_df = (
            df.select(
                [
                    pl.count().alias("total_requests"),
                    pl.col("cs_ip").n_unique().alias("unique_ips"),
                    pl.col("datetime").min().alias("start_date"),
                    pl.col("datetime").max().alias("end_date"),
                    pl.col("sc_status")
                    .is_in([400, 403, 404, 500, 502, 503, 504])
                    .mean()
                    .alias("error_rate"),
                    pl.col("sc_bytes").sum().alias("total_bytes"),
                    pl.col("time_taken").mean().alias("avg_response_time"),
                ]
            ).collect()
        )

        # Convert to dict
        stats = stats_df.to_dicts()[0]

        # Calculate duration
        if stats["start_date"] and stats["end_date"]:
            duration = stats["end_date"] - stats["start_date"]
            stats["duration_hours"] = duration.total_seconds() / 3600
        else:
            stats["duration_hours"] = 0

        logger.info(f"Processed {stats['total_requests']} requests")
        return stats

    def sample_for_display(
        self, df: pl.LazyFrame, n: int = 1000
    ) -> pl.DataFrame:
        """
        Sample data for display/preview

        Args:
            df: Log data
            n: Number of samples

        Returns:
            Sample DataFrame
        """
        return df.limit(n).collect()

    def save_processed_data(
        self, df: pl.LazyFrame, output_path: Path
    ) -> None:
        """
        Save processed data to parquet (much faster than CSV)

        Args:
            df: Processed data
            output_path: Output file path
        """
        logger.info(f"Saving processed data to {output_path}")
        df.collect().write_parquet(output_path, compression="zstd")
        logger.info("Data saved successfully")

    def load_processed_data(self, file_path: Path) -> pl.LazyFrame:
        """
        Load processed parquet file

        Args:
            file_path: Path to parquet file

        Returns:
            LazyFrame
        """
        logger.info(f"Loading processed data from {file_path}")
        return pl.scan_parquet(file_path)
