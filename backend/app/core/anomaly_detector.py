"""
Anomaly Detection Pipeline
Combines VAE + HDBSCAN clustering
"""
import numpy as np
import polars as pl
from typing import Tuple, Dict, Optional
import logging
import hdbscan
from .vae_model import VAEAnomalyDetector

logger = logging.getLogger(__name__)


class LogAnomalyDetectionPipeline:
    """Complete anomaly detection pipeline for web logs"""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        anomaly_threshold_pct: float = 0.02,
        min_cluster_size: int = 50,
        min_samples: int = 20,
    ):
        """
        Initialize anomaly detection pipeline

        Args:
            latent_dim: VAE latent dimension
            hidden_dim: VAE hidden layer dimension
            learning_rate: VAE learning rate
            anomaly_threshold_pct: Percentage of data to flag as anomalies
            min_cluster_size: HDBSCAN minimum cluster size
            min_samples: HDBSCAN minimum samples
        """
        self.vae = VAEAnomalyDetector(latent_dim, hidden_dim, learning_rate)
        self.anomaly_threshold_pct = anomaly_threshold_pct
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

        self.clusterer = None
        self.anomaly_threshold = None

    def train_vae(
        self,
        X: np.ndarray,
        epochs: int = 20,
        batch_size: int = 512,
        patience: int = 5,
    ) -> dict:
        """
        Train VAE model

        Args:
            X: Training data
            epochs: Training epochs
            batch_size: Batch size
            patience: Early stopping patience

        Returns:
            Training history
        """
        logger.info("Training VAE model...")
        history = self.vae.train(
            X, epochs=epochs, batch_size=batch_size, patience=patience
        )
        logger.info("VAE training completed")
        return history

    def detect_anomalies(
        self, X: np.ndarray, feature_names: list
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Detect anomalies using VAE

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Tuple of (anomaly_scores, is_anomaly, metadata)
        """
        logger.info("Detecting anomalies...")

        # Compute anomaly scores
        (
            anomaly_scores,
            reconstruction_errors,
            kl_divergences,
        ) = self.vae.compute_anomaly_scores(X)

        # Determine threshold (top N%)
        self.anomaly_threshold = np.percentile(
            anomaly_scores, (1 - self.anomaly_threshold_pct) * 100
        )

        # Flag anomalies
        is_anomaly = anomaly_scores >= self.anomaly_threshold

        # Metadata
        metadata = {
            "total_samples": len(X),
            "num_anomalies": int(is_anomaly.sum()),
            "anomaly_rate": float(is_anomaly.mean()),
            "threshold": float(self.anomaly_threshold),
            "mean_score": float(anomaly_scores.mean()),
            "std_score": float(anomaly_scores.std()),
            "max_score": float(anomaly_scores.max()),
            "min_score": float(anomaly_scores.min()),
        }

        logger.info(
            f"Detected {metadata['num_anomalies']} anomalies ({metadata['anomaly_rate']*100:.2f}%)"
        )

        return anomaly_scores, is_anomaly, metadata

    def cluster_latent_space(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform HDBSCAN clustering on VAE latent space

        Args:
            X: Feature matrix

        Returns:
            Tuple of (cluster_labels, cluster_metadata)
        """
        logger.info("Clustering latent space with HDBSCAN...")

        # Get latent representations
        latent_vectors = self.vae.get_latent_representations(X)

        # Adjust parameters based on dataset size
        n_samples = len(X)
        min_cluster_size = min(
            self.min_cluster_size, max(30, n_samples // 100)
        )
        min_samples = min(self.min_samples, max(10, n_samples // 200))

        # Perform clustering
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            prediction_data=True,
            core_dist_n_jobs=-1,
        )

        cluster_labels = self.clusterer.fit_predict(latent_vectors)

        # Metadata
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
        n_noise = (cluster_labels == -1).sum()

        metadata = {
            "n_clusters": len(unique_clusters),
            "n_noise_points": int(n_noise),
            "noise_rate": float(n_noise / len(X)),
            "cluster_sizes": {
                int(c): int((cluster_labels == c).sum())
                for c in unique_clusters
            },
        }

        logger.info(
            f"Found {metadata['n_clusters']} clusters, {n_noise} noise points"
        )

        return cluster_labels, metadata

    def analyze_clusters(
        self,
        df: pl.DataFrame,
        cluster_labels: np.ndarray,
        anomaly_scores: np.ndarray,
    ) -> Dict:
        """
        Analyze cluster characteristics

        Args:
            df: Original dataframe with features
            cluster_labels: Cluster assignments
            anomaly_scores: Anomaly scores

        Returns:
            Cluster analysis results
        """
        logger.info("Analyzing cluster characteristics...")

        # Add cluster labels and scores to dataframe
        df_with_clusters = df.with_columns(
            [
                pl.Series("cluster", cluster_labels),
                pl.Series("anomaly_score", anomaly_scores),
            ]
        )

        # Analyze each cluster
        cluster_stats = {}

        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])

        for cluster_id in unique_clusters:
            cluster_data = df_with_clusters.filter(
                pl.col("cluster") == cluster_id
            )

            if len(cluster_data) > 0:
                stats = {
                    "size": len(cluster_data),
                    "mean_anomaly_score": float(
                        cluster_data["anomaly_score"].mean()
                    ),
                    "max_anomaly_score": float(
                        cluster_data["anomaly_score"].max()
                    ),
                    "error_rate": float(
                        cluster_data.get_column("is_error").mean()
                        if "is_error" in cluster_data.columns
                        else 0
                    ),
                    "top_urls": (
                        cluster_data.get_column("cs_uri")
                        .value_counts()
                        .head(5)
                        .to_dicts()
                        if "cs_uri" in cluster_data.columns
                        else []
                    ),
                    "top_ips": (
                        cluster_data.get_column("cs_ip")
                        .value_counts()
                        .head(5)
                        .to_dicts()
                        if "cs_ip" in cluster_data.columns
                        else []
                    ),
                }

                cluster_stats[int(cluster_id)] = stats

        return cluster_stats

    def get_feature_reconstruction_errors(
        self, X: np.ndarray, feature_names: list
    ) -> np.ndarray:
        """
        Get reconstruction errors per feature (for explainability)

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Feature-wise reconstruction errors
        """
        return self.vae.get_reconstruction_errors_per_feature(X)

    def run_full_pipeline(
        self,
        X: np.ndarray,
        df: pl.DataFrame,
        feature_names: list,
        train_vae: bool = True,
        perform_clustering: bool = True,
    ) -> Dict:
        """
        Run complete anomaly detection pipeline

        Args:
            X: Feature matrix
            df: Original dataframe
            feature_names: List of feature names
            train_vae: Whether to train VAE (False to use pre-trained)
            perform_clustering: Whether to perform clustering

        Returns:
            Complete results dictionary
        """
        logger.info("Running full anomaly detection pipeline...")

        results = {}

        # 1. Train VAE
        if train_vae:
            training_history = self.train_vae(X)
            results["training_history"] = training_history

        # 2. Detect anomalies
        anomaly_scores, is_anomaly, anomaly_metadata = self.detect_anomalies(
            X, feature_names
        )

        results["anomaly_scores"] = anomaly_scores
        results["is_anomaly"] = is_anomaly
        results["anomaly_metadata"] = anomaly_metadata

        # 3. Clustering (optional)
        if perform_clustering:
            cluster_labels, cluster_metadata = self.cluster_latent_space(X)
            results["cluster_labels"] = cluster_labels
            results["cluster_metadata"] = cluster_metadata

            # Analyze clusters
            cluster_analysis = self.analyze_clusters(
                df, cluster_labels, anomaly_scores
            )
            results["cluster_analysis"] = cluster_analysis
        else:
            results["cluster_labels"] = np.zeros(len(X), dtype=int)
            results["cluster_metadata"] = {}
            results["cluster_analysis"] = {}

        # 4. Feature-level reconstruction errors (for explainability)
        feature_errors = self.get_feature_reconstruction_errors(
            X, feature_names
        )
        results["feature_reconstruction_errors"] = feature_errors

        logger.info("Pipeline completed successfully")
        return results
