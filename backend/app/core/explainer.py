"""
Hybrid Explainability Engine
Z-Score for all anomalies + SHAP for top 100
"""
import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
import shap

logger = logging.getLogger(__name__)


class AnomalyExplainer:
    """Explain why samples are flagged as anomalies"""

    def __init__(self, top_n_shap: int = 100, n_features_explain: int = 5):
        """
        Initialize explainer

        Args:
            top_n_shap: Number of top anomalies to explain with SHAP
            n_features_explain: Number of features to include in explanation
        """
        self.top_n_shap = top_n_shap
        self.n_features_explain = n_features_explain
        self.shap_values = None
        self.shap_explainer = None

    def compute_zscore_explanations(
        self,
        X: np.ndarray,
        feature_names: List[str],
        anomaly_indices: np.ndarray,
    ) -> Dict[int, Dict]:
        """
        Compute Z-score based explanations for all anomalies (FAST)

        Args:
            X: Feature matrix
            feature_names: List of feature names
            anomaly_indices: Indices of anomalies

        Returns:
            Dictionary mapping index to explanation
        """
        logger.info(
            f"Computing Z-score explanations for {len(anomaly_indices)} anomalies..."
        )

        # Compute global mean and std
        global_mean = np.mean(X, axis=0)
        global_std = np.std(X, axis=0)

        explanations = {}

        for idx in anomaly_indices:
            # Get sample
            sample = X[idx]

            # Compute Z-scores for each feature
            z_scores = np.abs((sample - global_mean) / (global_std + 1e-8))

            # Get top N features by Z-score
            top_feature_indices = np.argsort(z_scores)[-self.n_features_explain :][::-1]

            # Build explanation
            features_info = []
            for feat_idx in top_feature_indices:
                features_info.append(
                    {
                        "feature": feature_names[feat_idx],
                        "zscore": float(z_scores[feat_idx]),
                        "value": float(sample[feat_idx]),
                        "mean": float(global_mean[feat_idx]),
                        "deviation_pct": float(
                            (
                                (sample[feat_idx] - global_mean[feat_idx])
                                / (global_mean[feat_idx] + 1e-8)
                            )
                            * 100
                        ),
                    }
                )

            explanations[int(idx)] = {
                "method": "zscore",
                "top_features": features_info,
                "summary": self._create_zscore_summary(features_info),
            }

        logger.info("Z-score explanations computed")
        return explanations

    def compute_reconstruction_explanations(
        self,
        feature_reconstruction_errors: np.ndarray,
        feature_names: List[str],
        anomaly_indices: np.ndarray,
    ) -> Dict[int, Dict]:
        """
        Explain anomalies based on reconstruction errors (VAE-specific)

        Args:
            feature_reconstruction_errors: Per-feature reconstruction errors
            feature_names: List of feature names
            anomaly_indices: Indices of anomalies

        Returns:
            Reconstruction-based explanations
        """
        logger.info("Computing reconstruction-based explanations...")

        explanations = {}

        for idx in anomaly_indices:
            # Get reconstruction errors for this sample
            errors = feature_reconstruction_errors[idx]

            # Get top N features with highest reconstruction error
            top_feature_indices = np.argsort(errors)[-self.n_features_explain :][::-1]

            features_info = []
            for feat_idx in top_feature_indices:
                features_info.append(
                    {
                        "feature": feature_names[feat_idx],
                        "reconstruction_error": float(errors[feat_idx]),
                        "relative_error": float(
                            errors[feat_idx] / (errors.mean() + 1e-8)
                        ),
                    }
                )

            explanations[int(idx)] = {
                "method": "reconstruction",
                "top_features": features_info,
                "summary": self._create_reconstruction_summary(features_info),
            }

        return explanations

    def compute_shap_explanations(
        self,
        vae_model,
        X: np.ndarray,
        feature_names: List[str],
        anomaly_scores: np.ndarray,
        top_n: Optional[int] = None,
    ) -> Dict[int, Dict]:
        """
        Compute SHAP explanations for top N anomalies (SLOW but accurate)

        Args:
            vae_model: Trained VAE model
            X: Feature matrix
            feature_names: List of feature names
            anomaly_scores: Anomaly scores
            top_n: Number of top anomalies to explain (default: self.top_n_shap)

        Returns:
            SHAP-based explanations
        """
        if top_n is None:
            top_n = self.top_n_shap

        logger.info(f"Computing SHAP explanations for top {top_n} anomalies...")
        logger.info("This may take 5-10 minutes...")

        # Get top N anomalies
        top_anomaly_indices = np.argsort(anomaly_scores)[-top_n:][::-1]
        X_top = X[top_anomaly_indices]

        try:
            # Scale data using VAE's scaler
            X_scaled = vae_model.scaler.transform(X_top)

            # Create SHAP explainer
            # Use a background dataset (sample 100 points for efficiency)
            background_size = min(100, len(X))
            background_indices = np.random.choice(
                len(X), background_size, replace=False
            )
            background = vae_model.scaler.transform(X[background_indices])

            # Use DeepExplainer for neural networks
            self.shap_explainer = shap.DeepExplainer(
                vae_model.vae, background
            )

            # Compute SHAP values
            self.shap_values = self.shap_explainer.shap_values(X_scaled)

            # Create explanations
            explanations = {}

            for i, orig_idx in enumerate(top_anomaly_indices):
                # Get SHAP values for this sample
                shap_vals = self.shap_values[i]

                # Get top N features by absolute SHAP value
                abs_shap = np.abs(shap_vals)
                top_feature_indices = np.argsort(abs_shap)[
                    -self.n_features_explain :
                ][::-1]

                features_info = []
                for feat_idx in top_feature_indices:
                    features_info.append(
                        {
                            "feature": feature_names[feat_idx],
                            "shap_value": float(shap_vals[feat_idx]),
                            "feature_value": float(X_top[i, feat_idx]),
                            "importance": float(abs_shap[feat_idx]),
                        }
                    )

                explanations[int(orig_idx)] = {
                    "method": "shap",
                    "top_features": features_info,
                    "summary": self._create_shap_summary(features_info),
                }

            logger.info(f"SHAP explanations computed for {len(explanations)} samples")
            return explanations

        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            logger.info("Falling back to Z-score explanations for top anomalies")
            return {}

    def create_hybrid_explanations(
        self,
        vae_model,
        X: np.ndarray,
        feature_names: List[str],
        anomaly_scores: np.ndarray,
        is_anomaly: np.ndarray,
        feature_reconstruction_errors: np.ndarray,
    ) -> Dict[int, Dict]:
        """
        Create hybrid explanations:
        - Z-score + Reconstruction for all anomalies
        - SHAP for top N anomalies

        Args:
            vae_model: Trained VAE
            X: Feature matrix
            feature_names: Feature names
            anomaly_scores: Anomaly scores
            is_anomaly: Boolean mask of anomalies
            feature_reconstruction_errors: Per-feature reconstruction errors

        Returns:
            Complete explanations dictionary
        """
        logger.info("Creating hybrid explanations...")

        anomaly_indices = np.where(is_anomaly)[0]

        # 1. Z-score explanations for ALL anomalies (fast)
        zscore_explanations = self.compute_zscore_explanations(
            X, feature_names, anomaly_indices
        )

        # 2. Reconstruction explanations for ALL anomalies (fast)
        reconstruction_explanations = self.compute_reconstruction_explanations(
            feature_reconstruction_errors, feature_names, anomaly_indices
        )

        # 3. SHAP explanations for TOP N anomalies (slow)
        shap_explanations = self.compute_shap_explanations(
            vae_model, X, feature_names, anomaly_scores, top_n=self.top_n_shap
        )

        # Merge explanations
        all_explanations = {}

        for idx in anomaly_indices:
            explanation = {
                "index": int(idx),
                "anomaly_score": float(anomaly_scores[idx]),
                "zscore_explanation": zscore_explanations.get(idx, {}),
                "reconstruction_explanation": reconstruction_explanations.get(
                    idx, {}
                ),
            }

            # Add SHAP if available
            if idx in shap_explanations:
                explanation["shap_explanation"] = shap_explanations[idx]
                explanation["has_shap"] = True
            else:
                explanation["has_shap"] = False

            all_explanations[int(idx)] = explanation

        logger.info(f"Created explanations for {len(all_explanations)} anomalies")
        return all_explanations

    def _create_zscore_summary(self, features_info: List[Dict]) -> str:
        """Create human-readable summary from Z-score features"""
        if not features_info:
            return "No significant deviations"

        top_feat = features_info[0]
        return (
            f"{top_feat['feature']}: {top_feat['zscore']:.2f}σ "
            f"({top_feat['deviation_pct']:+.1f}% from mean)"
        )

    def _create_reconstruction_summary(
        self, features_info: List[Dict]
    ) -> str:
        """Create summary from reconstruction errors"""
        if not features_info:
            return "Low reconstruction error"

        top_feat = features_info[0]
        return f"{top_feat['feature']}: High reconstruction error ({top_feat['reconstruction_error']:.4f})"

    def _create_shap_summary(self, features_info: List[Dict]) -> str:
        """Create summary from SHAP values"""
        if not features_info:
            return "No significant SHAP values"

        top_feat = features_info[0]
        direction = "increased" if top_feat["shap_value"] > 0 else "decreased"
        return f"{top_feat['feature']} {direction} anomaly score (SHAP: {top_feat['shap_value']:.4f})"

    def generate_root_cause_analysis(
        self, all_explanations: Dict[int, Dict], feature_names: List[str]
    ) -> Dict:
        """
        Generate root cause analysis across all anomalies

        Args:
            all_explanations: All anomaly explanations
            feature_names: List of feature names

        Returns:
            Root cause analysis
        """
        logger.info("Generating root cause analysis...")

        # Count feature frequencies
        feature_frequency = {feat: 0 for feat in feature_names}

        for exp in all_explanations.values():
            # Count from Z-score explanations
            if "zscore_explanation" in exp:
                for feat_info in exp["zscore_explanation"].get(
                    "top_features", []
                ):
                    feature_frequency[feat_info["feature"]] += 1

        # Sort by frequency
        sorted_features = sorted(
            feature_frequency.items(), key=lambda x: x[1], reverse=True
        )

        # Create report
        root_cause = {
            "total_anomalies": len(all_explanations),
            "top_contributing_features": [
                {"feature": feat, "frequency": count, "percentage": count / len(all_explanations) * 100}
                for feat, count in sorted_features[:10]
                if count > 0
            ],
            "summary": self._create_root_cause_summary(sorted_features, len(all_explanations)),
        }

        return root_cause

    def _create_root_cause_summary(
        self, sorted_features: List[Tuple], total_anomalies: int
    ) -> str:
        """Create root cause summary text"""
        if not sorted_features or total_anomalies == 0:
            return "No clear patterns identified"

        top_3 = sorted_features[:3]
        summary_parts = []

        for feat, count in top_3:
            if count > 0:
                pct = (count / total_anomalies) * 100
                summary_parts.append(f"{feat} ({pct:.1f}%)")

        return "Top contributing factors: " + ", ".join(summary_parts)
