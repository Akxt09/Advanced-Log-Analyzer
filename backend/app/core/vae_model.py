"""
Variational Autoencoder (VAE) for Anomaly Detection
Optimized for 474K log entries with batch_size=512
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.warning(f"GPU setup failed: {e}")


class Sampling(layers.Layer):
    """Custom sampling layer for VAE latent space"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEAnomalyDetector:
    """Variational Autoencoder for log anomaly detection"""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
    ):
        """
        Initialize VAE

        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Dimension of hidden layers
            learning_rate: Learning rate for optimizer
        """
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        self.encoder = None
        self.decoder = None
        self.vae = None
        self.scaler = RobustScaler()
        self.history = None
        self.input_dim = None

    def build_model(self, input_dim: int) -> Model:
        """
        Build VAE architecture

        Args:
            input_dim: Number of input features

        Returns:
            Compiled VAE model
        """
        self.input_dim = input_dim

        # ============ ENCODER ============
        encoder_inputs = layers.Input(shape=(input_dim,), name="encoder_input")

        # Encoder network
        x = layers.Dense(self.hidden_dim, activation="relu")(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(self.hidden_dim // 2, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

        # Latent space parameters
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        # Sampling
        z = Sampling()([z_mean, z_log_var])

        # Build encoder model
        self.encoder = Model(
            encoder_inputs, [z_mean, z_log_var, z], name="encoder"
        )

        # ============ DECODER ============
        latent_inputs = layers.Input(shape=(self.latent_dim,), name="latent_input")

        # Decoder network
        x = layers.Dense(self.hidden_dim // 2, activation="relu")(latent_inputs)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(self.hidden_dim, activation="relu")(x)
        x = layers.BatchNormalization()(x)

        decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)

        # Build decoder model
        self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

        # ============ VAE (Full Model) ============
        outputs = self.decoder(z)
        self.vae = Model(encoder_inputs, outputs, name="vae")

        # Custom VAE loss (reconstruction + KL divergence)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_squared_error(encoder_inputs, outputs), axis=0
            )
        )

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

        vae_loss = reconstruction_loss + kl_loss
        self.vae.add_loss(vae_loss)

        # Compile
        self.vae.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )

        logger.info(f"VAE built: input_dim={input_dim}, latent_dim={self.latent_dim}")
        return self.vae

    def train(
        self,
        X: np.ndarray,
        epochs: int = 20,
        batch_size: int = 512,
        validation_split: float = 0.2,
        patience: int = 5,
    ) -> dict:
        """
        Train VAE on log data

        Args:
            X: Training data (n_samples, n_features)
            epochs: Number of training epochs
            batch_size: Batch size (512 optimal for 474K rows)
            validation_split: Validation data fraction
            patience: Early stopping patience

        Returns:
            Training history
        """
        logger.info(
            f"Training VAE on {X.shape[0]} samples with batch_size={batch_size}"
        )

        # Scale data (crucial for neural networks)
        X_scaled = self.scaler.fit_transform(X)

        # Build model if not already built
        if self.vae is None:
            self.build_model(X.shape[1])

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        # Train
        self.history = self.vae.fit(
            X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("VAE training completed")
        return self.history.history

    def compute_anomaly_scores(
        self, X: np.ndarray, use_kl: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute anomaly scores for data

        Args:
            X: Data to score
            use_kl: Include KL divergence in score

        Returns:
            Tuple of (total_scores, reconstruction_errors, kl_divergences)
        """
        logger.info(f"Computing anomaly scores for {X.shape[0]} samples")

        # Scale data
        X_scaled = self.scaler.transform(X)

        # Get reconstructions and latent parameters
        z_mean, z_log_var, _ = self.encoder.predict(X_scaled, verbose=0)
        X_reconstructed = self.vae.predict(X_scaled, verbose=0)

        # Compute reconstruction error (MSE per sample)
        reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

        # Compute KL divergence per sample
        kl_divergences = -0.5 * np.sum(
            1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=1
        )

        # Combined anomaly score
        if use_kl:
            # Normalize both components to [0, 1] range
            recon_norm = (
                reconstruction_errors - reconstruction_errors.min()
            ) / (reconstruction_errors.max() - reconstruction_errors.min() + 1e-8)
            kl_norm = (kl_divergences - kl_divergences.min()) / (
                kl_divergences.max() - kl_divergences.min() + 1e-8
            )

            # Weighted combination (reconstruction is more important)
            total_scores = 0.7 * recon_norm + 0.3 * kl_norm
        else:
            total_scores = reconstruction_errors

        logger.info(
            f"Anomaly scores - Mean: {total_scores.mean():.4f}, Std: {total_scores.std():.4f}"
        )

        return total_scores, reconstruction_errors, kl_divergences

    def get_latent_representations(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent space representations for clustering

        Args:
            X: Input data

        Returns:
            Latent representations (z_mean)
        """
        X_scaled = self.scaler.transform(X)
        z_mean, _, _ = self.encoder.predict(X_scaled, verbose=0)
        return z_mean

    def get_reconstruction_errors_per_feature(
        self, X: np.ndarray
    ) -> np.ndarray:
        """
        Get reconstruction error for each feature (for explainability)

        Args:
            X: Input data

        Returns:
            Feature-wise reconstruction errors (n_samples, n_features)
        """
        X_scaled = self.scaler.transform(X)
        X_reconstructed = self.vae.predict(X_scaled, verbose=0)

        # Absolute error per feature
        feature_errors = np.abs(X_scaled - X_reconstructed)

        return feature_errors

    def save_model(self, model_dir: Path):
        """Save VAE model and scaler"""
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        self.encoder.save(model_dir / "encoder.keras")
        self.decoder.save(model_dir / "decoder.keras")
        self.vae.save(model_dir / "vae.keras")

        # Save scaler
        import joblib

        joblib.dump(self.scaler, model_dir / "scaler.pkl")

        logger.info(f"VAE model saved to {model_dir}")

    def load_model(self, model_dir: Path):
        """Load VAE model and scaler"""
        import joblib

        self.encoder = keras.models.load_model(
            model_dir / "encoder.keras", custom_objects={"Sampling": Sampling}
        )
        self.decoder = keras.models.load_model(model_dir / "decoder.keras")
        self.vae = keras.models.load_model(
            model_dir / "vae.keras", custom_objects={"Sampling": Sampling}
        )
        self.scaler = joblib.load(model_dir / "scaler.pkl")

        logger.info(f"VAE model loaded from {model_dir}")
