"""Attribution-guided autoencoder implementation using PyTorch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger(__name__)


class _AutoEncoderNet(nn.Module):
    """Simple fully connected autoencoder network."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__()

        encoder_layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        decoder_layers: List[nn.Module] = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs forward pass through encoder and decoder."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


@dataclass(frozen=True)
class AutoEncoderConfig:
    """Configuration for attribution-guided autoencoder training."""

    hidden_dims: List[int]
    latent_dim: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    device: str


class AttributionGuidedAutoencoder:
    """Trains an autoencoder with per-feature weighted reconstruction loss."""

    def __init__(self, config: Dict) -> None:
        self.cfg = AutoEncoderConfig(
            hidden_dims=list(config.get("hidden_dims", [128, 64])),
            latent_dim=int(config.get("latent_dim", 32)),
            batch_size=int(config.get("batch_size", 256)),
            epochs=int(config.get("epochs", 20)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            weight_decay=float(config.get("weight_decay", 0.0)),
            device=str(config.get("device", "cpu")),
        )
        self.model: _AutoEncoderNet | None = None
        self.feature_names: List[str] = []
        self.feature_weights: np.ndarray | None = None

    def _to_tensor(self, frame: pd.DataFrame) -> torch.Tensor:
        """Converts a DataFrame to float tensor."""
        return torch.tensor(frame.values, dtype=torch.float32)

    def _weighted_reconstruction_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        feature_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Computes weighted mean squared reconstruction loss."""
        squared_error = (x - x_hat) ** 2
        weighted_error = squared_error * feature_weights
        return weighted_error.mean()

    def fit(self, x_train: pd.DataFrame, feature_weights: pd.Series | None = None) -> "AttributionGuidedAutoencoder":
        """Fits the autoencoder to provided training features."""
        self.feature_names = list(x_train.columns)
        input_dim = x_train.shape[1]

        if feature_weights is None:
            weight_arr = np.ones(input_dim, dtype=np.float32)
        else:
            aligned = feature_weights.reindex(self.feature_names).fillna(float(feature_weights.mean()))
            weight_arr = aligned.astype("float32").values
        self.feature_weights = weight_arr

        model = _AutoEncoderNet(
            input_dim=input_dim,
            hidden_dims=self.cfg.hidden_dims,
            latent_dim=self.cfg.latent_dim,
        )

        device = torch.device(self.cfg.device)
        model.to(device)

        dataset = TensorDataset(self._to_tensor(x_train))
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        feature_weight_tensor = torch.tensor(weight_arr, dtype=torch.float32, device=device)

        model.train()
        for epoch in range(self.cfg.epochs):
            epoch_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                reconstructed = model(batch_x)
                loss = self._weighted_reconstruction_loss(
                    x=batch_x,
                    x_hat=reconstructed,
                    feature_weights=feature_weight_tensor,
                )
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_x)

            mean_loss = epoch_loss / max(len(dataset), 1)
            LOGGER.info("Autoencoder epoch %d/%d loss=%.6f", epoch + 1, self.cfg.epochs, mean_loss)

        self.model = model
        return self

    def reconstruction_error(self, x: pd.DataFrame) -> np.ndarray:
        """Computes weighted reconstruction error per row."""
        if self.model is None:
            raise RuntimeError("Autoencoder must be fitted before scoring")
        if self.feature_weights is None:
            raise RuntimeError("Feature weights not initialized")

        missing = [col for col in self.feature_names if col not in x.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing[:5]}")

        x_aligned = x[self.feature_names]
        tensor = self._to_tensor(x_aligned)
        device = next(self.model.parameters()).device

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor.to(device)).cpu().numpy()

        original = x_aligned.values.astype("float32")
        weights = self.feature_weights.reshape(1, -1)
        weighted_sq_error = ((original - reconstructed) ** 2) * weights
        return weighted_sq_error.mean(axis=1)

    def save(self, output_path: str | Path) -> Path:
        """Saves model state and metadata."""
        if self.model is None or self.feature_weights is None:
            raise RuntimeError("Cannot save unfitted autoencoder")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "config": {
                "hidden_dims": self.cfg.hidden_dims,
                "latent_dim": self.cfg.latent_dim,
                "batch_size": self.cfg.batch_size,
                "epochs": self.cfg.epochs,
                "learning_rate": self.cfg.learning_rate,
                "weight_decay": self.cfg.weight_decay,
                "device": self.cfg.device,
            },
            "feature_names": self.feature_names,
            "feature_weights": self.feature_weights.tolist(),
        }
        torch.save(payload, path)
        LOGGER.info("Saved autoencoder artifact to %s", path)
        return path

    @staticmethod
    def load(input_path: str | Path) -> "AttributionGuidedAutoencoder":
        """Loads saved autoencoder from disk."""
        payload = torch.load(input_path, map_location="cpu")

        config = payload["config"]
        instance = AttributionGuidedAutoencoder(config)

        instance.feature_names = payload["feature_names"]
        instance.feature_weights = np.array(payload["feature_weights"], dtype=np.float32)

        model = _AutoEncoderNet(
            input_dim=len(instance.feature_names),
            hidden_dims=config["hidden_dims"],
            latent_dim=config["latent_dim"],
        )

        model.load_state_dict(payload["state_dict"])

        device = torch.device(config.get("device", "cpu"))
        model.to(device)

        instance.model = model
        return instance
