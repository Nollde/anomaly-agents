"""
Masked Autoregressive Flow (MAF) implementation for CATHODE.

This module implements the conditional density estimator used in the CATHODE method.
The MAF learns p(x|m) where x are features and m is the dijet mass mJJ.

Based on the CATHODE paper specifications:
- 15 MADE blocks with 128 hidden nodes each
- Batch normalization with momentum 1.0
- Unit normal base distribution
- Adam optimizer with lr=10^-4, batch size=256
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.normalization import BatchNorm


class ConditionalMAF:
    """
    Conditional Masked Autoregressive Flow for density estimation.

    Learns conditional density p(x|m) where x are features and m is a conditioning variable.
    """

    def __init__(
        self,
        features: int,
        context_features: int = 1,
        hidden_features: int = 128,
        num_layers: int = 15,
        use_batch_norm: bool = True,
        batch_norm_momentum: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize conditional MAF.

        Args:
            features: Number of input features (x dimension)
            context_features: Number of conditioning features (m dimension)
            hidden_features: Number of hidden units in MADE blocks (default: 128)
            num_layers: Number of MADE blocks (default: 15)
            use_batch_norm: Whether to use batch normalization (default: True)
            batch_norm_momentum: Momentum for batch normalization (default: 1.0)
            device: Device to use for computation
        """
        self.features = features
        self.context_features = context_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.device = device

        # Build the flow
        self.flow = self._build_flow()
        self.flow.to(device)

        # Training state
        self.optimizer: Optional[optim.Optimizer] = None
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def _build_flow(self) -> Flow:
        """
        Build the MAF architecture.

        Returns:
            Flow object with stacked MADE transforms
        """
        # Base distribution: standard normal
        base_dist = StandardNormal(shape=[self.features])

        # Build transform stack
        transforms = []

        for _ in range(self.num_layers):
            # MADE autoregressive transform with affine coupling
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=self.features,
                    hidden_features=self.hidden_features,
                    context_features=self.context_features,
                    num_blocks=2,  # Number of residual blocks in MADE
                    use_residual_blocks=False,
                    random_mask=False,
                    activation=nn.ReLU(),
                    dropout_probability=0.0,
                    use_batch_norm=False,  # We add batch norm separately
                )
            )

            # Batch normalization (if enabled)
            if self.use_batch_norm:
                transforms.append(
                    BatchNorm(
                        features=self.features,
                        momentum=self.batch_norm_momentum,
                    )
                )

        # Compose all transforms
        transform = CompositeTransform(transforms)

        # Create flow
        flow = Flow(transform, base_dist)

        return flow

    def _prepare_batch(
        self, X: np.ndarray, m: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare numpy arrays as torch tensors.

        Args:
            X: Features array of shape (n_samples, n_features)
            m: Conditioning array of shape (n_samples,) or (n_samples, 1)

        Returns:
            Tuple of (X_tensor, m_tensor) on the correct device
        """
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        # Ensure m has shape (n_samples, 1)
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        m_tensor = torch.tensor(m, dtype=torch.float32, device=self.device)

        return X_tensor, m_tensor

    def fit(
        self,
        X_train: np.ndarray,
        m_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        m_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-4,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the conditional MAF.

        Args:
            X_train: Training features of shape (n_train, n_features)
            m_train: Training conditioning of shape (n_train,)
            X_val: Validation features (optional)
            m_val: Validation conditioning (optional)
            epochs: Number of training epochs (default: 100)
            batch_size: Batch size for training (default: 256)
            learning_rate: Learning rate for Adam optimizer (default: 1e-4)
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history (train_loss, val_loss)
        """
        # Initialize optimizer
        self.optimizer = optim.Adam(self.flow.parameters(), lr=learning_rate)

        # Reset training history
        self.training_history = {"train_loss": [], "val_loss": []}

        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(X_train, m_train, batch_size)
            self.training_history["train_loss"].append(train_loss)

            # Validate
            if X_val is not None and m_val is not None:
                val_loss = self._validate(X_val, m_val, batch_size)
                self.training_history["val_loss"].append(val_loss)
            else:
                val_loss = None

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if val_loss is not None:
                    print(
                        f"Epoch {epoch+1:3d}/{epochs}: "
                        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                    )
                else:
                    print(f"Epoch {epoch+1:3d}/{epochs}: train_loss={train_loss:.4f}")

        return self.training_history

    def _train_epoch(
        self, X: np.ndarray, m: np.ndarray, batch_size: int
    ) -> float:
        """
        Train for one epoch.

        Args:
            X: Training features
            m: Training conditioning
            batch_size: Batch size

        Returns:
            Average training loss for the epoch
        """
        self.flow.train()

        n_samples = len(X)
        indices = np.random.permutation(n_samples)

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            X_batch = X[batch_indices]
            m_batch = m[batch_indices]

            # Prepare tensors
            X_tensor, m_tensor = self._prepare_batch(X_batch, m_batch)

            # Forward pass: compute negative log-likelihood
            log_prob = self.flow.log_prob(X_tensor, context=m_tensor)
            loss = -log_prob.mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        return epoch_loss / n_batches

    def _validate(
        self, X: np.ndarray, m: np.ndarray, batch_size: int
    ) -> float:
        """
        Validate on validation set.

        Args:
            X: Validation features
            m: Validation conditioning
            batch_size: Batch size

        Returns:
            Average validation loss
        """
        self.flow.eval()

        n_samples = len(X)
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                X_batch = X[i : i + batch_size]
                m_batch = m[i : i + batch_size]

                # Prepare tensors
                X_tensor, m_tensor = self._prepare_batch(X_batch, m_batch)

                # Compute loss
                log_prob = self.flow.log_prob(X_tensor, context=m_tensor)
                loss = -log_prob.mean()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def log_prob(self, X: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Compute log probability log p(x|m).

        Args:
            X: Features of shape (n_samples, n_features)
            m: Conditioning of shape (n_samples,)

        Returns:
            Log probabilities of shape (n_samples,)
        """
        self.flow.eval()

        X_tensor, m_tensor = self._prepare_batch(X, m)

        with torch.no_grad():
            log_prob = self.flow.log_prob(X_tensor, context=m_tensor)

        return log_prob.cpu().numpy()

    def sample(
        self, m: np.ndarray, num_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample from the conditional distribution p(x|m).

        Args:
            m: Conditioning values of shape (n_contexts,) or scalar
            num_samples: Number of samples per context (if None, one sample per context)

        Returns:
            Samples of shape (n_samples, n_features)
        """
        self.flow.eval()

        # Handle scalar m
        if np.isscalar(m):
            m = np.array([m])

        # Ensure m is 1D
        if m.ndim > 1:
            m = m.flatten()

        # Prepare conditioning
        if num_samples is None:
            # One sample per conditioning value
            m_tensor = torch.tensor(
                m.reshape(-1, 1), dtype=torch.float32, device=self.device
            )
            n_total = len(m)
        else:
            # Multiple samples per conditioning value
            m_repeated = np.repeat(m, num_samples)
            m_tensor = torch.tensor(
                m_repeated.reshape(-1, 1), dtype=torch.float32, device=self.device
            )
            n_total = len(m) * num_samples

        # Sample
        with torch.no_grad():
            samples = self.flow.sample(n_total, context=m_tensor)

        return samples.cpu().numpy()

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.flow.state_dict(),
            "features": self.features,
            "context_features": self.context_features,
            "hidden_features": self.hidden_features,
            "num_layers": self.num_layers,
            "use_batch_norm": self.use_batch_norm,
            "batch_norm_momentum": self.batch_norm_momentum,
            "training_history": self.training_history,
        }

        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "ConditionalMAF":
        """
        Load model from disk.

        Args:
            path: Path to saved model
            device: Device to load model on (if None, uses default)

        Returns:
            Loaded ConditionalMAF model
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        state = torch.load(path, map_location=device, weights_only=False)

        # Create model with saved parameters
        model = cls(
            features=state["features"],
            context_features=state["context_features"],
            hidden_features=state["hidden_features"],
            num_layers=state["num_layers"],
            use_batch_norm=state["use_batch_norm"],
            batch_norm_momentum=state["batch_norm_momentum"],
            device=device,
        )

        # Load state dict
        model.flow.load_state_dict(state["model_state_dict"])
        model.training_history = state["training_history"]

        return model
