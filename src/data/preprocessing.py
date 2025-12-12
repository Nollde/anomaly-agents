"""
Feature preprocessing for CATHODE analysis.

This module provides utilities for standardizing features and preparing
data for density estimation and classification.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class FeatureScaler:
    """
    Standard scaler for feature normalization.

    Standardizes features to have mean=0 and std=1, computed from
    the sideband data to avoid bias from signal.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.feature_names_ = None

    def fit(self, X: np.ndarray, feature_names: Optional[list] = None) -> "FeatureScaler":
        """
        Compute mean and std from training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            feature_names: Optional list of feature names

        Returns:
            self
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.feature_names_ = feature_names

        # Avoid division by zero
        self.std_[self.std_ < 1e-10] = 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features using fitted mean and std.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Standardized feature matrix
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")

        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray, feature_names: Optional[list] = None) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            feature_names: Optional list of feature names

        Returns:
            Standardized feature matrix
        """
        self.fit(X, feature_names)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse standardization.

        Args:
            X: Standardized feature matrix

        Returns:
            Original-scale feature matrix
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")

        return X * self.std_ + self.mean_

    def save(self, path: str) -> None:
        """
        Save scaler parameters to file.

        Args:
            path: Path to save the scaler
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "mean": self.mean_,
                    "std": self.std_,
                    "feature_names": self.feature_names_,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """
        Load scaler parameters from file.

        Args:
            path: Path to the saved scaler

        Returns:
            Loaded FeatureScaler instance
        """
        with open(path, "rb") as f:
            params = pickle.load(f)

        scaler = cls()
        scaler.mean_ = params["mean"]
        scaler.std_ = params["std"]
        scaler.feature_names_ = params["feature_names"]
        return scaler


def preprocess_data(
    data: Dict[str, np.ndarray],
    feature_names: list,
    scaler: Optional[FeatureScaler] = None,
    fit: bool = False,
) -> Tuple[np.ndarray, FeatureScaler]:
    """
    Preprocess data by extracting and standardizing features.

    Args:
        data: Dictionary with feature arrays
        feature_names: List of features to extract
        scaler: Optional pre-fitted scaler (if None and fit=False, raises error)
        fit: If True, fit a new scaler on this data

    Returns:
        Tuple of (standardized_features, scaler)
    """
    # Extract features
    features = []
    for name in feature_names:
        if name not in data:
            raise ValueError(f"Feature '{name}' not found in data")
        features.append(data[name])

    X = np.column_stack(features)

    # Standardize
    if fit:
        if scaler is None:
            scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(X, feature_names)
    else:
        if scaler is None:
            raise ValueError("Must provide scaler when fit=False")
        X_scaled = scaler.transform(X)

    return X_scaled, scaler
