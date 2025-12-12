"""
Plotting utilities for CATHODE analysis.

This module provides functions for creating publication-quality plots
comparing signal and background distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple


def setup_plot_style():
    """Set up matplotlib style for publication-quality plots."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
        }
    )


def plot_feature_distributions(
    bg_data: Dict[str, np.ndarray],
    sig_data: Dict[str, np.ndarray],
    features: List[str],
    output_path: str,
    bins: int = 50,
    xlabels: Optional[Dict[str, str]] = None,
    log_scale: bool = False,
) -> None:
    """
    Plot feature distributions comparing background and signal.

    Args:
        bg_data: Background data dictionary
        sig_data: Signal data dictionary
        features: List of feature names to plot
        output_path: Path to save the plot
        bins: Number of histogram bins
        xlabels: Optional dictionary mapping feature names to x-axis labels
        log_scale: If True, use log scale for y-axis
    """
    setup_plot_style()

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, feature in enumerate(features):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get data
        bg_vals = bg_data[feature]
        sig_vals = sig_data[feature]

        # Compute histogram range
        all_vals = np.concatenate([bg_vals, sig_vals])
        vmin, vmax = np.percentile(all_vals, [0.5, 99.5])

        # Plot histograms
        ax.hist(
            bg_vals,
            bins=bins,
            range=(vmin, vmax),
            alpha=0.6,
            label="Background",
            color="blue",
            density=True,
        )
        ax.hist(
            sig_vals,
            bins=bins,
            range=(vmin, vmax),
            alpha=0.6,
            label="Signal",
            color="red",
            density=True,
        )

        # Set labels
        if xlabels and feature in xlabels:
            ax.set_xlabel(xlabels[feature])
        else:
            ax.set_xlabel(feature)

        ax.set_ylabel("Normalized counts")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale("log")

    # Remove extra subplots
    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature distribution plot to {output_path}")


def plot_mjj_distribution(
    bg_data: Dict[str, np.ndarray],
    sig_data: Dict[str, np.ndarray],
    output_path: str,
    sr_low: float = 3.3,
    sr_high: float = 3.7,
    bins: int = 100,
) -> None:
    """
    Plot mJJ distribution with SR highlighted.

    Args:
        bg_data: Background data dictionary
        sig_data: Signal data dictionary
        output_path: Path to save the plot
        sr_low: Signal region lower bound in TeV
        sr_high: Signal region upper bound in TeV
        bins: Number of histogram bins
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get mJJ data
    bg_mjj = bg_data["mJJ"]
    sig_mjj = sig_data["mJJ"]

    # Plot histograms
    ax.hist(
        bg_mjj,
        bins=bins,
        range=(1.5, 5.5),
        alpha=0.6,
        label=f"Background (N={len(bg_mjj):,})",
        color="blue",
        density=True,
    )
    ax.hist(
        sig_mjj,
        bins=bins,
        range=(1.5, 5.5),
        alpha=0.6,
        label=f"Signal (N={len(sig_mjj):,})",
        color="red",
        density=True,
    )

    # Highlight signal region
    ax.axvline(sr_low, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(sr_high, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvspan(sr_low, sr_high, alpha=0.2, color="green", label="Signal Region")

    ax.set_xlabel(r"$m_{JJ}$ [TeV]")
    ax.set_ylabel("Normalized counts")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved mJJ distribution plot to {output_path}")


def plot_2d_distribution(
    data: Dict[str, np.ndarray],
    feature_x: str,
    feature_y: str,
    output_path: str,
    xlabel: str,
    ylabel: str,
    bins: int = 50,
    title: Optional[str] = None,
) -> None:
    """
    Plot 2D histogram of two features.

    Args:
        data: Data dictionary
        feature_x: Feature name for x-axis
        feature_y: Feature name for y-axis
        output_path: Path to save the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        bins: Number of bins for each dimension
        title: Optional plot title
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Get data
    x = data[feature_x]
    y = data[feature_y]

    # Plot 2D histogram
    h = ax.hist2d(x, y, bins=bins, cmap="viridis", cmin=1)
    plt.colorbar(h[3], ax=ax, label="Counts")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, color="white", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 2D distribution plot to {output_path}")
