#!/usr/bin/env env python
"""
Validate the trained CATHODE outer density estimator.

This script:
1. Loads the trained MAF model
2. Generates samples at different mJJ values
3. Compares generated samples to real data
4. Creates validation plots
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from pathlib import Path

from src.data.loader import load_data
from src.data.preprocessing import FeatureScaler
from src.cathode.maf import ConditionalMAF

# Configuration
DATA_FILE = "data/background.h5"
SCALER_FILE = "results/models/feature_scaler.pkl"
MODEL_FILE = "results/models/maf_best.pt"
OUTPUT_FILE = "results/plots/trained_maf_validation.png"

# Feature names
FEATURE_NAMES = ["mJ1", "delta_mJ", "tau21_J1", "tau21_J2"]

# Signal region for reference
SR_MIN, SR_MAX = 3.3, 3.7  # TeV

print("=" * 80)
print("CATHODE Trained MAF Validation")
print("=" * 80)
print()

# Load data
print("Loading data...")
bg_data, sig_data = load_data(DATA_FILE)

# Define sideband region (excluding signal region)
sb_mask = (bg_data["mJJ"] < SR_MIN) | (bg_data["mJJ"] > SR_MAX)
sb_data = {key: val[sb_mask] for key, val in bg_data.items()}

print(f"Sideband events: {len(sb_data['mJJ']):,}")

# Load scaler
print(f"\nLoading scaler from {SCALER_FILE}...")
with open(SCALER_FILE, "rb") as f:
    scaler_dict = pickle.load(f)

# Recreate scaler object from dict
scaler = FeatureScaler()
scaler.mean_ = scaler_dict["mean"]
scaler.std_ = scaler_dict["std"]
scaler.feature_names = scaler_dict["feature_names"]

print(f"Scaler loaded successfully")
print(f"Feature means: {scaler.mean_}")
print(f"Feature stds: {scaler.std_}")

# Prepare features
X_sb = np.column_stack([
    sb_data["mJ1"],
    sb_data["delta_mJ"],
    sb_data["tau21_J1"],
    sb_data["tau21_J2"]
])
m_sb = sb_data["mJJ"]

# Standardize
X_sb_std = scaler.transform(X_sb)

# Load trained MAF model
print(f"\nLoading trained MAF from {MODEL_FILE}...")
# Use CPU for sampling to avoid memory issues
device = "cpu"
maf = ConditionalMAF(
    features=4,
    context_features=1,
    hidden_features=128,
    num_layers=15,
    use_batch_norm=True,
    batch_norm_momentum=1.0,
    device=device
)

# Load state dict
checkpoint = torch.load(MODEL_FILE, map_location=device)
maf.flow.load_state_dict(checkpoint["model_state_dict"])
maf.flow.eval()

print(f"Model loaded successfully!")
if 'training_history' in checkpoint:
    history = checkpoint['training_history']
    best_epoch = np.argmin(history['val_loss']) + 1
    print(f"  Best epoch: {best_epoch}")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Best val loss: {min(history['val_loss']):.4f}")

# Generate samples at different mJJ values
print("\nGenerating samples...")
test_masses = [2.0, 2.5, 3.0, 4.0, 4.5]  # TeV
n_samples = 100

generated_samples = {}
for m in test_masses:
    print(f"  Generating {n_samples:,} samples at mJJ = {m} TeV...")
    samples_std = maf.sample(m=np.array([m]), num_samples=n_samples)
    # Inverse transform to original scale
    samples_original = scaler.inverse_transform(samples_std)
    generated_samples[m] = samples_original

# Get real data samples at similar masses for comparison
real_samples = {}
for m in test_masses:
    # Get real data within ±0.1 TeV of target mass
    mask = (np.abs(m_sb - m) < 0.1)
    if np.sum(mask) > 0:
        real_samples[m] = X_sb[mask]
        print(f"  Found {np.sum(mask):,} real events near mJJ = {m} TeV")

# Create validation plots
print("\nCreating validation plots...")
fig = plt.figure(figsize=(20, 12))

# Plot 1: Feature distributions at different masses (mJ1)
ax1 = plt.subplot(2, 4, 1)
for i, m in enumerate(test_masses):
    color = plt.cm.viridis(i / len(test_masses))

    # Real data
    if m in real_samples and len(real_samples[m]) > 0:
        counts, bin_edges = np.histogram(real_samples[m][:, 0], bins=50, density=True, range=(0, 800))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax1.plot(bin_centers, counts, '-', linewidth=2, color=color, alpha=0.7, label=f'Real m={m}')

    # Generated data
    counts, bin_edges = np.histogram(generated_samples[m][:, 0], bins=50, density=True, range=(0, 800))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax1.plot(bin_centers, counts, '--', linewidth=2, color=color, label=f'MAF m={m}')

ax1.set_xlabel("mJ1 [GeV]", fontsize=11)
ax1.set_ylabel("Density", fontsize=11)
ax1.set_title("Leading Jet Mass Distribution", fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# Plot 2: delta_mJ
ax2 = plt.subplot(2, 4, 2)
for i, m in enumerate(test_masses):
    color = plt.cm.viridis(i / len(test_masses))

    if m in real_samples and len(real_samples[m]) > 0:
        counts, bin_edges = np.histogram(real_samples[m][:, 1], bins=50, density=True, range=(-800, 600))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax2.plot(bin_centers, counts, '-', linewidth=2, color=color, alpha=0.7)

    counts, bin_edges = np.histogram(generated_samples[m][:, 1], bins=50, density=True, range=(-800, 600))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax2.plot(bin_centers, counts, '--', linewidth=2, color=color)

ax2.set_xlabel("delta_mJ [GeV]", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("Jet Mass Difference Distribution", fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: tau21_J1
ax3 = plt.subplot(2, 4, 3)
for i, m in enumerate(test_masses):
    color = plt.cm.viridis(i / len(test_masses))

    if m in real_samples and len(real_samples[m]) > 0:
        counts, bin_edges = np.histogram(real_samples[m][:, 2], bins=50, density=True, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax3.plot(bin_centers, counts, '-', linewidth=2, color=color, alpha=0.7)

    counts, bin_edges = np.histogram(generated_samples[m][:, 2], bins=50, density=True, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax3.plot(bin_centers, counts, '--', linewidth=2, color=color)

ax3.set_xlabel("tau21_J1", fontsize=11)
ax3.set_ylabel("Density", fontsize=11)
ax3.set_title("Leading Jet N-subjettiness", fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: tau21_J2
ax4 = plt.subplot(2, 4, 4)
for i, m in enumerate(test_masses):
    color = plt.cm.viridis(i / len(test_masses))

    if m in real_samples and len(real_samples[m]) > 0:
        counts, bin_edges = np.histogram(real_samples[m][:, 3], bins=50, density=True, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax4.plot(bin_centers, counts, '-', linewidth=2, color=color, alpha=0.7)

    counts, bin_edges = np.histogram(generated_samples[m][:, 3], bins=50, density=True, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax4.plot(bin_centers, counts, '--', linewidth=2, color=color)

ax4.set_xlabel("tau21_J2", fontsize=11)
ax4.set_ylabel("Density", fontsize=11)
ax4.set_title("Subleading Jet N-subjettiness", fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5-8: 2D scatter plots comparing real vs generated at one mass
comparison_mass = 3.0  # TeV
n_plot = 100  # subsample for scatter plots

ax5 = plt.subplot(2, 4, 5)
if comparison_mass in real_samples and len(real_samples[comparison_mass]) > 0:
    real_subsample = real_samples[comparison_mass][:n_plot]
    ax5.scatter(real_subsample[:, 0], real_subsample[:, 2], alpha=0.3, s=1, label='Real', c='blue')
gen_subsample = generated_samples[comparison_mass][:n_plot]
ax5.scatter(gen_subsample[:, 0], gen_subsample[:, 2], alpha=0.3, s=1, label='MAF', c='red')
ax5.set_xlabel("mJ1 [GeV]", fontsize=11)
ax5.set_ylabel("tau21_J1", fontsize=11)
ax5.set_title(f"mJ1 vs tau21_J1 (mJJ={comparison_mass} TeV)", fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 4, 6)
if comparison_mass in real_samples and len(real_samples[comparison_mass]) > 0:
    real_subsample = real_samples[comparison_mass][:n_plot]
    ax6.scatter(real_subsample[:, 1], real_subsample[:, 3], alpha=0.3, s=1, label='Real', c='blue')
gen_subsample = generated_samples[comparison_mass][:n_plot]
ax6.scatter(gen_subsample[:, 1], gen_subsample[:, 3], alpha=0.3, s=1, label='MAF', c='red')
ax6.set_xlabel("delta_mJ [GeV]", fontsize=11)
ax6.set_ylabel("tau21_J2", fontsize=11)
ax6.set_title(f"delta_mJ vs tau21_J2 (mJJ={comparison_mass} TeV)", fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

ax7 = plt.subplot(2, 4, 7)
if comparison_mass in real_samples and len(real_samples[comparison_mass]) > 0:
    real_subsample = real_samples[comparison_mass][:n_plot]
    ax7.scatter(real_subsample[:, 2], real_subsample[:, 3], alpha=0.3, s=1, label='Real', c='blue')
gen_subsample = generated_samples[comparison_mass][:n_plot]
ax7.scatter(gen_subsample[:, 2], gen_subsample[:, 3], alpha=0.3, s=1, label='MAF', c='red')
ax7.set_xlabel("tau21_J1", fontsize=11)
ax7.set_ylabel("tau21_J2", fontsize=11)
ax7.set_title(f"tau21_J1 vs tau21_J2 (mJJ={comparison_mass} TeV)", fontsize=12, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3)

# Plot 8: Statistical comparison
ax8 = plt.subplot(2, 4, 8)
feature_idx = 0  # mJ1
stats_real = []
stats_gen = []
masses_for_stats = []

for m in test_masses:
    if m in real_samples and len(real_samples[m]) > 100:
        real_mean = np.mean(real_samples[m][:, feature_idx])
        gen_mean = np.mean(generated_samples[m][:, feature_idx])
        stats_real.append(real_mean)
        stats_gen.append(gen_mean)
        masses_for_stats.append(m)

if len(stats_real) > 0:
    ax8.plot(masses_for_stats, stats_real, 'o-', linewidth=2, markersize=8, label='Real', color='blue')
    ax8.plot(masses_for_stats, stats_gen, 's--', linewidth=2, markersize=8, label='MAF', color='red')
    ax8.set_xlabel("mJJ [TeV]", fontsize=11)
    ax8.set_ylabel("Mean mJ1 [GeV]", fontsize=11)
    ax8.set_title("Mean mJ1 vs mJJ", fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

plt.suptitle("CATHODE Trained MAF Validation: Generated vs Real Data",
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

# Save plot
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
print(f"Validation plot saved to {OUTPUT_FILE}")

# Print statistics
print("\n" + "=" * 80)
print("Sample Statistics Comparison")
print("=" * 80)
for m in test_masses:
    print(f"\nmJJ = {m} TeV:")

    if m in real_samples and len(real_samples[m]) > 0:
        real = real_samples[m]
        print(f"  Real data samples: {len(real):,}")
        print(f"    mJ1:      mean={np.mean(real[:, 0]):.2f}, std={np.std(real[:, 0]):.2f}")
        print(f"    delta_mJ: mean={np.mean(real[:, 1]):.2f}, std={np.std(real[:, 1]):.2f}")
        print(f"    tau21_J1: mean={np.mean(real[:, 2]):.3f}, std={np.std(real[:, 2]):.3f}")
        print(f"    tau21_J2: mean={np.mean(real[:, 3]):.3f}, std={np.std(real[:, 3]):.3f}")

    gen = generated_samples[m]
    print(f"  Generated samples: {len(gen):,}")
    print(f"    mJ1:      mean={np.mean(gen[:, 0]):.2f}, std={np.std(gen[:, 0]):.2f}")
    print(f"    delta_mJ: mean={np.mean(gen[:, 1]):.2f}, std={np.std(gen[:, 1]):.2f}")
    print(f"    tau21_J1: mean={np.mean(gen[:, 2]):.3f}, std={np.std(gen[:, 2]):.3f}")
    print(f"    tau21_J2: mean={np.mean(gen[:, 3]):.3f}, std={np.std(gen[:, 3]):.3f}")

print("\n" + "=" * 80)
print("Validation complete!")
print("=" * 80)
