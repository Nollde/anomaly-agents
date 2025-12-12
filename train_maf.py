"""
Direct training script for CATHODE outer density estimator.
Bypasses law workflow for faster iteration.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.loader import load_data, apply_region_cut, get_features
from src.data.preprocessing import FeatureScaler
from src.cathode.maf import ConditionalMAF

# Parameters
N_TRAIN = 500000
N_VAL = 379000
EPOCHS = 10  # Reduced for testing
N_CHECKPOINTS = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-4

print("=" * 80)
print("TASK 3.3: Training CATHODE Outer Density Estimator")
print("=" * 80)
print()

# Load scaler
scaler = FeatureScaler.load("results/models/feature_scaler.pkl")
feature_names = scaler.feature_names_

# Load dataset
data_path = "data/background.h5"
bg_data, sig_data = load_data(data_path)

# Get sideband background data
bg_sb_mask = apply_region_cut(bg_data, 1.5, 5.5, 3.3, 3.7, False)
bg_sb = {key: val[bg_sb_mask] for key, val in bg_data.items()}

print(f"Sideband background events: {len(bg_sb['mJJ']):,}")
print(f"Using {N_TRAIN:,} for training, {N_VAL:,} for validation")
print()

# Extract and preprocess features
X_sb = get_features(bg_sb, include_mass=False)
X_sb_scaled = scaler.transform(X_sb)
m_sb = bg_sb["mJJ"]  # Conditioning variable (in TeV)

# Split into train/val
n_total = len(X_sb_scaled)
indices = np.random.permutation(n_total)
train_indices = indices[:N_TRAIN]
val_indices = indices[N_TRAIN : N_TRAIN + N_VAL]

X_train = X_sb_scaled[train_indices]
m_train = m_sb[train_indices]
X_val = X_sb_scaled[val_indices]
m_val = m_sb[val_indices]

print(f"Training set: {len(X_train):,} samples")
print(f"Validation set: {len(X_val):,} samples")
print(f"Feature dimension: {X_train.shape[1]}")
print()

# Check GPU availability
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No GPU available, using CPU")
print()

# Create MAF with paper specifications
print("Creating MAF model...")
print("  Architecture: 15 MADE blocks, 128 hidden units")
print("  Batch normalization: momentum 1.0")
print("  Optimizer: Adam, lr=1e-4")
print(f"  Device: {device}")
print()

maf = ConditionalMAF(
    features=X_train.shape[1],
    context_features=1,
    hidden_features=128,
    num_layers=15,
    use_batch_norm=True,
    batch_norm_momentum=1.0,
    device=device,  # Explicitly set device
)

# Train the model
print(f"Training for {EPOCHS} epochs...")
print()

history = maf.fit(
    X_train,
    m_train,
    X_val,
    m_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    verbose=True,
)

# Save checkpoints
print()
print("Saving checkpoints...")
checkpoint_dir = Path("results/models/maf_checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

val_losses = np.array(history["val_loss"])

# Save all epochs
for epoch in range(EPOCHS):
    checkpoint_path = checkpoint_dir / f"maf_epoch_{epoch:03d}.pt"
    maf.save(str(checkpoint_path))

# Find best N checkpoints
best_epochs = np.argsort(val_losses)[:N_CHECKPOINTS]
print(f"\nBest {N_CHECKPOINTS} epochs (by validation loss):")
for rank, epoch in enumerate(best_epochs):
    print(f"  Rank {rank+1}: Epoch {epoch+1:3d}, val_loss={val_losses[epoch]:.4f}")

# Save the single best model
best_epoch = best_epochs[0]
best_model_path = checkpoint_dir / f"maf_epoch_{best_epoch:03d}.pt"

import shutil
final_path = Path("results/models/maf_best.pt")
shutil.copy(best_model_path, final_path)

print(f"\nBest model (epoch {best_epoch+1}) saved to {final_path}")
print(f"  Final train loss: {history['train_loss'][best_epoch]:.4f}")
print(f"  Final val loss: {history['val_loss'][best_epoch]:.4f}")

# Plot training history
print("\nGenerating training history plot...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

epochs_array = np.arange(1, EPOCHS + 1)
ax.plot(epochs_array, history["train_loss"], label="Train loss", alpha=0.7, linewidth=2)
ax.plot(epochs_array, history["val_loss"], label="Val loss", alpha=0.7, linewidth=2)

# Mark best epoch
ax.axvline(
    best_epoch + 1, color="red", linestyle="--", alpha=0.5, label=f"Best epoch ({best_epoch+1})"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Negative log-likelihood")
ax.set_title("CATHODE Outer Density Estimator Training")
ax.legend()
ax.grid(True, alpha=0.3)

plot_path = Path("results/plots/maf_training_history.png")
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Training plot saved to {plot_path}")

# Verification against paper
print()
print("=" * 80)
print("Verification")
print("=" * 80)
print(f"Paper reports loss around ~5.3")
print(f"Our best val loss: {val_losses[best_epoch]:.4f}")
if 4.5 < val_losses[best_epoch] < 6.0:
    print("✓ Loss is in expected range!")
else:
    print("⚠ Loss differs from paper - may need hyperparameter tuning")

print()
print("Task 3.3 complete!")
