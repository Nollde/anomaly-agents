"""
Quick test of MAF implementation on toy data.

This script tests the MAF on a simple 2D dataset to verify the implementation works.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.cathode.maf import ConditionalMAF

# Generate toy data: x ~ N(m, 1) where m is the conditioning variable
np.random.seed(42)

# Create training data
n_train = 10000
m_train = np.random.uniform(0, 5, n_train)  # Conditioning variable
# 2D features: each dimension centered at m with std=1
X_train = np.column_stack([
    np.random.normal(m_train, 1.0),
    np.random.normal(m_train, 1.0)
])

# Create validation data
n_val = 2000
m_val = np.random.uniform(0, 5, n_val)
X_val = np.column_stack([
    np.random.normal(m_val, 1.0),
    np.random.normal(m_val, 1.0)
])

print("=" * 80)
print("Testing MAF on toy data")
print("=" * 80)
print(f"Training data: {n_train} samples, {X_train.shape[1]} features")
print(f"Validation data: {n_val} samples")
print()

# Create MAF with smaller architecture for testing
maf = ConditionalMAF(
    features=2,  # 2D features
    context_features=1,  # Conditioning on 1D variable (m)
    hidden_features=32,  # Smaller for quick testing
    num_layers=3,  # Fewer layers for quick testing
    use_batch_norm=True,
    batch_norm_momentum=1.0,
)

print(f"MAF architecture:")
print(f"  Features: {maf.features}")
print(f"  Context features: {maf.context_features}")
print(f"  Hidden features: {maf.hidden_features}")
print(f"  Num layers: {maf.num_layers}")
print(f"  Device: {maf.device}")
print()

# Train MAF
print("Training MAF...")
history = maf.fit(
    X_train,
    m_train,
    X_val,
    m_val,
    epochs=50,
    batch_size=256,
    learning_rate=1e-3,
    verbose=True,
)

print()
print("Training complete!")
print(f"Final train loss: {history['train_loss'][-1]:.4f}")
print(f"Final val loss: {history['val_loss'][-1]:.4f}")
print()

# Test sampling
print("Testing sampling...")
m_test = np.array([1.0, 2.0, 3.0])
samples = maf.sample(m_test, num_samples=1000)
print(f"Generated {len(samples)} samples conditioned on m={m_test}")
print()

# Test log probability
print("Testing log probability...")
log_probs = maf.log_prob(X_val[:10], m_val[:10])
print(f"Log probs for 10 validation samples: {log_probs}")
print()

# Plot training history
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(history["train_loss"], label="Train loss", alpha=0.7)
ax.plot(history["val_loss"], label="Val loss", alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Negative log-likelihood")
ax.set_title("MAF Training History (Toy Data)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/maf_toy_test.png", dpi=150, bbox_inches="tight")
print("Saved training history plot to results/plots/maf_toy_test.png")

# Test save/load
print()
print("Testing save/load...")
maf.save("results/models/maf_toy_test.pt")
print("Saved model to results/models/maf_toy_test.pt")

maf_loaded = ConditionalMAF.load("results/models/maf_toy_test.pt")
print("Loaded model successfully")

# Verify loaded model gives same results
log_probs_loaded = maf_loaded.log_prob(X_val[:10], m_val[:10])
diff = np.abs(log_probs - log_probs_loaded).max()
print(f"Max difference in log probs: {diff:.2e}")
assert diff < 1e-5, "Loaded model gives different results!"
print("Save/load test passed!")

print()
print("=" * 80)
print("All tests passed!")
print("=" * 80)
