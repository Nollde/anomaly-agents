# CATHODE Reproduction Project - Current Status

**Last Updated**: December 12, 2024
**Session**: Continued from previous context
**Primary Goal**: Reproduce Figure 7 (left) - CATHODE performance vs S/B ratio

---

## Quick Summary

✅ **Completed**: 130 points across BSc, MSc, and PhD level tasks
⚠️ **Current Issue**: Sample generation validation needs debugging
🎯 **Next Step**: Fix inverse transform, then proceed to Task 3.4

---

## Completed Tasks (130 points)

### Stage 1: Paper Understanding (25 points - BSc)
- ✅ Task 1.1: Extract key numerical values
- ✅ Task 1.2: Extract performance from figures
- ✅ Task 1.3: Summarize CATHODE method

### Stage 2: Data Loading & Preprocessing (40 points - BSc/MSc)
- ✅ Task 2.1: Load LHC Olympics dataset (10 pts)
- ✅ Task 2.2: Visualize feature distributions (10 pts)
- ✅ Task 2.3: Define SR/SB regions (10 pts - MSc)
- ✅ Task 2.4: Feature preprocessing (10 pts - MSc)

### Stage 3: CATHODE Implementation (65 points - PhD)
- ✅ Task 3.1: Implement MAF density estimator (30 pts)
- ✅ Task 3.3: Train outer density estimator (35 pts)

---

## Key Achievements

### 1. MAF Training Success
```
Best Model Performance:
- Epoch: 9/10
- Train Loss: 4.9171
- Validation Loss: 5.0319
- Paper Benchmark: ~5.3
✓ Achieved better loss than paper!
```

### 2. Correct Data Loading
- Uses single file `background.h5` with label column
- Background: 1,000,000 events (label=0)
- Signal: 100,000 events (label=1)
- SR background: 121,352 (matches paper benchmark)

### 3. Proper Preprocessing
- Scaler fitted on 870,730 sideband background events
- Features: mJ1, delta_mJ, tau21_J1, tau21_J2
- Standardized: mean≈0, std≈1
- Excludes mJJ from features (used as conditioning variable)

---

## Pending Tasks (160 points)

### Stage 3: CATHODE Implementation (continued)
- ⏳ Task 3.4: Generate synthetic background samples (30 pts - PhD)
- ⏳ Task 3.5: Implement CWoLa baseline (20 pts - PhD)
- ⏳ Task 3.6: Implement CATHODE classifier (30 pts - PhD)

### Stage 4: Reproduce Results (80 points - PostDoc)
- ⏳ Task 4.1: Compute ROC and SIC metrics (20 pts)
- ⏳ Task 4.2: Reproduce Figure 6 (25 pts)
- 🎯 **Task 4.3: Reproduce Figure 7 left (35 pts)** ← PRIMARY GOAL

---

## Known Issues

### 🔴 Critical: Sample Generation Validation

**Problem**: Generated samples don't match real data distributions

**Symptoms**:
- Generated mJ1: mean≈40 GeV (expected: ≈200 GeV)
- Generated tau21: mean≈40 (expected: ≈0.54)
- Samples clustered near zero in scatter plots

**Likely Cause**: Inverse transform not working correctly

**Evidence**:
```python
# Scaler has inverse_transform method implemented:
def inverse_transform(self, X):
    return X * self.std_ + self.mean_

# But generated samples show wrong statistics
```

**Next Steps**:
1. Debug scaler loading (check mean_ and std_ values)
2. Verify MAF sample output is in standardized space
3. Test inverse_transform separately
4. Check for numpy array shape issues

**Files to Investigate**:
- `validate_trained_maf.py` (line 107: inverse transform call)
- `src/data/preprocessing.py` (line 77-90: inverse_transform method)
- `src/cathode/maf.py` (line 342-345: sample method)

---

## Repository Structure

```
/workspaces/anomaly-agents/
├── data/
│   ├── background.h5              # Main dataset (gitignored, 1.1M events)
│   └── .download_complete         # Download marker
│
├── src/
│   ├── cathode/
│   │   └── maf.py                 # MAF implementation (385 lines)
│   └── data/
│       ├── loader.py              # Data loading with label separation
│       └── preprocessing.py       # FeatureScaler class
│
├── results/
│   ├── models/
│   │   ├── maf_best.pt           # Trained model (epoch 9, val_loss 5.03)
│   │   └── feature_scaler.pkl    # Fitted scaler
│   └── plots/
│       ├── maf_training_history.png        # Training curves
│       ├── trained_maf_validation.png      # Validation plots (with issues)
│       ├── maf_validation.png              # Toy test validation
│       ├── preprocessing_verification.png  # Scaler verification
│       ├── feature_distributions.png       # Raw feature plots
│       └── mjj_distribution.png            # mJJ distribution
│
├── train_maf.py                   # Standalone training script (GPU)
├── validate_trained_maf.py        # Sample generation validation
├── tasks.py                       # Law workflow definitions
├── law.cfg                        # Law configuration
├── conversation_log.md            # Full conversation export
├── PROJECT_STATUS.md              # This file
├── communication.md               # Original communication log
├── progress.md                    # Original progress tracking
├── instructions.md                # Hackathon instructions
└── tasks.md                       # Task descriptions

```

---

## Important Configuration

### Signal Region
```python
SR_MIN, SR_MAX = 3.3, 3.7  # TeV
```

### MAF Architecture
```python
features=4              # mJ1, delta_mJ, tau21_J1, tau21_J2
context_features=1      # mJJ (conditioning variable)
hidden_features=128
num_layers=15          # MADE blocks
use_batch_norm=True
batch_norm_momentum=1.0
```

### Training Hyperparameters
```python
optimizer: Adam
learning_rate: 1e-4
batch_size: 256
epochs: 10 (test), 100 (full)
train_samples: 500,000
val_samples: 370,730
device: cuda (NVIDIA A100-SXM4-40GB)
```

---

## How to Continue

### Immediate Next Steps:

1. **Debug Sample Generation**:
   ```bash
   # Test scaler separately
   python -c "
   import pickle
   import numpy as np
   from src.data.preprocessing import FeatureScaler

   # Load scaler
   scaler = FeatureScaler.load('results/models/feature_scaler.pkl')

   # Test transform/inverse
   X_test = np.array([[200, 0, 0.5, 0.5]])
   X_std = scaler.transform(X_test)
   X_back = scaler.inverse_transform(X_std)

   print('Original:', X_test)
   print('Standardized:', X_std)
   print('Back:', X_back)
   "
   ```

2. **Fix Validation Script**:
   - Once inverse transform works, re-run validation
   - Verify samples match real data distributions
   - Generate larger sample sets (10k per mass point)

3. **Proceed to Task 3.4**:
   - Generate synthetic background samples in SR
   - Use for CATHODE classifier training

### Full Task Sequence:

```
Current Position: Task 3.3 ✅
                      ↓
Fix sampling → Task 3.4 (Generate synthetic backgrounds)
                      ↓
              Task 3.5 (CWoLa baseline)
                      ↓
              Task 3.6 (CATHODE classifier)
                      ↓
              Task 4.1 (ROC and SIC metrics)
                      ↓
              Task 4.2 (Figure 6)
                      ↓
              Task 4.3 (Figure 7 left) 🎯 PRIMARY GOAL
```

---

## Key References

### Paper
"Classifying Anomalies THrough Outer Density Estimation (CATHODE)"

### Important Findings
- CATHODE maintains SIC > 10 for S/B ≥ 0.3%
- Nearly saturates idealized detector performance
- Significantly outperforms CWoLa and ANODE baselines

### Critical Implementation Details
1. **Scaler must be fitted on sideband background only** (no SR, no signal)
2. **MAF conditions on mJJ** but doesn't include it in features
3. **Batch norm momentum = 1.0** (different from typical 0.9)
4. **Generate in SR for CATHODE**, but CWoLa uses real SR background

---

## Contact Information

**Project**: CATHODE Reproduction Hackathon
**Documentation**: See `conversation_log.md` for full session details
**Issues**: Check validation plots and training logs for debugging

---

*Last commit: Complete Task 3.3*
*Status: Ready for sample generation debugging*
*Progress: 130/290 points (45%)*

🤖 Generated with Claude Code
