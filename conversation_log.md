# CATHODE Reproduction Hackathon - Full Conversation Log

## Session Information
- **Date**: December 12, 2024
- **Project**: CATHODE Paper Reproduction for Anomaly Detection Hackathon
- **Goal**: Reproduce Figure 7 (left) showing CATHODE performance vs S/B ratio
- **Primary Paper**: "Classifying Anomalies Through Outer Density Estimation (CATHODE)"

---

## Session Summary

This session continued from a previous conversation that ran out of context. The main accomplishment was successfully training the CATHODE outer density estimator (MAF) on sideband data.

### Key Accomplishments

1. **Task 3.3 Completed**: Trained MAF density estimator
   - 10-epoch training run on 500k sideband background samples
   - Achieved validation loss of 5.0319 (paper benchmark: ~5.3)
   - GPU training confirmed on NVIDIA A100-SXM4-40GB
   - Training history saved with loss curves

2. **Validation Attempt**: Created validation script for sample generation
   - Generated 100 samples at different mJJ values
   - Created comprehensive validation plots
   - **Issue Identified**: Inverse transform not working correctly, generated samples don't match real data distribution

### Training Results

**Best Model Performance:**
- Best Epoch: 9
- Final Train Loss: 4.9171
- Best Validation Loss: 5.0319
- Model saved to: `results/models/maf_best.pt`

**Training Configuration:**
- Architecture: 15 MADE blocks, 128 hidden units
- Batch normalization: momentum 1.0
- Optimizer: Adam, learning rate 1e-4
- Batch size: 256
- Training samples: 500,000
- Validation samples: 370,730

### Files Created/Modified

**New Files:**
1. `train_maf.py` - Standalone training script with GPU support
2. `validate_trained_maf.py` - Validation script for sample generation
3. `training_output_direct.log` - Training output log
4. `results/models/maf_best.pt` - Best trained model checkpoint
5. `results/plots/maf_training_history.png` - Training loss curves
6. `results/plots/trained_maf_validation.png` - Validation plots (with issues)

**Modified Files:**
1. `tasks.py` - Added TrainOuterDensityEstimator law task
2. `src/cathode/maf.py` - Complete MAF implementation (from previous session)

### Known Issues

1. **Sample Generation Issue**: The validation script shows that generated samples are not correctly inverse-transformed back to original scale. The samples appear clustered near zero instead of matching the real data distributions. This needs to be debugged before proceeding to Task 3.4 (generating synthetic background samples).

2. **Law Workflow Dependency**: The law workflow had dependency tracking issues, requiring standalone script approach.

### Project Status

**Completed Tasks:**
- ✅ Stage 1: Paper understanding (Tasks 1.1-1.3) - 25 points
- ✅ Task 2.1: Data loading - 10 points
- ✅ Task 2.2: Feature visualization - 10 points
- ✅ Task 2.3: SR/SB region definition - 10 points
- ✅ Task 2.4: Feature preprocessing - 10 points
- ✅ Task 3.1: MAF implementation - 30 points
- ✅ Task 3.3: Train outer density estimator - 35 points

**Total Points Earned**: 130 points (BSc + MSc + PhD level)

**Pending Tasks:**
- Task 3.4: Generate synthetic background samples (30 points - PhD)
- Task 3.5: Implement CWoLa baseline (20 points - PhD)
- Task 3.6: Implement CATHODE classifier (30 points - PhD)
- Task 4.1: Compute ROC and SIC metrics (20 points - PostDoc)
- Task 4.2: Reproduce Figure 6 (25 points - PostDoc)
- **Task 4.3: Reproduce Figure 7 left - PRIMARY TARGET (35 points - PostDoc)**

---

## Detailed Session Transcript

### 1. Session Continuation

**User Request**: "This session is being continued from a previous conversation that ran out of context."

**Action**: Reviewed summary showing previous work on data loading, preprocessing, and MAF implementation. Last activity was starting MAF training with GPU support.

### 2. Training Status Check

**Action**: Checked on the background training process
- Found training process running (PID 46251)
- GPU actively being used (561 MB memory, 36% utilization)
- Training output not captured due to Python buffering issue

**Waiting Period**: Monitored training progress over ~10 minutes
- Process remained active and stable
- GPU usage confirmed throughout

### 3. Training Completion

**Result**: Training completed successfully after the wait
- 10 epochs completed
- Best validation loss: 5.0319 at epoch 9
- All 10 checkpoints saved
- Training history plot generated

**Verification**:
```
Paper reports loss around ~5.3
Our best val loss: 5.0319
✓ Loss is in expected range!
```

### 4. Sample Generation Validation

**User Request**: "Please generate a few samples with the trained generative model, and make basic validation plots."

**Action**: Created `validate_trained_maf.py` script to:
1. Load trained MAF model
2. Generate samples at different mJJ values (2.0, 2.5, 3.0, 4.0, 4.5 TeV)
3. Compare generated vs real data distributions
4. Create comprehensive validation plots

**Issues Encountered**:
1. Scaler file path correction needed
2. Scaler format issue (saved as dict, not object)
3. CUDA out of memory error during GPU sampling
4. Switched to CPU sampling

**User Request**: "Generate only 100 samples for each mass point"

**Action**: Reduced sample count from 10,000 to 100 to avoid memory issues

### 5. Validation Results (with Issues)

**Observation**: Generated samples show incorrect distributions
- mJ1 values clustered near 0-50 GeV instead of 180-230 GeV
- tau21 values around 25-50 instead of expected 0.52-0.57
- Inverse transform appears to not be working correctly

**Plots Created**:
- Feature distributions at different masses
- 2D scatter plots comparing real vs generated
- Statistical comparisons

### 6. Final Cleanup Request

**User Request**: "Unfortunately we're running out of time. Commit and clean up everything such that all information is contained in the repository. Please also export the whole conversation we had to a md file."

---

## Technical Details

### MAF Architecture
```python
ConditionalMAF(
    features=4,              # mJ1, delta_mJ, tau21_J1, tau21_J2
    context_features=1,      # mJJ
    hidden_features=128,
    num_layers=15,
    use_batch_norm=True,
    batch_norm_momentum=1.0,
    device="cuda"
)
```

### Feature Preprocessing
```python
Scaler Statistics (fitted on 870,730 sideband background events):
- mJ1:      mean=208.22, std=154.69
- delta_mJ: mean=-11.38, std=206.78
- tau21_J1: mean=0.536,  std=0.188
- tau21_J2: mean=0.542,  std=0.188
```

### Signal Region Definition
- SR: mJJ ∈ [3.3, 3.7] TeV
- Sideband: mJJ ∉ SR in range [1.5, 5.5] TeV
- Background in SR: 121,352 events (matches paper benchmark)

---

## Key Code References

### Training Script
Location: [train_maf.py](train_maf.py)
- Loads data from `data/background.h5`
- Trains MAF on sideband background only
- Saves checkpoints and best model
- Generates training history plot

### Validation Script
Location: [validate_trained_maf.py](validate_trained_maf.py)
- Loads trained model and scaler
- Generates samples at different mJJ values
- Creates comparison plots
- **Note**: Has inverse transform issue that needs fixing

### MAF Implementation
Location: [src/cathode/maf.py](src/cathode/maf.py:1)
- Complete ConditionalMAF class
- Uses nflows library for normalizing flows
- Implements MADE-based autoregressive transforms
- Includes fit(), sample(), save(), load() methods

### Data Loader
Location: [src/data/loader.py](src/data/loader.py:1)
- Loads from single `background.h5` file
- Separates by label column (0=background, 1=signal)
- Computes 4-vector invariant masses

### Feature Scaler
Location: [src/data/preprocessing.py](src/data/preprocessing.py:1)
- StandardScaler implementation
- Fitted on sideband background only
- Includes inverse_transform method

---

## Next Steps (for future work)

1. **Fix Sample Generation**:
   - Debug inverse_transform issue
   - Verify generated samples match real data distributions
   - May need to check scaler loading or MAF sampling code

2. **Task 3.4**: Generate synthetic background samples
   - Once sampling is fixed, generate samples in signal region
   - Use for CATHODE method

3. **Task 3.5-3.6**: Implement classifiers
   - CWoLa baseline (signal vs signal region background)
   - CATHODE classifier (signal vs synthetic background)

4. **Task 4.1-4.3**: Reproduce paper results
   - Compute ROC and SIC metrics
   - Reproduce Figure 6 (benchmark performance)
   - **PRIMARY GOAL**: Reproduce Figure 7 left (S/B scan)

---

## Important Notes

- All commits include Claude co-authorship as requested
- Data uses single file approach with label column (0=background, 1=signal)
- Preprocessing scaler fitted on sideband background only (critical for CATHODE)
- Training achieved better loss than paper benchmark (5.03 vs 5.3)
- Law workflow has dependency tracking issues, standalone scripts preferred

---

## Repository Structure

```
/workspaces/anomaly-agents/
├── data/
│   └── background.h5                    # Dataset (1M background, 100k signal)
├── results/
│   ├── models/
│   │   ├── maf_best.pt                 # Best trained model
│   │   ├── feature_scaler.pkl          # Fitted scaler
│   │   └── maf_checkpoints/            # Training checkpoints
│   └── plots/
│       ├── maf_training_history.png    # Training loss curves
│       ├── trained_maf_validation.png  # Validation plots (with issues)
│       └── [other plots from previous session]
├── src/
│   ├── cathode/
│   │   └── maf.py                      # MAF implementation (385 lines)
│   └── data/
│       ├── loader.py                   # Data loading utilities
│       └── preprocessing.py            # Feature scaling
├── tasks.py                             # Law workflow tasks
├── train_maf.py                        # Standalone training script
├── validate_trained_maf.py             # Validation script
├── training_output_direct.log          # Training log
├── instructions.md                      # Hackathon instructions
├── tasks.md                            # Task descriptions
├── communication.md                    # Communication log
├── progress.md                         # Progress tracking
├── law.cfg                             # Law configuration
└── conversation_log.md                 # This file

```

---

## Conclusion

This session successfully completed the training of the CATHODE outer density estimator, achieving validation loss better than the paper benchmark. However, the sample generation validation revealed an issue with inverse transforming standardized samples back to original scale. This needs to be resolved before proceeding with the remaining tasks to reproduce Figure 7.

The project has completed 130 points worth of tasks across BSc, MSc, and PhD levels, with the primary goal (reproducing Figure 7 left) still pending.

---

*Generated by Claude Sonnet 4.5*
*Session Date: December 12, 2024*
