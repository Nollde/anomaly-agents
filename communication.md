# Communication Log - CATHODE Reproduction Hackathon

## Session 1: Project Setup and Initialization

**Date**: December 12, 2025

### Initial Setup

**User**: Please follow the instructions in `instructions.md`

**AI**: Understood! I've read the instructions and tasks. This is a hackathon project to reproduce the CATHODE (Classifying Anomalies THrough Outer Density Estimation) paper results. The main goal is to reproduce Figure 7 left, which shows performance vs signal strength.

**AI Actions**:
1. Read instructions.md and tasks.md to understand the full scope
2. Created comprehensive todo list with 15 main tasks across all stages
3. Setting up project structure:
   - Created directory structure: src/{data,cathode,cwola,utils}, results/{plots,models}
   - Created law.cfg for workflow management configuration
   - Created setup.sh for environment variable setup
   - Created this communication.md file to track our conversation

**Next Steps**:
- Complete project setup
- Read CATHODE paper and extract key information (Stage 1 tasks)
- Download and process the LHC Olympics dataset (Stage 2 tasks)
- Implement the CATHODE method (Stage 3 tasks)
- Reproduce the key figures (Stage 4 tasks)

---

### Stage 1 Completion: Paper Understanding

**AI Progress Update**:

Completed all three Stage 1 tasks (BSc level - 25 points):

1. **Task 1.1** ✅: Extracted key numerical values:
   - SR definition: mJJ ∈ [3.3, 3.7] TeV
   - Benchmark: 1M background, 1k signal → S/B = 0.6%, S/√B = 2.2
   - CATHODE max SIC: ~14

2. **Task 1.2** ✅: Extracted performance from figures:
   - Figure 6: CATHODE (14) >> CWoLa (11) >> ANODE (6.5)
   - Figure 7: CATHODE maintains SIC > 10 down to S/B ~ 0.25%

3. **Task 1.3** ✅: Summarized CATHODE method:
   - 3 steps: Density estimation → Sampling → Classification
   - Key advantage: Separates background learning (easy) from signal detection (hard)
   - Nearly saturates optimal performance (idealized detector)

**Deliverables Created**:
- `stage1_results.md`: Detailed extraction of all paper information
- `progress.md`: Comprehensive progress tracking with insights
- Project structure: src/, results/, tasks.py, law.cfg, setup.sh

**Key Insight**: CATHODE achieves near-optimal anomaly detection by combining the best aspects of density estimation (ANODE) and classification (CWoLa), while being robust to feature-mass correlations.

**Ready for Stage 2**: Data loading and visualization

---

### Stage 2 Progress: Data Loading and Visualization

**User**: Requested to continue with plotting after data was manually downloaded.

**AI Progress Update**:

Completed Tasks 2.1 and 2.2 (BSc level):

**Task 2.1** ✅: Data loading
- Fixed signal file structure issue: Uses 'output' key instead of 'df'
- Fixed mass point selection: Signal file has lowercase 'mx'/'my' columns
- Selected mass point: mX=500 GeV, mY=100 GeV
- Available mass points: mX ∈ [50, 600] GeV, mY ∈ [50, 750] GeV
- Signal events: 77,575 total, 58,176 in SR [3.3, 3.7] TeV
- Background events: 1.1M total, 196,651 in SR
- Computed S/B ratio in SR: ~29.6% (higher than paper's 0.6% benchmark)

**Task 2.2** ✅: Feature visualization
- Created plotting utilities in [src/utils/plotting.py](src/utils/plotting.py)
- Implemented VisualizeFeatures law task
- Generated feature distribution plots comparing background and signal
- Generated mJJ distribution with SR highlighted
- Plots saved to [results/plots/](results/plots/)

**Key Findings**:
- Signal clearly separable in mJ1, delta_mJ, and tau21 features
- High S/B ratio at chosen mass point will make detection easier
- Feature distributions match expected physics patterns

**Deliverables**:
- Data loader with parametric mass point selection
- Visualization utilities for feature comparisons
- Feature distribution and mJJ plots

**Committed**: Data loading and visualization implementation with proper formatting

---

### Stage 2 Completion: Preprocessing Verification

**User Request**: "Before moving on, please verify that the feature scaler works as expected. We'd especially like to see plots of the preprocessed features."

**AI Progress Update**:

**Task 2.4** ✅: Feature preprocessing verification completed
- Created VisualizePreprocessing law task in [tasks.py](tasks.py:290-377)
- Ran standalone verification script (law dependency tracking workaround)
- Generated before/after standardization plots: [results/plots/preprocessing_verification.png](results/plots/preprocessing_verification.png)
- Verified perfect standardization on sideband background (870,730 events):

```
Original features (SB background):
  mJ1         : mean= 208.216, std= 154.689
  delta_mJ    : mean= -11.379, std= 206.785
  tau21_J1    : mean=   0.536, std=   0.188
  tau21_J2    : mean=   0.542, std=   0.188

Standardized features (should be mean≈0, std≈1):
  mJ1         : mean=   0.000, std=   1.000
  delta_mJ    : mean=   0.000, std=   1.000
  tau21_J1    : mean=  -0.000, std=   1.000
  tau21_J2    : mean=   0.000, std=   1.000
```

**Key Findings**:
- Feature scaler works perfectly: all features have mean≈0, std≈1
- Standardization correctly fitted on sideband background only
- Plots show clear Gaussian-like distributions after standardization
- Ready for density estimation (Stage 3)

**Stage 2 Complete** ✅: All BSc level tasks (2.1-2.4) successfully finished

**Next**: Stage 3 - CATHODE Implementation (MSc Level)

---

### Stage 1 Corrections

**User Feedback**:
- Corrected Task 1.2 interpretation of Figure 7 left:
  - CATHODE maintains SIC > 10 for S/B ≳ 0.3% (not down to 0.25%)
  - Below S/B ~ 0.25%, all methods fail to reach 3σ
- Clarified correlation robustness statement:
  - CATHODE is "expected to be" robust (by design), then empirically validated
  - Not perfectly robust, but much better than CWoLa which breaks completely

**AI Actions**:
- Updated stage1_results.md with accurate figure readings
- Clarified language about robustness claims
- Ready to proceed to Stage 2

---
