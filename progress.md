# Progress Summary - CATHODE Reproduction Hackathon

**Last Updated**: December 12, 2025

---

## Overall Status

**Current Stage**: Stage 1 Complete ✅ | Moving to Stage 2

**Completion**:
- Stage 1 (BSc - Paper Understanding): **COMPLETE** (25/25 points)
- Stage 2 (MSc - Data Understanding): In Progress
- Stage 3 (PhD - Method Implementation): Pending
- Stage 4 (PostDoc - Paper Reproduction): Pending

**Primary Target**: Reproduce Figure 7 left showing CATHODE performance vs. S/B ratio

---

## Stage 1: Paper Understanding ✅ COMPLETE

### Task 1.1: Extract Basic Information ✅

Successfully extracted all key numerical values from the CATHODE paper:

**Dataset Configuration:**
- Signal Region (SR): mJJ ∈ [3.3, 3.7] TeV
- Sideband (SB): mJJ ∉ [3.3, 3.7] TeV, full range [1.5, 5.5] TeV
- Background events: 1M total, 121,352 in SR
- Signal events: 1,000 total, 772 in SR
- Benchmark S/B = 0.6%, S/√B = 2.2

**Key Result:**
- CATHODE maximum SIC: ~14 (nearly optimal)

### Task 1.2: Extract Information from Plots ✅

Extracted performance metrics from Figures 6 and 7:

**Figure 6 - Benchmark Comparison (S/B = 0.6%):**
- CATHODE: SIC ~14
- CWoLa Hunting: SIC ~11
- ANODE: SIC ~6.5
- Idealized AD: SIC ~14 (CATHODE saturates this!)
- Supervised: SIC ~18

**Figure 7 Left - S/B Scan:**
- CATHODE maintains SIC > 10 down to S/B ~ 0.25%
- Below S/B ~ 0.25%, no method achieves 3σ significance
- CATHODE saturates idealized detector across entire S/B range

### Task 1.3: Summarize the CATHODE Method ✅

**Three Main Steps:**

1. **Conditional Density Estimation**: Train MAF on sideband data to learn p_bg(x|m)
   - 15 MADE blocks, 128 hidden nodes each
   - 500k training, 379k validation events from SB
   - 100 epochs, Adam (lr=10⁻⁴), batch size 256

2. **Interpolation & Sampling**: Generate synthetic background in SR
   - Interpolate MAF into SR using KDE for mJJ distribution
   - Sample 400k synthetic events (10 models × 40k each)

3. **Classification**: Train classifier on SR data vs. synthetic background
   - 3 layers (64-32-1 nodes), ReLU activation
   - 60k SR data training, 60k validation
   - Ensemble of 10 best epochs

**Key Differences from Other Methods:**

*vs. CWoLa Hunting:*
- CATHODE: Generates synthetic background using density estimator → robust to correlations
- CWoLa: Direct SR vs. SB classification → breaks with correlations
- CATHODE can oversample (200k+ events), CWoLa limited to ~65k SB events

*vs. ANODE:*
- CATHODE: One density estimator (outer/background) + classifier
- ANODE: Two density estimators (outer + inner) with explicit likelihood ratio
- ANODE must learn sharp signal peaks (hard), CATHODE uses classifier (easier)

**Result**: CATHODE achieves 2× better performance than ANODE, 1.3× better than CWoLa

---

## Stage 2: Data Understanding (In Progress)

### Planned Tasks:

- [ ] **Task 2.1**: Download and load LHC Olympics R&D dataset
  - Background: zenodo.org/records/4536377
  - Signal: zenodo.org/records/11188685

- [ ] **Task 2.2**: Visualize feature distributions (reproduce Figure 3/4)

- [ ] **Task 2.3**: Define and validate SR/SB regions

- [ ] **Task 2.4**: Implement feature preprocessing pipeline

---

## Key Technical Decisions

### Data Sources (from instructions.md):
```
signal_url = "https://zenodo.org/records/11188685/files/events_anomalydetection_Z_XY_qq_parametric.h5"
background_url = "https://zenodo.org/records/4536377/files/events_anomalydetection_v2.features.h5"
```

**Note**: Signal mass point selection needed - will document choice and rationale

### Features Used:
- mJJ: Dijet invariant mass (for SR/SB definition)
- mJ1: Leading jet mass (lighter jet)
- ΔmJ = mJ2 - mJ1: Jet mass difference
- τ₂₁^J1, τ₂₁^J2: N-subjettiness ratios for both jets

### Project Structure:
```
anomaly-agents/
├── tasks.py                 # All law workflow tasks
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── cathode/            # CATHODE implementation (MAF, classifier)
│   ├── cwola/              # CWoLa baseline
│   └── utils/              # Metrics, plotting, helpers
├── results/
│   ├── plots/              # Generated figures
│   └── models/             # Trained model checkpoints
├── data/                   # Downloaded datasets
└── law.cfg                 # Workflow configuration
```

---

## Key Insights from Paper Analysis

1. **Why CATHODE Works So Well:**
   - Separates the hard problem (learning signal+background) from the easy problem (learning background)
   - Density estimator only learns smooth background (easy)
   - Classifier distinguishes data from synthetic background (easier than density estimation)
   - Oversampling provides more training data for classifier

2. **Robustness to Correlations:**
   - By training density estimator in SB, correlations between x and m in background are properly learned
   - Interpolation automatically handles correlations in SR
   - CWoLa fails because it learns spurious correlations between SR/SB split and features

3. **Nearly Optimal Performance:**
   - CATHODE saturates the idealized anomaly detector (which uses perfect simulation)
   - This is remarkable for a fully simulation-independent method!
   - Suggests density estimator achieves very high-fidelity background modeling

---

## Next Steps

1. ✅ Set up project structure and documentation
2. ✅ Complete Stage 1 paper understanding
3. ⏳ Download and explore datasets (Task 2.1 - NEXT)
4. ⏳ Implement data loading and preprocessing
5. ⏳ Create visualization pipeline
6. Implement CATHODE method components
7. Reproduce key figures (especially Figure 7 left)

---

## Questions / Decisions Needed

- [ ] Which signal mass point to choose from parametric signal dataset?
  - Options: Various mX and mY combinations
  - Need to match paper's mX = 500 GeV, mY = 100 GeV

- [ ] Computational resources available?
  - Training MAF: ~100 epochs on ~500k events
  - Training classifier: ~100 epochs, need to do 10 independent runs
  - Multiple signal injections for Figure 7 scan

---

## Git Commits Made

None yet - will commit after Stage 2 initial implementation

---

## References

- Main paper: `source/2109.00546v3.pdf` (CATHODE)
- Comparison paper: `source/2307.11157v2.pdf`
- Stage 1 detailed results: `stage1_results.md`
- Communication log: `communication.md`
