# Task Breakdown for CATHODE Reproduction

This document contains all tasks needed to reproduce the CATHODE paper results, organized by difficulty level.

---

## Stage 1: Paper Understanding (BSc Level)

### Task 1.1: Extract Basic Information from Text
**Difficulty**: BSc
**Weight**: 5 points
**Description**: Extract key numerical values and facts from the CATHODE paper text.

**Requirements**:
- Extract the signal region (SR) definition in mJJ
- Extract the number of background events in the LHCO R&D dataset
- Extract the number of signal events used
- Extract the S/B ratio and S/√B for the benchmark
- Extract the maximum SIC achieved by CATHODE

**Deliverable**: A summary table or structured output with these values

---

### Task 1.2: Extract Information from Plots
**Difficulty**: BSc
**Weight**: 10 points
**Description**: Extract numerical information from the paper's figures.

**Requirements**:
- From Figure 6 (right): Extract the maximum SIC value for CATHODE at the benchmark point
- From Figure 6: Extract approximate SIC values for CWoLa Hunting and ANODE
- From Figure 7 (left): Identify the S/B range where CATHODE maintains >10 SIC

**Deliverable**: Extracted values with figure references

---

### Task 1.3: Summarize the CATHODE Method
**Difficulty**: BSc
**Weight**: 10 points
**Description**: Provide a clear summary of how CATHODE works.

**Requirements**:
- Describe the three main steps of CATHODE
- Explain what makes it different from CWoLa Hunting
- Explain what makes it different from ANODE
- Describe the role of the density estimator

**Deliverable**: A concise method summary (1-2 paragraphs)

---

## Stage 2: Data Understanding (MSc Level)

### Task 2.1: Download and Load the LHC Olympics R&D Dataset
**Difficulty**: MSc
**Weight**: 15 points
**Description**: Access the LHC Olympics 2020 R&D dataset and load it into memory.

**Requirements**:
- Download the dataset from Zenodo (doi: 10.5281/zenodo.4536377)
- Load both background (QCD dijet) and signal (W' → XY) events
- Verify the dataset contains 1,000,000 background and 100,000 signal events
- Extract particle-level information (pT, η, φ) for up to 700 particles per event

**Deliverable**: A law task that downloads and loads the data, with basic statistics printed

---

### Task 2.2: Reconstruct Jets and Event Features
**Difficulty**: MSc
**Weight**: 20 points
**Description**: Reconstruct large-radius jets and compute the 5 auxiliary features used in the analysis.

**Requirements**:
- Cluster particles into R=1 anti-kT jets using FastJet or equivalent
- Apply pT > 1.2 TeV trigger requirement
- Compute the 5 features for each event:
  - mJ1: mass of the lighter jet
  - ∆mJ: difference in jet masses (mJ2 - mJ1)
  - τ21^J1: n-subjettiness ratio for jet 1
  - τ21^J2: n-subjettiness ratio for jet 2
  - mJJ: invariant mass of the dijet system
- Store features in a structured format (e.g., pandas DataFrame or HDF5)

**Deliverable**: A law task that produces a file with all reconstructed features

---

### Task 2.3: Visualize Feature Distributions
**Difficulty**: MSc
**Weight**: 15 points
**Description**: Create plots of the feature distributions matching Figure 3 and Figure 4 from the paper.

**Requirements**:
- Plot 1D histograms for all 5 features + mJJ
- Show separate distributions for background and signal
- Recreate Figure 3 (6-panel plot with background, signal, and simulation)
- Use proper normalization (events in arbitrary units)
- Save plots in a results directory

**Deliverable**: Plots matching the paper's Figure 3, plus a law task that generates them

---

### Task 2.4: Define Signal and Sideband Regions
**Difficulty**: MSc
**Weight**: 10 points
**Description**: Split the dataset into signal region (SR) and sideband (SB) based on mJJ.

**Requirements**:
- Define SR: mJJ ∈ [3.3, 3.7] TeV
- Define SB: mJJ ∉ [3.3, 3.7] TeV (using full range [1.5, 5.5] TeV)
- Count events in SR vs SB for background and signal
- Verify ~121k background and ~772 signal events in SR (matching paper)

**Deliverable**: Code that splits data into SR/SB with event counts validation

---

### Task 2.5: Feature Preprocessing
**Difficulty**: MSc
**Weight**: 10 points
**Description**: Implement the feature transformations used in CATHODE.

**Requirements**:
- Scale features to (0,1) range
- Apply logit transformation: logit(x) = ln(x/(1-x))
- Standardize features (subtract mean, divide by std)
- Use SB statistics for the transformation
- Implement inverse transformations

**Deliverable**: Preprocessing utilities in `src/data/preprocessing.py`

---

## Stage 3: Method Understanding (PhD Level)

### Task 3.1: Implement Masked Autoregressive Flow (MAF)
**Difficulty**: PhD
**Weight**: 30 points
**Description**: Implement or integrate a MAF density estimator for conditional density estimation.

**Requirements**:
- Implement MAF with affine transformations or use existing library (PyTorch)
- Use 15 MADE blocks with 128 hidden nodes each
- Implement batch normalization with momentum 1.0
- Use unit normal base distribution
- Train with Adam optimizer (lr=10^-4, batch size=256)

**Deliverable**: MAF implementation in `src/cathode/maf.py` with training code

---

### Task 3.2: Train Outer Density Estimator on Toy Data
**Difficulty**: PhD
**Weight**: 25 points
**Description**: Test the MAF on simple toy data before applying to real physics data.

**Requirements**:
- Generate 2D toy dataset (e.g., mixture of Gaussians) with conditional variable
- Train MAF to learn conditional density p(x|m)
- Validate that samples from trained MAF match training distribution
- Visualize learned density vs. true density
- Track training and validation loss

**Deliverable**: Notebook or script demonstrating MAF on toy data

---

### Task 3.3: Implement CATHODE Outer Density Estimator
**Difficulty**: PhD
**Weight**: 35 points
**Description**: Train the conditional density estimator on sideband data.

**Requirements**:
- Train MAF on SB data (500k training, ~379k validation)
- Condition on mJJ using kernel density estimate (KDE)
- Train for 100 epochs
- Implement model selection: keep 10 epochs with lowest validation loss
- Plot training and validation loss curves (reproduce Figure 3)
- Verify loss values match paper (~5.3 range)

**Deliverable**: Law task that trains outer density estimator with checkpoints

---

### Task 3.4: Interpolate and Sample Background Events
**Difficulty**: PhD
**Weight**: 30 points
**Description**: Interpolate the learned density into SR and generate synthetic background samples.

**Requirements**:
- Use trained MAF to sample in SR by querying at mJJ ∈ [3.3, 3.7] TeV
- Use KDE fit to mJJ distribution in SR for sampling m values
- Generate 400k samples (10 models × 40k samples each as ensemble)
- Apply inverse preprocessing to get physical feature values
- Validate samples match SR background distribution (reproduce Figure 4)

**Deliverable**: Law task that generates synthetic background samples

---

### Task 3.5: Implement CWOLA Binary Classifier (for comparison)
**Difficulty**: PhD
**Weight**: 20 points
**Description**: Implement the CWoLa Hunting baseline for comparison.

**Requirements**:
- Implement 3-layer network (64-32-1 nodes) with ReLU activation
- Train to distinguish SR data from Short Sideband (SSB) data
- SSB: 200 GeV strips adjacent to SR (mJJ ∈ [3.1, 3.3] ∪ [3.7, 3.9] TeV)
- Use binary cross-entropy loss
- Train with Adam (lr=10^-3, batch size=128, 100 epochs)
- Use 5-fold cross-validation

**Deliverable**: CWoLa implementation in `src/cwola/classifier.py`

---

### Task 3.6: Implement CATHODE Classifier
**Difficulty**: PhD
**Weight**: 35 points
**Description**: Train the final classifier to distinguish data from synthetic background.

**Requirements**:
- Use same architecture as CWoLa (3 layers: 64-32-1)
- Train to distinguish SR data (60k events) from synthetic background samples (200k)
- Reweight classes to contribute equally to loss
- Implement 5-fold cross-validation, keep best validation loss
- Ensemble predictions from 10 best epochs
- Track training curves (reproduce Figure 5)

**Deliverable**: Law task that trains CATHODE classifier with ensembling

---

## Stage 4: Paper Reproduction (PostDoc Level)

### Task 4.1: Compute ROC and SIC Curves
**Difficulty**: PostDoc
**Weight**: 25 points
**Description**: Evaluate classifier performance using ROC and significance improvement characteristic.

**Requirements**:
- Implement ROC curve calculation (TPR vs FPR)
- Implement SIC calculation: SIC = S/√B as function of signal efficiency
- Compute on held-out test set (340k background + 20k signal)
- Generate curves with uncertainty bands (from 10 independent trainings)
- Compare CATHODE, CWoLa, ANODE, idealized detector

**Deliverable**: Performance evaluation code in `src/cathode/metrics.py`

---

### Task 4.2: Reproduce Figure 6 (Benchmark Performance)
**Difficulty**: PostDoc
**Weight**: 40 points
**Description**: Reproduce the main performance comparison at S/B = 0.6%.

**Requirements**:
- Train all methods (CATHODE, CWoLa, ANODE) on 1000 signal + 1M background
- Verify SR contains 772 signal + 121k background (S/B = 0.6%, S/√B = 2.2)
- Generate ROC curves (Figure 6 left)
- Generate SIC curves (Figure 6 right)
- Verify CATHODE achieves max SIC ~14
- Verify CWoLa achieves max SIC ~11
- Verify ANODE achieves max SIC ~6.5
- Include uncertainty bands from 10 retrainings

**Deliverable**: Reproduction of Figure 6 with all methods

---

### Task 4.3: Reproduce Figure 7 Left (S/B Scan)
**Difficulty**: PostDoc
**Weight**: 50 points
**Description**: **PRIMARY TARGET** - Reproduce the key result showing performance vs signal strength.

**Requirements**:
- Scan over different signal injections: 0, 300, 500, 750, 1000, 1200, 1500, 2000, 2500, 3000 events
- For each point, train CATHODE and baselines with 10 different data realizations
- Compute maximum SIC for each method at each S/B
- Plot max(SIC) vs S/B (Figure 7 left)
- Plot maximum achieved significance vs S/B (Figure 7 right)
- Verify CATHODE saturates idealized detector performance
- Show CATHODE maintains >10 SIC down to S/B ~ 0.25%
- Include proper uncertainty bands

**Deliverable**: **Main hackathon result** - Figure 7 left reproduction

---

### Task 4.4: Reproduce Figure 8 (Correlation Robustness)
**Difficulty**: PostDoc
**Weight**: 30 points
**Description**: Test CATHODE's robustness to feature-mass correlations.

**Requirements**:
- Apply artificial correlation: mJ1 → mJ1 + 0.1·mJJ, ∆m → ∆m + 0.1·mJJ
- Retrain all methods on shifted dataset
- Compute SIC curves for shifted data
- Show CWoLa completely breaks down
- Show CATHODE maintains good performance
- Reproduce Figure 8 (left: SIC curves, right: ratio to unshifted)

**Deliverable**: Correlation robustness study matching Figure 8

---

### Task 4.5: Study Oversampling Benefits
**Difficulty**: PostDoc
**Weight**: 20 points
**Description**: Investigate the effect of generating different numbers of synthetic samples.

**Requirements**:
- Train CATHODE with 60k, 200k, and 800k synthetic samples
- Compare performance (reproduce Figure 9 left)
- Show performance saturates around 200k samples
- Compare with CWoLa Hunting (limited to ~65k SB events)

**Deliverable**: Oversampling study matching Figure 9

---

### Task 4.6: Background Estimation Study
**Difficulty**: PostDoc
**Weight**: 15 points
**Description**: Demonstrate that CATHODE doesn't sculpt the mass spectrum.

**Requirements**:
- Apply CATHODE classifier to background-only data
- Select events at different efficiency thresholds (20%, 5%)
- Plot mJJ distributions for selected events (Figure 10 left)
- Show no artificial bumps or features introduced
- Compute ratio of synthetic samples to data passing cuts (Figure 10 right)
- Verify ratio ~1.0 (unbiased background estimate)

**Deliverable**: Background sculpting study matching Figure 10

---

## Stage 5: Future Work (Professor Level - Optional)

### Task 5.1: Implement ANODE Method
**Difficulty**: Professor
**Weight**: 40 points (bonus)
**Description**: Implement the ANODE baseline for direct comparison.

**Requirements**:
- Train both inner (SR) and outer (SB) density estimators
- Compute likelihood ratio explicitly
- Compare performance with CATHODE

---

### Task 5.2: Implement Additional Methods (SALAD, CURTAINS, FETA)
**Difficulty**: Professor
**Weight**: 60 points (bonus)
**Description**: Implement the other three methods from the comparison paper.

**Requirements**:
- SALAD: Simulation-assisted likelihood-free anomaly detection
- CURTAINS: Normalizing flow transport between sidebands
- FETA: Flow-enhanced transportation using simulation
- Reproduce results from interplay paper (2307.11157)

---

## Summary

**Total Points Available**:
- Stage 1 (BSc): 25 points
- Stage 2 (MSc): 70 points
- Stage 3 (PhD): 175 points
- Stage 4 (PostDoc): 180 points
- **Total Core**: 450 points
- Stage 5 (Professor - Bonus): 100 points

**Critical Path for Hackathon**:
- **Must complete**: Tasks 4.3 (Figure 7 left) - the primary replication target
- **Should complete**: Tasks 4.1, 4.2 (performance evaluation and Figure 6)
- **Nice to have**: Tasks 4.4, 4.5, 4.6 (robustness studies)

**Time Estimates**:
- Stage 1: 1-2 hours
- Stage 2: 3-4 hours
- Stage 3: 6-8 hours
- Stage 4: 6-10 hours
- **Total**: 16-24 hours for full reproduction
