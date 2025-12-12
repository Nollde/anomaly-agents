# Stage 1: Paper Understanding Results

**Completed**: December 12, 2025

## Task 1.1: Extract Basic Information from Text

| Parameter | Value | Source |
|-----------|-------|--------|
| **Signal Region (SR) definition** | mJJ ∈ [3.3, 3.7] TeV | Page 3 |
| **Sideband (SB) definition** | mJJ ∉ [3.3, 3.7] TeV (full range [1.5, 5.5] TeV) | Page 3 |
| **Total background events (mock data)** | 1,000,000 events | Page 3 |
| **Background events in SR** | 121,352 events | Page 3 |
| **Total signal events injected** | 1,000 events | Page 3 |
| **Signal events in SR** | 772 events | Page 3 |
| **S/B ratio (benchmark)** | 0.6% (6 × 10⁻³) | Page 3 |
| **S/√B (benchmark)** | 2.2 | Page 3 |
| **Maximum SIC achieved by CATHODE** | ~14 | Page 6 |

## Task 1.2: Extract Information from Plots

### Figure 6 - Benchmark Performance (S/B = 0.6%)

| Method | Maximum SIC | Source |
|--------|-------------|--------|
| **CATHODE** | ~14 | Page 6, Figure 6 right |
| **CWoLa Hunting** | ~11 | Page 6 |
| **ANODE** | ~6.5 | Page 6 |
| **Idealized Anomaly Detector** | ~14 (similar to CATHODE) | Page 6 |
| **Fully Supervised** | ~18 | Figure 6 right |

**Key Finding**: CATHODE nearly saturates the idealized anomaly detector performance!

### Figure 7 Left - S/B Scan

| S/B Range | CATHODE Performance |
|-----------|---------------------|
| **S/B > 0.25%** | Maximum SIC > 10 |
| **S/B ~ 0.25%** | Performance threshold where methods start to fail |
| **S/B < 0.25%** | No method achieves 3σ significance |

**Key Finding**: CATHODE maintains >10 SIC down to S/B ~ 0.25%, and saturates the idealized detector performance across the entire S/B range.

## Task 1.3: Summarize the CATHODE Method

### Three Main Steps of CATHODE

1. **Conditional Density Estimation (Outer Estimator)**
   - Train a Masked Autoregressive Flow (MAF) on sideband (SB) data
   - Learn the background distribution: p_data(x|m ∉ SR) ≈ p_bg(x|m ∉ SR)
   - Uses 500,000 training events, 378,876 validation events from SB
   - Architecture: 15 MADE blocks with 128 hidden nodes each
   - Training: 100 epochs, Adam optimizer, learning rate 10⁻⁴, batch size 256

2. **Interpolation and Sampling**
   - Interpolate the trained MAF into the signal region (SR)
   - Sample synthetic background events in SR from the interpolated density
   - Use kernel density estimate (KDE) to match the mJJ distribution of SR data
   - Generate 400,000 synthetic events (ensemble of 10 models × 40,000 samples)
   - These samples follow p_bg(x|m ∈ SR)

3. **Classification**
   - Train a binary classifier to distinguish SR data from synthetic background samples
   - Architecture: 3 hidden layers (64-32-1 nodes) with ReLU activation
   - Training: 100 epochs, Adam optimizer, learning rate 10⁻³, batch size 128
   - Uses 60,000 SR data events for training, 60,000 for validation
   - Ensemble predictions from 10 best validation epochs
   - Classifier learns R(x) = p_data(x)/p_bg(x) (likelihood ratio)

### Difference from CWoLa Hunting

| Aspect | CATHODE | CWoLa Hunting |
|--------|---------|---------------|
| **Approach** | Generates synthetic background in SR using density estimator | Directly trains classifier on SR vs. Short Sideband (SSB) |
| **Background events** | Can oversample (200k+ synthetic events) | Limited to actual SB events (~65k) |
| **Correlation sensitivity** | Robust to x-m correlations (uses SB to learn background) | Very sensitive to x-m correlations |
| **Performance** | Max SIC ~14 | Max SIC ~11 |

**Key insight**: By using a density estimator trained on the sideband, CATHODE avoids learning correlations between features x and mass m that would degrade CWoLa Hunting.

### Difference from ANODE

| Aspect | CATHODE | ANODE |
|--------|---------|-------|
| **Density estimators needed** | One (outer/background only) | Two (outer + inner) |
| **What is learned** | p_bg(x\|m) in SB, interpolated to SR | Both p_bg(x\|m) and p_data(x\|m) in SR |
| **Likelihood ratio construction** | Implicit (via classifier) | Explicit (ratio of densities) |
| **Challenge** | Classification (easier) | Learning sharp signal peak in inner estimator (harder) |
| **Performance** | Max SIC ~14 | Max SIC ~6.5 |

**Key insight**: ANODE must learn the sharp signal peaks in the SR (inner estimator), which is difficult for density estimators. CATHODE avoids this by using a classifier instead, achieving much better performance.

### Role of the Density Estimator

The conditional density estimator (MAF) serves as the **background model**:

1. **Learning**: Trained on sideband data (where signal contamination is minimal) to learn the smooth background distribution p_bg(x|m) as a function of auxiliary features x and mass m

2. **Interpolation**: The learned function can be queried at any value of m, including values in the signal region, providing an estimate of the background density in the SR

3. **Sampling**: By inverting the learned bijective transformation, the MAF generates synthetic events that follow the background distribution in the SR

4. **Key advantage**: The smooth background is much easier to learn than the data distribution (which contains sharp signal peaks), leading to high-fidelity background modeling

## Summary of Key Findings

✅ **CATHODE nearly saturates optimal performance**: Matches the idealized anomaly detector (which uses perfect background simulation)

✅ **Significantly outperforms alternatives**:
- 2× better than ANODE (SIC 14 vs 6.5)
- 1.3× better than CWoLa Hunting (SIC 14 vs 11)

✅ **Robust to correlations**: Maintains performance when features are correlated with mass (unlike CWoLa Hunting which breaks down completely)

✅ **Works across wide S/B range**: Maintains SIC > 10 down to S/B ~ 0.25%

✅ **Benefits from oversampling**: Performance improves when generating more synthetic background events (up to ~200k samples)
