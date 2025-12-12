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
