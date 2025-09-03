# Extended Stochastic Block Models for Recommender Systems

This repository contains code and resources developed for my Master’s Thesis proposting an extension of **Extended Stochastic Block Model** (Legramanti et al. 2020) tailored for recommendser systems.

In particular, we introduce the **Degree-Corrected Extended Stochastic Block Model (DC-ESBM)**, which captures user/item popularity heterogeneity while leveraging nonparametric Bayesian modeling for community detection.

Key contributions:
- Extension of **ESBMs** to weighted, bipartite graphs.
- **Degree correction** for modeling popularity differences.
- Applications to synthetic data and a real-world book recommendation dataset.

## So... why does this matter? 
Recommendation algorithms decide what we watch, read, or buy but they are often black boxes. This project aims to develop an interpretable algorithm modeling user-item interactions as networks. Instead of blindly factorizing a ratings matrix, we discover communities of similar users and items, making recommendations easier to interpret and explain. Plus, with degree correction, we don’t get fooled by power-users or ultra-popular items, your “hidden gem” recommendations stay hidden gems.

---

## 📂 Repository Structure

```text
├── data/                # Datasets or links to download
├── src/                 # Core modeling code
│   ├── esbm.py          # Base ESBM implementation
│   ├── dc_esbm.py       # Degree-Corrected ESBM
│   └── inference.py     # Gibbs sampling and inference routines
├── notebooks/           # Analysis & experiment notebooks
├── results/             # Generated figures, evaluation results
└── README.md            # This file
