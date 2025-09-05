# Extended Stochastic Block Models for Recommender Systems

Code for my Matser's thesis extending **Extended Stochastic Block Model** (Legramanti et al. 2020) to recommender systems.  for recommendser systems.

---
## Project Description

This repository contains the code and analysis for my Master’s thesis, which explores **Extended Stochastic Block Models (ESBMs)** and their application to **recommender systems**. A stochastic block model (SBM) is a generative statistical model for networks, often used to cluster nodes into communities based on their connection patterns. Extended Stochastic Block Models (ESBMs) are a Bayesian nonparametric version of SBMs.

The project extends this version of SBMs to **weighted, bipartite networks** and introduces a **degree-corrected extension** to capture user/item popularity heterogeneity while leveraging nonparametric Bayesian modeling for community detection.

The codebase serves two main purposes:
1. **Simulation Studies:** Scripts to reproduce the synthetic experiments in the thesis, demonstrating the performance of ESBMs under different network settings.
2. **Goodreads Book Ratings Analysis:** An empirical study applying ESBM and its degree-corrected variant to a large-scale **Goodreads dataset** of book ratings.

This repository provides a **reproducible workflow** for both simulation experiments and real-world analysis.

### Data
The dataset is not stored in this repository due to its size. However:
- A **script is provided** to preprocess and build the dataset from the raw files.
- The raw Goodreads ratings data can be downloaded from [this page](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) (called 'goodreads_interactions_dedup.json').

### So... why does this matter? 
Recommendation algorithms decide what we watch, read, or buy but they are often black boxes. This project aims to develop an interpretable algorithm modeling user-item interactions as networks. Instead of blindly factorizing a ratings matrix, we discover communities of similar users and items, making recommendations easier to interpret and explain. Plus, with degree correction, we don’t get fooled by power-users or ultra-popular items, your “hidden gem” recommendations stay hidden gems.

---

## 📂 Repository Structure

```text
├── data/               # Raw and processed datasets
├── results/            # Generated figures and tables
├── src/                # Analysis and pipeline scripts
│   ├── analysis/       # Modeling and statistical methods
│   └── pipeline/       # Data loading and preprocessing
└── README.md           # This file
```
---

## Installation
Clone the repository and install requirements:
(pip)
```bash
git clone https://github.com/lorenzo-costa/REC-ESBM.git
cd REC-ESBM
pip install -r requirements.txt
```
(conda)
```bash
git clone https://github.com/lorenzo-costa/REC-ESBM.git
cd repo-name
conda env create -f environment.yml
conda activate REC-ESBM
```

---

## Usage
1. Preprocess the data: place Goodreads dataset in data/raw and run
```bash
python src/pipeline/pre_processing_functs.py
``` 
Note:
- this is not needed for simulations
- requires internet connection
- may be quite slow

2. Run simulations
```bash
python src/analysis/simulations.py\
```
3. Run data analysis:
```bash
python src/analysis/book_analysis.py
```
Note: this takes A LOT of time


To run the whole analysis with one command run
```bash
make all
```
