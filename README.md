# des-ml-evaluation-fragility

## When High Accuracy Misleads in Machine Learning for Deep Eutectic Solvent Recommendation

This repository contains the data, code, audit outputs, and reproducibility resources associated with the manuscript:

**“When High Accuracy Misleads in Machine Learning for Deep Eutectic Solvent Recommendation”**

The project investigates the reliability of machine-learning-based recommendation for deep eutectic solvent (DES) systems under literature-derived experimental conditions. The study focuses on leakage-aware evaluation, sparse comparative supervision, and the relationship between predictive accuracy and recommendation utility.

---

# Repository Structure

```text
des-ml-evaluation-fragility/
│
├── manuscript/
│   ├── Final_Revised_Manuscript.pdf
│   └── Final_Revised_Supporting_Information.pdf
│
├── data/
│   └── Dataset_S1.csv
│
├── code/
│   └── Code_S1.ipynb
│
├── outputs/
│   ├── audits/
│   ├── pairwise/
│   ├── ranking/
│   ├── statistics/
│   ├── uncertainty/
│   └── sanity_baselines/
│
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

---

# Repository Contents

## Manuscript

The `manuscript/` directory contains the final manuscript and Supporting Information associated with the study.

---

## Dataset

`Dataset_S1.csv` contains the curated experiment-level dataset reconstructed from literature-derived DES-based pesticide extraction studies.

The dataset includes:

* DES composition information;
* analyte descriptors;
* engineered interaction features;
* group structure information;
* leakage-clean modeling variables.

---

## Code

`Code_S1.ipynb` contains the complete leakage-clean computational workflow used in this study, including:

* DOI-grouped splitting;
* training-only preprocessing;
* feature filtering;
* pairwise dataset construction;
* model training;
* holdout inference;
* group-level ranking evaluation;
* uncertainty analysis;
* sanity baseline evaluation.

The workflow follows a split-first, train-only paradigm designed to minimize information leakage.

---

# Reproducibility

The repository includes the principal audit and reproducibility outputs required to verify the analyses reported in the manuscript, including:

* split audit summaries;
* feature filtering audits;
* pairwise dataset statistics;
* ranking outputs;
* statistical evaluation;
* uncertainty estimation;
* sanity baseline comparisons.

All analyses can be reproduced by running:

```text
code/Code_S1.ipynb
```

sequentially in a compatible Python environment.
The repository is intended to provide full reproducibility of the leakage-clean evaluation workflow and the principal results reported in the manuscript.
---

# Software Requirements

The software dependencies used in this study are listed in:

```text
requirements.txt
```

Core packages include:

* pandas
* numpy
* scikit-learn
* xgboost
* matplotlib
* RDKit

---

# Data and Software Availability

The data and code underlying this study are publicly available in this repository and archived through Zenodo.

---

# Citation

If you use this repository, please cite the associated manuscript and the archived Zenodo release DOI (to be added upon publication).

---

# License

This repository is distributed under the terms of the MIT License.
