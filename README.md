# Predicting Adult Income Level using Machine Learning

**Group ID:** 2026-Y2-S1-MLB-B1G1-07  
**Course:** IT2011 - Artificial Intelligence and Machine Learning  
**Institution:** SLIIT, Faculty of Computing — Year 2, Semester 1 (2026)  
**Project Stage:** Progress Review 1 — Data Cleaning, Preprocessing and Exploratory Data Analysis (EDA)  

---

## 1. Project Overview

This project applies machine learning to the **Adult Income Dataset** (UCI Machine Learning Repository, derived from the 1994 US Census Bureau) to predict whether an individual's annual income exceeds $50,000 based on census attributes.

- **Problem Domain:** Socio-Economic Workforce Classification & Salary Prediction
- **Dataset Size:** 32,561 records, 14 input features + 1 binary target (`income`)
- **Target Distribution:** `<=50K` (~75.9%) vs `>50K` (~24.1%)

---

## 2. Group Members & Assigned Preprocessing Techniques

As per the SLIIT IT2011 Progress Review 1 guide, each member implements one preprocessing technique:

| Member Full Name | IT Number | Assigned Preprocessing Technique |
|---|---|---|
| **Wickramasinghe G.D.M.H** | IT25102064 | Handling Missing Data |
| **Weerarathna N.H** | IT25103014 | Feature Scaling / Normalization |
| **Wickramasinghe M.P.T.H** | IT25300115 | Feature Engineering & Redundancy Removal |
| **Wickrama W.M.K.E** | IT25102219 | Encoding Categorical Variables |
| **Hansani J.A.T.H** | IT25300345 | Outlier Detection & Removal |
| **Lakshan H.M.K** | IT25101220 | Duplicate Removal & Class Imbalance Analysis |

---

## 3. Repository Structure

Conforming to the official SLIIT IT2011 group deliverable layout:

```
2026-Y2-S1-MLB-B1G1-07/
├── README.md                                    # Project documentation
├── .gitignore                                   # Standard Python / Jupyter git ignore
├── data/
│   ├── raw/
│   │   └── adult.csv                            # Assigned raw dataset as provided
│   └── external/                                # External reference datasets (if used)
├── notebooks/                                   # Individual member preprocessing notebooks
│   ├── IT25102064_MissingValues.ipynb
│   ├── IT25101220_DuplicateClassImbalance.ipynb
│   ├── IT25300345_OutlierRemoval.ipynb
│   ├── IT25300115_Engineering.ipynb
│   ├── IT25102219_Encoding.ipynb
│   └── IT25103014_FeatureScaling.ipynb
├── group_pipeline.ipynb                         # Integrated preprocessing pipeline (combined work)
└── results/
    ├── eda_visualizations/                      # Plots & EDA charts (PNG/JPEG)
    ├── logs/                                    # Execution logs (optional)
    └── outputs/                                 # Final processed dataset (adult_processed.csv)
```

---

## 4. Setup and How to Run

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Collaboration Workflow
1. Each group member clones this repository:
   ```bash
   git clone https://github.com/mln414/predicting-adult-income-level.git
   ```
2. Create a feature branch for your assigned technique:
   ```bash
   git checkout -b feat/<IT_Number>-<technique>
   ```
3. Add your individual notebook to `notebooks/` and any plots to `results/eda_visualizations/`.
4. Commit and push your branch, then open a Pull Request against `main`.
