# Predicting Adult Income Level using Machine Learning

**Group ID:** 2026-Y2-S1-MLB-B1G1-07  
**Course:** IT2011 - Artificial Intelligence and Machine Learning  
**Institution:** SLIIT, Faculty of Computing — Year 2, Semester 1 (2026)  
**Project Stages:** 
- Progress Review 1: Data Cleaning, Preprocessing and Exploratory Data Analysis (EDA)
- Progress Review 2: Model Design, Implementation, and Evaluation Preparation

---

## 1. Project Overview

This project applies machine learning to the **Adult Income Dataset** (UCI Machine Learning Repository, derived from the 1994 US Census Bureau) to predict whether an individual's annual income exceeds $50,000 based on census attributes.

- **Problem Domain:** Socio-Economic Workforce Classification & Salary Prediction
- **Dataset Size:** 32,561 records, 14 input features + 1 binary target (`income`)
- **Target Distribution:** `<=50K` (~75.9%) vs `>50K` (~24.1%)

---

## 2. Group Members & Assigned Roles

### Progress Review 1: Data Preprocessing
| Member Full Name | IT Number | Assigned Preprocessing Technique |
|---|---|---|
| **Wickramasinghe G.D.M.H** | IT25102064 | Handling Missing Data |
| **Weerarathna N.H** | IT25103014 | Feature Scaling / Normalization |
| **Wickramasinghe M.P.T.H** | IT25300115 | Feature Engineering & Redundancy Removal |
| **Wickrama W.M.K.E** | IT25102219 | Encoding Categorical Variables |
| **Hansani J.A.T.H** | IT25300345 | Outlier Detection & Removal |
| **Lakshan H.M.K** | IT25101220 | Duplicate Removal & Class Imbalance Analysis |

### Progress Review 2: Machine Learning Models (Individual Implementation)
| Member Full Name | IT Number | Assigned Model | Notebook Location |
|---|---|---|---|
| **Lakshan H.M.K** | IT25101220 | Decision Tree Classifier | `notebooks/models/IT25101220_Model_DecisionTree.ipynb` |
| **Hansani J.A.T.H** | IT25300345 | Random Forest Classifier | `notebooks/models/IT25300345_Model_RandomForest.ipynb` |
| **Wickrama W.M.K.E** | IT25102219 | Logistic Regression | `notebooks/models/IT25102219_Model_LogisticRegression.ipynb` |
| **Wickramasinghe G.D.M.H** | IT25102064 | K-Nearest Neighbors (KNN) | `notebooks/models/IT25102064_Model_KNN.ipynb` |
| **Weerarathna N.H** | IT25103014 | Support Vector Machine (SVM) | `notebooks/models/IT25103014_Model_SVM.ipynb` |
| **Wickramasinghe M.P.T.H** | IT25300115 | Gradient Boosting Classifier | `notebooks/models/IT25300115_Model_GradientBoosting.ipynb` |

---

## 3. Repository Structure

```
predicting-adult-income-level/
├── README.md                          # Project documentation & Progress Review 1 & 2 guide
├── requirements.txt                   # Standardized Python dependencies
├── group_pipeline.ipynb               # Review 1 integrated preprocessing pipeline
├── group_model_comparison.ipynb       # Review 2 GROUP notebook (to be added once all 6 models are completed)
│
├── data/
│   └── raw/
│       └── adult.csv                  # Assigned raw dataset
│
├── src/                               # Shared Python modules
│   └── common.py                      # Shared train-test split, evaluation metrics, and plots
│
├── notebooks/
│   ├── IT25101220_DuplicateClassImbalance.ipynb
│   ├── IT25102064_MissingValues.ipynb
│   ├── IT25102219_Encoding.ipynb
│   ├── IT25103014_FeatureScaling.ipynb
│   ├── IT25300115_Engineering.ipynb
│   ├── IT25300345_OutlierRemoval.ipynb
│   └── models/                        # Review 2 individual model notebooks
│       ├── .gitkeep
│       ├── IT25101220_Model_DecisionTree.ipynb
│       ├── IT25300345_Model_RandomForest.ipynb
│       ├── IT25102219_Model_LogisticRegression.ipynb
│       ├── IT25102064_Model_KNN.ipynb
│       ├── IT25103014_Model_SVM.ipynb
│       └── IT25300115_Model_GradientBoosting.ipynb
│
└── results/
    ├── outputs/                       # Review 1 output (adult_processed.csv)
    ├── eda_visualizations/            # Review 1 EDA charts
    │   └── models/                    # Review 2 model evaluation plots (.gitkeep)
    └── model_results/                 # Serialized individual JSON metrics for comparison
        ├── .gitkeep
        └── <IT_NUMBER>.json
```

---

## 4. Setup and How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Review 2 Individual Workflow
1. Use the shared utilities in `src/common.py` to ensure consistent data splitting and metric calculations:
   ```python
   from src.common import load_data, get_train_test_split, evaluate_model, save_member_result
   
   X, y = load_data()
   X_train, X_test, y_train, y_test = get_train_test_split(X, y)
   ```
2. Train and evaluate your model varieties in your designated notebook under `notebooks/models/`.
3. Save your best model's result using:
   ```python
   save_member_result("IT2510XXXX", "ModelName", best_params, metrics)
   ```
4. Run `group_model_comparison.ipynb` to aggregate all 6 models and generate the final group comparison table and charts for the viva.
