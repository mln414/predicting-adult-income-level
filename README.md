# Predicting Adult Income Level using Machine Learning

**Group ID:** `2026-Y2-S1-MLB-B1G1-07`  
**Course:** IT2011 - Artificial Intelligence and Machine Learning  
**Institution:** SLIIT, Faculty of Computing — Year 2, Semester 1 (2026)  
**Evaluation:** Progress Review 2 – Model Design, Implementation & Evaluation  

---

## 1. Project Overview

This project applies machine learning to the **Adult Income Dataset** (UCI Machine Learning Repository, 1994 US Census Bureau) to predict whether an individual's annual income exceeds **$50,000**.

- **Goal:** Binary classification (`<=50K` vs `>50K`).
- **Dataset Size:** 32,561 records with 14 census features (age, education, workclass, capital gain, etc.).
- **Class Imbalance:** 
  - `<=50K`: **75.9%** (majority class)
  - `>50K`: **24.1%** (minority class)
- **Key Takeaway on Metrics:** Because ~76% of people earn `<=50K`, a simple model could get 76% accuracy by guessing `<=50K` every time. Therefore, we prioritize **F1-Score, Balanced Accuracy, and ROC-AUC** to measure true performance on both classes.

---

## 2. Group Members & Assigned Roles

### Progress Review 1: Preprocessing Specializations
| Member Full Name | Student IT Number | Assigned Preprocessing Technique |
| :--- | :--- | :--- |
| **Wickramasinghe G.D.M.H** | IT25102064 | Handling Missing Data |
| **Weerarathna N.H** | IT25103014 | Feature Scaling / Normalization |
| **Wickramasinghe M.P.T.H** | IT25300115 | Feature Engineering & Redundancy Removal |
| **Wickrama W.M.K.E** | IT25102219 | Encoding Categorical Variables |
| **Hansani J.A.T.H** | IT25300345 | Outlier Detection & Removal |
| **Lakshan H.M.K** | IT25101220 | Duplicate Removal & Class Imbalance Analysis |

### Progress Review 2: Individual Machine Learning Models
| Member Full Name | Student IT Number | Assigned Model | Individual Notebook |
| :--- | :--- | :--- | :--- |
| **Lakshan H.M.K** | IT25101220 | **Decision Tree Classifier** | [`notebooks/models/IT25101220_Model_DecisionTree.ipynb`](notebooks/models/IT25101220_Model_DecisionTree.ipynb) |
| **Hansani J.A.T.H** | IT25300345 | **Random Forest Classifier** | [`notebooks/models/IT25300345_Model_RandomForest.ipynb`](notebooks/models/IT25300345_Model_RandomForest.ipynb) |
| **Wickrama W.M.K.E** | IT25102219 | **Logistic Regression** | [`notebooks/models/IT25102219_Model_LogisticRegression.ipynb`](notebooks/models/IT25102219_Model_LogisticRegression.ipynb) |
| **Wickramasinghe G.D.M.H** | IT25102064 | **K-Nearest Neighbors (KNN)** | [`notebooks/models/IT25102064_Model_KNN.ipynb`](notebooks/models/IT25102064_Model_KNN.ipynb) |
| **Weerarathna N.H** | IT25103014 | **Support Vector Machine (SVM)** | [`notebooks/models/IT25103014_Model_SVM.ipynb`](notebooks/models/IT25103014_Model_SVM.ipynb) |
| **Wickramasinghe M.P.T.H** | IT25300115 | **Gradient Boosting Classifier** | [`notebooks/models/IT25300115_Model_GradientBoosting.ipynb`](notebooks/models/IT25300115_Model_GradientBoosting.ipynb) |

---

## 3. Master Model Comparison Results

All 6 models were evaluated on the exact same held-out test set (80/20 stratified split, `random_state=42`, $N = 6,508$).

| Rank | Model Name | Student ID | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Key Highlight |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Gradient Boosting** | `IT25300115` | **86.73%** | **0.7629** | 0.6306 | 0.6904 | **0.9242** | **Best overall accuracy and highest ROC-AUC.** |
| **2** | **Random Forest** | `IT25300345` | 85.42% | 0.6759 | 0.7276 | **0.7008** | 0.9140 | **Best balanced F1-score across both classes.** |
| **3** | **Logistic Regression** | `IT25102219` | 81.13% | 0.5644 | **0.8585** | 0.6811 | 0.9050 | **Highest Recall (catches 85.8% of high earners).** |
| **4** | **Support Vector Machine** | `IT25103014` | 80.56% | 0.5642 | 0.8489 | 0.6779 | 0.9001 | Strong non-linear separation with RBF kernel. |
| **5** | **K-Nearest Neighbors** | `IT25102064` | 84.79% | 0.7011 | 0.6140 | 0.6546 | 0.8938 | High precision using distance-weighted voting. |
| **6** | **Decision Tree** | `IT25101220` | 80.04% | 0.5565 | 0.8450 | 0.6711 | 0.8899 | Fully transparent rule tree; prone to variance. |

---

## 4. Key Questions & Simple Viva Explanations

1. **Why did Gradient Boosting & Random Forest perform the best?**  
   They are **ensemble models** that combine hundreds of decision trees. Real census data has complex combinations of features (e.g., education + age + marital status together). Ensembles capture these patterns much better than a single linear line or single tree.
2. **Why do Logistic Regression, SVM, and Decision Tree have high Recall (~85%) but lower Precision (~56%)?**  
   They used `class_weight='balanced'`. Because high earners are only 24% of the dataset, balanced weights penalize the model heavily for missing high earners. This makes the model aggressive at catching them (85%+ Recall), with a small increase in false alarms.
3. **Why did KNN and SVM need Feature Scaling, but Decision Trees did not?**  
   KNN and SVM compute geometric distances in feature space. Without scaling, large numbers (like `capital-gain` up to 99,000) completely overwhelm smaller numbers (like `age` or `education-num`). Tree-based models only compare numbers at split thresholds (`age > 35`), so scale does not affect them.
4. **Final Recommendation:**  
   - **For maximum accuracy:** Deploy **Gradient Boosting** (ROC-AUC `0.9242`).
   - **For fast & explainable decisions:** Deploy **Logistic Regression** (<4 KB file size, microsecond predictions, and interpretable odds ratios).

---

## 5. Repository Structure

```
predicting-adult-income-level/
├── README.md                          # Project documentation & summary guide
├── requirements.txt                   # Standardized Python dependencies
├── group_pipeline.ipynb               # Review 1 integrated preprocessing pipeline
├── group_model_comparison.ipynb       # Review 2 GROUP comparison notebook & viva charts
│
├── data/
│   └── raw/
│       └── adult.csv                  # Raw UCI Adult dataset
│
├── app/
│   └── streamlit_app.py               # Interactive web app for live model demonstrations
│
├── src/                               # Shared Python modules
│   ├── common.py                      # Shared splits, metrics, and plotting helpers
│   └── prediction_demo.py             # Inference pipeline & model loading for Streamlit
│
├── notebooks/
│   ├── IT25101220_DuplicateClassImbalance.ipynb
│   ├── IT25102064_MissingValues.ipynb
│   ├── IT25102219_Encoding.ipynb
│   ├── IT25103014_FeatureScaling.ipynb
│   ├── IT25300115_Engineering.ipynb
│   ├── IT25300345_OutlierRemoval.ipynb
│   └── models/                        # Review 2 individual model notebooks
│       ├── IT25101220_Model_DecisionTree.ipynb
│       ├── IT25102064_Model_KNN.ipynb
│       ├── IT25102219_Model_LogisticRegression.ipynb
│       ├── IT25103014_Model_SVM.ipynb
│       ├── IT25300115_Model_GradientBoosting.ipynb
│       └── IT25300345_Model_RandomForest.ipynb
│
└── results/
    ├── outputs/                       # Preprocessed output (adult_processed.csv)
    ├── saved_models/                  # Serialized .joblib models for all 6 models
    ├── eda_visualizations/models/     # Saved evaluation plots and comparison charts
    └── model_results/                 # Serialized individual JSON metrics
```

---

## 6. Setup & How to Run

### Installation
```bash
pip install -r requirements.txt
```

### 1. View the Group Comparison Notebook
Open and run [`group_model_comparison.ipynb`](group_model_comparison.ipynb) in Jupyter or VS Code to see:
- Automatic ingestion of all 6 model results.
- Ranked master comparison table.
- Comparative bar charts, Precision-Recall trade-off plots, and 5-metric radar profiles.
- Detailed written answers for viva questions.

### 2. Run the Interactive Web Demonstration
Launch our Streamlit application to demonstrate live predictions with all 6 models:

```bash
streamlit run app/streamlit_app.py
```

- Select any of the 6 trained models from the dropdown.
- View the model's test metrics (Accuracy, F1-Score, ROC-AUC) and the assigned student's ID.
- Adjust census attributes (Age, Education, Marital Status, Hours Per Week, etc.) and click **Predict income** to see real-time classification probabilities.
