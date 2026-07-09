# internship_summer2026
DATASET OVERVIEW
==================================================
Rows: 7043
Columns: 21

First 5 rows:
   customerID  gender  SeniorCitizen  ... MonthlyCharges TotalCharges  Churn
0  7590-VHVEG  Female              0  ...          29.85        29.85     No
1  5575-GNVDE    Male              0  ...          56.95       1889.5     No
2  3668-QPYBK    Male              0  ...          53.85       108.15    Yes
3  7795-CFOCW    Male              0  ...          42.30      1840.75     No
4  9237-HQITU  Female              0  ...          70.70       151.65    Yes

[5 rows x 21 columns]

DATA CLEANING DONE
==================================================
Missing values after cleaning: 0

Churn distribution:
Churn
0    5174
1    1869
Name: count, dtype: int64

FEATURE SELECTION
==================================================
Selected features: ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling', 'TechSupport', 'OnlineSecurity']
Shape after one-hot encoding: (7043, 16)

TRAIN-TEST SPLIT
==================================================
Training samples: 5634
Testing samples: 1409


============================================================
LOGISTIC REGRESSION RESULTS
============================================================
Accuracy:  0.7977
Precision: 0.645
Recall:    0.5294
F1 Score:  0.5815

Confusion Matrix:
[[926 109]
 [176 198]]

Classification Report:
              precision    recall  f1-score   support

    No Churn       0.84      0.89      0.87      1035
       Churn       0.64      0.53      0.58       374

    accuracy                           0.80      1409
   macro avg       0.74      0.71      0.72      1409
weighted avg       0.79      0.80      0.79      1409


============================================================
DECISION TREE RESULTS
============================================================
Accuracy:  0.7225
Precision: 0.4785
Recall:    0.5053
F1 Score:  0.4915

Confusion Matrix:
[[829 206]
 [185 189]]

Classification Report:
              precision    recall  f1-score   support

    No Churn       0.82      0.80      0.81      1035
       Churn       0.48      0.51      0.49       374

    accuracy                           0.72      1409
   macro avg       0.65      0.65      0.65      1409
weighted avg       0.73      0.72      0.72      1409


============================================================
RANDOM FOREST RESULTS
============================================================
Accuracy:  0.7771
Precision: 0.5949
Recall:    0.5027
F1 Score:  0.5449

Confusion Matrix:
[[907 128]
 [186 188]]

Classification Report:
              precision    recall  f1-score   support

    No Churn       0.83      0.88      0.85      1035
       Churn       0.59      0.50      0.54       374

    accuracy                           0.78      1409
   macro avg       0.71      0.69      0.70      1409
weighted avg       0.77      0.78      0.77      1409


============================================================
K-NEAREST NEIGHBORS RESULTS
============================================================
Accuracy:  0.7679
Precision: 0.5765
Recall:    0.4733
F1 Score:  0.5198

Confusion Matrix:
[[905 130]
 [197 177]]

Classification Report:
              precision    recall  f1-score   support

    No Churn       0.82      0.87      0.85      1035
       Churn       0.58      0.47      0.52       374

    accuracy                           0.77      1409
   macro avg       0.70      0.67      0.68      1409
weighted avg       0.76      0.77      0.76      1409


============================================================
FINAL MODEL COMPARISON
============================================================
              Model  Accuracy  Precision  Recall  F1 Score
Logistic Regression    0.7977     0.6450  0.5294    0.5815
      Random Forest    0.7771     0.5949  0.5027    0.5449
K-Nearest Neighbors    0.7679     0.5765  0.4733    0.5198
      Decision Tree    0.7225     0.4785  0.5053    0.4915

============================================================
BEST PERFORMING MODEL
============================================================
Model: Logistic Regression
F1 Score: 0.5815
Accuracy: 0.7977

============================================================
SAMPLE PREDICTIONS (Logistic Regression)
============================================================
 Actual Churn  Predicted Churn  Churn Probability
            0                0             0.0495
            0                1             0.5957
            0                0             0.0839
            0                0             0.4372
            0                0             0.0195
            0                1             0.5451
            0                0             0.4510
            0                0             0.1238
            0                0             0.0035
            1                0             0.4816

============================================================
PROJECT SUMMARY
============================================================
Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv
Problem Type: Binary Classification
Target: Customer Churn (Yes/No)
Models Compared:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors Classifier
Evaluation Metrics: Accuracy, Precision, Recall, F1 Score
Best Model: Logistic Regression
Best F1 Score: 0.5815
============================================================
