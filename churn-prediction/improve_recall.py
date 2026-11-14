"""Try to catch more churners than the baseline Logistic Regression.

Three variants, all on the same train/test split as custfinal.py:
  1. baseline:          Logistic Regression, threshold 0.5
  2. balanced:          class_weight="balanced" (up-weights churners), threshold 0.5
  3. balanced + tuned:  as 2, but the threshold is chosen on training data only
                        (cross-validated), to maximize F1, then applied to test.

The threshold is never chosen on the test set, so the test numbers stay honest.
Run from inside churn-prediction/:  python improve_recall.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "TechSupport",
    "OnlineSecurity",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = pd.get_dummies(df[FEATURES], drop_first=True)
    y = df["Churn"]
    return X, y


def best_threshold(model, X, y):
    """Pick the threshold that maximizes F1 on out-of-fold training predictions."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.20, 0.81, 0.01):
        f1 = f1_score(y, (probs >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def report(name, y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "Model": name,
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred),
        "Recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, probs),
    }


def main():
    X, y = load_data()
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    rows = []

    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(train_X, train_y)
    rows.append(report("Baseline", test_y,
                       baseline.predict_proba(test_X)[:, 1], 0.5))

    balanced = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    balanced.fit(train_X, train_y)
    rows.append(report("Balanced", test_y,
                       balanced.predict_proba(test_X)[:, 1], 0.5))

    t = best_threshold(
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        train_X, train_y,
    )
    rows.append(report("Balanced + tuned threshold", test_y,
                       balanced.predict_proba(test_X)[:, 1], t))

    results = pd.DataFrame(rows).set_index("Model")
    print(results.round(4).to_string())


if __name__ == "__main__":
    main()
