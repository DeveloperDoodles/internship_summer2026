import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import warnings

warnings.filterwarnings("ignore")


customer_data = pd.read_csv(
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("DATASET OVERVIEW")
print("=" * 50)
print(f"Rows: {customer_data.shape[0]}")
print(f"Columns: {customer_data.shape[1]}")
print("\nFirst 5 rows:")
print(customer_data.head())
print()


customer_data["TotalCharges"] = pd.to_numeric(
    customer_data["TotalCharges"],
    errors="coerce"
)

customer_data["TotalCharges"] = customer_data[
    "TotalCharges"
].fillna(
    customer_data["TotalCharges"].median()
)

customer_data["Churn"] = customer_data["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("DATA CLEANING DONE")
print("=" * 50)

print(
    f"Missing values after cleaning: "
    f"{customer_data.isnull().sum().sum()}"
)

print(
    f"\nChurn distribution:\n"
    f"{customer_data['Churn'].value_counts()}"
)

print()


selected_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "TechSupport",
    "OnlineSecurity"
]

X = customer_data[selected_features].copy()

y = customer_data["Churn"]

X = pd.get_dummies(
    X,
    drop_first=True
)

print("FEATURE SELECTION")
print("=" * 50)

print(
    f"Selected features: "
    f"{selected_features}"
)

print(
    f"Shape after one-hot encoding: "
    f"{X.shape}"
)

print()


train_X, test_X, train_y, test_y = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("TRAIN-TEST SPLIT")
print("=" * 50)

print(
    f"Training samples: "
    f"{train_X.shape[0]}"
)

print(
    f"Testing samples: "
    f"{test_X.shape[0]}"
)

print()


scaler = StandardScaler()

train_X = pd.DataFrame(
    scaler.fit_transform(train_X),
    columns=X.columns
)

test_X = pd.DataFrame(
    scaler.transform(test_X),
    columns=X.columns
)


all_results = []


lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(
    train_X,
    train_y
)

lr_predictions = lr_model.predict(
    test_X
)

lr_accuracy = accuracy_score(
    test_y,
    lr_predictions
)

lr_precision = precision_score(
    test_y,
    lr_predictions,
    zero_division=0
)

lr_recall = recall_score(
    test_y,
    lr_predictions,
    zero_division=0
)

lr_f1 = f1_score(
    test_y,
    lr_predictions,
    zero_division=0
)


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(
    f"Accuracy:  "
    f"{round(lr_accuracy, 4)}"
)

print(
    f"Precision: "
    f"{round(lr_precision, 4)}"
)

print(
    f"Recall:    "
    f"{round(lr_recall, 4)}"
)

print(
    f"F1 Score:  "
    f"{round(lr_f1, 4)}"
)

print(
    f"\nConfusion Matrix:\n"
    f"{confusion_matrix(test_y, lr_predictions)}"
)

print(
    f"\nClassification Report:\n"
    f"{classification_report(
        test_y,
        lr_predictions,
        target_names=['No Churn', 'Churn'],
        zero_division=0
    )}"
)


all_results.append({
    "Model": "Logistic Regression",
    "Accuracy": round(lr_accuracy, 4),
    "Precision": round(lr_precision, 4),
    "Recall": round(lr_recall, 4),
    "F1 Score": round(lr_f1, 4)
})


dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(
    train_X,
    train_y
)

dt_predictions = dt_model.predict(
    test_X
)

dt_accuracy = accuracy_score(
    test_y,
    dt_predictions
)

dt_precision = precision_score(
    test_y,
    dt_predictions,
    zero_division=0
)

dt_recall = recall_score(
    test_y,
    dt_predictions,
    zero_division=0
)

dt_f1 = f1_score(
    test_y,
    dt_predictions,
    zero_division=0
)


print("\n" + "=" * 60)
print("DECISION TREE RESULTS")
print("=" * 60)

print(
    f"Accuracy:  "
    f"{round(dt_accuracy, 4)}"
)

print(
    f"Precision: "
    f"{round(dt_precision, 4)}"
)

print(
    f"Recall:    "
    f"{round(dt_recall, 4)}"
)

print(
    f"F1 Score:  "
    f"{round(dt_f1, 4)}"
)

print(
    f"\nConfusion Matrix:\n"
    f"{confusion_matrix(test_y, dt_predictions)}"
)

print(
    f"\nClassification Report:\n"
    f"{classification_report(
        test_y,
        dt_predictions,
        target_names=['No Churn', 'Churn'],
        zero_division=0
    )}"
)


all_results.append({
    "Model": "Decision Tree",
    "Accuracy": round(dt_accuracy, 4),
    "Precision": round(dt_precision, 4),
    "Recall": round(dt_recall, 4),
    "F1 Score": round(dt_f1, 4)
})


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    train_X,
    train_y
)

rf_predictions = rf_model.predict(
    test_X
)

rf_accuracy = accuracy_score(
    test_y,
    rf_predictions
)

rf_precision = precision_score(
    test_y,
    rf_predictions,
    zero_division=0
)

rf_recall = recall_score(
    test_y,
    rf_predictions,
    zero_division=0
)

rf_f1 = f1_score(
    test_y,
    rf_predictions,
    zero_division=0
)


print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(
    f"Accuracy:  "
    f"{round(rf_accuracy, 4)}"
)

print(
    f"Precision: "
    f"{round(rf_precision, 4)}"
)

print(
    f"Recall:    "
    f"{round(rf_recall, 4)}"
)

print(
    f"F1 Score:  "
    f"{round(rf_f1, 4)}"
)

print(
    f"\nConfusion Matrix:\n"
    f"{confusion_matrix(test_y, rf_predictions)}"
)

print(
    f"\nClassification Report:\n"
    f"{classification_report(
        test_y,
        rf_predictions,
        target_names=['No Churn', 'Churn'],
        zero_division=0
    )}"
)


all_results.append({
    "Model": "Random Forest",
    "Accuracy": round(rf_accuracy, 4),
    "Precision": round(rf_precision, 4),
    "Recall": round(rf_recall, 4),
    "F1 Score": round(rf_f1, 4)
})


knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    train_X,
    train_y
)

knn_predictions = knn_model.predict(
    test_X
)

knn_accuracy = accuracy_score(
    test_y,
    knn_predictions
)

knn_precision = precision_score(
    test_y,
    knn_predictions,
    zero_division=0
)

knn_recall = recall_score(
    test_y,
    knn_predictions,
    zero_division=0
)

knn_f1 = f1_score(
    test_y,
    knn_predictions,
    zero_division=0
)


print("\n" + "=" * 60)
print("K-NEAREST NEIGHBORS RESULTS")
print("=" * 60)

print(
    f"Accuracy:  "
    f"{round(knn_accuracy, 4)}"
)

print(
    f"Precision: "
    f"{round(knn_precision, 4)}"
)

print(
    f"Recall:    "
    f"{round(knn_recall, 4)}"
)

print(
    f"F1 Score:  "
    f"{round(knn_f1, 4)}"
)

print(
    f"\nConfusion Matrix:\n"
    f"{confusion_matrix(test_y, knn_predictions)}"
)

print(
    f"\nClassification Report:\n"
    f"{classification_report(
        test_y,
        knn_predictions,
        target_names=['No Churn', 'Churn'],
        zero_division=0
    )}"
)


all_results.append({
    "Model": "K-Nearest Neighbors",
    "Accuracy": round(knn_accuracy, 4),
    "Precision": round(knn_precision, 4),
    "Recall": round(knn_recall, 4),
    "F1 Score": round(knn_f1, 4)
})


comparison_table = pd.DataFrame(
    all_results
)

comparison_table = comparison_table.sort_values(
    "F1 Score",
    ascending=False
).reset_index(
    drop=True
)


print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    comparison_table.to_string(
        index=False
    )
)

print()


best_row = comparison_table.iloc[0]


print("=" * 60)
print("BEST PERFORMING MODEL")
print("=" * 60)

print(
    f"Model: "
    f"{best_row['Model']}"
)

print(
    f"F1 Score: "
    f"{best_row['F1 Score']}"
)

print(
    f"Accuracy: "
    f"{best_row['Accuracy']}"
)

print()


model_lookup = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "K-Nearest Neighbors": knn_model
}


best_model = model_lookup[
    best_row["Model"]
]


sample_predictions = best_model.predict(
    test_X[:10]
)


sample_probabilities = best_model.predict_proba(
    test_X[:10]
)[:, 1]


sample_output = pd.DataFrame({
    "Actual Churn": test_y.values[:10],
    "Predicted Churn": sample_predictions,
    "Churn Probability": np.round(
        sample_probabilities,
        4
    )
})


print("=" * 60)

print(
    f"SAMPLE PREDICTIONS "
    f"({best_row['Model']})"
)

print("=" * 60)

print(
    sample_output.to_string(
        index=False
    )
)

print()


print("=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print(
    "Dataset: "
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print(
    "Problem Type: "
    "Binary Classification"
)

print(
    "Target: "
    "Customer Churn (Yes/No)"
)

print(
    "Models Compared:"
)

print(
    "1. Logistic Regression"
)

print(
    "2. Decision Tree Classifier"
)

print(
    "3. Random Forest Classifier"
)

print(
    "4. K-Nearest Neighbors Classifier"
)

print(
    "Evaluation Metrics: "
    "Accuracy, Precision, Recall, F1 Score"
)

print(
    f"Best Model: "
    f"{best_row['Model']}"
)

print(
    f"Best F1 Score: "
    f"{best_row['F1 Score']}"
)

print("=" * 60)