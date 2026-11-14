import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

customerdata = pd.read_csv("customer_data.csv")

customerdata = customerdata.dropna(axis=0)
customerdata['Churn'] = customerdata['Churn'].map({'Yes': 1, 'No': 0})
y = customerdata.Churn
customerfeat = ['tenure', 'SeniorCitizen', 'MonthlyCharges']
X = customerdata[customerfeat]

train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=0
)
customermodel = LogisticRegression()
customermodel.fit(train_X, train_y)
predictions = customermodel.predict(val_X)
print("The predictions are:")
print(predictions)
print("Accuracy:")
print(accuracy_score(val_y, predictions))