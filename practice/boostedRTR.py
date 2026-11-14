import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melbdata = pd.read_csv("melb_data.csv")

melbdata = pd.get_dummies(melbdata)

melbdata = melbdata.dropna()

y = melbdata.Price

X = melbdata.drop(['Price'], axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

melbmodel = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    random_state=1
)

melbmodel.fit(train_X, train_y)

predictprices = melbmodel.predict(val_X)

print(mean_absolute_error(val_y, predictprices))