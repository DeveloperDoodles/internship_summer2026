import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melbdata = pd.read_csv("melb_data.csv")

melbdata = melbdata.dropna(subset=['Price'])

y = melbdata.Price

melbfeat = [
    'Rooms',
    'Bathroom',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Type',
    'Method',
    'Lattitude',
    'Longtitude'
]

X = melbdata[melbfeat]

X = pd.get_dummies(X)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

train_X, val_X = train_X.align(val_X, join='left', axis=1, fill_value=0)

train_X = train_X.fillna(train_X.median(numeric_only=True))
val_X = val_X.fillna(train_X.median(numeric_only=True))

melbmodel = RandomForestRegressor(random_state=1)

melbmodel.fit(train_X, train_y)

predictprices = melbmodel.predict(val_X)

print(mean_absolute_error(val_y, predictprices))