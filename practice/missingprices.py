import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melbdata = pd.read_csv("melb_data.csv")

y = melbdata.Price

melbfeat = [
    'Rooms',
    'Bathroom',
    'Landsize',
    'BuildingArea',
    'YearBuilt',
    'Lattitude',
    'Longtitude'
]

X = melbdata[melbfeat]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

train_X = train_X.fillna(train_X.median(numeric_only=True))
val_X = val_X.fillna(train_X.median(numeric_only=True))

melbmodel = RandomForestRegressor(random_state=1)
melbmodel.fit(train_X, train_y)

predictprices = melbmodel.predict(val_X)

print(mean_absolute_error(val_y, predictprices))