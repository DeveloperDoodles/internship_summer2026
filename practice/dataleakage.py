import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

melbdata = pd.read_csv("melb_data.csv")

melbdata = melbdata.dropna()

y = melbdata.Price

melbfeat = [
    'Rooms',
    'Bathroom',
    'Landsize',
    'Lattitude',
    'Longtitude'
]

X = melbdata[melbfeat]

train_X, val_X, train_y, val_y = train_test_split(
    X,
    y,
    random_state=0
)

melbmodel = RandomForestRegressor(random_state=1)

melbmodel.fit(train_X, train_y)

predictprices = melbmodel.predict(val_X)

print(mean_absolute_error(val_y, predictprices))