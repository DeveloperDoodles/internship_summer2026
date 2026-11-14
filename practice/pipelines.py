import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

melbdata = pd.read_csv("melb_data.csv")

melbdata = melbdata.dropna(subset=['Price'])

y = melbdata.Price

X = melbdata.drop(['Price'], axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

catcols = [c for c in train_X.columns if train_X[c].dtype == "object"]
numcols = [c for c in train_X.columns if train_X[c].dtype in ['int64', 'float64']]

numtrans = SimpleImputer(strategy='median')

cattrans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numtrans, numcols),
        ('cat', cattrans, catcols)
    ]
)

melbmodel = RandomForestRegressor(random_state=1)

mypipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', melbmodel)
])

mypipeline.fit(train_X, train_y)

predictprices = mypipeline.predict(val_X)

print(mean_absolute_error(val_y, predictprices))