import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

melbdata = pd.read_csv("melb_data.csv")

melbdata = melbdata.dropna(subset=['Price'])

y = melbdata.Price

X = melbdata.drop(['Price'], axis=1)

catcols = [c for c in X.columns if X[c].dtype == "object"]
numcols = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]

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

scores = -1 * cross_val_score(
    mypipeline,
    X,
    y,
    cv=5,
    scoring='neg_mean_absolute_error'
)

print(scores)
print(scores.mean())