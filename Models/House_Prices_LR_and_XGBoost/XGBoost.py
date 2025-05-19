#XGBoost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

X_train = pd.read_csv("data/train.csv")
X_test  = pd.read_csv("data/test.csv")

y = X_train["SalePrice"]
X_train = X_train.drop(["SalePrice", "Id"], axis=1)
X_test = X_test.drop("Id", axis=1)

num_cols = X_train.select_dtypes(["int64", "float64"]).columns
cat_cols = X_train.select_dtypes("object").columns

# highly skewed numeric features, learned this through GPT o3, not my work
skewed_cols = (
    X_train[num_cols]
    .apply(lambda s: s.skew())
    .abs()
    .loc[lambda s: s > 0.75]
    .index
    .tolist()
)
num_non_skew = [c for c in num_cols if c not in skewed_cols]

numeric_pipe = Pipeline(
    [("impute", SimpleImputer(strategy="median"))]
)

log_pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ]
)

categorical_pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    [
        ("num",  numeric_pipe, num_non_skew),
        ("log",  log_pipe,     skewed_cols),
        ("cat",  categorical_pipe, cat_cols),
    ]
)

xgb_params = dict(
    objective="reg:squarederror",
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=62,
)

model = XGBRegressor(**xgb_params)

pipe = Pipeline(
    [
        ("prep", preprocess),
        ("reg",  model),
    ]
)

y_log = np.log1p(y)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_log, test_size=0.2, random_state=62
)

pipe.fit(X_tr, y_tr)

cv_rmse = -cross_val_score(
    pipe, X_train, y_log,
    cv=5,
    scoring="neg_root_mean_squared_error",
).mean()
print(cv_rmse)
#0.1238 RMSE

pipe.fit(X_train, y_log)
test_pred = np.expm1(pipe.predict(X_test))

submission = pd.read_csv('data/sample_submission.csv')

submission['SalePrice'] = test_pred

submission.to_csv('data/xgb_submission.csv', index=False)
#0.12961 on test set. Not bad but still more room for improvement.