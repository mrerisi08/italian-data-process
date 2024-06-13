# with potential data leak

import pandas as pd
import numpy as np
import xgboost as xgb
import random
import os
from sklearn.metrics import roc_auc_score as auc
import shap


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(42)

# process and load to numpy
df = pd.read_csv("Case I 6-12 binary ft >10 positives and normalized.csv")
df = df.drop("Unnamed: 0", axis=1)

df = df.drop("GLYCATED HEMOGLOBIN", axis=1)

X = df.drop("LABEL", axis=1).to_numpy()
y = df["LABEL"].to_numpy()

# permutation/randomizer
p = np.random.permutation(len(y))
X = X[p]
y = y[p]

# split based on test_percent
test_percent = .2
numTest = round(len(y) * test_percent)
numTrain = round(len(y) * (1 - test_percent))  # not actually relevant
X_test, y_test = X[:numTest], y[:numTest]
X_train, y_train = X[numTest:], y[numTest:]

# get ft names

# DMatrices
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
eval_list = [(dtrain, 'train'), (dtest, 'eval')]

# paramaters
param = {
    "max_depth": 10,  # 10
    "eta": 0.05,  # 0.05
    "subsample": 1,  # 1.0
    "colsample_bytree": 1,  # 1.0
    "colsample_bylevel": 1,  # 1.0
    "min_child_weight": 1,  # 1
    "silent": 1,
    "objective": "binary:logistic",  # HIS CODE DIDN'T HAVE IT, I THINK THIS IS WHAT IT SAID BUT IDK FOR SURE
    "eval_metric": "auc",
    # "tree_method": "gpu_hist",  # use "exact" on small data THE GPU PART THROWS AN ERROR
    "tree_method": "exact",
    "alpha": 0.01,  # 0.01
    "gamma": 0.1,  # 0.1
    "lambda": 0.01,  # 0.01
    "scale_pos_weight": 1,
    "nthread": 8
}

bst = xgb.train(param, dtrain, 400, eval_list, early_stopping_rounds=1000)
preds_val = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
auc_val = auc(y_test, preds_val)
print(preds_val)
preds_val = np.round(preds_val)
acc_val = 100 * np.sum(preds_val == y_test) / len(y_test)
print(auc_val)
print(acc_val)

ft_imp = bst.get_score(importance_type='weight')
print(ft_imp)
maximp = 0
maxft = ""
for a in ft_imp:
    if ft_imp[a] > maximp:
        maximp = ft_imp[a]
        maxft = a
print(maximp, maxft)

explainer = shap.Explainer(bst, feature_names=list(df.drop("LABEL", axis=1).columns))
Xd = xgb.DMatrix(X, label=y)
explanation = explainer(X_train)
shap.plots.beeswarm(explanation)

