# with potential data leak

import pandas as pd
import numpy as np
import xgboost as xgb
import random
import os
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import precision_recall_fscore_support as prf1
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


k = 5
fold_size = int(len(X) / k)
print(fold_size * k)

all_preds = []
SHAPS = []
gainImp = []

for j in range(k):
    start = j*fold_size
    end = (j+1)*fold_size
    if j!=4:
        X_train = [*X[:start], *X[end:]]
        y_train = [*y[:start], *y[end:]]
        X_test = X[start:end]
        y_test = y[start:end]
    else:
        X_train = [*X[:start]]
        y_train = [*y[:start]]
        X_test = X[start:]
        y_test = y[start:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    eval_list = [(dtrain, 'train'), (dtest, 'eval')]

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

    bst = xgb.train(param, dtrain, 10, eval_list, early_stopping_rounds=1000)
    preds_val = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    all_preds += list(preds_val)
    explainer = shap.Explainer(bst)
    shap_values = explainer(X_train)
    SHAPS.append(shap_values)
    gainImp.append(bst.get_score(importance_type='gain'))
    print(bst.get_score(importance_type='gain'))


auc_val = auc(y, all_preds)
all_preds = np.round(all_preds)
acc_val = 100 * np.sum(all_preds == y) / len(y)
print(auc_val)
print(acc_val)

lotsofmetrics = prf1(y, all_preds) # prints precision, recall, fbeta_score (f1) and support
# 1d arrays bc each corresponds to the different classes. I'm assuming [0] == 0 and [1] == 1
# meaning that the first value is for non diabetes
for a, b in zip(["precision", "recall", "f1", "support"], lotsofmetrics):
    print(f"{a}: {b}")

gainImpOut = {}
for x in gainImp:
    for y in x:
        if y not in gainImpOut:
            gainImpOut[y] = []
        gainImpOut[y].append(x[y])

print(gainImpOut)

for x in gainImpOut:
    while len(gainImpOut[x]) != 5:
        gainImpOut[x].append(np.nan)



impDf = pd.DataFrame(gainImpOut)
print(impDf)
impDf.to_csv("weight_ft_imp_by_k-fold.csv")

# shap.plots.beeswarm(SHAPS)


# explainer = shap.Explainer(bst, feature_names=list(df.drop("LABEL", axis=1).columns))
# Xd = xgb.DMatrix(X, label=y)
# explanation = explainer(X_train)
# shap.plots.beeswarm(explanation)

