
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
print(len(list(df.columns)))

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


for gmma in []:
    all_preds = []
    PARAM = {
            "max_depth": 10,
            "eta": 0.5, # learning rate
            "subsample": 1,  # 1.0
            "colsample_bytree": 1,  # 1.0
            "colsample_bylevel": 1,  # 1.0
            "min_child_weight": 1,  # 1
            "objective": "binary:logistic",  # HIS CODE DIDN'T HAVE IT, I THINK THIS IS WHAT IT SAID BUT IDK FOR SURE
            "eval_metric": "auc",
            # "tree_method": "gpu_hist",  # use "exact" on small data THE GPU PART THROWS AN ERROR
            "tree_method": "exact",
            "alpha": 0.01,  # 0.01
            "gamma": 10,  # 0.1
            "lambda": 0.01,  # 0.01
            "scale_pos_weight": 1,
            "nthread": 8,
            "verbosity": gmma, # Verbosity of printing messages. Valid values of 0 (silent), 1 (warning), 2 (info), and 3 (debug).
            "num_parallel_tree": 1,
            "disable_default_eval_metric": True
        }

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

        param = PARAM

        bst = xgb.train(param, dtrain, 10, eval_list, early_stopping_rounds=1000)
        preds_val = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
        all_preds += list(preds_val)


    auc_val = auc(y, all_preds)
    all_preds = np.round(all_preds)
    acc_val = 100 * np.sum(all_preds == y) / len(y)
    print(acc_val)

    file = open("hyp_tun_auc.txt", 'a')
    file.write(f"{auc_val},{acc_val},{PARAM}\n")

    print(auc_val)

