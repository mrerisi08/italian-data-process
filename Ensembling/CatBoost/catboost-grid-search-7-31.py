import time
import numpy as np
import random
import os
from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
import itertools
import requests


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

def send_ifttt_notification(message):
    url = f"https://maker.ifttt.com/trigger/python_notif_tester/with/key/cUlA4Bn82wLJshLLMLQwBt"
    data = {"value1": message}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print("Notification sent successfully!")
    else:
        print(f"Failed to send notification: {response.status_code}, {response.text}")



df = pd.read_csv("Case I 6-12 binary ft >10 positives and normalized.csv").drop("Unnamed: 0",
                                                                                axis=1)  # minimum processing

df = df.drop(["GLYCATED HEMOGLOBIN"], axis=1)  # data leaks

categorcial_features = list(df.drop(['Min Diastolic', 'Max Diastolic', 'Mean Diastolic',
                                     'Mean Systolic', 'Min Systolic', 'Max Systolic',
                                     'Age', 'LABEL'], axis=1).columns)

df = df.astype({x: "category" for x in categorcial_features})
# print(df.dtypes)

X = df.drop("LABEL", axis=1).to_numpy()
y = df["LABEL"].to_numpy()

p = np.random.permutation(len(y))
X = X[p]
y = y[p]

K_FOLDS = 5
fold_size = int(len(X) / K_FOLDS)
FOLD_ARRAYS = {}
for fold in range(4):
    start = fold * fold_size
    end = (fold + 1) * fold_size
    FOLD_ARRAYS[fold] = ([*X[:start], *X[end:]], [*y[:start], *y[end:]], X[start:end], y[start:end])
start = 4 * fold_size
end = 5 * fold_size
FOLD_ARRAYS[4] = ([*X[:start], *X[end:]], [*y[:start], *y[end:]], X[start:], y[start:])

param_list = {"depth": [3, 5, 10],
              "iterations": [10, 50, 100, 250, 1000],
              "learning_rate": [0.001, 0.01, 0.1, 0.5],
              "l2_leaf_reg": [0, 1, 2, 3, 5, 10, 25],
              "scale_pos_weight": [1, 10, 100],
              "verbose": True
              }


def get_param_list(params):
    products = list(itertools.product(*[params[k] for k in params if isinstance(params[k], list)]))
    list_names = [k for k in params if isinstance(params[k], list)]
    not_list_names = [k for k in params if not isinstance(params[k], list)]
    out_list = []
    for param_set in products:
        dict = {}
        for dex in range(len(param_set)):
            dict[list_names[dex]] = param_set[dex]
        for name in not_list_names:
            dict[name] = params[name]
        out_list.append(dict)
    return out_list

PARAMS = get_param_list(param_list)

total_start = time.time()
bst_auc = -1
for dex, param in enumerate(PARAMS):
    dex += 1
    all_preds = []
    for fold in range(K_FOLDS):
        X_train, y_train, X_test, y_test = FOLD_ARRAYS[fold]

        mdl = CatBoostClassifier(**param, eval_metric='AUC')
        mdl.fit(X_train, y_train)
        predictions = np.rot90(mdl.predict_proba(X_test))[0]
        all_preds += list(predictions)

    auc_val = auc(y, all_preds)
    with open("catboost-grid-search-7-31-data.csv", "a") as file:
        str = f"\n{auc_val}"
        for x in param:
            str = f"{str},{param[x]}"
        file.write(str)

    if auc_val > bst_auc:
        bst_auc = auc_val

    if dex % 3 == 0:
        file = open("iter.txt", 'w')
        str = (f"{dex} / {len(PARAMS)}\n"
               f"best auc: {bst_auc}\n"
               f"time elapsed: {(time.time()-total_start)/3600:.3f} hrs\n"
               f"estim remaining: {(len(PARAMS)-dex)*(time.time()-total_start)/dex/3600:.3f} hrs")
        file.write(str)


send_ifttt_notification(f"FINISHED. Best AUC: {bst_auc}")