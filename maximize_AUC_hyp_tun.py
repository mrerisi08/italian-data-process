# with potential data leak

import pandas as pd
import numpy as np
import xgboost as xgb
import random
import os
from sklearn.metrics import roc_auc_score as auc
import itertools
from sklearn.metrics import precision_recall_fscore_support as prf1
import shap
import time


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(42)

import requests

def send_ifttt_notification(message):
    url = f"https://maker.ifttt.com/trigger/python_notif_tester/with/key/cUlA4Bn82wLJshLLMLQwBt"
    data = {"value1": message}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print("Notification sent successfully!")
    else:
        print(f"Failed to send notification: {response.status_code}, {response.text}")





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

RAND_ITERS = 15000
random_vals = {"mdep": [random.randint(1, 50) for _ in range(RAND_ITERS)],
               "eta": [random.uniform(0, 1) for _ in range(RAND_ITERS)],
               "min_weight": [random.randint(0,100) for _ in range(RAND_ITERS)],
               "alpha": [random.uniform(0, 5) for _ in range(RAND_ITERS)],
               "gamma": [random.uniform(0, 25) for _ in range(RAND_ITERS)],
               "lam": [random.uniform(0, 5) for _ in range(RAND_ITERS)],
               "scale_pos": [random.uniform(0, 10) for _ in range(RAND_ITERS)]
               }
params = ["mdep", "eta", "min_weight", "alpha", "gamma", "lam", "scale_pos"]
param_sets = []
params_val = {"mdep":4, "eta":0.298855629, "min_weight":5, "alpha":0.209002515, "gamma":6.886596893, "lam":0.631963664, "scale_pos":8.429048193}

# do the same thing but hold each subset of variables constant
# itertools.combinations(input_set, r) generates all combos of input_set for length r
# input_set needs to be "set" type.
# use lst.extend for appending
out = []
[out.extend(itertools.combinations(set(params), x+1)) for x in range(7)]
BEST_ACCURACY = 0
BES_AUC = 0
for sub in out:
    for _ in range(150):
        rand = {"mdep": random.randint(1, 50),
                        "eta": random.uniform(0, 1),
                        "min_weight": random.randint(0, 100),
                        "alpha": random.uniform(0, 5),
                        "gamma": random.uniform(0, 25),
                        "lam": random.uniform(0, 5),
                        "scale_pos": random.uniform(0, 10)
                                }
        for param in sub:
            rand[param] = params_val[param]
        param_sets.append(tuple([rand[a] for a in params]))

TOTAL_ITERS = len(param_sets)
TOTAL_START = time.time()
DEX = 0
for mdep, eta, min_weight, alpha, gamma, lam, scale_pos in param_sets:
    DEX += 1
    all_preds = []
    PARAM = {
            "max_depth": mdep,
            "eta": eta, # learning rate
            "subsample": 1,  # 1.0
            "colsample_bytree": 1,  # 1.0
            "colsample_bylevel": 1,  # 1.0
            "min_child_weight": min_weight,  # 1
            "silent": 1,
            "objective": "binary:logistic",  # HIS CODE DIDN'T HAVE IT, I THINK THIS IS WHAT IT SAID BUT IDK FOR SURE
            "eval_metric": "auc",
            # "tree_method": "gpu_hist",  # use "exact" on small data THE GPU PART THROWS AN ERROR
            "tree_method": "exact",
            "alpha": alpha,  # 0.01
            "gamma": gamma,  # 0.1
            "lambda": lam,  # 0.01
            "scale_pos_weight": scale_pos,
            "nthread": 8,
            "verbosity": 0, # Verbosity of printing messages. Valid values of 0 (silent), 1 (warning), 2 (info), and 3 (debug).
            "num_parallel_tree": 1
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

        bst = xgb.train(param, dtrain, 100, eval_list, early_stopping_rounds=1000)
        preds_val = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
        all_preds += list(preds_val)


    auc_val = auc(y, all_preds)
    all_preds = np.round(all_preds)
    acc_val = 100 * np.sum(all_preds == y) / len(y)
    print(acc_val)

    file = open("max_auc_tun_with_subsets.txt", 'a')
    file.write(f"{auc_val},{acc_val},{PARAM}\n")

    print(auc_val)
    if auc_val > BES_AUC:
        BES_AUC = auc_val
        send_ifttt_notification(f"New best AUC reached at index {DEX}. {auc_val:.5f},{PARAM}")
    if acc_val > BEST_ACCURACY:
        BEST_ACCURACY = acc_val
        send_ifttt_notification(f"New best accuracy reached at index {DEX}. {acc_val:.5f},{PARAM}")
    if DEX % 500 == 0:
        LAST_TIME = time.time()
        send_ifttt_notification(f"Reached {DEX}th iteration ({DEX/TOTAL_ITERS*100:.2f}%). Best AUC: {BES_AUC}, Best accuracy: {BEST_ACCURACY}, Time elapsed: {(time.time()-TOTAL_START)/3600:.2f} hrs, Time since last: {(time.time()-LAST_TIME)/60:.2f} min.")


send_ifttt_notification(f"Completed. Operation at 100% completion.")