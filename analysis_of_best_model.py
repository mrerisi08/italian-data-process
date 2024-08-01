# 23099,0.910253623188406,84.01150842581175,4.0,0.298855629,1.0,1.0,1.0,5.0,1.0,"[""'binary:logistic'""]","[""'auc'""]","[""'exact'""]",0.209002515,7.464987507779308,0.631963664,8.429048193


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
import requests
import pickle
import matplotlib.pyplot as plt

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


all_preds = []
PARAM = {
        "max_depth": 4,
        "eta": 0.298855629,
        "subsample": 1,
        "colsample_bytree": 1,
        "colsample_bylevel": 1,
        "min_child_weight": 5,
        "silent": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "exact",
        "alpha": 0.209002515,
        "gamma": 7.464987507779308,
        "lambda": 0.631963664,
        "scale_pos_weight": 8.429048193,
        "nthread": 8,
        "verbosity": 0,
        "num_parallel_tree": 1
    }
CONTRIB = []
models = []
expected_values = []
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
    #
    param = PARAM
    #
    bst = xgb.train(param, dtrain, 500, eval_list, early_stopping_rounds=1000)
    # with open(f'model_files/bst_mdl{j}.pkl', 'rb') as file:
    #     bst = pickle.load(file)
    preds_val = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    all_preds += list(preds_val)
    contribs = bst.predict(dtest, pred_contribs=True)
    contribs = contribs[:, :-1]
    CONTRIB.append(contribs)
    expected_value = np.mean(bst.predict(dtest))
    expected_values.append(expected_value)
    # with open(f'model_files/bst_mdl{j}.pkl', 'wb') as file:
    #     pickle.dump(bst, file)


auc_val = auc(y, all_preds)
all_preds = np.round(all_preds)
acc_val = 100 * np.sum(all_preds == y) / len(y)
print(acc_val)
print(auc_val)

shap_vals = np.concatenate(tuple(CONTRIB))

print(shap_vals.shape)
combined_expected_value = np.mean(expected_values)

shap_exp = shap.Explanation(values=shap_vals, base_values=combined_expected_value, data=X,feature_names=list(df.columns))


shap.plots.beeswarm(shap_exp) # displays it
# shap.plots.beeswarm(shap_exp, show=False,max_display=25)
# plt.savefig("extended_features.png",bbox_inches="tight")
# ^^ will not display it but will save to the file. bbox_inches makes everything fit properly


