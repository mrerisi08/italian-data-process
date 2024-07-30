# not as good as manual grid search

import time

import numpy as np
import random
import os
from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.metrics import roc_auc_score as auc



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

df = pd.read_csv("Case I 6-12 binary ft >10 positives and normalized.csv")
df = df.drop("Unnamed: 0", axis = 1)

df = df.drop("GLYCATED HEMOGLOBIN", axis=1)
X = df.drop("LABEL", axis=1).to_numpy()
y = df["LABEL"].to_numpy()
p = np.random.permutation(len(y))
X = X[p]
y = y[p]

K_FOLDS = 5
fold_size = int(len(X) / K_FOLDS)


params = [{"iterations":100,
           "depth":5,
           "learning_rate":1,
           "loss_function":'Logloss',
           "verbose":False,
           "eval_metric":"AUC",
           "random_seed":42,
           "reg_lambda":1}]
bst_auc = -1
bst_param = None
# strt = time.time()
# for iters in [5, 10, 50, 100, 500]:
#     for etas in [0.001,0.01,0.1,0.5,1]:
#         all_preds = []
#         for fold in range(K_FOLDS):
#             # print(fold)
#             start = fold * fold_size
#             end = (fold + 1) * fold_size
#             if fold != 4:
#                 X_train = [*X[:start], *X[:end]]
#                 y_train = [*y[:start], *y[:end]]
#                 X_test = X[start:end]
#                 y_test = y[start:end]
#             else:
#                 X_train = X[:start]
#                 y_train = y[:start]
#                 X_test = X[start:]
#                 y_test = y[start:]
#
#             # initialize data
#
#
#             model = CatBoostClassifier(eval_metric = "AUC", eta=etas, iterations=iters, verbose=False)
#             # train the model
#             model.fit(X_train, y_train)
#             # make the prediction using the resulting model
#             preds_class = model.predict(X_test) # gives 0 or 1
#             preds_proba = model.predict_proba(X_test) # gives prob (no shit)
#             preds_proba = np.hsplit(preds_proba,2)[1]
#             all_preds += list(preds_proba)
#             # np.concatenate((all_preds, preds_proba), axis=0)
#
#
#         # print(all_preds)
#         # preds_proba = np.hsplit(preds_proba,2)[1]
#         auc_val = auc(y, all_preds)
#         print(auc_val)
#         if auc_val > bst_auc:
#             bst_auc = auc_val
#             bst_param = {"eta":etas, "iters":iters}
#
#
# print(bst_auc, bst_param, time.time()-strt)
# strt = time.time()
param_grid = {"iterations":[5,10, 50, 100],"eta":[0.001,0.01,0.1,0.5,1]}

# mdl = CatBoostClassifier(eval_metric="AUC").grid_search(param_grid, X, y=y, cv=5, shuffle=False, partition_random_seed=42, verbose=False)
# print(mdl)
# the claim is that the best is eta:0.5, iterations:10
# print(time.time()-strt)
# eta 1 iters 5