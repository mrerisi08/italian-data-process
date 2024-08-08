import pandas as pd
import numpy as np
import shap
from sklearn.metrics import roc_auc_score as auc
import random
import os
import re
import pickle as pkl
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(42)

df = pd.read_csv("Case I 6-12 binary ft >10 positives and normalized.csv", na_values=np.nan)
df = df.drop("Unnamed: 0", axis=1)

df = df.drop(["GLYCATED HEMOGLOBIN"], axis=1) # data leaks

df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '',x))
# https://stackoverflow.com/questions/60582050/lightgbmerror-do-not-support-special-json-characters-in-feature-name-the-same
# looks like what this does is remove spaces in feature names (at a high level), used to fix an lgbm error


X = df.drop("LABEL", axis=1).to_numpy()
y = df["LABEL"].to_numpy()

p = np.random.permutation(len(y))
X = X[p]
y = y[p]

fold_size = int(len(y) / 5)

PARAMS = {
          "lgb": {'objective': "binary", 'metric': "auc", 'verbosity': -1,'num_threads': 9, 'eta': 0.1, 'max_leaf': 10,'min_data': 5, 'feature_fraction': 0.25, 'reg_alpha': 1,'lambda': 0, 'min_split_gain': 0, 'max_bin': 255, "seed": 64},
          "xgb": {"max_depth": 4, "eta": 0.298855629,"subsample": 1,"colsample_bytree": 1,"colsample_bylevel": 1,"min_child_weight": 5,"silent": 1,"objective": "binary:logistic","eval_metric": "auc","tree_method": "exact","alpha": 0.209002515,"gamma": 7.464987507779308,"lambda": 0.631963664,"scale_pos_weight": 8.429048193,"nthread": 8,"verbosity": 0,"num_parallel_tree": 1},
          "cat": {"depth": 10,"iterations": 1000,"learning_rate": 0.01,"l2_leaf_reg": 10,"scale_pos_weight": 1,"verbose": False} # was true previously
         }

BINARY_FT = list(df.drop(["LABEL","MinDiastolic","MaxDiastolic","MeanDiastolic","MinSystolic","MaxSystolic","MeanSystolic", "Age"], axis=1).columns)

PREDICTIONS = {"lgb": [], "xgb": [], "cat": [], "all": []}

AUC_VALS = {"lgb": {}, "xgb": {}, "cat": {}, "all": {}}

for j in range(5):
    print(j)
    start = j * fold_size
    end = (j + 1) * fold_size
    if j != 4:
        X_train, X_test = [*X[:start], *X[end:]], X[start:end]
        y_train, y_test = [*y[:start], *y[end:]], y[start:end]
    else:
        X_train, X_test = X[:start], X[start:]
        y_train, y_test = y[:start], y[start:]

    # lgb train
    lgb_X_train = np.array(X_train)
    lgb_y_train = np.array(y_train)
    lgb_train_data = lgb.Dataset(lgb_X_train, label=lgb_y_train, feature_name=list(df.drop("LABEL", axis=1).columns), categorical_feature=BINARY_FT)
    lgb_mdl = lgb.train(PARAMS["lgb"], lgb_train_data)
    print("lgb trained")

    # xgb train
    xgb_dtrain = xgb.DMatrix(X_train, label=y_train)
    xgb_dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_eval_list = [(xgb_dtrain, 'train'), (xgb_dtest, 'eval')]
    xgb_mdl = xgb.train(PARAMS["xgb"], xgb_dtrain, 500) # removed arg that contained xgb_eval_list
    print("xgb trained")

    # cat train
    cat_mdl = CatBoostClassifier(**PARAMS["cat"], eval_metric='AUC')
    cat_mdl.fit(X_train, y_train)
    print("cat trained")

    # save to pickles
    with open(f'models/lgb/lgb_{j}.pkl', 'wb') as file:
        pkl.dump(lgb_mdl, file)
    with open(f'models/xgb/xgb_{j}.pkl', 'wb') as file:
        pkl.dump(xgb_mdl, file)
    with open(f'models/cat/cat_{j}.pkl', 'wb') as file:
        pkl.dump(cat_mdl, file)
    print("pickled")

    # preds
    lgb_pred = list(lgb_mdl.predict(X_test))
    print("lgb pred")
    xgb_pred = list(xgb_mdl.predict(xgb_dtest))
    print("xgb pred")
    cat_pred = list(np.rot90(cat_mdl.predict_proba(X_test))[0])
    print("cat pred")
    all_pred = list(np.mean(np.array([lgb_pred, xgb_pred, cat_pred]), axis=0))
    print("all pred")

    PREDICTIONS["lgb"] += lgb_pred
    PREDICTIONS["xgb"] += xgb_pred
    PREDICTIONS["cat"] += cat_pred
    PREDICTIONS["all"] += all_pred


    # sub aucs
    AUC_VALS["lgb"][j] = auc(y_test, lgb_pred)
    print("lgb auc")
    AUC_VALS["xgb"][j] = auc(y_test, xgb_pred)
    print("xgb auc")
    AUC_VALS["cat"][j] = auc(y_test, cat_pred)
    print("cat auc")
    AUC_VALS["all"][j] = auc(y_test, all_pred)
    print("all auc")

print("done with all folds")

AUC_VALS["lgb"]["all"] = auc(y, PREDICTIONS["lgb"])
AUC_VALS["xgb"]["all"] = auc(y, PREDICTIONS["xgb"])
AUC_VALS["cat"]["all"] = auc(y, PREDICTIONS["cat"])
AUC_VALS["all"]["all"] = auc(y, PREDICTIONS["all"])

print(AUC_VALS)