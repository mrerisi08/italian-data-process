import numpy as np
import random
import os
from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
import itertools
import time


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




def hyp_param_grid():
    param_ranges = {"iterations":[500, 1, 5, 50, 100],
           "depth":[1, 5, 10, 25, 50, 100],
           "learning_rate":[0.01,0.1,0.5,1],
           "loss_function":'Logloss',
           "verbose":False,
           "eval_metric":"AUC",
           "random_seed":42,
           "reg_lambda":[0,0.1,1,10],
           "scale_pos_weight":[0.1,1,10],
           "min_data_in_leaf":[5, 10, 50, 100],
                    "thread_count":7}


    constant_hyperparams = {k: v for k, v in param_ranges.items() if not isinstance(v, list)}
    list_hyperparams = {k: v for k, v in param_ranges.items() if isinstance(v, list)}

    # Generate all combinations of list hyperparameters
    list_keys = list(list_hyperparams.keys())
    list_values = list(list_hyperparams.values())
    combinations = list(itertools.product(*list_values))
    final_hyperparams = []
    for combination in combinations:
        config = constant_hyperparams.copy()
        config.update(zip(list_keys, combination))
        final_hyperparams.append(config)
    return final_hyperparams

HYPERPARAMETERS = hyp_param_grid()
print(len(HYPERPARAMETERS))
round = 0
bst_auc = -1
start_time = time.time()
for param in HYPERPARAMETERS:
    round += 1
    all_preds = []
    try:
        for fold in range(K_FOLDS):
            # print(fold)
            start = fold * fold_size
            end = (fold + 1) * fold_size
            if fold != 4:
                X_train = [*X[:start], *X[:end]]
                y_train = [*y[:start], *y[:end]]
                X_test = X[start:end]
                y_test = y[start:end]
            else:
                X_train = X[:start]
                y_train = y[:start]
                X_test = X[start:]
                y_test = y[start:]

            # initialize data


            model = CatBoostClassifier(**param)
            # train the model
            model.fit(X_train, y_train)
            # make the prediction using the resulting model
            preds_class = model.predict(X_test) # gives 0 or 1
            preds_proba = model.predict_proba(X_test) # gives prob (no shit)
            preds_proba = np.hsplit(preds_proba,2)[1]
            all_preds += list(preds_proba)
            # np.concatenate((all_preds, preds_proba), axis=0)

        # print(all_preds)
        # preds_proba = np.hsplit(preds_proba,2)[1]
        auc_val = auc(y, all_preds)
        # print(round, auc_val)
        with open("catboost-grid-search.csv",'a') as file:
            str = f"\n{auc_val}"
            for a in param:
                str = f"{str},{param[a]}"
            file.write(str)

        if auc_val > bst_auc:
            bst_auc = auc_val
            print(f"Best AUC now {auc_val:.5f} ({round})")

        if round % 25 == 0:
            print(f"{round} ({100*round/len(HYPERPARAMETERS):.2f}%) - {(time.time()-start_time)/60:.2f} min, estim remaining: "
                  f"{(len(HYPERPARAMETERS)-round)*((time.time()-start_time)/round)/60:.2f} min")
    except:
        print(f"ERROR ({round}): {param}")



# check how many it writes to csv bc its unclear if it is or isnt writing the bad rows to file