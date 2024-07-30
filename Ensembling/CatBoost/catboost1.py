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
all_preds = []
for fold in range(K_FOLDS):
    start = fold * fold_size
    end = (fold + 1) * fold_size
    print(fold, start, end)
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


    model = CatBoostClassifier(iterations=2,
                               depth=2,
                               learning_rate=1,
                               loss_function='Logloss',
                               verbose=True)
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
print(auc_val)

