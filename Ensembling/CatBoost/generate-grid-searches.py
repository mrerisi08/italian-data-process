import itertools


RANGES = {"depth":[3,5,10],
          "iterations":[10,50,100,250,1000],
          "learning_rate":[0.001,0.01,0.1,0.5],
          "l2_leaf_reg":[0,1,2,3,5,10,25]
          }



# RANGES = {"eta": [0.001, 0.01, 0.1, 0.5],
#                 "max_leaf": [5, 10, 15, 20, 31, 50],
#                 "min_data": [5, 10, 20, 50],
#                 "feature_fraction": [0.1, 0.25, 0.5, 0.75, 1],
#                 "reg_alpha": [0.0, 0.01, 1, 5, 25],
#                 "lambda": [0.0, 0.01, 1, 5, 25],
#                 "min_split_gain": [0, 50, 100, 250],
#                 "max_bin": [100, 255, 500],
#                 "verbosity": -1,
#                 "objective": "binary",
#                 "metric": "auc",
#                 "num_threads": 5
#                 }
products = list(itertools.product(*[RANGES[k] for k in RANGES if isinstance(RANGES[k], list)]))
list_names = [k for k in RANGES if isinstance(RANGES[k], list)]
not_list_names = [k for k in RANGES if not isinstance(RANGES[k], list)]
out_list = []
for params in products:
    dict = {}
    for dex in range(len(params)):
        dict[list_names[dex]] = params[dex]
    for name in not_list_names:
        dict[name] = RANGES[name]
    out_list.append(dict)

print(out_list)
print(len(out_list))

