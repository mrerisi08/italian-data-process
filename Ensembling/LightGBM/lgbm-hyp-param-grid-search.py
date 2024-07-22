import random
import os
import numpy as np
import itertools


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(42)

param = {"num_iterations": 100, "learning_rate": random.uniform(0, 1), "max_leaf": random.randint(5, 50),
         "min_data": random.randint(0, 50), "feature_fraction": random.randint(1, 10) / 10,
         "reg_alpha": random.uniform(0, 10), "lambda": random.uniform(0, 10),
         "min_split_gain": random.uniform(0, 100), "verbosity": -1, "max_bin": random.randint(1, 50) * 10,
         "objective": "binary", "metric": "auc"}

param_ranges = {"eta": [0.001, 0.01, 0.1, 0.5],
                "max_leaf": [5, 10, 15, 20, 31, 50],
                "min_data": [5, 10, 20, 50],
                "feature_fraction": [0.1, 0.25, 0.5, 0.75, 1],
                "reg_alpha": [0.0, 0.01, 1, 5, 25],
                "lambda": [0.0, 0.01, 1, 5, 25],
                "min_split_gain": [0, 50, 100, 250],
                "verbosity": -1,
                "max_bin": [100, 255, 500],
                "objective": "binary",
                "metric": "auc",
                "num_threads": 5
                }


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
for idx, config in enumerate(final_hyperparams):
    print(f"Configuration {idx + 1}: {config}")