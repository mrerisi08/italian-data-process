import itertools
import random


input_set = {"mdep", "eta", "min_weight", "alpha", "gamma", "lam", "scale_pos"}
params_val = {"mdep":4, "eta":0.298855629, "min_weight":5, "alpha":0.209002515, "gamma":6.886596893, "lam":0.631963664, "scale_pos":8.429048193}
out = []


[out.extend(itertools.combinations(input_set, x+1)) for x in range(7)]


params = ["mdep", "eta", "min_weight", "alpha", "gamma", "lam", "scale_pos"]
param_sets = []
params_val = {"mdep":4, "eta":0.298855629, "min_weight":5, "alpha":0.209002515, "gamma":6.886596893, "lam":0.631963664, "scale_pos":8.429048193}

# for const in params:
#     for _ in range(500):
#         rand = {"mdep": random.randint(1, 50),
#                 "eta": random.uniform(0, 1),
#                 "min_weight": random.randint(0, 100),
#                 "alpha": random.uniform(0, 5),
#                 "gamma": random.uniform(0, 25),
#                 "lam": random.uniform(0, 5),
#                 "scale_pos": random.uniform(0, 10)
#                        }
#         rand[const] = params_val[const]
#         param_sets.append(tuple([rand[a] for a in params]))

print(out)
out_vals = []
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
        out_vals.append(tuple([rand[a] for a in params]))

print(len(out_vals))