import pandas as pd

file = open("max_auc_tun_with_subsets copy.csv",'r')
f_ile = file.readlines()
file.close()
file = f_ile

out = []

for row in file:
    fin = []
    row = row.split(",")
    fin.append(float(row[0]))
    fin.append(float(row[1]))
    row = row[2:]
    row = row[:-3]
    for val in row:
        splt = val.split(": ")
        try:
            fin.append(float(splt[1]))
        except ValueError:
            fin.append(splt[1:])


    out.append(fin)
    # print(fin)

df = pd.DataFrame(out, columns=["AUC", "ACC", "max_depth","eta","subsample", "colsamp_bytree", "colsamp_bylvl", "min_child_weight", "silent", "objective", "eval_metric", "tree_method","alpha","gamma", "lambda", "scale_pos_weight"])
print(df)
df.to_csv("max_auc_tun_with_subsets copy.csv")