import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


df = pd.read_csv("correlationAnalysis.csv")
df = df.drop("Unnamed: 0", axis=1)
print(df.drop("names", axis=1))
print([x for x in df])
# for a in df[["posFtNegLbl", "posFtPosLbl", "negFtNegLbl","negFtPosLbl", "elseCase"]]:
#     print(a, df.mean())

#print(df.drop("names",axis=1).mean())

for a, b in zip(df["names"].iloc, df["elseCase"].iloc):
    if b != 0:
        print(a, b)

