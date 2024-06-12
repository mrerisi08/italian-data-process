import pandas as pd
import numpy as np


df = pd.read_csv("Processed_Case1_6-6.csv")
df = df.drop("Unnamed: 0", axis=1)


less10 = []
for col in df.drop(["Min Diastolic","Max Diastolic","Mean Diastolic","Min Systolic","Max Systolic","Mean Systolic","Age","Gender","LABEL"], axis=1):
    zeros = 0
    ones = 0
    for cell in df[col].iloc:
        if cell == 0:
            zeros += 1
            continue
        if cell == 1:
            ones += 1
            continue
    if ones < 10:
        print(col, zeros, ones)
        less10.append(col)

print(less10)
df = df.drop(less10, axis=1)
df.to_csv("Case I 6-12 binary ft >10 positives.csv")