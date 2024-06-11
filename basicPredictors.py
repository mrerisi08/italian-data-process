import pandas as pd
import numpy as np


df = pd.read_csv("./Italian Data/FIMMG dataset_ok(CASE1).csv")
file = open("Data Name - Sheet2.csv", 'r')
dict = {}
lines = file.readlines()

for x in lines:
    splt = x.split(",")
    splt[1] = splt[1][:-1]
    dict[splt[0]] = splt[1]

df.rename(columns=dict, inplace=True)
df = df.drop("Unnamed: 1862", axis=1)
df = df.where(df != 999, np.nan)
df = df[["Min Diastolic", "Max Diastolic", "Mean Diastolic", "Mean Systolic", "Min Systolic", "Max Systolic", "Gender", "Age", "LABEL"]]
print(df)


