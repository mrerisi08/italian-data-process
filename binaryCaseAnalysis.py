import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import random

df = pd.read_csv("Processed_Case1_6-6.csv", na_values=np.nan)
df = df.drop("Unnamed: 0", axis=1)
numericaldf = df[["Min Systolic", "Max Systolic", "Mean Systolic", "Min Diastolic", "Max Diastolic","Mean Diastolic", "Age", "Gender"]]
df = df.drop(["Min Systolic", "Max Systolic", "Mean Systolic", "Min Diastolic", "Max Diastolic","Mean Diastolic", "Age", "Gender"], axis=1)

def binary_analysis():
    outDict = {}
    for dex, x in enumerate(df.columns):
        outDict[x] = {"total 0":0, "total 1":0, "0 & diabetes":0, "0 & no diabetes":0, "1 & diabetes":0, "1 & no diabetes":0}
        for y in range(len(list(df[x].iloc))):
            if df[x].iloc[y] == 1:
                outDict[x]["total 1"] += 1
                if df["LABEL"].iloc[y] == 1:
                    outDict[x]["1 & diabetes"] += 1
                elif df["LABEL"].iloc[y] == 0:
                    outDict[x]["1 & no diabetes"] += 1
                else:
                    print("ERROR", x, y)
            elif df[x].iloc[y] == 0:
                outDict[x]["total 0"] += 1
                if df["LABEL"].iloc[y] == 1:
                    outDict[x]["0 & diabetes"] += 1
                elif df["LABEL"].iloc[y] == 0:
                    outDict[x]["0 & no diabetes"] += 1
                else:
                    print("ERROR", x, y)
            else:
                print("ERROR", x, y)
        print(dex, x, outDict[x]["1 & diabetes"])

    outFile = open("binaryAnalysis.csv", 'w')
    outFile.write("ft name; total 0; total 1; 0 & diabetes; 0 & no diabetes; 1 & diabetes; 1 & no diabetes\n")
    for dex, t in enumerate(outDict):
        s = outDict[t]
        print(dex, t, round(s["0 & diabetes"]/s["total 0"],3), round(s["1 & diabetes"]/s["total 1"],3))
        outFile.write(f"{t};{s["total 0"]};{s["total 1"]};{s["0 & diabetes"]};{s["0 & no diabetes"]};{s["1 & diabetes"]};{s["1 & no diabetes"]}\n")



def pcc_for_numerical_features():
    for x in numericaldf.columns:
        badIndex = []
        # the nan_mask was GPT
        nan_mask = np.isnan(np.array(numericaldf[x]))
        vals = np.array(numericaldf[x])[~nan_mask]
        label = np.array(df["LABEL"])[~nan_mask]
        print(f"{x}, {scipy.stats.pearsonr(vals, label)}")