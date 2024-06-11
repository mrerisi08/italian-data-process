# convert to binary and count for each feature its comparison to LABEL

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

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

# print(sum(df["SODIUM"]))
# print([x for x in df["SODIUM"]])
# print(df)
print([a for a in df["OBESITY"].iloc[0:]])

for x in df.drop(["Min Diastolic","Max Diastolic","Mean Diastolic","Min Systolic","Max Systolic","Age","Gender","Mean Systolic"],axis=1):
    df[x] = df[x].where(df[x] == 0, 1)

print([a for a in df["OBESITY"].iloc[0:]])
# print(df)
#
# out = {}
# for x in df.drop(["Min Diastolic","Max Diastolic","Mean Diastolic","Min Systolic","Max Systolic","Age","LABEL","Gender","Mean Systolic"],axis=1):
#     strt = time.time()
#     out[x] = {"posFtNegLbl":0, "posFtPosLbl":0, "negFtNegLbl":0, "negFtPosLbl":0, "elseCase":0, "timing":0}
#     for y, z in zip(df[x].iloc[0:], df["LABEL"].iloc[0:]):
#         if y == 1 and z == 0:
#             out[x]["posFtNegLbl"] += 1
#         elif y == 1 and z == 1:
#             out[x]["posFtPosLbl"] += 1
#         elif y == 0 and z == 0:
#             out[x]["negFtNegLbl"] += 1
#         elif y == 0 and z == 1:
#             out[x]["negFtPosLbl"] += 1
#         else:
#             out[x]["elseCase"] += 1
#
#     out[x]["timing"] = time.time() - strt
#
# # print(out)
# names = []
# posFtNegLbl = []
# posFtPosLbl = []
# negFtNegLbl = []
# negFtPosLbl = []
# lblDict = {"posFtNegLbl":[], "posFtPosLbl":[], "negFtNegLbl":[], "negFtPosLbl":[], "elseCase":[], "timing":[]}
# finalDict = {"names":[],"posFtNegLbl":[], "posFtPosLbl":[], "negFtNegLbl":[], "negFtPosLbl":[], "elseCase":[], "timing":[]}
# elseCase = []
# timing = []
# for a in out:
#     finalDict["names"].append(a)
#     for b in out[a]:
#         finalDict[b].append(out[a][b])
#
# rafa = pd.DataFrame(finalDict)
# rafa.to_csv("correlationAnalysis.csv")
