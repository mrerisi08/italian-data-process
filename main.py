import pandas as pd
import matplotlib.pyplot as plt
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
for x in df.drop(["Min Diastolic","Max Diastolic","Mean Diastolic","Min Systolic","Max Systolic","Age"],axis=1):
    df[x] = df[x].where(df[x] == 0, 1)
print(set(df))
print([x for x in df])
print(len(df))
# for a in df:
#     df.loc[df[a] == 999, df[a]] = np.nan
print(df)
# counts, bins = np.histogram(df["Max Diastolic"][np.isfinite(df["Max Diastolic"])])
counts, bins = np.histogram(df["Age"])
print(counts)
print(bins)
# plt.stairs(counts, bins)
# plt.hist(bins[:-1], bins, weights=counts)
# plt.scatter(df["Age"],df["LABEL"])
print(df["Age"])
print(df["BMI<BODY MASS INDEX>"])
print(set(df["BMI<BODY MASS INDEX>"]))

# CODE FOR A BEESWARM
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style("whitegrid")
# ax = sns.swarmplot(x="Age", y="LABEL", data=df)
# # ax = sns.boxplot(x="Age", y="LABEL", data=df,
# #         showcaps=False,boxprops={'facecolor':'None'},
# #         showfliers=False,whiskerprops={'linewidth':0})

df2 = df[["Min Diastolic","Max Diastolic","Mean Diastolic",	"Mean Systolic", "Min Systolic", "Max Systolic", "Gender", "Age", "LABEL"]]

# CODE FOR COUNTING HOW MANY NON ZEROS IN EACH FEATURE
# ftCounts = {}
# for x in df:
#     ct = 0
#     for y in range(2432):
#         if df[x].iloc[y].any() != np.int64(0):
#             ct += 1
#     print(x, ct)
#     ftCounts[x] = ct
#
# dfDict = {"ft":[],"cts":[]}
# for a in ftCounts:
#     dfDict["ft"].append(a)
#     dfDict["cts"].append(ftCounts[a])
#
# ftCt = pd.DataFrame.from_dict(dfDict)
# # ftCt.to_csv("feature_counts.csv")


ftData = pd.read_csv("feature_counts.csv")
# ftData = ftData[ftData['Counts'] < 50]
ftCounts, ftBins = np.histogram(ftData['Counts'])
plt.stairs(ftCounts, ftBins)
plt.hist(ftBins[:-1], ftBins, weights=ftCounts)
bigFtData = ftData[ftData["Counts"] > 300]
bigFtData = bigFtData[bigFtData["Counts"] < 2400]
bigFtData = bigFtData.drop("Unnamed: 0", axis=1)
print(bigFtData)

print(set(df["OBESITY"]))
print([a for a in df['OBESITY'].iloc[0:]])
graham = open("graham.txt", 'w')
[graham.write(f"{x},") for x in df]
# plt.show()
