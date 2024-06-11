# this file is for fixing the duplicates in column names (for obesity at first, then generally)
import pandas as pd

df = pd.read_csv("Italian Data/FIMMG dataset_ok(CASE1).csv")
df = df.drop("Unnamed: 1862", axis=1)
print(df)
print(df[r"'''278.0'''"])
print(df[r"'''278.00'''"])
print(df[r"'''278.01'''"])


abc = 0
ab = 0
d = 0
for a, b, c in zip(df[r"'''278.0'''"],df[r"'''278.00'''"],df[r"'''278.01'''"]):
    if a == b == c:
        abc += 1
    elif a == b:
        ab += 1
    d += 1
print(d, abc, ab)

for x in range(len(df.iloc[0:])):
    if df[r"'''278.0'''"].iloc[x] != df[r"'''278.00'''"].iloc[x]:
        print([a for a in df[[r"'''278.0'''", r"'''278.00'''"]].loc[x]])



# '''278.0''',OBESITY
# '''278.00''',OBESITY
# '''278.01''',SEVERE OBESITY

df2 = pd.read_csv("Data Name - Sheet2.csv")
print(len(df2['name']))
print(len(set(df2['name'])))

seen = []
for x in df2['name'].iloc[0:]:
    if x in seen:
        print(x)
    if x not in seen:
        seen.append(x)
