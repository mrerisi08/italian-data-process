import pandas as pd

df = pd.read_csv("Case I 6-12 binary ft >10 positives and normalized.csv")
df = df.drop("Unnamed: 0", axis=1)
print(df)
print(list(df.columns))