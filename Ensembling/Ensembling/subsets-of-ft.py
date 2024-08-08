import numpy as np
import pandas as pd

df = pd.read_csv("Data Categorization_In-Depth Feature Analysis - catgeorization - directly for model.csv")
data = pd.read_csv("Case I 6-12 binary ft >10 positives and normalized.csv").drop("Unnamed: 0", axis=1)

print(df)
print(data)

subs_dict = {a:[] for a in ["?", "Medical History", "Test", "Procedure", "Medication", "SDoH", "trash"]}


for dex, a in enumerate(list(data.columns[:-1])):
    b = df.loc[dex][0][1:]
    if a != b:
        b = b[1:-1]
    if a != b:
        b = f"'{b}'"
    if a != b:
        if b[:3] == "'AL":
            b = "ALZHEIMER'S DISEASE"
        if b[:5] == "'LICO":
            b = "HELICOBACTER PYLORI INFECTION"
        if b[:4] == "'PAR":
            b = "PARKINSON'S DISEASE"
        if b[:6] == "'Hashi":
            b = "Hashimoto's thyroiditis"
    if a!=b:
        print(dex, a)
    else:
        print(dex)
    set1 = df.loc[dex][1]
    set2 = df.loc[dex][2] if not pd.isna(df.loc[dex][2]) else "trash"
    subs_dict[set1].append(a)
    subs_dict[set2].append(a)

del subs_dict["trash"]
print(subs_dict)
for x in subs_dict:
    print(x, len(subs_dict[x]))
# file = open("subsets_processed.txt", 'w')
# for key in subs_dict:
#     file.write(f"{key},")
#
# file.write("\n")
#
# for key in subs_dict:
#     str = ""
#     for ft in subs_dict[key]:
#         str = f"{str},{ft}"
#     str = f"{str}\n"
#     file.write(str)
# file.close()