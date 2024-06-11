import pandas as pd

case1 = pd.read_csv("./Italian Data/FIMMG dataset_ok(CASE1).csv")
case2 = pd.read_csv("./Italian Data/FIMMG dataset_ok(CASE2).csv")
case3 = pd.read_csv("./Italian Data/FIMMG dataset_ok(CASE3).csv")

case1 = case1.drop("Unnamed: 1862", axis=1)
case2 = case2.drop("Unnamed: 1841", axis=1)
case3 = case3.drop("Unnamed: 1841", axis=1)

data = [case1, case2, case3]

print(list(case1.columns))
print(list(case2.columns))
print(list(case3.columns))
print(case1)
print(case2)
print(case3)


for a, b, c in zip(*[list(x.columns) for x in data]):
    if a == b == c:
        print(1, a, b, c)
    elif b == c:
        print(2, a, b, c)
    else:
        print(3, a, b, c)