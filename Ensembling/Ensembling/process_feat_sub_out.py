import math

file = open("feat-subsets-out.txt", 'r')
file = file.read()[1:]
out = []
for x in file.split("}}], ")[:1]:
    a = x.split(":")
    b = ":".join(a[1:])
    c, b = b.split("), ")[0][3:], b.split("), ")[1]
    c = f"|{c}|"
    d = [z for z in b.split("}")]
    d = [z.split(": {") for z in d]
    d = [[y.split(",") for y in z] for z in d]
    d = [[[w.split(":") for w in y] for y in z] for z in d]
    # print(a[0], c, d)
    # out.append([a[0], c, d])
    temp_lst = []
    for w in d:
        for i in w[1]:
            u = i[0].strip()
            if u == "'all'":
                u = "all"
            else:
                pass
            temp_lst.append([u, float(i[1])])
    mdls = ["lgb", "xgb", "cat", "all"]
    for dex, g in enumerate(temp_lst):
        dex2 = int(math.floor(dex/6))
