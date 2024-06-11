import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import random

df = pd.read_csv("./Italian Data/FIMMG dataset_ok(CASE1).csv")
def initiate_and_rename_df():
    global df
    df = df.drop("Unnamed: 1862", axis=1)
    names = pd.read_csv("Data Name - Sheet2.csv")
    codesNames = {}
    namesDict = names.to_dict('dict')
    for x in range(len(namesDict['code'])):
        codesNames[namesDict['code'][x]] = namesDict['name'][x]
    df.rename(columns=codesNames, inplace=True)
initiate_and_rename_df()

def get_indices_of_duplicate_columns_non_zero():
    duplicateNames = [x for x in df]
    print(duplicateNames)
    print(list(set(duplicateNames)))
    print(len(duplicateNames))
    print(len(set(duplicateNames)))

    for x in set(duplicateNames):
        try:
            duplicateNames.remove(x)
        except:
            print(x)
    print(duplicateNames)
    # print([x for x in df["ARTERIOSCLEROTIC DEMENTIA"].iloc])
    print(df["ARTERIOSCLEROTIC DEMENTIA"].iloc[117])
    print(df["ARTERIOSCLEROTIC DEMENTIA"].iloc[117][0])
    print(df["ARTERIOSCLEROTIC DEMENTIA"].iloc[117][1])
    print(len(df["ARTERIOSCLEROTIC DEMENTIA"].iloc[117]))

    duplicateConsistency = {}
    for x in list(set(duplicateNames)):
         print(x, list(set(duplicateNames)).index(x), len(set(duplicateNames)))
         duplicateConsistency[x] = {}
         for y in range(len(df["LABEL"])):
             for z in range(len(df[x].iloc[y])):
                 if z not in duplicateConsistency[x]:
                     duplicateConsistency[x][z] = []
                 if df[x].iloc[y][z] != 0:
                     duplicateConsistency[x][z].append(y)
    print(duplicateConsistency)
    duplicateConsistency = {'GAIT DISORDERS': {0: [2129], 1: [71, 159, 1732, 1905]}, 'ATRIAL FIBRILLATION': {0: [35, 123, 159, 370, 381, 392, 414, 482, 694, 1183, 1274, 1346, 1602, 1667, 1711, 1721, 1797, 1807, 1933, 2257, 2263], 1: [9, 10, 22, 106, 155, 162, 169, 213, 226, 268, 275, 325, 333, 439, 477, 495, 583, 632, 648, 684, 820, 874, 883, 917, 923, 930, 943, 987, 1045, 1048, 1155, 1178, 1255, 1294, 1323, 1348, 1373, 1394, 1418, 1524, 1732, 1735, 1746, 1751, 1762, 1810, 1840, 1846, 1920, 1925, 1938, 1948, 1951, 2064, 2067, 2125, 2129, 2134, 2135, 2182, 2298]}, 'ARTERIOSCLEROTIC DEMENTIA': {0: [154, 215, 232, 365, 377, 546, 725, 825, 1274, 1389, 1450, 1608, 1670, 1699, 1759, 1905, 1934, 1957, 1997, 2379], 1: [414, 1355]}, 'HORSESHOE KIDNEY': {0: [2043], 1: [2213]}, 'ACUTE MYOCARDIAL INFARCTION': {0: [1814, 1948, 1966, 2261], 1: [1857], 2: [39, 117, 149, 160, 161, 262, 295, 355, 400, 519, 688, 830, 849, 947, 1075, 1136, 1165, 1340, 1354, 1404, 1452, 1473, 1544, 1565, 1725, 1749, 1777, 1790, 1826, 1840, 1861, 1874, 1916, 1949, 1969, 2015, 2053, 2111, 2135, 2143], 3: [363]}, "CROHN'S DISEASE": {0: [105, 1530], 1: [28, 73, 168, 287, 710, 1184, 1227, 1530]}, 'ANXIETY': {0: [1655, 2194, 2393], 1: [28, 157, 283, 616, 783, 978, 1206, 1311, 2146, 2152]}, 'HEAD TRAUMA': {0: [1664, 2236], 1: [16, 68, 69, 176, 903, 953, 1057, 1504]}, 'K UTERUS': {0: [725, 1299, 1717], 1: [1788, 1807, 1966, 2261]}, 'ROTATOR CUFF SYNDROME': {0: [281, 1949], 1: [271]}, 'ACUTE DUODENAL ULCER WITH HEMORRHAGE': {0: [375, 1086, 1988, 2134, 2163], 1: [1541]}, 'TENSION HEADACHE': {0: [1165], 1: [509, 862, 1009]}, 'AMYOTROPHIC LATERAL SCLEROSIS': {0: [409, 1103], 1: [1103]}, 'CHRONIC ALCOHOLISM': {0: [2135], 1: [1525]}, 'TRANSIENT CEREBRAL ISCHEMIA < TIA >': {0: [126, 215, 1719, 1796, 2260], 1: [254, 2263]}, 'K TESTICLE': {0: [947], 1: [914]}, 'CHURG-STRAUSS SYNDROME': {0: [1045], 1: [1045]}, 'DEPRESSION': {0: [30, 33, 35, 41, 132, 163, 177, 209, 235, 245, 310, 328, 377, 380, 386, 390, 397, 457, 492, 537, 584, 607, 611, 773, 819, 857, 920, 958, 1047, 1104, 1146, 1273, 1319, 1372, 1392, 1410, 1487, 1555, 1560, 1568, 1596, 1597, 1651, 1685, 1699, 1734, 1782, 1802, 1847, 1849, 1917, 1950, 1969, 2119, 2132, 2311, 2374], 1: [2043, 2115]}, 'SLEEP APNEA': {0: [1, 34, 326, 374, 960, 1048, 1058, 1373, 1629, 1658], 1: [1381]}, 'OBESITY': {0: [35, 227, 245, 283, 385, 567, 622, 677, 693, 788, 864, 878, 960, 1139, 1357, 1396, 1439, 1505, 1515, 1519, 1693, 1713, 1801, 1864, 2111, 2194, 2327, 2413], 1: [2109]}, 'MANIC-DEPRESSIVE PSYCHOSIS': {0: [2149], 1: [191]}, 'TOOTH ABSCESS': {0: [5], 1: [893, 2209]}, 'CHICKENPOX': {0: [1760], 1: [401, 1041, 1198, 1328, 1521]}, 'GENERALIZED ARTHROSIS': {0: [7, 9, 18, 21, 37, 40, 79, 98, 100, 116, 137, 154, 160, 188, 260, 284, 298, 300, 311, 320, 340, 358, 360, 365, 379, 382, 390, 415, 426, 434, 450, 468, 478, 557, 781, 1259, 1294, 1315, 1335, 1339, 1349, 1361, 1421, 1425, 1441, 1517, 1522, 1523, 1548, 1655, 1667, 1670, 1673, 1688, 1732, 1742, 1759, 1785, 1807, 1808, 1828, 1847, 1892, 1901, 1905, 1916, 1920, 1944, 1963, 1987, 2045, 2051, 2097, 2130, 2257, 2259, 2268, 2281, 2288], 1: [684, 849, 1207, 1241, 2139]}, 'BRONCHITIS': {0: [17, 119, 133, 947], 1: [445, 1011]}, 'ACUTE APPENDICITIS': {0: [494, 646, 1967], 1: [834]}, 'K PROSTATE': {0: [30, 128, 134, 149, 170, 185, 206, 336, 348, 375, 408, 414, 439, 443, 498, 577, 581, 596, 640, 1055, 1293, 1333, 1353, 1362, 1495, 1541, 1640, 1646, 1816, 1826, 1841, 2065, 2134], 1: [18, 43, 182, 341, 344, 627, 688, 1165, 1277, 1315, 1344, 1452, 1539, 1607, 1655, 1797, 1852, 1893, 1934, 1997, 2115]}, 'GALLBLADSTONE STONES': {0: [15, 17, 18, 20, 25, 40, 46, 50, 59, 106, 153, 169, 170, 174, 185, 186, 188, 208, 215, 238, 245, 249, 268, 272, 287, 298, 336, 341, 414, 469, 498, 546, 575, 637, 655, 683, 693, 698, 742, 777, 781, 814, 890, 943, 960, 1008, 1022, 1027, 1040, 1047, 1054, 1146, 1170, 1189, 1191, 1251, 1256, 1273, 1305, 1355, 1359, 1362, 1414, 1421, 1429, 1439, 1497, 1505, 1528, 1530, 1533, 1543, 1552, 1568, 1586, 1596, 1611, 1626, 1683, 1699, 1727, 1737, 1741, 1763, 1795, 1798, 1802, 1810, 1817, 1818, 1820, 1828, 1847, 1854, 1916, 1924, 1948, 1989, 2021, 2102, 2109, 2135, 2139, 2170, 2172, 2180, 2184, 2286, 2298, 2306, 2382, 2398], 1: [682, 688, 1562, 1842, 2108]}, 'only Case I': {0: [2185], 1: [1505], 2: [20], 3: [210, 375, 398, 761, 832, 1058, 1374, 1395, 1686], 4: [88, 210, 474, 1162, 1629, 1661, 1701, 2151, 2185, 2424], 5: [790], 6: [161, 750, 790, 807, 928, 1371, 1374, 1530, 1765, 1815, 2114, 2185], 7: [928], 8: [161, 750, 928, 1371, 1374, 1765, 2114], 9: [688], 10: [1, 58, 125, 150, 162, 210, 225, 258, 272, 288, 325, 347, 359, 374, 375, 398, 417, 474, 688, 785, 835, 1058, 1065, 1162, 1273, 1395, 1411, 1422, 1497, 1505, 1519, 1583, 1629, 1646, 1661, 1686, 1701, 1747, 1782, 1790, 1795, 1817, 1824, 1880, 1933, 1945, 1956, 2105, 2151, 2238, 2327, 2424], 11: [417], 12: [1058], 13: [20, 359, 1956, 2105], 14: [1701], 15: [210, 2185], 16: [807, 2124]}}
    with open('duplicateConsistency.json', 'w') as convert_file:
        convert_file.write(json.dumps(duplicateConsistency))

duplicateConsistency = None
def load_dupe_columns_from_json():
    global duplicateConsistency
    with open("duplicateConsistency.json", 'r') as file:
        duplicateConsistency = json.load(file)
load_dupe_columns_from_json()

def positives_by_instance_for_dupe_columns():
    for a in duplicateConsistency:
        print(a, [len(duplicateConsistency[a][b]) for b in duplicateConsistency[a]])
def failed_binary():
    for a in df.drop(["Min Diastolic","Max Diastolic","Mean Diastolic","Min Systolic","Max Systolic","Mean Systolic","Age","Gender","LABEL"],axis=1):
        indices = []
        for b,c in enumerate(df[a].iloc):
            if type(c) != type(pd.Series([0])):
                if c != 0:
                    indices.append(b)
            else:
                for d in range(len(c)):
                    for e, f in enumerate(df[a].iloc[d].iloc):
                        if f != 0:
                            indices.append(e)

        print(a, len(indices), indices)
        for w in indices:
            df.loc[w, a] = 1
def binary2():
    global df
    for x in df.drop(["Min Diastolic","Max Diastolic","Mean Diastolic","Min Systolic","Max Systolic","Mean Systolic","Age","Gender","LABEL"],axis=1):
        df[x] = df[x].where(df[x] == 0, 1)
def tally(inp):
    out = {}
    for x in inp:
        if x not in out:
            out[x] = 0
        out[x] += 1
    return out
def rename_dupes():
    # used gpt for this code
    new_columns = []
    columns_count = {}
    for col in df.columns:
        if col in columns_count:
            columns_count[col] += 1
            new_columns.append(f"{col}_{columns_count[col]} (my edit)")
        else:
            columns_count[col] = 0
            new_columns.append(col)
    df.columns = new_columns
rename_dupes()
binary2()

def print_non_binary_sets():
    for x in df.columns:
        if set(df[x]) != {0, 1}:
            print(x, set(df[x]))

def print_rows_where_999s_dont_carry_across():
    for a, b, c, d, e, f, ind in zip(*[df[x].iloc for x in ["Min Diastolic","Max Diastolic","Mean Diastolic","Min Systolic","Max Systolic","Mean Systolic"]], range(len(list(df["Max Systolic"].iloc)))):
        if 999 in [a, b, c, d, e, f] and not a == b == c == d == e == f:
            print(ind, [a, b, c, d, e, f])

def nans_from_999s():
    global df
    for x in df[["Min Diastolic","Max Diastolic","Mean Diastolic","Min Systolic","Max Systolic","Mean Systolic"]]:
        df[x] = df[x].where(df[x] != 999, np.nan)
nans_from_999s()

df.to_csv("Processed_Case1_6-6.csv")