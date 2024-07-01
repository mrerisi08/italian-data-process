import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("max_auc_tun_with_subsets copy.csv")

df = df.drop([
            "Unnamed: 0", "subsample","eval_metric","colsamp_bylvl",
            "min_child_weight", "silent","objective","colsamp_bytree","tree_method"
            ], axis=1)

print(df.nlargest(10, "AUC"))
print(df.nlargest(10, "ACC"))


# plt.scatter(df["scale_pos_weight"], df["AUC"], c=df["ACC"], alpha=0.5)
# plt.scatter(df["lambda"], df["AUC"], c=df["ACC"], alpha=0.5)
# plt.show()

# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# ax1.scatter(df["lambda"], df["AUC"], c=df["ACC"], alpha=0.5)
# ax1.set_title("Lambda")
# fig1.show()
# ax2.scatter(df["scale_pos_weight"], df["AUC"], c=df["ACC"], alpha=0.5)
# ax2.set_title("Scale_pos_weight")
# fig2.show()
# plt.show()

for a in list(df.columns)[2:]:
    print(a)
    fig, ax = plt.subplots()
    ax.scatter(df[a], df["AUC"], c=df["ACC"], alpha=0.5)  # darker = lower, lighter = higher
    ax.set_title(a)
    fig.show()
    fig.savefig(f"plots for hyp param subsets/{a}_plot.png")
plt.show()