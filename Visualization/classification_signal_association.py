import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def pairwise_ttests(df, column, group):
    # Define a function to perform pairwise t-tests

    p_values = {}
    groups = df[group].unique()
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j:
                data1 = df[df[group] == group1][column]
                data2 = df[df[group] == group2][column]
                t_stat, p_val = ttest_ind(data1, data2)
                p_values[(group1, group2)] = p_val

    return p_values

df_path = input("Please enter the dataframe path:  ")
df = pd.read_csv(df_path)

font = "Montserrat"
palette = {"TP": "g",
           "TN": "b",
           "FP": "r",
           "FN": "y"}

sns.set_context("poster")
sns.set_style("whitegrid")
sns.set(font=font)

fig, ax = plt.subplots()
ax.set_facecolor('white')
fig.set_facecolor('white')

sns.swarmplot(data = df, y = "Max Mean Signal", x = "Type", size = 6, palette = palette, zorder = 1)

type_order = ["TP", "TN", "FP", "FN"]
means = df.groupby("Type")["Max Mean Signal"].mean().reindex(type_order)
stds = df.groupby("Type")["Max Mean Signal"].std().reindex(type_order)

for i, (mean, std) in enumerate(zip(means, stds)):
    ax.hlines(mean, i-0.05, i+0.05, color = "black", linewidth = 2, zorder = 2)  # for mean
    ax.hlines(mean + std, i-0.1, i+0.1, color = "black", linewidth = 2, zorder = 2)  # for mean + standard deviation
    ax.hlines(mean - std, i-0.1, i+0.1, color = "black", linewidth = 2, zorder = 2)  # for mean - standard deviation

for i, (mean, std) in enumerate(zip(means, stds)):
    ax.vlines(i, mean - std, mean + std, color = "black", linewidth = 1, zorder = 2)

y_max = df["Max Mean Signal"].max()
p_values = pairwise_ttests(df, "Max Mean Signal", "Type")
alpha_1 = 0.001
alpha_2 = 0.01
alpha_3 = 0.05

for (group1, group2), p_val in p_values.items():
    if p_val <= alpha_1:
        x1 = type_order.index(group1)
        x2 = type_order.index(group2)
        ax.plot([x1, x2], [y_max + 3, y_max + 3], color = "black", linewidth = 1, zorder = 3)
        ax.text((x1 + x2) / 2, y_max + 4, "***", ha = "center", va = "center", color = "black", fontsize = 20, zorder = 3)
        y_max += 10
    elif p_val <= alpha_2:
        x1 = type_order.index(group1)
        x2 = type_order.index(group2)
        ax.plot([x1, x2], [y_max + 3, y_max + 3], color = "black", linewidth = 1, zorder = 3)
        ax.text((x1 + x2) / 2, y_max + 4, "**", ha = "center", va = "center", color = "black", fontsize = 20, zorder = 3)
        y_max += 10
    elif p_val <= alpha_3:
        x1 = type_order.index(group1)
        x2 = type_order.index(group2)
        ax.plot([x1, x2], [y_max + 3, y_max + 3], color = "black", linewidth = 1, zorder = 3)
        ax.text((x1 + x2) / 2, y_max + 4, "*", ha = "center", va = "center", color = "black", fontsize = 20, zorder = 3)
        y_max += 10

ax.yaxis.grid(True, linestyle = "-", which = "major", color = "gray", alpha = 0.25)

plt.xlabel("Type", fontname = font, fontsize = 0)
plt.ylabel("Max Mean Signal", fontname = font, fontsize = 18)

plt.xticks(fontsize=18)
plt.yticks(fontsize=14)

ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontname(font)
for label in ax.get_yticklabels():
    label.set_fontname(font)

plt.show()