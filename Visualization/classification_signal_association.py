import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

sns.swarmplot(data = df, y = "Max Mean Signal", x = "Type", size = 6, palette = palette)

type_order = ["TP", "TN", "FP", "FN"]
means = df.groupby("Type")["Max Mean Signal"].mean().reindex(type_order)
stds = df.groupby("Type")["Max Mean Signal"].std().reindex(type_order)
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.hlines(mean, i-0.05, i+0.05, color="black", linewidth=2)  # for mean
    ax.hlines(mean + std, i-0.1, i+0.1, color="black", linewidth=2)  # for mean + standard deviation
    ax.hlines(mean - std, i-0.1, i+0.1, color="black", linewidth=2)  # for mean - standard deviation

ax.yaxis.grid(True, linestyle = "-", which = "major", color = "gray", alpha = 0.25)

plt.xlabel("Type", fontname = font, fontsize = 18)
plt.ylabel("Max Mean Signal", fontname = font, fontsize = 18)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontname(font)
for label in ax.get_yticklabels():
    label.set_fontname(font)

plt.show()