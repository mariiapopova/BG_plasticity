#%%
# Paired t-test: PreMOTA vs VibrationOFF
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt


# Load band power files
premota_path = r"....xlsx"
vibrationoff_path  = r"....xlsx"

df_pre  = pd.read_excel(premota_path, index_col=0)
df_post = pd.read_excel(vibrationoff_path, index_col=0)

# Safety checks
assert df_pre.shape == df_post.shape, "Pre/Post shapes do not match!"
assert all(df_pre.index == df_post.index), "Channel mismatch!"

bands = df_pre.columns

# Paired t-test
results = []

for band in bands:
    t_stat, p_val = ttest_rel(df_pre[band], df_post[band])
    results.append({
        "Band": band,
        "t-value": t_stat,
        "p-value": p_val,
        "Mean PreMOTA": df_pre[band].mean(),
        "Mean VibrationOFF": df_post[band].mean(),
        "Difference (VibrationOFF - PreMOTA)": df_post[band].mean() - df_pre[band].mean()
    })

stats_df = pd.DataFrame(results)

print("\nPaired t-test: PreMOTA vs VibrationOFF\n")
print(stats_df)

# Save statistics
stats_df.to_excel("Paired_ttest_PreMOTA_vs_VibrationOFF.xlsx", index=False)


# Visualization 1: Means
x = np.arange(len(bands))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, stats_df["Mean PreMOTA"], width, label="PreMOTA")
plt.bar(x + width/2, stats_df["Mean VibrationOFF"], width, label="VibrationOFF")

plt.xticks(x, bands)
plt.ylabel("Mean Band Power (µV²)")
plt.title("PreMOTA vs VibrationOFF (Paired t-test)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Visualization 2: Difference
plt.figure(figsize=(7,4))
plt.bar(bands, stats_df["Difference (VibrationOFF - PreMOTA)"])
plt.axhline(0, linestyle='--')
plt.ylabel("VibrationOFF − PreMOTA (µV²)")
plt.title("Band Power Differences")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
