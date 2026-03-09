#%%
# Paired t-test: PreMOTA vs VibrationOFF
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt


# Load band power files
premota_path = r"....xlsx"
vibrationoff_path  = r"....xlsx"

# Sheet names in Excel
sheets = ["Raw_Power_uV2", "Log_Power_dB", "Relative_Power_percent"]

for sheet in sheets:

    print(f"{sheet}")

    df_pre  = pd.read_excel(premota_path, sheet_name=sheet, index_col=0)
    df_post = pd.read_excel(vibrationoff_path, sheet_name=sheet, index_col=0)

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
            "Median PreMOTA": df_pre[band].median(),
            "Median VibrationOFF": df_post[band].median(),
            "Difference (VibrationOFF - PreMOTA)": df_post[band].median() - df_pre[band].median()
        })

    stats_df = pd.DataFrame(results)

    print("\nPaired t-test: PreMOTA vs VibrationOFF\n")
    print(stats_df)

    # Save statistics
    stats_df.to_excel(f"Paired_ttest_{sheet}.xlsx", index=False)


    # Visualization 1: Medians
    x = np.arange(len(bands))
    width = 0.35

    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, stats_df["Median PreMOTA"], width, label="PreMOTA")
    plt.bar(x + width/2, stats_df["Median VibrationOFF"], width, label="VibrationOFF")

    plt.xticks(x, bands)
    plt.ylabel("Median Band Power")
    plt.title(f"PreMOTA vs VibrationOFF ({sheet})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Visualization 2: Difference
    plt.figure(figsize=(7,4))
    plt.bar(bands, stats_df["Difference (VibrationOFF - PreMOTA)"])
    plt.axhline(0, linestyle='--')
    plt.ylabel("VibrationOFF − PreMOTA")
    plt.title(f"Band Power Differences ({sheet})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    print(df_pre.describe())
    print(df_post.describe())
