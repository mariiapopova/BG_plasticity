#%%
"""
vCR ON vs OFF Comparison — Beta Power

"""

import mne
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Settings — reading from preprocessed folder
DATA_FOLDER = r"..."

# Matched pairs: (label, vCR OFF file, vCR ON file)
PAIRS = [
    (
        "Eyes Open - Resting",
        "fatemeh1eyeopen1min0809_clean_manual.fif",
        "fatemeh1vcreyesopen0809_clean_manual.fif",
    ),
    (
        "Eyes Closed - Resting",
        "fatemeh1eyeclosed1min0809_clean_manual.fif",
        "fatemeh1vcreyeclosed0809_clean_manual.fif",
    ),
    (
        "Finger Extension",
        "fingerextensionandreleasingeyesopen_clean_manual.fif",
        "vcrextensionandreleasingeyesopen_clean_manual.fif",
    ),
    (
        "Counting Silent 30s",
        "countingbackwardssilent30sec_clean_manual.fif",
        "countingbackwardssilentwithvcr30sec_clean_manual.fif",
    ),
]

# Channels to focus on (sensorimotor ROI)
ROI_CHANNELS = ["C3", "C4", "CP3", "CP4", "Cz"]

# Beta frequency bands
LOW_BETA  = (13, 20)
HIGH_BETA = (21, 30)

# Epoch length for PSD (in seconds)
EPOCH_LENGTH = 2.0
# ============================================================


def get_beta_power(filepath, roi_channels, band):
    """
    Load a cleaned .fif file and return mean beta power
    per ROI channel.
    """
    try:
        raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None

    # Keep only channels present in this file
    available = [ch for ch in roi_channels if ch in raw.ch_names]
    if not available:
        print(f"  WARNING: No ROI channels found")
        print(f"  Available (first 10): {raw.ch_names[:10]}")
        return None

    raw_roi = raw.copy().pick_channels(available, verbose=False)

    # Cut into short epochs
    epochs = mne.make_fixed_length_epochs(
        raw_roi, duration=EPOCH_LENGTH, verbose=False
    )

    # Compute PSD and average across frequencies and epochs
    psd   = epochs.compute_psd(method="welch",
                                fmin=band[0], fmax=band[1],
                                verbose=False)
    power = psd.get_data().mean(axis=-1).mean(axis=0) * 1e12

    return dict(zip(available, power))


# ============================================================
# Run comparison for all pairs
# ============================================================

print("\n" + "="*60)
print("   vCR ON vs OFF — BETA POWER COMPARISON")
print("   (preprocessed data)")
print("="*60)

results = []

for label, file_off, file_on in PAIRS:

    print(f"\n--- {label} ---")

    for band_name, band in [("Low beta  (13-20 Hz)", LOW_BETA),
                             ("High beta (21-30 Hz)", HIGH_BETA)]:

        power_off = get_beta_power(DATA_FOLDER + file_off,
                                   ROI_CHANNELS, band)
        power_on  = get_beta_power(DATA_FOLDER + file_on,
                                   ROI_CHANNELS, band)

        if power_off is None or power_on is None:
            print(f"  Skipping {band_name}")
            continue

        shared_ch = [ch for ch in power_off if ch in power_on]

        print(f"\n  {band_name}")
        print(f"  {'Channel':<8} {'vCR OFF':>12} {'vCR ON':>12} "
              f"{'Change':>10} {'Direction':>12}")
        print(f"  {'─'*58}")

        for ch in shared_ch:
            p_off     = power_off[ch]
            p_on      = power_on[ch]
            change    = ((p_on - p_off) / p_off) * 100
            direction = "↓ decreased" if change < 0 else "↑ increased"

            print(f"  {ch:<8} {p_off:>12.6f} {p_on:>12.6f} "
                  f"{change:>9.1f}% {direction:>12}")

            results.append({
                "condition":  label,
                "band":       band_name,
                "channel":    ch,
                "power_off":  p_off,
                "power_on":   p_on,
                "change_pct": change,
            })

# ============================================================
# Plot — change heatmap per band
# ============================================================

if not results:
    print("\nNo results to plot — check file paths and names")
else:
    for band_name in ["Low beta  (13-20 Hz)", "High beta (21-30 Hz)"]:

        band_results = [r for r in results if r["band"] == band_name]
        if not band_results:
            continue

        conditions = list(dict.fromkeys(r["condition"] for r in band_results))
        channels   = list(dict.fromkeys(r["channel"]   for r in band_results))

        # Build change % matrix
        matrix = np.full((len(conditions), len(channels)), np.nan)
        for r in band_results:
            row = conditions.index(r["condition"])
            col = channels.index(r["channel"])
            matrix[row, col] = r["change_pct"]

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(matrix, aspect="auto",
                       cmap="RdBu", vmin=-50, vmax=50)
        ax.set_xticks(range(len(channels)))
        ax.set_xticklabels(channels, fontsize=10)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions, fontsize=9)
        ax.set_title(
            f"vCR Effect — {band_name}\n"
            f"Blue = beta decreased, Red = beta increased"
        )
        plt.colorbar(im, ax=ax, label="Change (%)")

        # Add % numbers inside cells
        for i in range(len(conditions)):
            for j in range(len(channels)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}%",
                            ha="center", va="center",
                            fontsize=9, color="black")

        plt.tight_layout()
        fname = f"vcr_effect_{band_name[:8].strip().replace(' ','_')}.png"
        plt.savefig(fname, dpi=150)
        plt.show(block=True)
        print(f"\nPlot saved: {fname}")

    # Bar chart — average change per condition
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, band_name in zip(axes, ["Low beta  (13-20 Hz)",
                                     "High beta (21-30 Hz)"]):
        band_results = [r for r in results if r["band"] == band_name]
        conditions   = list(dict.fromkeys(
            r["condition"] for r in band_results))
        mean_changes = [
            np.mean([r["change_pct"] for r in band_results
                     if r["condition"] == c])
            for c in conditions
        ]
        colors = ["steelblue" if c < 0 else "tomato"
                  for c in mean_changes]
        ax.bar(range(len(conditions)), mean_changes, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=30,
                           ha="right", fontsize=9)
        ax.set_ylabel("Mean beta power change (%)")
        ax.set_title(f"{band_name}\n"
                     f"Blue = vCR decreased beta  |  Red = vCR increased beta")

    plt.suptitle("vCR Effect on Beta Power — ROI Average\n"
                 "(preprocessed data)", fontsize=12)
    plt.tight_layout()
    plt.savefig("vcr_effect_summary_clean.png", dpi=150)
    plt.show(block=True)
    print("Plot saved: vcr_effect_summary_clean.png")

print("\n" + "="*60)
print("   DONE")
print("="*60 + "\n")

