#%%
import mne
import numpy as np
import matplotlib.pyplot as plt

DATA_FOLDER = r"..."

#match pairs
PAIRS = [
    (
        "Resting - Eyes Open",
        "restingeyesopen1min_clean_manual.fif",
        "20amplitudevcreyesopen1min_clean_manual.fif",
    ),
    (
        "Resting - Eyes Closed",
        "restingeyesclosed_clean_manual.fif",
        "20amplitudevcreyesclosed_clean_manual.fif",
    ),
    (
        "Hold & Let Go - Eyes Open",
        "Second2minholdandletgoeyesopen_clean_manual.fif",
        "Second2minholdandletgoeyesopenVCR20Amp_clean_manual.fif",
    ),
    (
        "Hold & Let Go - Eyes Closed",
        "Second2minholdandletgoeyesclosed_clean_manual.fif",
        "Second2minholdandletgoeyesclosedVCR20Amp_clean_manual.fif",
    ),
    (
        "Finger Extension",
        "Secondindexfingerextensioneyesopen_clean_manual.fif",
        "SecondindexfingerextensioneyesopenVCR20Amp_clean_manual.fif",
    ),
    (
        "Countdown",
        "countfroma100back7steps1mineyesopen_clean_manual.fif",
        "countfrom100back7steps1mineyesopenwithVCR20Amp_clean_manual.fif",
    ),
]

ROI_CHANNELS = ["C3", "C4", "CP3", "CP4", "Cz"]
LOW_BETA     = (13, 20)
HIGH_BETA    = (21, 30)
EPOCH_LENGTH = 2.0

def get_beta_power(filepath, roi_channels, band):
    try:
        raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    available = [ch for ch in roi_channels if ch in raw.ch_names]
    if not available:
        return None

    raw_roi = raw.copy().pick_channels(available, verbose=False)
    epochs  = mne.make_fixed_length_epochs(raw_roi, duration=EPOCH_LENGTH, verbose=False)
    psd     = epochs.compute_psd(method="welch", fmin=band[0], fmax=band[1], verbose=False)

    # Convert to µV²/Hz
    power = psd.get_data().mean(axis=-1).mean(axis=0) * 1e12
    return dict(zip(available, power))


# Run comparison
print("\n" + "="*65)
print("   vCR ON vs OFF — BETA POWER COMPARISON")
print("   Session 2  | preprocessed data")
print("="*65)

results = []

for label, file_off, file_on in PAIRS:
    print(f"\n--- {label} ---")

    for band_name, band in [("Low beta  (13-20 Hz)", LOW_BETA),
                             ("High beta (21-30 Hz)", HIGH_BETA)]:

        power_off = get_beta_power(DATA_FOLDER + file_off, ROI_CHANNELS, band)
        power_on  = get_beta_power(DATA_FOLDER + file_on,  ROI_CHANNELS, band)

        if power_off is None or power_on is None:
            print(f"  Skipping {band_name} — file issue")
            continue

        shared = [ch for ch in power_off if ch in power_on]

        print(f"\n  {band_name}")
        print(f"  {'Channel':<8} {'vCR OFF':>12} {'vCR ON':>12} {'Change':>10} {'Direction':>12}")
        print(f"  {'─'*58}")

        for ch in shared:
            p_off     = power_off[ch]
            p_on      = power_on[ch]
            change    = ((p_on - p_off) / p_off) * 100
            direction = "↓ decreased" if change < 0 else "↑ increased"
            print(f"  {ch:<8} {p_off:>12.4f} {p_on:>12.4f} {change:>9.1f}% {direction:>12}")
            results.append({
                "condition":  label,
                "band":       band_name,
                "channel":    ch,
                "power_off":  p_off,
                "power_on":   p_on,
                "change_pct": change,
            })

# Plots
if results:
    for band_name in ["Low beta  (13-20 Hz)", "High beta (21-30 Hz)"]:

        band_results = [r for r in results if r["band"] == band_name]
        if not band_results:
            continue

        conditions = list(dict.fromkeys(r["condition"] for r in band_results))
        channels   = list(dict.fromkeys(r["channel"]   for r in band_results))

        matrix = np.full((len(conditions), len(channels)), np.nan)
        for r in band_results:
            row = conditions.index(r["condition"])
            col = channels.index(r["channel"])
            matrix[row, col] = r["change_pct"]

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu", vmin=-50, vmax=50)
        ax.set_xticks(range(len(channels)))
        ax.set_xticklabels(channels, fontsize=10)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions, fontsize=9)
        ax.set_title(f"vCR Effect — {band_name} | Session 2 \n"
                     f"Blue = decreased, Red = increased")
        plt.colorbar(im, ax=ax, label="Change (%)")

        for i in range(len(conditions)):
            for j in range(len(channels)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}%", ha="center",
                            va="center", fontsize=9, color="black")

        plt.tight_layout()
        fname = f"session2_vcr_effect_{band_name[:8].strip().replace(' ','_')}.png"
        plt.savefig(fname, dpi=150)
        plt.show(block=True)
        print(f"\nSaved: {fname}")

    # Bar chart summary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, band_name in zip(axes, ["Low beta  (13-20 Hz)", "High beta (21-30 Hz)"]):
        band_results = [r for r in results if r["band"] == band_name]
        conditions   = list(dict.fromkeys(r["condition"] for r in band_results))
        mean_changes = [np.mean([r["change_pct"] for r in band_results
                                 if r["condition"] == c]) for c in conditions]
        colors = ["steelblue" if c < 0 else "tomato" for c in mean_changes]
        ax.bar(range(len(conditions)), mean_changes, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean beta change (%)")
        ax.set_title(f"{band_name}\nBlue = vCR decreased beta | Red = increased")

    plt.suptitle("vCR Effect on Beta Power — Session 2 ", fontsize=12)
    plt.tight_layout()
    plt.savefig("session2_vcr_effect_summary.png", dpi=150)
    plt.show(block=True)
    print("Saved: session2_vcr_effect_summary.png")

print("\n" + "="*65)
print("   DONE — Session 2 comparison complete")
print("="*65 + "\n")

# %%
