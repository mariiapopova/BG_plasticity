#%%
"""
Signal Quality Check — Pilot Data

"""

import mne
import numpy as np
import matplotlib.pyplot as plt

DATA_FOLDER = r"..."

# All your pilot files and a short label for each
FILES = {
    "Baseline - Eyes Open":           "fatemeh1eyeopen1min0809.vhdr",
    "Baseline - Eyes Closed":         "fatemeh1eyeclosed1min0809.vhdr",
    "Baseline - vCR Eyes Open":       "fatemeh1vcreyesopen0809.vhdr",
    "Baseline - vCR Eyes Closed":     "fatemeh1vcreyeclosed0809.vhdr",
    "Movement - Extension (vCR OFF)": "fingerextensionandreleasingeyesopen.vhdr",
    "Movement - Extension (vCR ON)":  "vcrextensionandreleasingeyesopen.vhdr",
    "Hold & Let Go (vCR OFF open)":   "vcrwitheyesopenholdandletgo.vhdr",
    "Hold & Let Go (vCR ON open)":    "holdandletgoeyesclosed2min.vhdr",
    "Counting Loud (vCR OFF)":        "countingbackwardsloud.vhdr",
    "Counting Silent 30s (vCR OFF)":  "countingbackwardssilent30sec.vhdr",
    "Counting Silent 30s (vCR ON)":   "countingbackwardssilentwithvcr30sec.vhdr",
    "vCR 40Hz - Resting":             "VCR40restingopeneyes1min.vhdr",
    "vCR 60Hz - Resting":             "VCR60restingopeneyes1min.vhdr",
    "vCR 80Hz - Resting":             "VCR80restingopeneyes1min.vhdr",
    "vCR 100Hz - Resting":            "VCR100restingopeneyes1min.vhdr",
}

# Thresholds for signal quality
FLAT_THRESHOLD_UV  = 1
NOISY_THRESHOLD_UV = 40
# ============================================================


def check_one_file(label, filename):
    """Load one file and return signal quality info."""

    filepath = DATA_FOLDER + filename

    # Load file
    try:
        raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
    except FileNotFoundError:
        return None, f"FILE NOT FOUND: {filename}"
    except Exception as e:
        return None, f"ERROR loading {filename}: {e}"

    # Keep only EEG channels
    raw_eeg = raw.copy().pick_types(eeg=True, verbose=False)
    data_uv = raw_eeg.get_data() * 1e6
    ch_names = raw_eeg.ch_names

    # Check each channel
    flat_ch  = []
    noisy_ch = []
    good_ch  = []

    for i, ch in enumerate(ch_names):
        std = np.std(data_uv[i])
        if std < FLAT_THRESHOLD_UV:
            flat_ch.append((ch, std))
        elif std > NOISY_THRESHOLD_UV:
            noisy_ch.append((ch, std))
        else:
            good_ch.append((ch, std))

    result = {
        "label":      label,
        "filename":   filename,
        "duration":   raw.times[-1],
        "sfreq":      raw.info["sfreq"],
        "n_channels": len(ch_names),
        "good_ch":    good_ch,
        "flat_ch":    flat_ch,
        "noisy_ch":   noisy_ch,
        "bad_pct":    (len(flat_ch) + len(noisy_ch)) / len(ch_names) * 100,
        "all_stds":   [np.std(data_uv[i]) for i in range(len(ch_names))],
        "ch_names":   ch_names,
    }

    return result, None


# ============================================================
# Run quality check on all files
# ============================================================

print("\n" + "="*65)
print("   SIGNAL QUALITY CHECK — ALL PILOT FILES")
print("="*65)

all_results = {}

for label, filename in FILES.items():
    result, error = check_one_file(label, filename)
    if error:
        print(f"\n {error}")
    else:
        all_results[label] = result

# ============================================================
# Print summary for each file
# ============================================================

print("\n" + "="*65)
print("   RESULTS PER FILE")
print("="*65)

for label, r in all_results.items():
    status = "OK" if r["bad_pct"] <= 20 else "BAD"
    print(f"\n[{status}] {label}")
    print(f"      File     : {r['filename']}")
    print(f"      Duration : {r['duration']:.1f} sec")
    print(f"      Good     : {len(r['good_ch'])} / {r['n_channels']} channels")
    print(f"      Flat     : {len(r['flat_ch'])} channels", end="")
    if r["flat_ch"]:
        print(f"  → {[c[0] for c in r['flat_ch']]}", end="")
    print()
    print(f"      Noisy    : {len(r['noisy_ch'])} channels", end="")
    if r["noisy_ch"]:
        print(f"  → {[c[0] for c in r['noisy_ch']]}", end="")
    print()
    print(f"      Bad %    : {r['bad_pct']:.0f}%")

# ============================================================
# Plot 1 — Bad channel % per recording (overview bar chart)
# ============================================================

labels     = list(all_results.keys())
bad_pcts   = [all_results[l]["bad_pct"] for l in labels]
colors     = ["green" if p <= 20 else "red" for p in bad_pcts]

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(range(len(labels)), bad_pcts, color=colors)
ax.axhline(20, color="red", linestyle="--", label="20% threshold")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Bad channels (%)")
ax.set_title("Signal Quality Overview — Bad Channel % per Recording\n(Green = acceptable, Red = too many bad channels)")
ax.legend()
plt.tight_layout()
plt.savefig("quality_overview.png", dpi=150)
plt.show(block=True)
print("\nPlot saved: quality_overview.png")

# ============================================================
# Plot 2 — Channel std heatmap across all recordings
# ============================================================

# Build matrix: rows = channels, columns = recordings
# Use first result to get channel names
first_result = list(all_results.values())[0]
ch_names     = list(first_result["ch_names"])
n_ch         = len(ch_names)
n_files      = len(all_results)

std_matrix   = np.zeros((n_ch, n_files))

for col, (label, r) in enumerate(all_results.items()):
    std_matrix[:, col] = r["all_stds"]

fig, ax = plt.subplots(figsize=(16, 10))
im = ax.imshow(std_matrix, aspect="auto", cmap="RdYlGn_r",
               vmin=0, vmax=100)
ax.set_xticks(range(n_files))
ax.set_xticklabels(list(all_results.keys()), rotation=45,
                   ha="right", fontsize=7)
ax.set_yticks(range(n_ch))
ax.set_yticklabels(ch_names, fontsize=6)
ax.set_title("Channel Quality Heatmap\n(Green = good, Red = noisy, Dark = flat)")
plt.colorbar(im, ax=ax, label="Signal std (µV)")
plt.tight_layout()
plt.savefig("quality_heatmap.png", dpi=150)
plt.show(block=True)
print("Plot saved: quality_heatmap.png")

# ============================================================
# Find consistently bad channels across all recordings
# ============================================================

print("\n" + "="*65)
print("   CONSISTENTLY BAD CHANNELS (bad in 3+ recordings)")
print("="*65)

bad_count = {}
for label, r in all_results.items():
    for ch, _ in r["flat_ch"] + r["noisy_ch"]:
        bad_count[ch] = bad_count.get(ch, 0) + 1

consistent_bad = {ch: count for ch, count in bad_count.items() if count >= 3}
if consistent_bad:
    print("\n  These channels were bad in multiple recordings:")
    for ch, count in sorted(consistent_bad.items(),
                            key=lambda x: x[1], reverse=True):
        print(f"  → {ch:<8} bad in {count} / {len(all_results)} recordings")
    print("\n  These likely had high impedance from the start.")
    print("  Consider interpolating these channels during preprocessing.")
else:
    print("\n  No consistently bad channels found across recordings.")

print("\n" + "="*65)
print("   DONE")
print("="*65 + "\n")
