#%%

import mne
import numpy as np
import matplotlib.pyplot as plt

DATA_FOLDER = r"...."

FILES = {
    "01 Resting - Eyes Open":              "restingeyesopen1min.vhdr",
    "02 Resting - Eyes Closed":            "restingeyesclosed.vhdr",
    "03 Stim Resting - vCR 20 Open":       "20amplitudevcreyesopen1min.vhdr",
    "04 Stim Resting - vCR 20 Closed":     "20amplitudevcreyesclosed.vhdr",
    "05 Hold & Let Go - OFF Open":         "Second2minholdandletgoeyesopen.vhdr",
    "06 Hold & Let Go - OFF Closed":       "Second2minholdandletgoeyesclosed.vhdr",
    "07 Hold & Let Go - vCR 20 Open":      "Second2minholdandletgoeyesopenVCR20Amp.vhdr",
    "08 Hold & Let Go - vCR 20 Closed":    "Second2minholdandletgoeyesclosedVCR20Amp.vhdr",
    "09 Finger Extension - OFF":           "Secondindexfingerextensioneyesopen.vhdr",
    "10 Finger Extension - vCR 20":        "SecondindexfingerextensioneyesopenVCR20Amp.vhdr",
    "11 Countdown - OFF":                  "countfroma100back7steps1mineyesopen.vhdr",
    "12 Countdown - vCR 20":              "countfrom100back7steps1mineyesopenwithVCR20Amp.vhdr",
    "13 Stim Resting - vCR 40":           "VCR40AMP1mineyesopen.vhdr",
    "14 Stim Resting - vCR 60":           "VCR60AMP1mineyesopen.vhdr",
    "15 Stim Resting - vCR 80":           "VCR80AMP1mineyesopen.vhdr",
    "16 Stim Resting - vCR 100":          "VCR100AMP1mineyesopen.vhdr",
}

FLAT_THRESHOLD_UV  = 1.0
NOISY_THRESHOLD_UV = 40.0

def check_one_file(label, filename):
    filepath = DATA_FOLDER + filename
    try:
        raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
    except FileNotFoundError:
        return None, f"FILE NOT FOUND: {filename}"
    except Exception as e:
        return None, f"ERROR: {e}"

    raw_eeg = raw.copy().pick_types(eeg=True, verbose=False)
    data_uv = raw_eeg.get_data() * 1e6

    flat_ch  = []
    noisy_ch = []
    good_ch  = []

    for i, ch in enumerate(raw_eeg.ch_names):
        std = np.std(data_uv[i])
        if std < FLAT_THRESHOLD_UV:
            flat_ch.append((ch, std))
        elif std > NOISY_THRESHOLD_UV:
            noisy_ch.append((ch, std))
        else:
            good_ch.append(ch)

    return {
        "label":     label,
        "filename":  filename,
        "duration":  raw.times[-1],
        "n_ch":      len(raw_eeg.ch_names),
        "good_ch":   good_ch,
        "flat_ch":   flat_ch,
        "noisy_ch":  noisy_ch,
        "bad_pct":   (len(flat_ch) + len(noisy_ch)) / len(raw_eeg.ch_names) * 100,
        "all_stds":  [np.std(data_uv[i]) for i in range(len(raw_eeg.ch_names))],
        "ch_names":  raw_eeg.ch_names,
    }, None


# Run checks
print("\n" + "="*65)
print("   SIGNAL QUALITY CHECK — SESSION 2 (MP)")
print("="*65)

all_results = {}
for label, filename in FILES.items():
    result, error = check_one_file(label, filename)
    if error:
        print(f"\n  {error}")
    else:
        all_results[label] = result

# ─── Print results ───
print("\n" + "="*65)
print("   RESULTS PER FILE")
print("="*65)

for label, r in all_results.items():
    status = "OK " if r["bad_pct"] <= 20 else "BAD"
    print(f"\n[{status}] {label}")
    print(f"      File     : {r['filename']}")
    print(f"      Duration : {r['duration']:.1f} sec")
    print(f"      Good     : {len(r['good_ch'])} / {r['n_ch']} channels")
    print(f"      Flat     : {len(r['flat_ch'])} channels", end="")
    if r["flat_ch"]:
        print(f"  → {[c[0] for c in r['flat_ch']]}", end="")
    print()
    print(f"      Noisy    : {len(r['noisy_ch'])} channels", end="")
    if r["noisy_ch"]:
        print(f"  → {[c[0] for c in r['noisy_ch']]}", end="")
    print()
    print(f"      Bad %    : {r['bad_pct']:.0f}%")

# Plot 1: Bad % per file
labels   = list(all_results.keys())
bad_pcts = [all_results[l]["bad_pct"] for l in labels]
colors   = ["green" if p <= 20 else "red" for p in bad_pcts]

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(labels)), bad_pcts, color=colors)
ax.axhline(20, color="red", linestyle="--", label="20% threshold")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Bad channels (%)")
ax.set_title("Signal Quality — Session 2 \nBad Channel % per Recording")
ax.legend()
plt.tight_layout()
plt.savefig("session2_quality_overview.png", dpi=150)
plt.show(block=True)
print("\nSaved: session2_quality_overview.png")

# Plot 2: Heatmap
first = list(all_results.values())[0]
ch_names = list(first["ch_names"])
n_ch     = len(ch_names)
n_files  = len(all_results)

matrix = np.zeros((n_ch, n_files))
for col, (label, r) in enumerate(all_results.items()):
    matrix[:, col] = r["all_stds"]

fig, ax = plt.subplots(figsize=(16, 10))
im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
ax.set_xticks(range(n_files))
ax.set_xticklabels(list(all_results.keys()), rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(n_ch))
ax.set_yticklabels(ch_names, fontsize=6)
ax.set_title("Channel Quality Heatmap — Session 2 \n(Green=good, Red=noisy)")
plt.colorbar(im, ax=ax, label="Signal std (µV)")
plt.tight_layout()
plt.savefig("session2_quality_heatmap.png", dpi=150)
plt.show(block=True)
print("Saved: session2_quality_heatmap.png")

#  Consistently bad channels
print("\n" + "="*65)
print("   CONSISTENTLY BAD CHANNELS (bad in 3+ recordings)")
print("="*65)

bad_count = {}
for label, r in all_results.items():
    for ch, _ in r["flat_ch"] + r["noisy_ch"]:
        bad_count[ch] = bad_count.get(ch, 0) + 1

consistent_bad = {ch: c for ch, c in bad_count.items() if c >= 3}
if consistent_bad:
    for ch, c in sorted(consistent_bad.items(), key=lambda x: x[1], reverse=True):
        print(f"  → {ch:<8} bad in {c} / {len(all_results)} recordings")
    print("\n  These will be interpolated during preprocessing.")
else:
    print("  No consistently bad channels found.")

print("\n" + "="*65 + "\n")

# %%
