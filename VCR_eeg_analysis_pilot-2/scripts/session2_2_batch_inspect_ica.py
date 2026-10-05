#%%
import mne
import numpy as np
import os

DATA_FOLDER   = r"...."
OUTPUT_FOLDER = DATA_FOLDER + "preprocessed\\"

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

BAD_CHANNELS = ["Fp1", "Fp2", "AF7", "AF8"]
N_COMPONENTS = 20

def load_and_prepare(filepath):
    raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
    raw.pick_types(eeg=True, verbose=False)
    raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
    raw.notch_filter(freqs=50, verbose=False)
    bad_ch = [ch for ch in BAD_CHANNELS if ch in raw.ch_names]
    if bad_ch:
        raw.info["bads"] = bad_ch
        raw.interpolate_bads(reset_bads=True, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    return raw


def fit_ica(raw):
    ica = mne.preprocessing.ICA(
        n_components=N_COMPONENTS, method="fastica",
        random_state=42, verbose=False
    )
    ica.fit(raw, verbose=False)
    eog_ch = [ch for ch in ["Fp1", "Fp2"] if ch in raw.ch_names]
    auto_remove = []
    if eog_ch:
        try:
            eog_idx, _ = ica.find_bads_eog(raw, ch_name=eog_ch, verbose=False)
            auto_remove = eog_idx
        except:
            pass
    return ica, auto_remove


def get_user_selection(auto_remove):
    print("\n" + "-"*50)
    print("  Which components to remove?")
    if auto_remove:
        print(f"  Auto-detected: {auto_remove}")
    print("  → Type numbers: 0, 1, 3")
    print("  → Press Enter to use auto-detected")
    print("  → Type 'skip' to skip this file")
    print("  → Type 'quit' to stop\n")
    user_input = input("  Your selection: ").strip().lower()
    if user_input == "quit":
        return "quit"
    elif user_input == "skip":
        return "skip"
    elif user_input == "":
        return auto_remove
    else:
        try:
            return [int(x.strip()) for x in user_input.split(",") if x.strip().isdigit()]
        except:
            return auto_remove


# ─── Main loop ───
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

already_done = []
for label, filename in FILES.items():
    if os.path.exists(OUTPUT_FOLDER + filename.replace(".vhdr", "_clean_manual.fif")):
        already_done.append(label)

print("\n" + "="*60)
print("   BATCH ICA INSPECTION — SESSION 2 ")
print("="*60)
print(f"  Total files  : {len(FILES)}")
print(f"  Already done : {len(already_done)}")
print(f"  Remaining    : {len(FILES) - len(already_done)}")

if already_done:
    print(f"\n  Already done (will skip):")
    for l in already_done:
        print(f"    ✓ {l}")

print("\n  Press Enter to start...", end="")
input()

results_log = {}

for i, (label, filename) in enumerate(FILES.items()):

    manual_path = OUTPUT_FOLDER + filename.replace(".vhdr", "_clean_manual.fif")
    if os.path.exists(manual_path):
        print(f"\n  [{i+1}/{len(FILES)}] Skipping (done): {label}")
        results_log[label] = "already done"
        continue

    print(f"\n{'='*60}")
    print(f"  [{i+1}/{len(FILES)}]  {label}")
    print(f"  File: {filename}")
    print(f"{'='*60}")

    print("\n  Loading and preparing...")
    try:
        raw = load_and_prepare(DATA_FOLDER + filename)
        print(f"  OK — {len(raw.ch_names)} channels, {raw.times[-1]:.1f} sec")
    except Exception as e:
        print(f"  ERROR: {e}")
        results_log[label] = f"error: {e}"
        continue

    print("  Fitting ICA...")
    ica, auto_remove = fit_ica(raw)
    print(f"  OK — auto-detected: {auto_remove}")

    print("\n  Opening plots — close Plot 1, then Plot 2, then select\n")
    ica.plot_components(picks=range(N_COMPONENTS),
                        title=f"[{i+1}/{len(FILES)}] {label} — Topographies")
    ica.plot_sources(raw, title=f"[{i+1}/{len(FILES)}] {label} — Time Courses")

    selection = get_user_selection(auto_remove)

    if selection == "quit":
        print("\n  Stopped — progress saved")
        break
    elif selection == "skip":
        results_log[label] = "skipped"
        continue

    raw_clean = raw.copy()
    if selection:
        print(f"\n  Removing: {selection}")
        ica.exclude = selection
        ica.apply(raw_clean, verbose=False)
        ica.plot_overlay(raw, exclude=selection,
                         title=f"Before vs After — {label}")

    raw_clean.save(manual_path, overwrite=True, verbose=False)
    print(f"  Saved: {filename.replace('.vhdr', '_clean_manual.fif')}")
    results_log[label] = f"removed: {selection}"

# ─── Summary ───
print("\n" + "="*60)
print("   SUMMARY — SESSION 2 ICA")
print("="*60)
for label, status in results_log.items():
    print(f"  {label}")
    print(f"    → {status}")

remaining = [l for l, f in FILES.items()
             if not os.path.exists(OUTPUT_FOLDER + f.replace(".vhdr", "_clean_manual.fif"))]
if remaining:
    print(f"\n  Still remaining: {len(remaining)} files")
    print("  Run again to continue")
else:
    print("\n  All done!")
print("="*60 + "\n")

# %%
