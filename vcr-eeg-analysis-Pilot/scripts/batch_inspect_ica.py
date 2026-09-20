#%%
"""
Batch ICA Inspection

"""

import mne
import numpy as np
import os

# ============================================================
# Settings
DATA_FOLDER   = r"..."
OUTPUT_FOLDER = DATA_FOLDER + "preprocessed\\"

FILES = {
    "01 - Baseline Eyes Open":           "fatemeh1eyeopen1min0809.vhdr",
    "02 - Baseline Eyes Closed":         "fatemeh1eyeclosed1min0809.vhdr",
    "03 - Baseline vCR Eyes Open":       "fatemeh1vcreyesopen0809.vhdr",
    "04 - Baseline vCR Eyes Closed":     "fatemeh1vcreyeclosed0809.vhdr",
    "05 - Finger Extension vCR OFF":     "fingerextensionandreleasingeyesopen.vhdr",
    "06 - Finger Extension vCR ON":      "vcrextensionandreleasingeyesopen.vhdr",
    "07 - Hold & Let Go vCR OFF":        "holdandletgo2min.vhdr",
    "08 - Hold & Let Go vCR ON":         "vcrwitheyesopenholdandletgo.vhdr",
    "09 - Counting Loud vCR OFF":        "countingbackwardsloud.vhdr",
    "10 - Counting Silent vCR OFF":      "countingbackwardssilent30sec.vhdr",
    "11 - Counting Silent vCR ON":       "countingbackwardssilentwithvcr30sec.vhdr",
    "12 - vCR 40Hz Resting":             "VCR40restingopeneyes1min.vhdr",
    "13 - vCR 60Hz Resting":             "VCR60restingopeneyes1min.vhdr",
    "14 - vCR 80Hz Resting":             "VCR80restingopeneyes1min.vhdr",
    "15 - vCR 100Hz Resting":            "VCR100restingopeneyes1min.vhdr",
}

BAD_CHANNELS = ["Fp1", "Fp2", "AF7", "AF8"]
N_COMPONENTS = 20

def load_and_prepare(filepath):
    """Load and preprocess one file ready for ICA."""
    raw = mne.io.read_raw_brainvision(
        filepath, preload=True, verbose=False
    )
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
    """Fit ICA and return ica object + auto-detected components."""
    ica = mne.preprocessing.ICA(
        n_components=N_COMPONENTS,
        method="fastica",
        random_state=42,
        verbose=False
    )
    ica.fit(raw, verbose=False)

    # Auto-detect eye components
    eog_ch = [ch for ch in ["Fp1", "Fp2"] if ch in raw.ch_names]
    auto_remove = []
    if eog_ch:
        try:
            eog_idx, _ = ica.find_bads_eog(
                raw, ch_name=eog_ch, verbose=False
            )
            auto_remove = eog_idx
        except:
            pass

    return ica, auto_remove


def get_user_selection(auto_remove):
    """Ask user which components to remove."""
    print("\n" + "-"*50)
    print("  Which components do you want to remove?")
    print("-"*50)

    if auto_remove:
        print(f"\n  Auto-detected: {auto_remove}")

    print("\n  Options:")
    print("  → Type numbers separated by commas: 0, 1, 3")
    print("  → Press Enter to use auto-detected")
    print("  → Type 'skip' to skip this file")
    print("  → Type 'quit' to stop and save progress\n")

    user_input = input("  Your selection: ").strip().lower()

    if user_input == "quit":
        return "quit"
    elif user_input == "skip":
        return "skip"
    elif user_input == "":
        return auto_remove
    else:
        try:
            return [int(x.strip()) for x in user_input.split(",")
                    if x.strip().isdigit()]
        except:
            print("  Could not parse — using auto-detected")
            return auto_remove


# ============================================================
# Main loop — go through all files
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Check which files are already done
already_done = []
for label, filename in FILES.items():
    manual_file = OUTPUT_FOLDER + filename.replace(
        ".vhdr", "_clean_manual.fif"
    )
    if os.path.exists(manual_file):
        already_done.append(label)

print("\n" + "="*55)
print("   BATCH ICA INSPECTION")
print("   vCR-EEG Project")
print("="*55)
print(f"\n  Total files   : {len(FILES)}")
print(f"  Already done  : {len(already_done)}")
print(f"  Remaining     : {len(FILES) - len(already_done)}")

if already_done:
    print(f"\n  Already inspected (will skip):")
    for label in already_done:
        print(f"    ✓ {label}")

print("\n  Press Enter to start...", end="")
input()

# ─── Loop through files ───
results_log = {}
file_list   = list(FILES.items())

for i, (label, filename) in enumerate(file_list):

    # Skip if already done
    manual_path = OUTPUT_FOLDER + filename.replace(
        ".vhdr", "_clean_manual.fif"
    )
    if os.path.exists(manual_path):
        print(f"\n  [{i+1}/{len(FILES)}] Skipping (already done): {label}")
        results_log[label] = "already done"
        continue

    # ─── Header ───
    print(f"\n{'='*55}")
    print(f"  [{i+1}/{len(FILES)}]  {label}")
    print(f"  File: {filename}")
    print(f"{'='*55}")

    # ─── Load ───
    print("\n  Loading and preparing...")
    try:
        raw = load_and_prepare(DATA_FOLDER + filename)
        print(f"  OK — {len(raw.ch_names)} channels, "
              f"{raw.times[-1]:.1f} sec")
    except Exception as e:
        print(f"  ERROR: {e}")
        results_log[label] = f"load error: {e}"
        continue

    # ─── ICA ───
    print("  Fitting ICA...")
    ica, auto_remove = fit_ica(raw)
    print(f"  OK — auto-detected: {auto_remove}")

    # ─── Show plots ───
    print("\n  Opening plots...")
    print("  → Close Plot 1 (topographies) first")
    print("  → Then close Plot 2 (time courses)")
    print("  → Then enter your selection\n")

    ica.plot_components(
        picks=range(N_COMPONENTS),
        title=f"[{i+1}/{len(FILES)}] {label} — Topographies"
    )
    ica.plot_sources(
        raw,
        title=f"[{i+1}/{len(FILES)}] {label} — Time Courses"
    )

    # ─── User selection ───
    selection = get_user_selection(auto_remove)

    if selection == "quit":
        print("\n  Stopping — progress saved so far")
        break
    elif selection == "skip":
        print("  Skipped")
        results_log[label] = "skipped"
        continue

    # ─── Apply and save ───
    if not selection:
        print("  No components selected — saving without ICA removal")
        raw_clean = raw.copy()
    else:
        print(f"\n  Removing components: {selection}")
        raw_clean = raw.copy()
        ica.exclude = selection
        ica.apply(raw_clean, verbose=False)

        # Show before/after
        print("  Showing before/after — close to continue")
        ica.plot_overlay(
            raw,
            exclude=selection,
            title=f"Before vs After — {label}"
        )

    # Save
    raw_clean.save(manual_path, overwrite=True, verbose=False)
    print(f"  Saved: {filename.replace('.vhdr', '_clean_manual.fif')}")
    results_log[label] = f"removed: {selection}"


# ─── Final summary ───
print("\n" + "="*55)
print("   BATCH INSPECTION SUMMARY")
print("="*55)

for label, status in results_log.items():
    print(f"\n  {label}")
    print(f"    → {status}")

done_count = sum(
    1 for s in results_log.values()
    if "removed" in s or "already done" in s
)
print(f"\n  Completed: {done_count} / {len(FILES)} files")

remaining = [
    label for label, filename in FILES.items()
    if not os.path.exists(
        OUTPUT_FOLDER + filename.replace(".vhdr", "_clean_manual.fif")
    )
]
if remaining:
    print(f"\n  Still remaining ({len(remaining)} files):")
    for r in remaining:
        print(f"  → {r}")
    print("\n  Run the script again to continue where you left off")
else:
    print("\n  All files done!")
    print("  Next step: run compare_vcr_on_off.py")
    print("  (update filenames to _clean_manual.fif)")

print("="*55 + "\n")

# %%
