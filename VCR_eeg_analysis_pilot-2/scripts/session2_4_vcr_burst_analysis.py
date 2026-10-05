#%%

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from mne.filter import filter_data

# Settings
DATA_FOLDER   = r"..."
OUTPUT_FOLDER = r"..."
VCR_FILES = [
    ("Resting - vCR 20 (eyes open)",
     "20amplitudevcreyesopen1min_clean_manual.fif"),
    ("Resting - vCR 20 (eyes closed)",
     "20amplitudevcreyesclosed_clean_manual.fif"),
    ("Hold & Let Go - vCR 20 (eyes open)",
     "Second2minholdandletgoeyesopenVCR20Amp_clean_manual.fif"),
    ("Hold & Let Go - vCR 20 (eyes closed)",
     "Second2minholdandletgoeyesclosedVCR20Amp_clean_manual.fif"),
    ("Finger Extension - vCR 20",
     "SecondindexfingerextensioneyesopenVCR20Amp_clean_manual.fif"),
    ("Countdown - vCR 20",
     "countfrom100back7steps1mineyesopenwithVCR20Amp_clean_manual.fif"),
    ("Resting - vCR 40",
     "VCR40AMP1mineyesopen_clean_manual.fif"),
    ("Resting - vCR 60",
     "VCR60AMP1mineyesopen_clean_manual.fif"),
    ("Resting - vCR 80",
     "VCR80AMP1mineyesopen_clean_manual.fif"),
    ("Resting - vCR 100",
     "VCR100AMP1mineyesopen_clean_manual.fif"),
]

# ROI: sensorimotor channels
ROI_CHANNELS = ["C3", "C4", "CP3", "CP4", "Cz"]

# Epoch window
TMIN = -0.5   # short pre-trigger window
TMAX =  2.0   # post-trigger window

BASELINE = (-0.5, -0.05)

# Beta bands
LOW_BETA  = (13, 20)
HIGH_BETA = (21, 30)

# vCR trigger label
VCR_TRIGGER = "Stimulus/S  1"

def compute_instantaneous_power(data, sfreq, band):
    original_shape = data.shape
    data_2d        = data.reshape(-1, data.shape[-1])

    # Step 1: bandpass filter
    data_filtered = filter_data(
        data_2d,
        sfreq=sfreq,
        l_freq=band[0],
        h_freq=band[1],
        verbose=False
    )

    analytic = hilbert(data_filtered, axis=-1)
    power    = np.abs(analytic) ** 2

    return power.reshape(original_shape)


def load_and_epoch(filepath, roi_channels, tmin, tmax):
    
    try:
        raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return None, 0

    # Keep only available ROI channels
    available = [ch for ch in roi_channels if ch in raw.ch_names]
    if not available:
        print(f"  ERROR: no ROI channels found")
        return None, 0

    raw_roi = raw.copy().pick_channels(available, verbose=False)

    # Extract S1 trigger times from annotations
    s1_times = [
        ann["onset"] for ann in raw.annotations
        if VCR_TRIGGER in ann["description"]
    ]

    if not s1_times:
        print(f"  WARNING: no S1 triggers found")
        return None, 0

    print(f"  S1 triggers found: {len(s1_times)}")

    # Check baseline overlap risk
    gaps    = [s1_times[i+1] - s1_times[i]
               for i in range(len(s1_times) - 1)]
    min_gap = min(gaps)
    print(f"  Min inter-trigger gap: {min_gap:.3f} sec  "
          f"(baseline ends at {BASELINE[1]} sec → safe ✓)")

    # Build MNE events array: [sample, 0, event_id]
    sfreq  = raw_roi.info["sfreq"]
    events = np.array([[int(t * sfreq), 0, 1] for t in s1_times])

    # Cut epochs — no automatic baseline correction
    epochs = mne.Epochs(
        raw_roi,
        events,
        event_id={"S1": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    print(f"  Epochs kept: {len(epochs)} / {len(s1_times)}")
    return epochs, len(s1_times)


def get_beta_timecourse(epochs, band):


    sfreq = epochs.info["sfreq"]
    data  = epochs.get_data()    # (n_epochs, n_channels, n_times)
    times = epochs.times

    # Instantaneous power via Hilbert
    power = compute_instantaneous_power(data, sfreq, band)

    # Average across epochs and channels -> (n_times,)
    mean_power = power.mean(axis=0).mean(axis=0)

    # Smooth with 100ms sliding window
    window   = int(0.1 * sfreq)
    smoothed = np.convolve(mean_power,
                           np.ones(window) / window,
                           mode="same")

    # Baseline correction: % change relative to pre-burst window
    bl_mask    = (times >= BASELINE[0]) & (times <= BASELINE[1])
    bl_mean    = smoothed[bl_mask].mean()
    pct_change = ((smoothed - bl_mean) / bl_mean) * 100

    return times, pct_change

print("\n" + "="*65)
print("   vCR BURST ANALYSIS — TRIGGER-BASED EPOCHS")
print("   Session 2  | Hilbert power | Safe baseline")
print("="*65)
print(f"\n  Epoch window  : {TMIN} to {TMAX} sec")
print(f"  Baseline      : {BASELINE[0]} to {BASELINE[1]} sec")
print(f"  Power method  : Hilbert transform (instantaneous power)")
print(f"  ROI channels  : {ROI_CHANNELS}")
print(f"  Beta bands    : Low {LOW_BETA} Hz | High {HIGH_BETA} Hz")

all_results = {}

for label, filename in VCR_FILES:

    print(f"\n--- {label} ---")
    epochs, n_triggers = load_and_epoch(
        DATA_FOLDER + filename, ROI_CHANNELS, TMIN, TMAX
    )

    if epochs is None or len(epochs) == 0:
        print("  Skipping")
        continue

    # Compute timecourses for both bands
    times_low,  pct_low  = get_beta_timecourse(epochs, LOW_BETA)
    times_high, pct_high = get_beta_timecourse(epochs, HIGH_BETA)

    # Pre-burst window: matches baseline (-0.4 to -0.05 sec)
    # Post-burst window: immediate post-burst response (0.1 to 1.0 sec)
    pre_mask  = (times_low >= -0.4) & (times_low <  -0.05)
    post_mask = (times_low >=  0.1) & (times_low <=  1.0)

    pre_low   = pct_low[pre_mask].mean()
    post_low  = pct_low[post_mask].mean()
    pre_high  = pct_high[pre_mask].mean()
    post_high = pct_high[post_mask].mean()

    delta_low  = post_low  - pre_low
    delta_high = post_high - pre_high

    print(f"\n  Low beta  (13-20 Hz): "
          f"pre={pre_low:+.1f}%  post={post_low:+.1f}%  "
          f"Δ={delta_low:+.1f}%")
    print(f"  High beta (21-30 Hz): "
          f"pre={pre_high:+.1f}%  post={post_high:+.1f}%  "
          f"Δ={delta_high:+.1f}%")

    all_results[label] = {
        "times_low":  times_low,
        "times_high": times_high,
        "pct_low":    pct_low,
        "pct_high":   pct_high,
        "n_epochs":   len(epochs),
        "pre_low":    pre_low,    "post_low":  post_low,
        "pre_high":   pre_high,   "post_high": post_high,
        "delta_low":  delta_low,  "delta_high": delta_high,
    }
# Plot 1 — Beta timecourse per condition
if all_results:

    n   = len(all_results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for row, (label, r) in enumerate(all_results.items()):
        for col, (band_name, times, pct, color) in enumerate([
            ("Low beta (13-20 Hz)",  r["times_low"],  r["pct_low"],  "steelblue"),
            ("High beta (21-30 Hz)", r["times_high"], r["pct_high"], "tomato"),
        ]):
            ax = axes[row][col]
            ax.plot(times, pct, color=color, linewidth=1.5)
            ax.axvline(0,   color="red",  linestyle="--",
                       linewidth=1.2, label="S1 (vCR burst)")
            ax.axhline(0,   color="gray", linewidth=0.8)
            ax.axvspan(BASELINE[0], BASELINE[1],
                       alpha=0.15, color="gray", label="baseline")
            ax.set_ylabel("Beta change (%)")
            ax.set_title(f"{label}\n{band_name}", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Time relative to S1 trigger (sec)")
    axes[-1][1].set_xlabel("Time relative to S1 trigger (sec)")
    plt.suptitle(
        "Beta Power Time-Locked to vCR Bursts (S1) — Session 2 \n"
        "Hilbert power | Baseline: -0.5 to -0.05 sec | Red = burst onset",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig("session2_vcr_burst_timecourse.png",
                dpi=150, bbox_inches="tight")
    plt.show(block=True)
    print("\nSaved: session2_vcr_burst_timecourse.png")

# Plot 2 — Pre vs post bar chart
if all_results:

    labels = list(all_results.keys())
    x      = np.arange(len(labels))
    width  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pre_key, post_key, title, c_pre, c_post in [
        (axes[0], "pre_low",  "post_low",
         "Low beta (13-20 Hz)",  "lightsteelblue", "steelblue"),
        (axes[1], "pre_high", "post_high",
         "High beta (21-30 Hz)", "lightsalmon",    "tomato"),
    ]:
        pre_vals  = [all_results[l][pre_key]  for l in labels]
        post_vals = [all_results[l][post_key] for l in labels]

        ax.bar(x - width/2, pre_vals,  width,
               label="Pre-burst (-0.4 to -0.05 sec)",
               color=c_pre,  edgecolor="black")
        ax.bar(x + width/2, post_vals, width,
               label="Post-burst (0.1 to 1.0 sec)",
               color=c_post, edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Beta change from baseline (%)")
        ax.set_title(f"{title}\nPre vs Post vCR burst")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Pre vs Post vCR Burst Beta Power — Session 2 \n"
        "Hilbert power | Baseline: -0.5 to -0.05 sec",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("session2_vcr_burst_summary.png", dpi=150)
    plt.show(block=True)
    print("Saved: session2_vcr_burst_summary.png")

# Summary table

print("\n" + "="*65)
print("   SUMMARY — Pre vs Post vCR Burst Beta Change")
print("="*65)
print(f"\n  {'Condition':<42} {'LB Δ':>8} {'HB Δ':>8}  {'epochs':>6}")
print(f"  {'─'*72}")

for label, r in all_results.items():
    print(f"  {label[:40]:<42} "
          f"{r['delta_low']:>+7.1f}% "
          f"{r['delta_high']:>+7.1f}%  "
          f"{r['n_epochs']:>6}")

print(f"\n  LB = Low beta (13-20 Hz)")
print(f"  HB = High beta (21-30 Hz)")
print(f"  Δ  = post minus pre (% change from baseline)")
print("\n" + "="*65 + "\n")


