

import mne
import numpy as np
import matplotlib.pyplot as plt

# Settings
DATA_FOLDER   = r"..."
OUTPUT_FOLDER = DATA_FOLDER

# ROI channels — sensorimotor cortex
ROI_CHANNELS = ["C3", "C4", "CP3", "CP4", "Cz"]

# PSD settings
FMIN = 1.0
FMAX = 60.0

# Group 1 — Minimal: resting OFF vs ON (main request)
MINIMAL_CONDITIONS = [
    ("Resting — vCR OFF (eyes open)",
     "fatemeh1eyeopen1min0809_clean_manual.fif",
     "black", "-"),
    ("Resting — vCR ON 20 (eyes open)",
     "fatemeh1vcreyesopen0809_clean_manual.fif",
     "red", "-"),
]

# Group 2 — Resting: all vCR amplitudes
RESTING_CONDITIONS = [
    ("Baseline (vCR OFF, eyes open)",
     "fatemeh1eyeopen1min0809_clean_manual.fif",
     "black", "-"),
    ("Baseline (vCR OFF, eyes closed)",
     "fatemeh1eyeclosed1min0809_clean_manual.fif",
     "gray", "--"),
    ("vCR 20 (eyes open)",
     "fatemeh1vcreyesopen0809_clean_manual.fif",
     "red", "-"),
    ("vCR 20 (eyes closed)",
     "fatemeh1vcreyeclosed0809_clean_manual.fif",
     "blue", "-"),
    ("vCR 40",
     "VCR40restingopeneyes1min_clean_manual.fif",
     "green", "-"),
    ("vCR 60",
     "VCR60restingopeneyes1min_clean_manual.fif",
     "orange", "-"),
    ("vCR 80",
     "VCR80restingopeneyes1min_clean_manual.fif",
     "purple", "-"),
    ("vCR 100",
     "VCR100restingopeneyes1min_clean_manual.fif",
     "cyan", "-"),
]

# Group 3 — Motor tasks: vCR OFF vs ON
MOTOR_CONDITIONS = [
    ("Hold & Let Go — vCR OFF (open)",
     "holdandletgo2min_clean_manual.fif",
     "black", "-"),
    ("Hold & Let Go — vCR ON (open)",
     "vcrwitheyesopenholdandletgo_clean_manual.fif",
     "red", "-"),
    ("Finger Extension — vCR OFF",
     "fingerextensionandreleasingeyesopen_clean_manual.fif",
     "blue", "--"),
    ("Finger Extension — vCR ON",
     "vcrextensionandreleasingeyesopen_clean_manual.fif",
     "orange", "--"),
]

def compute_psd(filepath, roi_channels, fmin, fmax):
    """
    Load preprocessed file and compute mean PSD
    averaged across ROI channels.
    Returns freqs (Hz) and psd (µV²/Hz).
    """
    try:
        raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None

    available = [ch for ch in roi_channels if ch in raw.ch_names]
    if not available:
        print(f"  ERROR: no ROI channels found")
        return None, None

    raw_roi = raw.copy().pick_channels(available, verbose=False)

    psd_obj = raw_roi.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=int(raw_roi.info["sfreq"] * 2),
        n_overlap=int(raw_roi.info["sfreq"]),
        verbose=False
    )

    freqs   = psd_obj.freqs
    psd_uv  = psd_obj.get_data() * 1e12   # convert V²/Hz to µV²/Hz
    return freqs, psd_uv.mean(axis=0)


def compute_psd_per_channel(filepath, roi_channels, fmin, fmax):
    """
    Same as compute_psd but returns a dict {channel: psd_array}.
    """
    try:
        raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None

    available = [ch for ch in roi_channels if ch in raw.ch_names]
    if not available:
        return None, None

    raw_roi = raw.copy().pick_channels(available, verbose=False)

    psd_obj  = raw_roi.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=int(raw_roi.info["sfreq"] * 2),
        n_overlap=int(raw_roi.info["sfreq"]),
        verbose=False
    )

    freqs    = psd_obj.freqs
    psd_data = psd_obj.get_data() * 1e12

    return freqs, {ch: psd_data[i] for i, ch in enumerate(available)}


def add_band_shading(ax):
    """Add frequency band shading to a PSD axis."""
    ax.axvspan( 8, 12, alpha=0.08, color="green")
    ax.axvspan(13, 20, alpha=0.08, color="blue")
    ax.axvspan(21, 30, alpha=0.08, color="red")


# Plot 1 — Minimal: resting OFF vs ON (for email)

print("\n" + "="*60)
print("   PSD PLOTS — Session 1 (FS)")
print("="*60)

print("\n  Plot 1: Resting — vCR OFF vs ON (minimal)")

fig, ax = plt.subplots(figsize=(10, 6))

for label, filename, color, ls in MINIMAL_CONDITIONS:
    print(f"  Loading: {label}")
    freqs, psd = compute_psd(DATA_FOLDER + filename,
                             ROI_CHANNELS, FMIN, FMAX)
    if psd is not None:
        ax.plot(freqs, psd, color=color, linestyle=ls,
                linewidth=2, label=label)

add_band_shading(ax)
ax.set_xlabel("Frequency (Hz)", fontsize=12)
ax.set_ylabel("PSD (µV²/Hz)", fontsize=12)
ax.set_title("Power Spectral Density — Resting State\n"
             "Session 1 (FS) | ROI: C3, C4, CP3, CP4, Cz",
             fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(FMIN, FMAX)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "session1_psd_resting_minimal.png", dpi=150)
plt.show(block=True)
print("  Saved: session1_psd_resting_minimal.png")


# Plot 2 — Resting: all vCR amplitudes

print("\n  Plot 2: Resting — all vCR amplitudes")

fig, ax = plt.subplots(figsize=(12, 6))

for label, filename, color, ls in RESTING_CONDITIONS:
    print(f"  Loading: {label}")
    freqs, psd = compute_psd(DATA_FOLDER + filename,
                             ROI_CHANNELS, FMIN, FMAX)
    if psd is not None:
        ax.plot(freqs, psd, color=color, linestyle=ls,
                linewidth=1.8, label=label)

add_band_shading(ax)
ax.set_xlabel("Frequency (Hz)", fontsize=12)
ax.set_ylabel("PSD (µV²/Hz)", fontsize=12)
ax.set_title("Power Spectral Density — Resting State (all vCR amplitudes)\n"
             "Session 1 (FS) | ROI: C3, C4, CP3, CP4, Cz",
             fontsize=12)
ax.legend(fontsize=8)
ax.set_xlim(FMIN, FMAX)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "session1_psd_resting_all.png", dpi=150)
plt.show(block=True)
print("  Saved: session1_psd_resting_all.png")

# Plot 3 — Motor tasks: vCR OFF vs ON

print("\n  Plot 3: Motor tasks — vCR OFF vs ON")

fig, ax = plt.subplots(figsize=(12, 6))

for label, filename, color, ls in MOTOR_CONDITIONS:
    print(f"  Loading: {label}")
    freqs, psd = compute_psd(DATA_FOLDER + filename,
                             ROI_CHANNELS, FMIN, FMAX)
    if psd is not None:
        ax.plot(freqs, psd, color=color, linestyle=ls,
                linewidth=1.8, label=label)

add_band_shading(ax)
ax.set_xlabel("Frequency (Hz)", fontsize=12)
ax.set_ylabel("PSD (µV²/Hz)", fontsize=12)
ax.set_title("Power Spectral Density — Motor Tasks\n"
             "Session 1 (FS) | ROI: C3, C4, CP3, CP4, Cz",
             fontsize=12)
ax.legend(fontsize=8)
ax.set_xlim(FMIN, FMAX)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "session1_psd_motor.png", dpi=150)
plt.show(block=True)
print("  Saved: session1_psd_motor.png")

# Plot 4 — Per channel: resting OFF vs ON

print("\n  Plot 4: Per-channel PSD — resting vCR OFF vs ON")

fig, axes = plt.subplots(1, len(ROI_CHANNELS),
                         figsize=(16, 5), sharey=True)

for col, ch in enumerate(ROI_CHANNELS):
    ax = axes[col]

    for label, filename, color, ls in MINIMAL_CONDITIONS:
        freqs, psd_dict = compute_psd_per_channel(
            DATA_FOLDER + filename, ROI_CHANNELS, FMIN, FMAX
        )
        if psd_dict is not None and ch in psd_dict:
            ax.plot(freqs, psd_dict[ch],
                    color=color, linestyle=ls,
                    linewidth=1.8,
                    label=label if col == 0 else "")

    add_band_shading(ax)
    ax.set_title(ch, fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_xlim(FMIN, FMAX)
    ax.grid(True, alpha=0.3)

    if col == 0:
        ax.set_ylabel("PSD (µV²/Hz)", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

plt.suptitle(
    "PSD per Channel — Resting State | vCR OFF vs ON\n"
    "Session 1 (FS) | Green=alpha | Blue=low beta | Red=high beta",
    fontsize=11
)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "session1_psd_per_channel.png",
            dpi=150, bbox_inches="tight")
plt.show(block=True)
print("  Saved: session1_psd_per_channel.png")

# Done
print("\n" + "="*60)
print("   ALL PSD PLOTS SAVED — Session 1 (FS)")
print("="*60)
print("""
  Files saved to preprocessed/ folder:
  1. session1_psd_resting_minimal.png   ← OFF vs ON 20 only
  2. session1_psd_resting_all.png       ← all vCR amplitudes
  3. session1_psd_motor.png             ← motor tasks OFF vs ON
  4. session1_psd_per_channel.png       ← one panel per channel
""")

