#%%

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from mne.filter import filter_data

# Settings
DATA_FOLDER   = r".."
OUTPUT_FOLDER = DATA_FOLDER

FILES = {
    "Finger Extension — vCR OFF":
        "Secondindexfingerextensioneyesopen_clean_manual.fif",
    "Finger Extension — vCR ON (20 Hz)":
        "SecondindexfingerextensioneyesopenVCR20Amp_clean_manual.fif",
}

# Movement trigger — finger release (movement offset)
MOVEMENT_TRIGGER = "Comment/r"

# Epoch window
TMIN = -2.0
TMAX =  2.0

# Baseline window for normalization
BASELINE = (-1.5, -0.5)

# Single channel — contralateral M1 (like Little et al.)
CHANNEL = "C3"

# Beta band for timecourse
BETA_BAND = (13, 30)

# TFR frequencies — 8 to 48 Hz like figure
FREQS    = np.arange(8, 49, 1)
N_CYCLES = FREQS / 2

def load_and_epoch(filepath, trigger_label, tmin, tmax):
    """Load file and cut epochs time-locked to trigger."""
    try:
        raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    trigger_times = [
        ann["onset"] for ann in raw.annotations
        if trigger_label in ann["description"]
    ]

    if not trigger_times:
        print(f"  WARNING: '{trigger_label}' not found")
        avail = set(a["description"] for a in raw.annotations)
        print(f"  Available triggers: {avail}")
        return None

    print(f"  Trigger '{trigger_label}': {len(trigger_times)} found")

    sfreq  = raw.info["sfreq"]
    events = np.array([[int(t * sfreq), 0, 1]
                       for t in trigger_times])

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"movement": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    print(f"  Epochs kept: {len(epochs)}")
    return epochs


def get_beta_timecourse(epochs, channel, band, baseline):
    """
    Compute trial-averaged beta amplitude timecourse.
    Uses Hilbert transform for instantaneous power.
    Returns times and % change from baseline.
    Also returns raw envelopes for individual trial plots.
    """
    ep    = epochs.copy().pick_channels([channel], verbose=False)
    data  = ep.get_data()[:, 0, :]   # (n_trials, n_times)
    sfreq = epochs.info["sfreq"]
    times = epochs.times

    # Bandpass + Hilbert envelope
    data_filt = filter_data(data, sfreq=sfreq,
                            l_freq=band[0], h_freq=band[1],
                            verbose=False)
    envelopes = np.abs(hilbert(data_filt, axis=-1))

    # Trial average
    mean_env = envelopes.mean(axis=0)

    # Smooth 100ms
    w        = int(0.1 * sfreq)
    smoothed = np.convolve(mean_env, np.ones(w)/w, mode="same")

    # Baseline normalize → % change
    bl_mask = (times >= baseline[0]) & (times <= baseline[1])
    bl_mean = smoothed[bl_mask].mean()
    pct     = ((smoothed - bl_mean) / bl_mean) * 100

    return times, pct, envelopes


def get_tfr(epochs, channel, freqs, n_cycles, baseline):
    """
    Compute TFR using Morlet wavelets.
    Uses 'logratio' baseline correction for better visual contrast
    (same as many published EEG papers).
    Returns TFR object.
    """
    ep  = epochs.copy().pick_channels([channel], verbose=False)
    tfr = mne.time_frequency.tfr_morlet(
        ep,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        verbose=False
    )
    # logratio gives better contrast than percent:
    # positive = power increase, negative = decrease
    # unit: log(power/baseline)
    tfr.apply_baseline(baseline=baseline, mode="logratio", verbose=False)
    return tfr


def plot_tfr(ax, tfr_data, freqs, tmin, tmax, vmin, vmax):
    """
    Plot TFR heatmap with blue background style like Little et al.
    Blue = below baseline, red = above baseline.
    """
    im = ax.imshow(
        tfr_data,
        aspect="auto",
        origin="lower",
        extent=[tmin, tmax, freqs[0], freqs[-1]],
        cmap="RdBu_r",    # blue-white-red colormap
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear"
    )
    ax.axvline(0, color="green", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Hz", fontsize=9)
    ax.set_xlim(tmin, tmax)
    return im



# Main analysis

print("\n" + "="*65)
print("   PMBR ANALYSIS — MOVEMENT-LOCKED BETA")
print("   Session 2 (MP) | Little et al. (2019) style")
print("="*65)
print(f"\n  Trigger  : {MOVEMENT_TRIGGER} (finger release)")
print(f"  Window   : {TMIN} to {TMAX} sec")
print(f"  Baseline : {BASELINE[0]} to {BASELINE[1]} sec")
print(f"  Channel  : {CHANNEL}")

all_data = {}   # store results for comparison plot

for label, filename in FILES.items():

    print(f"\n--- {label} ---")

    epochs = load_and_epoch(
        DATA_FOLDER + filename, MOVEMENT_TRIGGER, TMIN, TMAX
    )
    if epochs is None:
        continue

    if CHANNEL not in epochs.ch_names:
        print(f"  WARNING: {CHANNEL} not in file")
        print(f"  Channels: {epochs.ch_names[:10]}")
        continue

    # Beta timecourse
    times, pct, envelopes = get_beta_timecourse(
        epochs, CHANNEL, BETA_BAND, BASELINE
    )

    # TFR
    print("  Computing TFR...")
    tfr = get_tfr(epochs, CHANNEL, FREQS, N_CYCLES, BASELINE)

    all_data[label] = {
        "epochs":    epochs,
        "times":     times,
        "pct":       pct,
        "envelopes": envelopes,
        "tfr":       tfr,
    }

    # Figure A: Trial average
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Trial Average — {label}\n"
        f"Session 2 (MP) | Channel: {CHANNEL} | n={len(epochs)} trials",
        fontsize=12
    )

    # Left: beta timecourse
    axes[0].plot(times, pct, color="red", linewidth=2)
    axes[0].axvline(0, color="green", linestyle="--",
                    linewidth=1.5, label="Movement offset")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].axvspan(BASELINE[0], BASELINE[1],
                    alpha=0.1, color="gray", label="baseline")
    axes[0].set_xlabel("Time from response (s)", fontsize=11)
    axes[0].set_ylabel("Beta amplitude (% change)", fontsize=11)
    axes[0].set_title("Trial-averaged beta amplitude", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(TMIN, TMAX)
    axes[0].grid(True, alpha=0.3)

    # Right: TFR heatmap
    # vmin/vmax tuned for logratio: ±0.5 gives good contrast
    im = plot_tfr(axes[1], tfr.data[0], FREQS, TMIN, TMAX,
                  vmin=-0.5, vmax=0.5)
    axes[1].set_xlabel("Time from response (s)", fontsize=11)
    axes[1].set_title("Time-frequency representation\n"
                      "(blue=ERD, red=ERS)", fontsize=11)
    plt.colorbar(im, ax=axes[1], label="Log power ratio")

    plt.tight_layout()
    safe  = label.replace(" ", "_").replace("(","").replace(")","").replace("/","")
    fname = OUTPUT_FOLDER + f"session2_pmbr_{safe}_average.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show(block=True)
    print(f"  Saved: session2_pmbr_{safe}_average.png")

    #Figure B: Individual trials
    n_show = min(8, len(epochs))
    fig2, axes2 = plt.subplots(n_show, 2,
                               figsize=(13, 2.5 * n_show))
    fig2.suptitle(
        f"Individual Trials — {label}\n"
        f"Session 2 (MP) | Channel: {CHANNEL}",
        fontsize=12
    )

    for idx in range(n_show):

        ax_l = axes2[idx][0]
        ax_r = axes2[idx][1]

        # Single trial beta envelope normalized
        env     = envelopes[idx]
        bl_mask = (times >= BASELINE[0]) & (times <= BASELINE[1])
        bl_mean = env[bl_mask].mean()
        trial_pct = ((env - bl_mean) / bl_mean) * 100

        ax_l.plot(times, trial_pct, color="red", linewidth=1)
        ax_l.axvline(0, color="green", linestyle="--", linewidth=1)
        ax_l.axhline(0, color="gray", linewidth=0.5)
        ax_l.set_ylabel("Beta", fontsize=8)
        ax_l.set_xlim(TMIN, TMAX)
        ax_l.grid(True, alpha=0.2)
        if idx == 0:
            ax_l.set_title("Beta amplitude (single trial)", fontsize=10)
        if idx == n_show - 1:
            ax_l.set_xlabel("Time from response (s)", fontsize=9)

        # Single trial TFR
        tfr_s = mne.time_frequency.tfr_morlet(
            epochs[idx:idx+1].copy().pick_channels(
                [CHANNEL], verbose=False),
            freqs=FREQS,
            n_cycles=N_CYCLES,
            return_itc=False,
            verbose=False
        )
        tfr_s.apply_baseline(baseline=BASELINE,
                             mode="logratio", verbose=False)

        im_s = plot_tfr(ax_r, tfr_s.data[0], FREQS, TMIN, TMAX,
                        vmin=-1.0, vmax=1.0)
        ax_r.set_xlim(TMIN, TMAX)
        if idx == 0:
            ax_r.set_title("Time-frequency (single trial)", fontsize=10)
            plt.colorbar(im_s, ax=ax_r, label="Log ratio")
        if idx == n_show - 1:
            ax_r.set_xlabel("Time from response (s)", fontsize=9)

    plt.tight_layout()
    fname2 = OUTPUT_FOLDER + f"session2_pmbr_{safe}_trials.png"
    plt.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.show(block=True)
    print(f"  Saved: session2_pmbr_{safe}_trials.png")


# Comparison: vCR OFF vs ON

if len(all_data) == 2:

    print("\n  Comparison plot: vCR OFF vs ON")

    labels = list(all_data.keys())
    colors = {"Finger Extension — vCR OFF": "black",
              "Finger Extension — vCR ON (20 Hz)": "red"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "PMBR — vCR OFF vs ON | Session 2 (MP)\n"
        f"Channel: {CHANNEL} | Movement offset at time 0",
        fontsize=12
    )

    # Panel 1: full timecourse overlay
    for label, d in all_data.items():
        axes[0].plot(d["times"], d["pct"],
                     color=colors.get(label, "blue"),
                     linewidth=2, label=label)
    axes[0].axvline(0, color="green", linestyle="--",
                    linewidth=1.5, label="Release")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_xlabel("Time from response (s)", fontsize=11)
    axes[0].set_ylabel("Beta (% change)", fontsize=11)
    axes[0].set_title("Full timecourse", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(TMIN, TMAX)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: TFR — vCR OFF
    label_off = labels[0]
    im0 = plot_tfr(axes[1], all_data[label_off]["tfr"].data[0],
                   FREQS, TMIN, TMAX, vmin=-0.5, vmax=0.5)
    axes[1].set_xlabel("Time from response (s)", fontsize=11)
    axes[1].set_title(f"TFR — vCR OFF\n(blue=ERD, red=ERS)", fontsize=11)
    plt.colorbar(im0, ax=axes[1], label="Log ratio")

    # Panel 3: TFR — vCR ON
    label_on = labels[1]
    im1 = plot_tfr(axes[2], all_data[label_on]["tfr"].data[0],
                   FREQS, TMIN, TMAX, vmin=-0.5, vmax=0.5)
    axes[2].set_xlabel("Time from response (s)", fontsize=11)
    axes[2].set_title(f"TFR — vCR ON\n(blue=ERD, red=ERS)", fontsize=11)
    plt.colorbar(im1, ax=axes[2], label="Log ratio")

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER + "session2_pmbr_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.show(block=True)
    print("  Saved: session2_pmbr_comparison.png")


print("\n" + "="*65)
print("   DONE")
print("="*65)
print("""
  What to look for:
  → ERD (blue):  beta drops BEFORE movement (negative time)
  → PMBR (red):  beta rebounds AFTER movement (positive time)
  → Compare OFF vs ON: does vCR change the rebound?
""")