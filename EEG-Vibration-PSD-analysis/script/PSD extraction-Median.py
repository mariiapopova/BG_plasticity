#%%
#Bahar-edit
#PSD Analysis - Using MEDIAN method

import mne
import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_welch
from pathlib import Path

DATA_PATH = r"...vhdr"
OUTPUT_FOLDER = r"..Folder"
OUTPUT_NAME = "....xlsx"

# PSD settings
N_FFT = 4096
FMIN = 0.5
FMAX = 40.0

# Frequency bands
BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 40.0),
}

# STEP 1: LOAD DATA

raw = mne.io.read_raw_brainvision(DATA_PATH, preload=True, verbose=False)
print(f"Original channels: {len(raw.ch_names)}")

# Keep only EEG
raw.pick(picks="eeg")
print(f"EEG channels kept: {len(raw.ch_names)}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]/60:.1f} minutes")

# STEP 2: AMPLITUDE CHECKS
data = raw.get_data()

# RMS (Root Mean Square)
rms_uV = 1e6 * np.sqrt(np.median(data**2, axis=1))
print(f"Median RMS: {np.median(rms_uV):.2f} µV")

# Peak-to-peak (1st to 99th percentile)
p2p_uV = 1e6 * (np.percentile(data, 99, axis=1) - np.percentile(data, 1, axis=1))
print(f"Median peak-to-peak: {np.median(p2p_uV):.2f} µV")
print(f"Max peak-to-peak: {np.max(p2p_uV):.2f} µV")

# STEP 3: OUTLIER CHANNELS

# MAD (Median Absolute Deviation) method
p2p = np.percentile(data, 99, axis=1) - np.percentile(data, 1, axis=1)
median = np.median(p2p)
mad = np.median(np.abs(p2p - median)) + 1e-20
z_score = (p2p - median) / mad

# Find outliers (z > 8)
outlier_idx = np.where(z_score > 8)[0]
outlier_names = [raw.ch_names[i] for i in outlier_idx]

if outlier_names:
    print(f" Found {len(outlier_names)} outlier channels (MAD z>8):")
    for ch in outlier_names:
        idx = raw.ch_names.index(ch)
        print(f"  {ch}: p2p={p2p_uV[idx]:.1f} µV, z-score={z_score[idx]:.1f}")
else:
    print("No outlier channels detected")

# STEP 4: COMPUTE PSD (MEDIAN METHOD - KEY DIFFERENCE!)


def compute_median_psd(raw, fmin=0.5, fmax=40.0, n_fft=4096):
    """
    Compute PSD using median across channels (like reference code)
    """
    data = raw.get_data()

    # Use MNE's psd_array_welch with MEDIAN averaging
    psd, freqs = psd_array_welch(
        data,
        sfreq=raw.info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        average="median"
    )

    return psd, freqs

# Compute PSD for all channels
psd_data, freqs = compute_median_psd(raw, fmin=FMIN, fmax=FMAX, n_fft=N_FFT)

print(f"✓ PSD computed with Welch method (n_fft={N_FFT})")
print(f"  Frequency points: {len(freqs)}")
print(f"  Frequency resolution: {freqs[1]-freqs[0]:.4f} Hz")
print(f"  PSD range: {psd_data.min():.2e} to {psd_data.max():.2e} V²/Hz")

# Also compute the median PSD across all channels (for visualization)
psd_median_across_channels = np.median(psd_data, axis=0)
print(f"  Median PSD across channels: {psd_median_across_channels.mean():.2e} V²/Hz")


# STEP 5: EXTRACT BAND POWERS (per channel)

print("\n" + "="*70)
print("Extracting band powers...")
print("="*70)

band_power_raw = {}
band_power_log = {}
band_power_rel = {}

# First compute total power per channel
total_power_per_ch = np.zeros(len(raw.ch_names))
for i in range(len(raw.ch_names)):
    total_power_per_ch[i] = np.trapz(psd_data[i], freqs) * 1e12  # V² → µV²

# Now extract band powers
for band_name, (fmin, fmax) in BANDS.items():
    idx = (freqs >= fmin) & (freqs <= fmax)

    if idx.sum() == 0:
        print(f"  No frequencies for {band_name}")
        continue

    # Integration for each channel
    power_uV2 = np.zeros(len(raw.ch_names))
    for i in range(len(raw.ch_names)):
        power_uV2[i] = np.trapz(psd_data[i, idx], freqs[idx]) * 1e12

    # Log transform
    log_power = 10 * np.log10(power_uV2 + 1e-20)

    # Relative power (%)
    rel_power = (power_uV2 / total_power_per_ch) * 100

    # Store
    band_power_raw[band_name] = power_uV2
    band_power_log[band_name] = log_power
    band_power_rel[band_name] = rel_power

    print(f"{band_name:8s} ({fmin:4.1f}-{fmax:4.1f} Hz): "
          f"median={np.median(power_uV2):10.2f} µV², "
          f"mean={power_uV2.mean():10.2f} µV²")

# STEP 6: RMS PER CHANNEL
rms_dict = dict(zip(raw.ch_names, rms_uV))
top_rms = sorted(rms_dict.items(), key=lambda x: x[1], reverse=True)[:5]

for ch, rms in top_rms:
    print(f"  {ch:8s}: {rms:8.2f} µV")

# STEP 7: BAND POWER RATIOS
# Delta/Alpha ratio
delta_alpha_ratio = np.median(band_power_raw["Delta"]) / np.median(band_power_raw["Alpha"])
print(f"Delta/Alpha ratio (median): {delta_alpha_ratio:.2f}")

if delta_alpha_ratio > 2.0:
    print("High Delta/Alpha ratio - possible slow drift or artifacts")
else:
    print(" Normal Delta/Alpha ratio")

# STEP 8: SAVE TO EXCEL

df_raw = pd.DataFrame(band_power_raw, index=raw.ch_names)
df_log = pd.DataFrame(band_power_log, index=raw.ch_names)
df_rel = pd.DataFrame(band_power_rel, index=raw.ch_names)

df_raw.index.name = 'Channel'
df_log.index.name = 'Channel'
df_rel.index.name = 'Channel'

output_path = str(Path(OUTPUT_FOLDER) / OUTPUT_NAME)
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_raw.to_excel(writer, sheet_name='Raw_Power_uV2')
    df_log.to_excel(writer, sheet_name='Log_Power_dB')
    df_rel.to_excel(writer, sheet_name='Relative_Power_percent')

print(f"✓ Saved to: {output_path}")

# STEP 9: SUMMARY
print(f"\nFile: {Path(DATA_PATH).name}")
print(f"Channels: {len(raw.ch_names)}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f} sec ({raw.times[-1]/60:.1f} min)")
print(f"\nMedian RMS: {np.median(rms_uV):.2f} µV")
print(f"Median peak-to-peak: {np.median(p2p_uV):.2f} µV")
print(f"\nOutlier channels: {len(outlier_names)}")
if outlier_names:
    print(f"  {', '.join(outlier_names)}")
print(f"\nDelta/Alpha ratio: {delta_alpha_ratio:.2f}")

# which channels might be problematic
print("\n DIAGNOSTIC HINTS:")

if delta_alpha_ratio > 2.0:
    print("  → High Delta/Alpha suggests slow drift or movement artifacts")
    print("  → Consider applying 1 Hz high-pass filter")

if outlier_names:
    print(f"  → {len(outlier_names)} channels have extremely high amplitude")
    print(f"  → Consider removing: {', '.join(outlier_names)}")

if "M2" in raw.ch_names or "M1" in raw.ch_names:
    print("  → M1/M2 (mastoid) channels are present")
    print("  → These should typically be removed before analysis")

# %%
