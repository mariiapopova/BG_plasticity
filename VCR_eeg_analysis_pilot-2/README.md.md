# vCR-EEG Analysis Pipeline — Session 2 

**EEG analysis of vibrotactile Coordinated Reset (vCR) effects on cortical beta oscillations**

> Pilot Session 2 | Subject: MP | Date: 14.09.2026
> Institute for Clinical Neuroscience (ICNS), UKE Hamburg

## Overview

This folder contains the analysis pipeline for Session 2 of the healthy vCR-EEG pilot study. Session 2 was recorded from subject MP and includes 16 runs across resting, motor, and cognitive conditions, with vCR stimulation at amplitudes of 20, 40, 60, 80, and 100 Hz carrier frequency.

Compared to Session 1 (FS), Session 2 shows substantially better signal quality due to improved impedance control during recording.

---

## Run the scripts in order

```
python session2_1_check_signal_quality.py
python session2_2_batch_inspect_ica.py
python session2_3_compare_vcr_on_off.py
python session2_4_vcr_burst_analysis.py
```

---

## Script 1 — `session2_1_check_signal_quality.py`

### What it does
Checks signal quality for all 16 raw recordings before any preprocessing. Reports which channels are good, flat, or noisy in each condition and identifies channels that are consistently bad across multiple files.

### How it works
```
For each of the 16 .vhdr files:
  → Load raw EEG
  → Compute std per channel (in µV)
  → Flag flat channels  (std < 1 µV  → probably disconnected)
  → Flag noisy channels (std > 40 µV → high impedance or artifact)
  → Report bad channel %
```

### Outputs
- Printed summary per file (good / flat / noisy channel counts)
- `session2_quality_overview.png` — bad channel % per recording (bar chart)
- `session2_quality_heatmap.png` — channel quality across all files (heatmap)

### Session 2 results
| Metric | Result |
|--------|--------|
| Total recordings | 16 |
| Passed quality threshold (<20% bad) | 16 / 16 |
| Worst recording | Countdown OFF — 3% bad (Fp1, Fp2) |
| Best recordings | 14 / 16 files — 0% bad channels |
| Consistently bad channels | None (Fp1/Fp2 only in 1–2 files) |

Session 2 signal quality is substantially better than Session 1 (FS), where Fp1/Fp2/AF7 were noisy in 8–11 out of 15 files. The improvement is attributed to better impedance control during the recording session.

---

## Script 2 — `session2_2_batch_inspect_ica.py`

### What it does
Applies the full preprocessing pipeline to all 16 files and allows manual inspection of ICA components for each one. Saves cleaned files as `_clean_manual.fif` in the `preprocessed/` subfolder.

### Preprocessing steps applied
```
1. Load raw file — EEG channels only (drops EMG, accelerometer)
2. Bandpass filter   — 1 to 40 Hz
3. Notch filter      — 50 Hz (power line noise)
4. Bad channel interpolation — spherical spline (if any bad channels found)
5. Average re-reference
6. ICA — 20 components, FastICA, random_state=42
7. Auto-detect eye artifact components using Fp1/Fp2
8. Show component topographies and time courses
9. User selects components to remove
10. Show before/after overlay
11. Save as _clean_manual.fif
```

### Key feature — resume support
The script automatically skips files that already have a `_clean_manual.fif` saved. If you stop mid-way, just run again and it continues from where it left off.

### ICA component guide
```
Remove:
  → Large blob at front (Fp1/Fp2 area)  = eye blink artifact
  → Left-right asymmetry at front        = horizontal eye movement
  → High-frequency spiky pattern at edges = muscle artifact

Keep:
  → Smooth, symmetric patterns over motor cortex
  → Alpha pattern at the back (occipital)
  → Central sensorimotor dipolar patterns
```

### Outputs
- `preprocessed/*_clean_manual.fif` — one cleaned file per recording
- Progress summary printed at the end

### Note on bad channels for Session 2
Because Session 2 has no consistently bad channels, `BAD_CHANNELS` is set to `[]`. The ICA step handles any residual ocular artifacts. If you want to interpolate Fp1/Fp2 in the few files where they were noisy, set:
```python
BAD_CHANNELS = ["Fp1", "Fp2"]
```

---

## Script 3 — `session2_3_compare_vcr_on_off.py`

### What it does
Compares beta power between matched vCR ON and OFF file pairs using the preprocessed cleaned data. This gives a whole-file comparison of how vCR stimulation affects average beta power across conditions.

### Analysis approach
```
For each matched pair (vCR OFF file vs vCR ON file):
  → Load both preprocessed .fif files
  → Cut into 2-second epochs (Welch PSD)
  → Compute mean power in low beta (13-20 Hz) and high beta (21-30 Hz)
  → Compare OFF vs ON: report absolute power and % change
  → ROI: C3, C4, CP3, CP4, Cz
```

### Matched pairs
| Condition | vCR OFF file | vCR ON file |
|-----------|-------------|------------|
| Resting — Eyes Open | restingeyesopen1min | 20amplitudevcreyesopen1min |
| Resting — Eyes Closed | restingeyesclosed | 20amplitudevcreyesclosed |
| Hold & Let Go — Eyes Open | Second2minholdandletgoeyesopen | Second2minholdandletgoeyesopenVCR20Amp |
| Hold & Let Go — Eyes Closed | Second2minholdandletgoeyesclosed | Second2minholdandletgoeyesclosedVCR20Amp |
| Finger Extension | Secondindexfingerextensioneyesopen | SecondindexfingerextensioneyesopenVCR20Amp |
| Countdown | countfroma100back7steps1mineyesopen | countfrom100back7steps1mineyesopenwithVCR20Amp |

### Important note on analysis approach
vCR ON and OFF conditions were recorded as **separate files** in this pilot session. Beta power is therefore computed as the mean across the entire file duration and compared between matched pairs. This is a whole-file comparison, not a within-file time-locked analysis. For recordings where vCR ON/OFF occur within the same file, trigger-based segmentation (Script 4) should be used instead.

### Outputs
- Printed table: power values and % change per channel per condition
- `session2_vcr_effect_Low_beta.png` — heatmap, low beta change per channel
- `session2_vcr_effect_High_beta.png` — heatmap, high beta change per channel
- `session2_vcr_effect_summary.png` — bar chart, ROI average per condition

---

## Script 4 — `session2_4_vcr_burst_analysis.py`

### What it does
A more precise, trigger-based analysis that cuts epochs time-locked to each individual S1 trigger (each vCR burst) and computes how beta power changes immediately before and after each burst. This reveals the dynamic brain response to vCR stimulation at the level of individual bursts.

### Why this is better than whole-file comparison
```
Script 3 (whole-file):            Script 4 (trigger-based):
──────────────────────────────────────────────────────────
Compares entire files             Locks to each S1 trigger
Includes non-stimulation periods  Only analyses ±2 sec around burst
Less precise timing               Millisecond-level precision
Shows cumulative effect           Shows immediate burst response
```

### vCR trigger pattern found in data
```
[S1]──0.80s──[S1]──0.80s──[S1]──2.10s──[S1]──...
      ↑              ↑              ↑
  burst 1        burst 2        cycle gap

Groups of 3 S1 = one vCR cycle (Stimulation Pattern 3:2)
Mean inter-trigger gap : ~1.24 sec
Min inter-trigger gap  : ~0.78 sec  ← determines safe baseline window
```

### Methodological decisions

**Power estimation: Hilbert transform**
Instantaneous beta power is computed using the Hilbert transform rather than simply squaring the filtered signal. Squaring produces noisy, high-variance power estimates. The Hilbert analytic signal gives a smooth, accurate power envelope.

```
Steps:
1. Bandpass filter to beta band (13-20 or 21-30 Hz)
2. Hilbert transform → analytic signal
3. Absolute value → amplitude envelope
4. Square → instantaneous power
```

**Baseline window: -0.5 to -0.05 sec**
The minimum inter-trigger gap is ~0.78 sec. A baseline of -1.0 to 0.0 sec (used in some studies) would overlap with the previous trigger. The safe baseline used here is -0.5 to -0.05 sec, which stays well within the gap and avoids contamination from adjacent bursts.

**Pre and post windows**
```
Pre-burst  : -0.4 to -0.05 sec  (matches baseline, avoids overlap)
Post-burst :  0.1 to  1.0  sec  (immediate post-burst response)
```

### Epoch structure
```
      baseline         trigger    post-burst window
   ───────────────────────────────────────────────
   -0.5s          -0.05s  0s              +2.0s
                           ↑
                      S1 (vCR burst)
```

### Outputs
- Printed summary: pre/post beta change (Δ%) per condition
- `session2_vcr_burst_timecourse.png` — beta timecourse per condition (all bands)
- `session2_vcr_burst_summary.png` — pre vs post bar chart per condition

---

## Folder structure

```
subj_2/
├── *.vhdr / *.eeg / *.vmrk          ← raw data (not in GitHub)
├── preprocessed/
│   ├── *_clean_manual.fif            ← cleaned files (not in GitHub)
│   ├── session2_quality_overview.png
│   ├── session2_quality_heatmap.png
│   ├── session2_vcr_effect_*.png
│   ├── session2_vcr_burst_timecourse.png
│   └── session2_vcr_burst_summary.png
├── session2_1_check_signal_quality.py
├── session2_2_batch_inspect_ica.py
├── session2_3_compare_vcr_on_off.py
└── session2_4_vcr_burst_analysis.py
```

---

## Session 2 recordings

| Run | Condition | vCR | Eyes | Duration |
|-----|-----------|-----|------|----------|
| 101 | Resting | OFF | Open | 62 sec |
| 102 | Resting | OFF | Closed | 60 sec |
| 103 | Stim Resting | ON 20 | Open | 61 sec |
| 104 | Stim Resting | ON 20 | Closed | 60 sec |
| 105 | Hold & Let Go | OFF | Open | 122 sec |
| 106 | Hold & Let Go | OFF | Closed | 122 sec |
| 107 | Hold & Let Go | ON 20 | Open | 122 sec |
| 108 | Hold & Let Go | ON 20 | Closed | 123 sec |
| 109 | Finger Extension | OFF | Open | 123 sec |
| 110 | Finger Extension | ON 20 | Open | 123 sec |
| 111 | Countdown (loud) | OFF | Open | 61 sec |
| 112 | Countdown (loud) | ON 20 | Open | 60 sec |
| 113 | Stim Resting | ON 40 | Open | 63 sec |
| 114 | Stim Resting | ON 60 | Open | 61 sec |
| 115 | Stim Resting | ON 80 | Open | 62 sec |
| 116 | Stim Resting | ON 100 | Open | 62 sec |

---

## Trigger legend (Session 2)

| Trigger | Meaning |
|---------|---------|
| `Stimulus/S  1` | vCR stimulation burst (one finger, part of 3:2 pattern) |
| `Comment/e` | Finger extension onset |
| `Comment/r` | Finger release |
| `Comment/h` | Grip hold onset |
| `Comment/l` | Grip letting go |
| `New Segment/` | Recording segment start (standard BrainVision marker) |

Note: release marker is `r` in Session 2 (different from Session 1 where it was `f`).

---

## Requirements

```bash
pip install mne numpy matplotlib scipy
```

Python 3.10+

---

## Authors

Bahar Khoshkroodian — EEG analysis
ICNS, UKE Hamburg / University of Bremen
September 2026
