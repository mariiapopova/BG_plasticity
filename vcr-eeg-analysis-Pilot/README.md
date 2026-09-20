# vCR-EEG Analysis Pipeline

**EEG analysis of vibrotactile Coordinated Reset (vCR) effects on cortical beta oscillations**

> Healthy pilot study — September 2026


## Repository Structure

```
vcr-eeg-analysis/
├── scripts/
│   ├── 1_check_signal_quality.py    # Signal quality check across all files
│   ├── 2_batch_inspect_ica.py       # Preprocessing + manual ICA inspection
│   └── 3_compare_vcr_on_off.py      # Beta power: vCR ON vs vCR OFF
├── results/
│   ├── quality_overview.png         # Bad channel % per recording
│   ├── quality_heatmap.png          # Channel quality heatmap
│   └── vcr_effect_summary_clean.png # Beta change vCR ON vs OFF
├── .gitignore
└── README.md
```

> **Raw data, preprocessed files, and participant information are stored on the UKE server and are NOT included in this repository.**

---

## How to Run

Run the three scripts in order:

```bash
# Step 1 — Check signal quality of all raw files
python scripts/1_check_signal_quality.py

# Step 2 — Preprocess and manually inspect ICA components
python scripts/2_batch_inspect_ica.py

# Step 3 — Compare beta power vCR ON vs OFF
python scripts/3_compare_vcr_on_off.py
```

---

## Pipeline Overview

### Script 1 — Signal Quality Check
Loops through all raw recordings and reports:
- Number of good, flat, and noisy channels per file
- Channels consistently bad across recordings
- Overview bar chart and heatmap

Thresholds: flat < 1 µV std, noisy > 40 µV std, warning if > 20% bad

### Script 2 — Preprocessing + ICA Inspection
Applies full preprocessing to each file:
1. Bandpass filter: 1–40 Hz
2. Notch filter: 50 Hz
3. Bad channel interpolation (spherical spline)
4. Average re-reference
5. ICA (20 components, FastICA)
6. Manual component inspection — plots shown, user selects which to remove
7. Save cleaned file as `_clean_manual.fif`

The script remembers which files are already done — you can stop and resume anytime.

### Script 3 — vCR ON vs OFF Comparison
Compares beta power between matched file pairs:

| vCR OFF file | vCR ON file |
|---|---|
| fatemeh1eyeopen...vhdr | fatemeh1vcreyesopen...vhdr |
| fatemeh1eyeclosed...vhdr | fatemeh1vcreyeclosed...vhdr |
| fingerextension...vhdr | vcrextension...vhdr |
| countingsilent30sec...vhdr | countingsilentwithvcr...vhdr |

> **Note on analysis approach:** vCR ON and OFF were recorded as separate files in this pilot session. Each file contains only one condition for its full duration. Beta power was therefore computed as the mean across the entire file and compared between matched pairs. For future recordings where vCR ON/OFF occur within the same file, trigger-based segmentation (using S1 markers) will be used instead.

---

## Pilot Recordings

15 conditions recorded from one healthy participant (08.09.2026):

| # | Condition | vCR | Duration |
|---|-----------|-----|----------|
| 01 | Baseline — Eyes Open | OFF | 69 sec |
| 02 | Baseline — Eyes Closed | OFF | 61 sec |
| 03 | Baseline — Eyes Open | ON | 66 sec |
| 04 | Baseline — Eyes Closed | ON | 66 sec |
| 05 | Finger Extension | OFF | 123 sec |
| 06 | Finger Extension | ON | 122 sec |
| 07 | Hold & Let Go | OFF | 123 sec |
| 08 | Hold & Let Go | ON | 123 sec |
| 09 | Counting Backwards (loud) | OFF | 90 sec |
| 10 | Counting Backwards (silent) | OFF | 31 sec |
| 11 | Counting Backwards (silent) | ON | 31 sec |
| 12 | vCR 40 Hz — Resting | ON | 60 sec |
| 13 | vCR 60 Hz — Resting | ON | 60 sec |
| 14 | vCR 80 Hz — Resting | ON | 63 sec |
| 15 | vCR 100 Hz — Resting | ON | 60 sec |

---

## Signal Quality Results

| Metric | Result |
|--------|--------|
| Total recordings | 15 |
| Passed quality threshold (<20% bad) | 15 / 15 |
| Worst recording | Counting Loud — 11% bad channels |
| Best recordings | Eyes Closed conditions — 0% bad |
| Consistently bad channels | Fp1 (11/15), Fp2 (10/15), AF7 (8/15) |
| Root cause | Impedance not controlled (>20 kOhm on several channels) |
| Resolution | Interpolation + ICA artifact removal |

> **For future recordings:** Ensure impedance < 5–10 kOhm per channel before starting. This will substantially improve raw signal quality.

---

## Beta Power Results (vCR ON vs OFF)

ROI channels: **C3, C4, CP3, CP4, Cz**
Method: Welch PSD on 2-second epochs, preprocessed data

| Condition | Low Beta (13–20 Hz) | High Beta (21–30 Hz) | Interpretation |
|-----------|--------------------|-----------------------|----------------|
| Eyes Open — Resting | ↑ up to +109% | ↑ up to +80% | vCR drives sensorimotor synchronization at rest |
| Eyes Closed — Resting | ↑ up to +29% | ↑ up to +27% | Same effect, smaller magnitude |
| Finger Extension | ↓ up to −66% | ↓ up to −74% | vCR enhances ERD during movement |
| Counting Silent | Mixed | Mixed | Cognitive load interacts with vCR |

**Key finding:** vCR shows a state-dependent effect on beta oscillations. During rest, it increases beta synchronization. During active movement, it strongly enhances beta desynchronization (ERD) — up to −74% high beta at C4. This is directly relevant to the PD hypothesis, where releasing pathological beta synchrony during movement is the therapeutic target.

---

## Recording Setup

- EEG: BrainProducts ActiChamp, 64-channel active cap
- Format: BrainVision (.vhdr / .eeg / .vmrk)
- Triggers: Triggerbox Plus via C# control script
- vCR device: Tass gloves (C# control script)
- Sampling rate: 1000 Hz

---

## Authors

Bahar Khoshkroodian — EEG analysis
ICNS, UKE Hamburg / University of Bremen
September 2026
