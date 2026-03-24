import numpy as np
import scipy.signal as sig
from scipy.ndimage import gaussian_filter1d


# 1. mean frequency

# per-neuron metrics
def detect_spikes_from_voltage(v, dt_ms, spike_height=-20.0, refractory_ms=2.0):
    refractory_samples = max(1, int(round(refractory_ms / dt_ms)))
    peaks, props = sig.find_peaks(v, height=spike_height, distance=refractory_samples)
    spike_times_ms = peaks * dt_ms
    return peaks, spike_times_ms

def compute_firing_rate_from_voltage(v, dt_ms, spike_height=-20.0, refractory_ms=2.0):
    peaks, spike_times_ms = detect_spikes_from_voltage(
        v=v,
        dt_ms=dt_ms,
        spike_height=spike_height,
        refractory_ms=refractory_ms,
    )

    duration_s = (len(v) * dt_ms) / 1000.0
    firing_rate_hz = len(peaks) / duration_s if duration_s > 0 else np.nan
    return firing_rate_hz


# population metrics (one nucleus)
def compute_population_metrics(v_pop, dt_ms, spike_height=-20.0, refractory_ms=2.0):
    n_timepoints, n_neurons = v_pop.shape
    rates_hz = np.zeros(n_neurons, dtype=float)

    for i in range(n_neurons):
        v = v_pop[:, i]
        rates_hz[i] = compute_firing_rate_from_voltage(
            v=v,
            dt_ms=dt_ms,
            spike_height=spike_height,
            refractory_ms=refractory_ms,
        )

    return {
        "rates_hz": rates_hz,
        "mean_rate_hz": float(np.mean(rates_hz)) if len(rates_hz) else np.nan,
        "median_rate_hz": float(np.median(rates_hz)) if len(rates_hz) else np.nan,
        "std_rate_hz": float(np.std(rates_hz)) if len(rates_hz) else np.nan,
    }

# mulitple nuclei
def compute_metrics_all_populations(
    population_voltages,
    dt_ms,
    spike_height_map=None,
    refractory_ms=2.0,
):
    results = {}

    for pop_name, v_pop in population_voltages.items():
        spike_height = 0.0
        if spike_height_map is not None and pop_name in spike_height_map:
            spike_height = spike_height_map[pop_name]

        results[pop_name] = compute_population_metrics(
            v_pop=v_pop,
            dt_ms=dt_ms,
            spike_height=spike_height,
            refractory_ms=refractory_ms,
        )

    return results


# 2. ISI CSV irregularity analysis
# for one neuron
def compute_isi_metrics_from_voltage(v, dt_ms, spike_height=0.0, refractory_ms=2.0):

    peaks, spike_times_ms = detect_spikes_from_voltage(
        v=v,
        dt_ms=dt_ms,
        spike_height=spike_height,
        refractory_ms=refractory_ms,
    )
    n_spikes = len(spike_times_ms)
    isis_ms = np.diff(spike_times_ms)
    mean_isi_ms = np.mean(isis_ms)
    std_isi_ms = np.std(isis_ms)
    cv_isi = std_isi_ms / mean_isi_ms if mean_isi_ms > 0 else np.nan

    return {
        "isis_ms": isis_ms,
        "mean_isi_ms": float(mean_isi_ms),
        "std_isi_ms": float(std_isi_ms),
        "cv_isi": float(cv_isi),
        "spike_times_ms": spike_times_ms,
        "n_spikes": n_spikes,
        "n_isis": len(isis_ms),

    }

# population irregularity metrics
def compute_population_irregularity_metrics(
    v_pop,
    dt_ms,
    spike_height=0.0,
    refractory_ms=2.0,
    min_spikes_for_cv=2,
):

    n_timepoints, n_neurons = v_pop.shape

    cv_isi_values = np.full(n_neurons, np.nan, dtype=float)
    mean_isi_values = np.full(n_neurons, np.nan, dtype=float)
    std_isi_values = np.full(n_neurons, np.nan, dtype=float)
    n_spikes_values = np.zeros(n_neurons, dtype=int)
    spike_times_all = []
    isis_all = []

    for i in range(n_neurons):
        v = v_pop[:, i]

        out = compute_isi_metrics_from_voltage(
            v=v,
            dt_ms=dt_ms,
            spike_height=spike_height,
            refractory_ms=refractory_ms,
        )

        n_spikes_values[i] = out["n_spikes"]
        spike_times_all.append(out["spike_times_ms"])
        isis_all.append(out["isis_ms"])
        mean_isi_values[i] = out["mean_isi_ms"]
        std_isi_values[i] = out["std_isi_ms"]

        if out["n_spikes"] >= min_spikes_for_cv:
            cv_isi_values[i] = out["cv_isi"]

    valid = np.isfinite(cv_isi_values)

    return {
        "cv_isi": cv_isi_values,
        "mean_isi_ms": mean_isi_values,
        "std_isi_ms": std_isi_values,
        "n_spikes": n_spikes_values,
        "spike_times_ms": spike_times_all,
        "isis_ms": isis_all,
        "n_neurons": n_neurons,
        "n_valid_cv": int(np.sum(valid)),
        "mean_cv_isi": float(np.nanmean(cv_isi_values)) if np.any(valid) else np.nan,
        "median_cv_isi": float(np.nanmedian(cv_isi_values)) if np.any(valid) else np.nan,
        "std_cv_isi": float(np.nanstd(cv_isi_values)) if np.any(valid) else np.nan,
    }

# multiple populations
def compute_irregularity_all_populations(
    population_voltages,
    dt_ms,
    spike_height_map=None,
    refractory_ms=2.0,
    min_spikes_for_cv=2,
):
    results = {}

    for pop_name, v_pop in population_voltages.items():
        spike_height = 0.0
        if spike_height_map is not None and pop_name in spike_height_map:
            spike_height = spike_height_map[pop_name]

        results[pop_name] = compute_population_irregularity_metrics(
            v_pop=v_pop,
            dt_ms=dt_ms,
            spike_height=spike_height,
            refractory_ms=refractory_ms,
            min_spikes_for_cv=min_spikes_for_cv,
        )

    return results


# 3. PSD 
# detecting spikes
def extract_population_spike_times(V_pop, dt_ms, spike_height=0.0, refractory_ms=2.0): 
    V_pop = np.asarray(V_pop) # V_pop shape: (n_timepoints, n_neurons)
    spike_times_list = []

    for i in range(V_pop.shape[1]):   # loop over neurons = columns
        _, spike_times_ms = detect_spikes_from_voltage(
            V_pop[:, i],
            dt_ms=dt_ms,
            spike_height=spike_height,
            refractory_ms=refractory_ms
        )
        spike_times_list.append(spike_times_ms)

    return spike_times_list

# bining spikes in time windows
def population_rate_from_spike_times(spike_times_list, tmax_ms, bin_ms=1.0, n_neurons=None):
    if n_neurons is None:
        n_neurons = len(spike_times_list)

    bins = np.arange(0.0, tmax_ms + bin_ms, bin_ms)

    all_spikes = np.concatenate([st for st in spike_times_list if len(st) > 0]) \
        if any(len(st) > 0 for st in spike_times_list) else np.array([])

    counts, edges = np.histogram(all_spikes, bins=bins)
    bin_s = bin_ms / 1000.0
    pop_rate_hz = counts / (n_neurons * bin_s)

    t_bins = edges[:-1]
    return t_bins, pop_rate_hz

# smoothing
def smooth_rate(rate_hz, sigma_ms=2.0, bin_ms=1.0):
    sigma_bins = sigma_ms / bin_ms
    return gaussian_filter1d(rate_hz, sigma=sigma_bins)

# Welch PSD
def welch_psd(signal, dt_ms, nperseg=64, noverlap=32): # npersig: raise later for larger time window
    fs = 1000.0 / dt_ms  # sampling frequency in Hz
    x = sig.detrend(signal)
    nperseg = min(nperseg, len(x))
    noverlap = min(noverlap, nperseg // 2)
    freqs, psd = sig.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return freqs, psd

