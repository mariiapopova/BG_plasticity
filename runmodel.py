#%%
# choose device to do calculations on
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import jax
from jax import jit
from jax import random
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from checkfreq import *
from revised_jaxmodel import *
import matplotlib.mlab as mlab

jax.config.update("jax_enable_x64", True)
print(jax.devices())


# shared validation setup
population_order = [
    "GPe", "STN", "GPi", "TH",
    "PYR M1", "FSI M1", "PYR S1", "FSI S1",
    "dStr", "iStr", "SNc"
]

spike_height_map = {
    "GPe": -20.0,
    "GPi": -20.0,
    "TH": -20.0,
    "STN": -20.0,
    "PYR M1": 0.0,
    "FSI M1": -20.0,
    "PYR S1": 0.0,
    "FSI S1": -20.0,
    "dStr": -20.0,
    "iStr": -20.0,
    "SNc": -20.0,
}

def build_population_voltages(res):
    return {
        "TH": res["V1_th"],
        "STN": res["V2_stn"],
        "GPe": res["V3_gpe"],
        "GPi": res["V4_gpi"],
        "dStr": res["V5_dstr"],
        "iStr": res["V5_istr"],
        "PYR M1": res["V6_ctx"],
        "FSI M1": res["V7_ctx"],
        "PYR S1": res["V8_ctx"],
        "FSI S1": res["V9_ctx"],
        "SNc": res["V10_snc"],
    }


def run_condition(
    pd_flag,
    tmax,
    chunk_size,
    dt0,
    dt_save,
    stim=0,
    freq=130.0,
    sw=0,
    n=4,
    seed=0,
):
    key = jax.random.PRNGKey(seed)

    cfg = make_condition_config(
        pd=pd_flag,
        stim=stim,
        freq=freq,
        tmax=tmax,
        dt=dt0,
        sw=sw,
    )

    params = make_params(
        cfg=cfg,
        key0=key,
        n=n,
    )

    y0 = make_initial_state(
        params=params,
        key=key,
    )

    res = simulate_last_chunk_euler(
        y0=y0,
        params=params,
        tmax=tmax,
        dt=dt0,
        dt_save=dt_save,
        chunk_length=chunk_size,
    )

    ts = res["ts"]
    dt_ms = float(ts[1] - ts[0]) if len(ts) > 1 else float(dt_save)
    tmax_window_ms = float(len(ts) * dt_ms)

    return {
        "pd_flag": pd_flag,
        "cfg": cfg,
        "params": params,
        "y0": y0,
        "res": res,
        "ts": ts,
        "dt_ms": dt_ms,
        "tmax_window_ms": tmax_window_ms,
        "population_voltages": build_population_voltages(res),
    }



#%% run both healthy and PD
healthy_run = run_condition(
    pd_flag=0,
    tmax=1000,
    chunk_size=100,
    dt0=0.1,
    dt_save=1.0,
    stim=0,
    freq=130.0,
    sw=0,
    n=4,
    seed=0,
)

pd_run = run_condition(
    pd_flag=1,
    tmax=1000,
    chunk_size=100,
    dt0=0.1,
    dt_save=1.0,
    stim=0,
    freq=130.0,
    sw=0,
    n=4,
    seed=0,
)


#%%

def validate_condition(run_dict):
    population_voltages = run_dict["population_voltages"]
    dt_ms = run_dict["dt_ms"]
    tmax_window_ms = run_dict["tmax_window_ms"]

    rate_results = compute_metrics_all_populations(
        population_voltages=population_voltages,
        dt_ms=dt_ms,
        spike_height_map=spike_height_map,
        refractory_ms=2.0,
    )

    irregularity_results = compute_irregularity_all_populations(
        population_voltages=population_voltages,
        dt_ms=dt_ms,
        spike_height_map=spike_height_map,
        refractory_ms=2.0,
        min_spikes_for_cv=2,
    )

    mean_rates = {
        pop: rate_results[pop]["mean_rate_hz"]
        for pop in rate_results
    }

    V4 = population_voltages["GPi"]
    n_gpi = V4.shape[1]

    gpi_spike_times = extract_population_spike_times(
        V4,
        dt_ms=dt_ms,
        spike_height=spike_height_map["GPi"],
        refractory_ms=2.0,
    )

    t_rate, gpi_rate = population_rate_from_spike_times(
        gpi_spike_times,
        tmax_ms=tmax_window_ms,
        bin_ms=dt_ms,
        n_neurons=n_gpi,
    )

    gpi_rate_smooth = smooth_rate(
        gpi_rate,
        sigma_ms=2.0,
        bin_ms=dt_ms,
    )

    freqs, psd = welch_psd(
        gpi_rate_smooth,
        dt_ms=dt_ms,
        nperseg=min(512, len(gpi_rate_smooth)),
        noverlap=min(256, max(0, len(gpi_rate_smooth) // 2)),
    )

    out = dict(run_dict)
    out["rate_results"] = rate_results
    out["irregularity_results"] = irregularity_results
    out["mean_rates"] = mean_rates
    out["gpi_spike_times"] = gpi_spike_times
    out["gpi_rate_t"] = t_rate
    out["gpi_rate"] = gpi_rate
    out["gpi_rate_smooth"] = gpi_rate_smooth
    out["gpi_psd_freqs"] = freqs
    out["gpi_psd"] = psd

    return out



healthy_val = validate_condition(healthy_run)
pd_val = validate_condition(pd_run)

#%% validate model
# print mean rates
def print_mean_rates_comparison(*validated_runs):
    for run in validated_runs:
        for pop in population_order:
            if pop in run["mean_rates"]:
                print(f"{pop}: {run['mean_rates'][pop]:.3f} Hz")


print_mean_rates_comparison(healthy_val, pd_val)

#compare mean rates side by side
def plot_mean_rate_comparison(run_a, run_b, population_order=population_order):
    labels = [pop for pop in population_order if pop in run_a["mean_rates"] and pop in run_b["mean_rates"]]
    vals_a = [run_a["mean_rates"][pop] for pop in labels]
    vals_b = [run_b["mean_rates"][pop] for pop in labels]

    y = np.arange(len(labels))
    h = 0.35

    plt.figure(figsize=(8, 6))
    plt.barh(y - h/2, vals_a, height=h, label="healthy")
    plt.barh(y + h/2, vals_b, height=h, label="pd")
    plt.yticks(y, labels)
    plt.xlabel("Mean firing rate (Hz)")
    plt.ylabel("Population")
    plt.title("Healthy vs PD mean firing rates")
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_mean_rate_comparison(healthy_val, pd_val)

#boxplots: rates
def plot_population_boxplots(results, title="", population_order=None):
    if population_order is None:
        population_order = list(results.keys())

    labels = []
    data = []

    for pop in population_order:
        if pop not in results:
            continue
        vals = np.asarray(results[pop]["rates_hz"])
        vals = vals[np.isfinite(vals)]
        labels.append(pop)
        data.append(vals)

    plt.figure(figsize=(8, 6))
    plt.boxplot(
        data,
        vert=False,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="*", markeredgecolor="black", markersize=7),
        flierprops=dict(marker="+", markeredgecolor="black", markersize=6),
    )
    plt.xlabel("Rate (Hz)")
    plt.ylabel("Population")
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_population_boxplots(
    healthy_val["rate_results"],
    title="Healthy: firing rates",
    population_order=population_order,
)

plot_population_boxplots(
    pd_val["rate_results"],
    title="PD: firing rates",
    population_order=population_order,
)

# boxplots: CV of ISI
def plot_irregularity_boxplots(
    results,
    title="Irregularity by population (CV_ISI)",
    population_order=None,
    xlim=None,
):
    if population_order is None:
        population_order = list(results.keys())

    labels = []
    data = []

    for pop in population_order:
        if pop not in results:
            continue
        vals = np.asarray(results[pop]["cv_isi"])
        vals = vals[np.isfinite(vals)]
        labels.append(pop)
        data.append(vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        data,
        vert=False,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="*", markeredgecolor="black", markersize=7),
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.6),
        capprops=dict(linewidth=1.6),
        boxprops=dict(linewidth=1.6),
        flierprops=dict(marker="+", markeredgecolor="black", markersize=6),
    )

    ax.set_title(title)
    ax.set_xlabel("CV of ISI")
    ax.set_ylabel("Population")
    ax.grid(axis="x", alpha=0.3)

    if xlim is not None:
        ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


plot_irregularity_boxplots(
    healthy_val["irregularity_results"],
    title="Healthy: CV of ISI",
    population_order=population_order,
    xlim=(0, 2.5),
)

plot_irregularity_boxplots(
    pd_val["irregularity_results"],
    title="PD: CV of ISI",
    population_order=population_order,
    xlim=(0, 2.5),
)


# compare GPi PSD
def plot_gpi_psd_comparison(run_a, run_b, xlim=(0, 50)):
    plt.figure(figsize=(6, 4))
    plt.plot(run_a["gpi_psd_freqs"], run_a["gpi_psd"], label="healthy")
    plt.plot(run_b["gpi_psd_freqs"], run_b["gpi_psd"], label="pd")
    plt.xlim(*xlim)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("GPi population-rate PSD")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_gpi_psd_comparison(healthy_val, pd_val)

#compare voltage traces
def plot_trace_comparison(run_a, run_b, pop_name, neuron_idx=0):
    plt.figure(figsize=(8, 3))
    plt.plot(run_a["ts"], run_a["population_voltages"][pop_name][:, neuron_idx], label="healthy")
    plt.plot(run_b["ts"], run_b["population_voltages"][pop_name][:, neuron_idx], label="pd")
    plt.xlabel("t (ms)")
    plt.ylabel("V (mV)")
    plt.title(f"{pop_name}, neuron {neuron_idx}")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_trace_comparison(healthy_val, pd_val, "STN", neuron_idx=0)
plot_trace_comparison(healthy_val, pd_val, "GPe", neuron_idx=0)
plot_trace_comparison(healthy_val, pd_val, "GPi", neuron_idx=0)
plot_trace_comparison(healthy_val, pd_val, "TH", neuron_idx=0)

# %%
