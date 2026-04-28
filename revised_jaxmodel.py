
#%%
# choose device to do calculations on
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import jax
from jax import jit
from jax import random
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from additional_functions import *
from checkfreq import *
from createdbs_jax import *
import matplotlib.mlab as mlab

#jax.config.update("jax_enable_x64", True)
print(jax.devices())

# dbs train creating function + look up function (temporary in this script)
def createdbs_jax(freq, tmax, dt, pulse_width=0.3, amplitude=300.0):

    n_steps = int(round(tmax / dt)) + 1 # do we need +1?
    t_idx = jnp.arange(n_steps)

    if freq <= 0.0:
        return jnp.zeros(n_steps)

    isi = 1000.0 / freq

    # discretization
    isi_steps = int(round(isi / dt))
    pw_steps = int(round(pulse_width / dt))

    # pulse condition
    waveform = jnp.where(
        (t_idx % isi_steps) < pw_steps,
        amplitude,
        0.0
    )

    return waveform

def dbs_current(t, cond):
    dt = cond["dt"]
    Idbs = cond["Idbs"]

    idx = jnp.clip(
        jnp.floor(t / dt).astype(jnp.int32),
        0,
        Idbs.shape[0] - 1
    )

    return Idbs[idx]


# cfg function
def make_condition_config(pd, stim, freq, tmax, dt):
    DA = 1.0 if pd == 0 else 0.1

    n_steps = int(round(tmax / dt))
    stim_ts = jnp.arange(n_steps) * dt

    if stim == 0:
        Idbs = jnp.zeros_like(n_steps)
    else:
       Idbs = createdbs_jax(freq, tmax, dt)

    return {
        "pd": jnp.asarray(pd),
        "stim": jnp.asarray(stim),
        "DA": jnp.asarray(DA),
        "freq": jnp.asarray(freq),
        "tmax": jnp.asarray(tmax),
        "dt": jnp.asarray(dt),
        "stim_ts": stim_ts,
        "Idbs": Idbs,
    }

#%% params function
def make_params(cfg, key0=None, n=4):
    if n != 4:
        raise ValueError("This version currently assumes n=4 because several connectivity matrices are hard-coded as 4x4.")

    if key0 is None:
        key0 = jax.random.PRNGKey(0)


    pd = cfg["pd"]
    stim = cfg["stim"]
    DA = cfg["DA"]
    freq = cfg["freq"]
    sw = cfg["sw"]
    tmax = cfg["tmax"]
    dt = cfg["dt"]
    stim_ts = cfg["stim_ts"]
    Idbs = jnp.asarray(cfg["Idbs"])

    # population sizes
    n_th = n
    n_stn = n
    n_gpe = n
    n_gpi = n
    n_dstr = n
    n_istr = n
    n_ctx_fsi = n * 2
    n_ctx_pyr = n * 20
    n_snc = n * 4

    # random keys for connectivity
    (
        key_pyr,
        key_pyr_th,
        key_th_pyr,
        key_pyr_stn,
        key_th_fsi,
    ) = jax.random.split(key0, 5)

    params = {
        "condition": {
            "pd": pd,
            "stim": stim,
            "DA": DA,
            "freq": freq,
            "sw": sw,
            "tmax": tmax,
            "dt": dt,
            "stim_ts": stim_ts,
            "Idbs": Idbs,
        },

        "sizes": {
            "th": n_th,
            "stn": n_stn,
            "gpe": n_gpe,
            "gpi": n_gpi,
            "dstr": n_dstr,
            "istr": n_istr,
            "ctx_fsi": n_ctx_fsi,
            "ctx_pyr": n_ctx_pyr,
            "snc": n_snc,
        },

        "neurons": {
            "th": {
                "n": n_th,
                "Cm": 1.0,
                "gl": 0.05,
                "El": -70.0,
                "gna": 3.0,
                "Ena": 50.0,
                "gk": 5.0,
                "Ek": -75.0,
                "gt": 5.0,
                "Et": 0.0,
                "Iapp": 1.7,
                "spike_threshold": -20.0,
            },

            "stn": {
                "n": n_stn,
                "Cm": 1.0,
                "gl": 2.25,
                "El": -60.0,
                "gna": 37.0,
                "Ena": 55.0,
                "gk": 45.0,
                "Ek": -80.0,
                "gt": 0.5,
                "Et": 0.0,
                "gca": 2.0,
                "Eca": 140.0,
                "gahp": 20.0,
                "k1": 15.0,
                "kca": 22.5,
                "Iapp": 35.0,
                "spike_threshold": -20.0,
            },

            "gpe": {
                "n": n_gpe,
                "Cm": 1.0,
                "gl": 0.1,
                "El": -65.0,
                "gna": 120.0,
                "Ena": 55.0,
                "gk": 30.0,
                "Ek": -80.0,
                "gt": 0.5,
                "Et": 0.0,
                "gca": 0.15,
                "Eca": 120.0,
                "gahp": 10.0,
                "k1": 10.0,
                "kca": 15.0,
                "Iapp": 15.0,
                "spike_threshold": -20.0,
            },

            "gpi": {
                "n": n_gpi,
                "Cm": 1.0,
                "gl": 0.1,
                "El": -65.0,
                "gna": 120.0,
                "Ena": 55.0,
                "gk": 30.0,
                "Ek": -80.0,
                "gt": 0.5,
                "Et": 0.0,
                "gca": 0.15,
                "Eca": 120.0,
                "gahp": 10.0,
                "k1": 10.0,
                "kca": 15.0,
                "Iapp": 15.0,
                "spike_threshold": -20.0,
            },

            "dstr": {
                "n": n_dstr,
                "Cm": 1.0,
                "gl": 0.1,
                "El": -67.0,
                "gna": 100.0,
                "Ena": 50.0,
                "gk": 80.0,
                "Ek": -100.0,
                "gm_base": 1.0,
                "gm_effective": (2.6 - 1.1 * pd) * 1.0,
                "Em": -100.0,
                "Iapp": 0.0,
                "spike_threshold": -20.0,
            },

            "istr": {
                "n": n_istr,
                "Cm": 1.0,
                "gl": 0.1,
                "El": -67.0,
                "gna": 100.0,
                "Ena": 50.0,
                "gk": 80.0,
                "Ek": -100.0,
                "gm_base": 1.0,
                "gm_effective": (2.6 - 1.1 * pd) * 1.0,
                "Em": -100.0,
                "Iapp": 0.0,
                "spike_threshold": -20.0,
            },

            "m1_pyr": {
                "n": n_ctx_pyr,
                "Cm": 1.0,
                "gl": 2.25,
                "El": -60.0,
                "gna": 37.0,
                "Ena": 55.0,
                "gk": 45.0,
                "Ek": -80.0,
                "gt": 0.5,
                "Et": 0.0,
                "gca": 2.0,
                "Eca": 140.0,
                "gahp": 20.0,
                "k1": 15.0,
                "kca": 22.5,
                "Iapp": 10.0,
                "spike_threshold": 0.0,
            },

            "m1_fsi": {
                "n": n_ctx_fsi,
                "Cm": 1.0,
                "gl": 2.25,
                "El": -60.0,
                "gna": 37.0,
                "Ena": 55.0,
                "gk": 45.0,
                "Ek": -80.0,
                "gt": 0.5,
                "Et": 0.0,
                "gca": 2.0,
                "Eca": 140.0,
                "gahp": 20.0,
                "k1": 15.0,
                "kca": 22.5,
                "Iapp": 10.0,
                "spike_threshold": -20.0,
            },

            "s1_pyr": {
                "n": n_ctx_pyr,
                "Cm": 1.0,
                "gl": 2.25,
                "El": -60.0,
                "gna": 37.0,
                "Ena": 55.0,
                "gk": 45.0,
                "Ek": -80.0,
                "gt": 0.5,
                "Et": 0.0,
                "gca": 2.0,
                "Eca": 140.0,
                "gahp": 20.0,
                "k1": 15.0,
                "kca": 22.5,
                "Iapp": 5.0,
                "spike_threshold": 0.0,
            },

            "s1_fsi": {
                "n": n_ctx_fsi,
                "Cm": 1.0,
                "gl": 2.25,
                "El": -60.0,
                "gna": 37.0,
                "Ena": 55.0,
                "gk": 45.0,
                "Ek": -80.0,
                "gt": 0.5,
                "Et": 0.0,
                "gca": 2.0,
                "Eca": 140.0,
                "gahp": 20.0,
                "k1": 15.0,
                "kca": 22.5,
                "Iapp": 5.0,
                "spike_threshold": -20.0,
            },

            "snc": {
                "n": n_snc,
                "Cm": 1.0,
                "gl": 0.01,
                "El": -50.0,
                "gna": 100.0,
                "Ena": 60.0,
                "Ek": -90.0,
                "gca": 0.15,
                "Eca": 120.0,
                "gsk": 0.25,
                "gkdr": 3.5,
                "Iapp": 10.0,
                "spike_threshold": -20.0,
            },
        },

        "synapses": {
            "gpi_to_th": {
                "A": 2.0,
                "B": 0.04,
                "theta": 20.0,
                "g": 0.08,
                "E": -85.0,
                "scale": 1.4,
                #"type": "first_order",
            },

            "gpe_to_stn": {
                "A": 2.0,
                "B": 0.04,
                "theta": 20.0,
                "g": 1.0,
                "E": -85.0,
                "scale": 0.5,
                #"type": "first_order",
            },

            "stn_to_gpe": {
                "A": 3.0,
                "B": 0.1,
                "theta": 30.0,
                "g": 0.3,
                "E": 0.0,
                "scale": 0.5,
                #"type": "first_order",
            },

            "gpe_to_gpe": {
                "A": 2.0,
                "B": 0.04,
                "theta": 20.0,
                "g": 1.0,
                "E": -85.0,
                "scale": 0.5,
                #"type": "first_order",
                "pd_multiplier": 1.0,
            },

            "gpe_to_gpi": {
                "A": 2.0,
                "B": 0.04,
                "theta": 20.0,
                "g": 1.0,
                "E": -85.0,
                "scale": 0.5,
                #"type": "first_order",
            },

            "stn_to_gpi": {
                "A": 3.0,
                "B": 0.1,
                "theta": 30.0,
                "g": 0.3,
                "E": 0.0,
                "scale": 0.5,
                #"type": "first_order",
            },

            "dstr_to_dstr": {
                "ggaba": 0.1,
                "E": -80.0,
                "tau_i": 13.0,
                "normalization": 3.0,
                #"type": "recurrent_gaba",
            },

            "istr_to_istr": {
                "ggaba": 0.1,
                "E": -85.0,
                "tau_i": 13.0,
                "normalization": 4.0,
                #"type": "recurrent_gaba",
            },

            "istr_to_gpe": {
                "g": 0.8,
                "E": -80.0,
                #"type": "alpha_second_order",
            },

            "dstr_to_gpi": {
                "g": 0.5,
                "E": -85.0,
                #"type": "alpha_second_order",
            },

            "pyr_to_pyr": {
                "A": 3.0,
                "B": 0.1,
                "theta": 30.0,
                "g": 0.3,
                "E": 0.0,
                "scale": 0.025,
                #"type": "first_order",
            },

            "pyr_to_fsi": {
                "g": 0.3,
                "E": 0.0,
                "scale": 0.025,
                #"type": "algebraic",
            },

            "fsi_to_fsi": {
                "g": 1.0,
                "E": -80.0,
                "scale": 0.25,
                #"type": "algebraic",
            },

            "fsi_to_pyr": {
                "g": 1.0,
                "E": -80.0,
                "scale": 0.25,
                #"type": "algebraic",
            },

            "pyr_to_dstr": {
                "g": (0.07 - 0.044 * pd),
                "E": 0.0,
                "scale": 0.2,
                #"type": "algebraic",
            },

            "pyr_to_istr": {
                "g": (0.07 - 0.044 * pd),
                "E": 0.0,
                "scale": 0.2,
                #"type": "algebraic",
            },

            "pyr_to_stn": {
                "g": 0.3,
                "E": 0.0,
                "scale": 0.2,
                #"type": "algebraic",
            },

            "pyr_to_th": {
                "g": 0.08,
                "E": 0.0,
                "scale": 0.25,
                #"type": "algebraic",
            },

            "th_to_pyr": {
                "g": 0.3,
                "E": 0.0,
                #"type": "algebraic",
            },

            "th_to_fsi": {
                "g": 0.3,
                "E": 0.0,
                #"type": "algebraic",
            },

            "alpha_shared": {
                "tau": 5.0,
                "gpeak": 0.43,
                "gpeak1": 0.3,
            },
        },

        "plasticity": {
            "gpe_to_stn_stdp": {
                "enabled": True,
                "tau_pre": 12.0,
                "tau_post": 27.5,
                "A_plus": 0.002,
                "A_minus": 0.002 * 1.1,
                "pre_threshold": -20.0,
                "post_threshold": -20.0,
                "W_init_min": 0.05,
                "W_init_max": 0.50,
                "W_min": 0.0,
                "W_max": 1.0,
            }
        },

        "connectivity": {
            "gpi_to_th": jnp.eye(n_th, n_gpi),

            "istr_to_gpe": jnp.eye(n_gpe, n_istr),
            "dstr_to_gpi": jnp.eye(n_gpi, n_dstr),

            "stn_to_gpe": jnp.array([
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
            ], dtype=jnp.float32),

            "stn_to_gpi": jnp.array([
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
            ], dtype=jnp.float32),

            "gpe_to_gpi": jnp.array([
                [0, 0, 1, 1],
                [1, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
            ], dtype=jnp.float32),

            "gpe_to_gpe": jnp.array([
                [0, 0, 1, 1],
                [1, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
            ], dtype=jnp.float32),

            "dstr_to_dstr": jnp.array([
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
            ], dtype=jnp.float32),

            "istr_to_istr": jnp.ones((n_istr, n_istr), dtype=jnp.float32),

            "fsi_to_fsi": (
                jnp.ones((n_ctx_fsi, n_ctx_fsi), dtype=jnp.float32)
                - jnp.eye(n_ctx_fsi, dtype=jnp.float32)
            ),

            "pyr_to_pyr": init_connectivity_divergence(
                key=key_pyr,
                n_post=n_ctx_pyr,
                n_pre=n_ctx_pyr,
                wsyn=9,
                divergence=6,
            ),

            "pyr_to_fsi": jnp.ones((n_ctx_fsi, n_ctx_pyr), dtype=jnp.float32),

            "fsi_to_pyr": jnp.ones((n_ctx_pyr, n_ctx_fsi), dtype=jnp.float32),

            "pyr_to_dstr": (
                jnp.ones((n_dstr, n_ctx_pyr), dtype=jnp.float32) * cD1(DA)
            ),

            "pyr_to_istr": (
                jnp.ones((n_istr, n_ctx_pyr), dtype=jnp.float32) * cD1(DA)
            ),

            "pyr_to_th": init_connectivity_convergence(
                key=key_pyr_th,
                n_post=n_th,
                n_pre=n_ctx_pyr,
                wsyn=1,
                convergence=4,
            ),

            "th_to_pyr": init_connectivity_divergence(
                key=key_th_pyr,
                n_post=n_ctx_pyr,
                n_pre=n_th,
                wsyn=5,
                divergence=6,
            ),

            "pyr_to_stn": w_matrix_random(
                key_pyr_stn,
                n=n_ctx_pyr,
                p=n_stn,
                k=5,
            ),

            "th_to_fsi": init_connectivity_divergence(
                key=key_th_fsi,
                n_post=n_ctx_fsi,
                n_pre=n_th,
                wsyn=1,
                divergence=2,
            ),
        },
    }

    return params


#%% main function
# defining the functions for ODEterm
def bg_rhs(t, y, args):
    params = args
    cond = params["condition"]

    # state variables

    # TH
    V1 = y["V1_th"]
    H1 = y["H1_th"]
    R1 = y["R1_th"]
    S1 = y["S1_th"]

    # STN
    V2 = y["V2_stn"]
    N2 = y["N2_stn"]
    H2 = y["H2_stn"]
    R2 = y["R2_stn"]
    C2 = y["C2_stn"]
    CA2 = y["CA2_stn"]
    S2 = y["S2_stn"]
    Z2 = y["Z2_stn"]

    # GPe
    V3 = y["V3_gpe"]
    N3 = y["N3_gpe"]
    H3 = y["H3_gpe"]
    R3 = y["R3_gpe"]
    CA3 = y["CA3_gpe"]
    S3 = y["S3_gpe"]

    # GPi
    V4 = y["V4_gpi"]
    N4 = y["N4_gpi"]
    H4 = y["H4_gpi"]
    R4 = y["R4_gpi"]
    CA4 = y["CA4_gpi"]
    S4 = y["S4_gpi"]
    Z4 = y["Z4_gpi"]

    # direct striatum
    V5d = y["V5_dstr"]
    m5d = y["m5_dstr"]
    h5d = y["h5_dstr"]
    n5d = y["n5_dstr"]
    p5d = y["p5_dstr"]
    S5d = y["S5_dstr"]
    S5d_2 = y["S52_dstr"]
    Z5d_2 = y["Z52_dstr"]

    # indirect striatum
    V5i = y["V5_istr"]
    m5i = y["m5_istr"]
    h5i = y["h5_istr"]
    n5i = y["n5_istr"]
    p5i = y["p5_istr"]
    S5i = y["S5_istr"]
    S5i_2 = y["S52_istr"]
    Z5i_2 = y["Z52_istr"]

    # M1 cortex PYR
    V6 = y["V6_ctx"]
    N6 = y["N6_ctx"]
    H6 = y["H6_ctx"]
    R6 = y["R6_ctx"]
    C6 = y["C6_ctx"]
    CA6 = y["CA6_ctx"]
    S6 = y["S6_ctx"]

    # M1 cortex FSI
    V7 = y["V7_ctx"]
    N7 = y["N7_ctx"]
    H7 = y["H7_ctx"]
    R7 = y["R7_ctx"]
    C7 = y["C7_ctx"]
    CA7 = y["CA7_ctx"]
    S7 = y["S7_ctx"]

    # S1 cortex PYR
    V8 = y["V8_ctx"]
    N8 = y["N8_ctx"]
    H8 = y["H8_ctx"]
    R8 = y["R8_ctx"]
    C8 = y["C8_ctx"]
    CA8 = y["CA8_ctx"]

    # S1 cortex FSI
    V9 = y["V9_ctx"]
    N9 = y["N9_ctx"]
    H9 = y["H9_ctx"]
    R9 = y["R9_ctx"]
    C9 = y["C9_ctx"]
    CA9 = y["CA9_ctx"]

    # SNc
    V10 = y["V10_snc"]
    M10_na = y["M10_na_snc"]
    H10_na = y["H10_na_snc"]
    M10_ca = y["M10_ca_snc"]
    M10_k = y["M10_k_snc"]
    CA10 = y["CA10_snc"]

    # STDP state variables
    x_pre = y["x_pre"]
    x_post = y["x_post"]
    W = y["W"]


    # parameter block

    # condition
    cond = params["condition"]
    pd = cond["pd"]
    DA = cond["DA"]
    Idbs = cond["Idbs"]

    # neurons
    p_th = params["neurons"]["th"]
    p_stn = params["neurons"]["stn"]
    p_gpe = params["neurons"]["gpe"]
    p_gpi = params["neurons"]["gpi"]
    p_dstr = params["neurons"]["dstr"]
    p_istr = params["neurons"]["istr"]
    p_m1_pyr = params["neurons"]["m1_pyr"]
    p_m1_fsi = params["neurons"]["m1_fsi"]
    p_s1_pyr = params["neurons"]["s1_pyr"]
    p_s1_fsi = params["neurons"]["s1_fsi"]
    p_snc = params["neurons"]["snc"]

    # synapses
    syn = params["synapses"]
    syn_gpi_th = syn["gpi_to_th"]
    syn_gpe_stn = syn["gpe_to_stn"]
    syn_stn_gpe = syn["stn_to_gpe"]
    syn_gpe_gpe = syn["gpe_to_gpe"]
    syn_gpe_gpi = syn["gpe_to_gpi"]
    syn_stn_gpi = syn["stn_to_gpi"]

    syn_dstr_dstr = syn["dstr_to_dstr"]
    syn_istr_istr = syn["istr_to_istr"]
    syn_istr_gpe = syn["istr_to_gpe"]
    syn_dstr_gpi = syn["dstr_to_gpi"]

    syn_pyr_pyr = syn["pyr_to_pyr"]
    syn_pyr_fsi = syn["pyr_to_fsi"]
    syn_fsi_fsi = syn["fsi_to_fsi"]
    syn_fsi_pyr = syn["fsi_to_pyr"]

    syn_pyr_dstr = syn["pyr_to_dstr"]
    syn_pyr_istr = syn["pyr_to_istr"]
    syn_pyr_stn = syn["pyr_to_stn"]
    syn_pyr_th = syn["pyr_to_th"]

    syn_th_pyr = syn["th_to_pyr"]
    syn_th_fsi = syn["th_to_fsi"]

    alpha_shared = syn["alpha_shared"]

    # plasticity
    plast = params["plasticity"]["gpe_to_stn_stdp"]

    # connectivity
    conn = params["connectivity"]
    w_gpi_th = conn["gpi_to_th"]

    w_istr_gpe = conn["istr_to_gpe"]
    w_dstr_gpi = conn["dstr_to_gpi"]

    w_stn_gpe = conn["stn_to_gpe"]
    w_stn_gpi = conn["stn_to_gpi"]

    w_gpe_gpi = conn["gpe_to_gpi"]
    w_gpe_gpe = conn["gpe_to_gpe"]

    w_dstr = conn["dstr_to_dstr"]
    w_istr = conn["istr_to_istr"]

    w_fsi = conn["fsi_to_fsi"]
    w_pyr = conn["pyr_to_pyr"]
    w_pyr_fsi = conn["pyr_to_fsi"]
    w_fsi_pyr = conn["fsi_to_pyr"]

    w_pyr_dstr = conn["pyr_to_dstr"]
    w_pyr_istr = conn["pyr_to_istr"]
    w_pyr_th = conn["pyr_to_th"]
    w_pyr_stn = conn["pyr_to_stn"]

    w_th_pyr = conn["th_to_pyr"]
    w_th_fsi = conn["th_to_fsi"]

    # synapse constants
    tau = alpha_shared["tau"]
    gpeak = alpha_shared["gpeak"]
    gpeak1 = alpha_shared["gpeak1"]

    # STDP constants
    tau_pre = plast["tau_pre"]
    tau_post = plast["tau_post"]
    A_plus = plast["A_plus"]
    A_minus = plast["A_minus"]
    pre_threshold = plast["pre_threshold"]
    post_threshold = plast["post_threshold"]



    # gating steady states & taus from V
    r1   = th_rinf(V1)
    p1   = th_pinf(V1)
    m1   = th_minf(V1)
    h1   = th_hinf(V1)
    th1  = th_tauh(V1)
    tr1  = th_taur(V1)

    r2   = stn_rinf(V2)
    n2   = stn_ninf(V2)
    m2   = stn_minf(V2)
    h2   = stn_hinf(V2)
    c2   = stn_cinf(V2)
    a2   = stn_ainf(V2)
    b2   = stn_binf(V2)

    tr2  = stn_taur(V2)
    tn2  = stn_taun(V2)
    th2  = stn_tauh(V2)
    tc2  = stn_tauc(V2)

    m3  = gpe_minf(V3)
    n3  = gpe_ninf(V3)
    h3  = gpe_hinf(V3)
    a3  = gpe_ainf(V3)
    s3  = gpe_sinf(V3)
    r3  = gpe_rinf(V3)
    tn3 = gpe_taun(V3)
    th3 = gpe_tauh(V3)
    tr3 = 30.0

    m4  = gpe_minf(V4)
    n4  = gpe_ninf(V4)
    h4  = gpe_hinf(V4)
    a4  = gpe_ainf(V4)
    s4  = gpe_sinf(V4)
    r4  = gpe_rinf(V4)
    tn4 = gpe_taun(V4)
    th4 = gpe_tauh(V4)
    tr4 = 30.0

    # cortex uses stn like gating - maybe change later?
    r6   = stn_rinf(V6)
    n6   = stn_ninf(V6)
    m6   = stn_minf(V6)
    h6   = stn_hinf(V6)
    c6   = stn_cinf(V6)
    a6   = stn_ainf(V6)
    b6   = stn_binf(V6)

    tr6  = stn_taur(V6)
    tn6  = stn_taun(V6)
    th6  = stn_tauh(V6)
    tc6  = stn_tauc(V6)

    r7   = stn_rinf(V7)
    n7   = stn_ninf(V7)
    m7   = stn_minf(V7)
    h7   = stn_hinf(V7)
    c7   = stn_cinf(V7)
    a7   = stn_ainf(V7)
    b7   = stn_binf(V7)

    tr7  = stn_taur(V7)
    tn7  = stn_taun(V7)
    th7  = stn_tauh(V7)
    tc7 = stn_tauc(V7)

    r8   = stn_rinf(V8)
    n8   = stn_ninf(V8)
    m8   = stn_minf(V8)
    h8   = stn_hinf(V8)
    c8   = stn_cinf(V8)
    a8   = stn_ainf(V8)
    b8   = stn_binf(V8)

    tr8  = stn_taur(V8)
    tn8  = stn_taun(V8)
    th8  = stn_tauh(V8)
    tc8  = stn_tauc(V8)

    r9   = stn_rinf(V9)
    n9   = stn_ninf(V9)
    m9   = stn_minf(V9)
    h9   = stn_hinf(V9)
    c9   = stn_cinf(V9)
    a9   = stn_ainf(V9)
    b9   = stn_binf(V9)

    tr9  = stn_taur(V9)
    tn9  = stn_taun(V9)
    th9  = stn_tauh(V9)
    tc9 = stn_tauc(V9)

    m10 = snc_minf(V10)
    tm10 = snc_taum(V10)
    m10_ca = snc_minf_ca(V10)
    tm10_ca = snc_taum_ca(V10)


    # ionic currents
    # TH 
    Il1 = p_th["gl"] * (V1 - p_th["El"])
    Ina1 = p_th["gna"] * (m1**3) * H1 * (V1 - p_th["Ena"])
    Ik1 = p_th["gk"] * ((0.75 * (1 - H1))**4) * (V1 - p_th["Ek"])  # as in original code
    It1 = p_th["gt"] * (p1**2) * R1 * (V1 - p_th["Et"])

    Iapp_th = p_th["Iapp"]

    # STN
    Il2 = p_stn["gl"] * (V2 - p_stn["El"])
    Ik2 = p_stn["gk"] * (N2**4) * (V2 - p_stn["Ek"])
    Ina2 = p_stn["gna"] * (m2**3) * H2 * (V2 - p_stn["Ena"])
    It2 = p_stn["gt"] * (a2**3) * (b2**2) * (V2 - p_stn["Eca"])
    Ica2 = p_stn["gca"] * (c2**2) * (V2 - p_stn["Eca"])
    Iahp2 = p_stn["gahp"] * (V2 - p_stn["Ek"]) * (CA2 / (CA2 + p_stn["k1"]))

    Iappstn = p_stn["Iapp"]

    I_dbs = dbs_current(t, cond) #DBS current

    # GPe 
    Il3 = p_gpe["gl"] * (V3 - p_gpe["El"])
    Ik3 = p_gpe["gk"] * (N3**4) * (V3 - p_gpe["Ek"])
    Ina3 = p_gpe["gna"] * (m3**3) * H3 * (V3 - p_gpe["Ena"])
    It3 = p_gpe["gt"] * (a3**3) * R3 * (V3 - p_gpe["Eca"])
    Ica3 = p_gpe["gca"] * (s3**2) * (V3 - p_gpe["Eca"])
    Iahp3 = p_gpe["gahp"] * (V3 - p_gpe["Ek"]) * (CA3 / (CA3 + p_gpe["k1"]))

    Iappgpe = p_gpe["Iapp"]

    # GPi
    Il4 = p_gpi["gl"] * (V4 - p_gpi["El"])
    Ik4 = p_gpi["gk"] * (N4**4) * (V4 - p_gpi["Ek"])
    Ina4 = p_gpi["gna"] * (m4**3) * H4 * (V4 - p_gpi["Ena"])
    It4 = p_gpi["gt"] * (a4**3) * R4 * (V4 - p_gpi["Eca"])
    Ica4 = p_gpi["gca"] * (s4**2) * (V4 - p_gpi["Eca"])
    Iahp4 = p_gpi["gahp"] * (V4 - p_gpi["Ek"]) * (CA4 / (CA4 + p_gpi["k1"]))

    Iappgpi = p_gpi["Iapp"]

    # direct striatum
    Ina5d = p_dstr["gna"] * (m5d**3) * h5d * (V5d - p_dstr["Ena"])
    Ik5d = p_dstr["gk"] * (n5d**4) * (V5d - p_dstr["Ek"])
    Il5d = p_dstr["gl"] * (V5d - p_dstr["El"])
    Im5d = p_dstr["gm_effective"] * p5d * (V5d - p_dstr["Em"])

    Iappstrd = p_dstr["Iapp"]

    # indirect striatum
    Ina5i = p_istr["gna"] * (m5i**3) * h5i * (V5i - p_istr["Ena"])
    Ik5i = p_istr["gk"] * (n5i**4) * (V5i - p_istr["Ek"])
    Il5i = p_istr["gl"] * (V5i - p_istr["El"])
    Im5i = p_istr["gm_effective"] * p5i * (V5i - p_istr["Em"])

    Iappstri = p_istr["Iapp"]

    # M1 PYR 
    # currently STN-like 
    Il6 = p_m1_pyr["gl"] * (V6 - p_m1_pyr["El"])
    Ik6 = p_m1_pyr["gk"] * (N6**4) * (V6 - p_m1_pyr["Ek"])
    Ina6 = p_m1_pyr["gna"] * (m6**3) * H6 * (V6 - p_m1_pyr["Ena"])
    It6 = p_m1_pyr["gt"] * (a6**3) * (b6**2) * (V6 - p_m1_pyr["Eca"])
    Ica6 = p_m1_pyr["gca"] * (c6**2) * (V6 - p_m1_pyr["Eca"])
    Iahp6 = p_m1_pyr["gahp"] * (V6 - p_m1_pyr["Ek"]) * (CA6 / (CA6 + p_m1_pyr["k1"]))

    Iappctx6 = p_m1_pyr["Iapp"]

    #  M1 FSI
    # currently STN-like 
    Il7 = p_m1_fsi["gl"] * (V7 - p_m1_fsi["El"])
    Ik7 = p_m1_fsi["gk"] * (N7**4) * (V7 - p_m1_fsi["Ek"])
    Ina7 = p_m1_fsi["gna"] * (m7**3) * H7 * (V7 - p_m1_fsi["Ena"])
    It7 = p_m1_fsi["gt"] * (a7**3) * (b7**2) * (V7 - p_m1_fsi["Eca"])
    Ica7 = p_m1_fsi["gca"] * (c7**2) * (V7 - p_m1_fsi["Eca"])
    Iahp7 = p_m1_fsi["gahp"] * (V7 - p_m1_fsi["Ek"]) * (CA7 / (CA7 + p_m1_fsi["k1"]))

    Iappctx7 = p_m1_fsi["Iapp"]

    # S1 PYR 
    # currently STN-like
    Il8 = p_s1_pyr["gl"] * (V8 - p_s1_pyr["El"])
    Ik8 = p_s1_pyr["gk"] * (N8**4) * (V8 - p_s1_pyr["Ek"])
    Ina8 = p_s1_pyr["gna"] * (m8**3) * H8 * (V8 - p_s1_pyr["Ena"])
    It8 = p_s1_pyr["gt"] * (a8**3) * (b8**2) * (V8 - p_s1_pyr["Eca"])
    Ica8 = p_s1_pyr["gca"] * (c8**2) * (V8 - p_s1_pyr["Eca"])
    Iahp8 = p_s1_pyr["gahp"] * (V8 - p_s1_pyr["Ek"]) * (CA8 / (CA8 + p_s1_pyr["k1"]))

    Iappctx8 = p_s1_pyr["Iapp"]

    # S1 FSI 
    # currently STN-like
    Il9 = p_s1_fsi["gl"] * (V9 - p_s1_fsi["El"])
    Ik9 = p_s1_fsi["gk"] * (N9**4) * (V9 - p_s1_fsi["Ek"])
    Ina9 = p_s1_fsi["gna"] * (m9**3) * H9 * (V9 - p_s1_fsi["Ena"])
    It9 = p_s1_fsi["gt"] * (a9**3) * (b9**2) * (V9 - p_s1_fsi["Eca"])
    Ica9 = p_s1_fsi["gca"] * (c9**2) * (V9 - p_s1_fsi["Eca"])
    Iahp9 = p_s1_fsi["gahp"] * (V9 - p_s1_fsi["Ek"]) * (CA9 / (CA9 + p_s1_fsi["k1"]))

    Iappctx9 = p_s1_fsi["Iapp"]

    # SNc 
    Il10 = p_snc["gl"] * (V10 - p_snc["El"])
    Ina10 = p_snc["gna"] * (M10_na**3) * H10_na * (V10 - p_snc["Ena"])
    Ikdr10 = p_snc["gkdr"] * (M10_k**3) * (V10 - p_snc["Ek"])
    Ica10 = p_snc["gca"] * M10_ca * h10_ca(CA10) * (V10 - p_snc["Eca"])
    Isk10 = p_snc["gsk"] * s_SK(CA10) * (V10 - p_snc["Ek"])

    Iappsnc = p_snc["Iapp"]

  
    # synapses with connectivity matrices


    # presynaptic activation
    H_th = Hinf(V1, theta=p_th["spike_threshold"])
    H_gpe = Hinf(V3, theta=syn_stn_gpe["theta"])
    H_stn = Hinf(V2, theta=syn_gpe_stn["theta"])
    H_gpi = Hinf(V4, theta=syn_gpi_th["theta"])
    H_istr = Hinf(V5i, theta=p_istr["spike_threshold"])
    H_dstr = Hinf(V5d, theta=p_dstr["spike_threshold"])
    H_ctxpyr = Hinf(V6, theta=syn_pyr_pyr["theta"])
    H_ctxfi = Hinf(V7, theta=syn_pyr_pyr["theta"])

    # TH synapse gating
    # first-order synapse
    dS1dt = syn_gpi_th["A"] * (1.0 - S1) * H_th - syn_gpi_th["B"] * S1

    # STN synapse gating
    # second-order alpha synapse
    u2 = gpeak / (tau * jnp.exp(-1.0)) * H_stn
    dS2dt = Z2
    dZ2dt = u2 - (2.0 / tau) * Z2 - (1.0 / tau**2) * S2

    # GPi synapse gating 
    # second-order alpha synapse
    u4 = gpeak1 / (tau * jnp.exp(-1.0)) * H_gpi
    dS4dt = Z4
    dZ4dt = u4 - (2.0 / tau) * Z4 - (1.0 / tau**2) * S4

    # GPe synapse gating 
    dS3dt = syn_gpe_stn["A"] * (1.0 - S3) * H_gpe - syn_gpe_stn["B"] * S3

    #  GPe - STN STDP
    pre_spike = spike_event(V3, plast["pre_threshold"])
    post_spike = spike_event(V2, plast["post_threshold"])

    dx_pre_dt = -x_pre / tau_pre + pre_spike
    dx_post_dt = -x_post / tau_post + post_spike

    dWdt = (
        A_plus * jnp.outer(post_spike, x_pre)
        - A_minus * jnp.outer(x_post, pre_spike)
    )

    # striatum output synapse gating
    # indirect striatum -> GPe
    u3 = gpeak1 / (tau * jnp.exp(-1.0)) * H_istr
    dS52idt = Z5i_2
    dZ52idt = u3 - (2.0 / tau) * Z5i_2 - (1.0 / tau**2) * S5i_2

    # direct striatum -> GPi
    u5 = gpeak1 / (tau * jnp.exp(-1.0)) * H_dstr
    dS52ddt = Z5d_2
    dZ52ddt = u5 - (2.0 / tau) * Z5d_2 - (1.0 / tau**2) * S5d_2

    # recurrent striatal GABA
    dS5idt = (w_istr @ str_Ggaba(V5i)) * (1.0 - S5i) - (S5i / syn_istr_istr["tau_i"])
    dS5ddt = (w_dstr @ str_Ggaba(V5d)) * (1.0 - S5d) - (S5d / syn_dstr_dstr["tau_i"])

    # cortical synapse gating
    # currently using Rubin/Terman-like first-order kinetics
    dS6dt = syn_pyr_pyr["A"] * (1.0 - S6) * H_ctxpyr - syn_pyr_pyr["B"] * S6
    dS7dt = syn_pyr_pyr["A"] * (1.0 - S7) * H_ctxfi - syn_pyr_pyr["B"] * S7


    # synaptic currents
    # BG loop
    Igith = syn_gpi_th["scale"] * (
        syn_gpi_th["g"] * (V1 - syn_gpi_th["E"]) * (w_gpi_th @ S4))

    Igesn = syn_gpe_stn["scale"] * (
        syn_gpe_stn["g"] * (V2 - syn_gpe_stn["E"]) * (W @ S3))

    Isnge = syn_stn_gpe["scale"] * (
        syn_stn_gpe["g"] * (V3 - syn_stn_gpe["E"]) * (w_stn_gpe @ S2))

    Igege = syn_gpe_gpe["scale"] * (
        syn_gpe_gpe["g"] * (V3 - syn_gpe_gpe["E"]) * (w_gpe_gpe @ S3))

    Igegi = syn_gpe_gpi["scale"] * (
        syn_gpe_gpi["g"] * (V4 - syn_gpe_gpi["E"]) * (w_gpe_gpi @ S3))

    Isngi = syn_stn_gpi["scale"] * (
        syn_stn_gpi["g"] * (V4 - syn_stn_gpi["E"]) * (w_stn_gpi @ S2))

    # striatum outputs
    Istrge = syn_istr_gpe["g"] * (V3 - syn_istr_gpe["E"]) * (w_istr_gpe @ S5i_2)
    Istrgi = syn_dstr_gpi["g"] * (V4 - syn_dstr_gpi["E"]) * (w_dstr_gpi @ S5d_2)

    # recurrent striatal inhibition
    Istrd = (syn_dstr_dstr["ggaba"] / syn_dstr_dstr["normalization"]) * (
        V5d - syn_dstr_dstr["E"]) * (w_dstr @ S5d)

    Istri = (syn_istr_istr["ggaba"] / syn_istr_istr["normalization"]) * (
        V5i - syn_istr_istr["E"]) * (w_istr @ S5i)

    # cortex internal
    Ipypy = syn_pyr_pyr["scale"] * syn_pyr_pyr["g"] * (V6 - syn_pyr_pyr["E"]) * (w_pyr @ S6)
    Ipyfi = syn_pyr_fsi["scale"] * syn_pyr_fsi["g"] * (V7 - syn_pyr_fsi["E"]) * (w_pyr_fsi @ S6)
    Ififi = syn_fsi_fsi["scale"] * syn_fsi_fsi["g"] * (V7 - syn_fsi_fsi["E"]) * (w_fsi @ S7)
    Ifipy = syn_fsi_pyr["scale"] * syn_fsi_pyr["g"] * (V6 - syn_fsi_pyr["E"]) * (w_fsi_pyr @ S7)

    # cortical projections to BG / TH
    Ipystrd = syn_pyr_dstr["scale"] * syn_pyr_dstr["g"] * (V5d - syn_pyr_dstr["E"]) * (w_pyr_dstr @ S6)
    Ipystri = syn_pyr_istr["scale"] * syn_pyr_istr["g"] * (V5i - syn_pyr_istr["E"]) * (w_pyr_istr @ S6)
    Ipysn = syn_pyr_stn["scale"] * syn_pyr_stn["g"] * (V2 - syn_pyr_stn["E"]) * (w_pyr_stn @ S6)
    Ipyth = syn_pyr_th["scale"] * syn_pyr_th["g"] * (V1 - syn_pyr_th["E"]) * (w_pyr_th @ S6)

    # thalamocortical projections
    Ithpy = syn_th_pyr["g"] * (V6 - syn_th_pyr["E"]) * (w_th_pyr @ S1)
    Ithfi = syn_th_fsi["g"] * (V7 - syn_th_fsi["E"]) * (w_th_fsi @ S1)


    # differential equations

    # TH
    dV1dt = (-Il1 - Ik1 - Ina1 - It1 - Igith + Iapp_th) / p_th["Cm"]
    dH1dt = (h1 - H1) / th1
    dR1dt = (r1 - R1) / tr1

    # STN
    dV2dt = (-Il2 - Ik2 - Ina2 - It2 - Ica2 - Iahp2 - Igesn + Iappstn + I_dbs) / p_stn["Cm"]
    dN2dt = 0.75 * (n2 - N2) / tn2
    dH2dt = 0.75 * (h2 - H2) / th2
    dR2dt = 0.2 * (r2 - R2) / tr2
    dCA2dt = 3.75e-5 * (-Ica2 - It2 - p_stn["kca"] * CA2)
    dC2dt = 0.08 * (c2 - C2) / tc2

    # GPe
    dV3dt = (-Il3 - Ik3 - Ina3 - It3 - Ica3 - Iahp3 - Isnge - Igege - Istrge + Iappgpe) / p_gpe["Cm"]
    dN3dt = 0.1 * (n3 - N3) / tn3
    dH3dt = 0.05 * (h3 - H3) / th3
    dR3dt = 1.0 * (r3 - R3) / tr3
    dCA3dt = 1e-4 * (-Ica3 - It3 - p_gpe["kca"] * CA3)

    # GPi
    dV4dt = (-Il4 - Ik4 - Ina4 - It4 - Ica4 - Iahp4 - Igegi - Isngi - Istrgi + Iappgpi) / p_gpi["Cm"]
    dN4dt = 0.1 * (n4 - N4) / tn4
    dH4dt = 0.05 * (h4 - H4) / th4
    dR4dt = 1.0 * (r4 - R4) / tr4
    dCA4dt = 1e-4 * (-Ica4 - It4 - p_gpi["kca"] * CA4)

    # direct striatum
    dV5ddt = (-Il5d - Ik5d - Ina5d - Im5d - Istrd - Ipystrd + Iappstrd) / p_dstr["Cm"]
    dm5ddt = str_alpham(V5d) * (1.0 - m5d) - str_betam(V5d) * m5d
    dh5ddt = str_alphah(V5d) * (1.0 - h5d) - str_betah(V5d) * h5d
    dn5ddt = str_alphan(V5d) * (1.0 - n5d) - str_betan(V5d) * n5d
    dp5ddt = str_alphap(V5d) * (1.0 - p5d) - str_betap(V5d) * p5d

    # indirect striatum 
    dV5idt = (-Il5i - Ik5i - Ina5i - Im5i - Istri - Ipystri + Iappstri) / p_istr["Cm"]
    dm5idt = str_alpham(V5i) * (1.0 - m5i) - str_betam(V5i) * m5i
    dh5idt = str_alphah(V5i) * (1.0 - h5i) - str_betah(V5i) * h5i
    dn5idt = str_alphan(V5i) * (1.0 - n5i) - str_betan(V5i) * n5i
    dp5idt = str_alphap(V5i) * (1.0 - p5i) - str_betap(V5i) * p5i

    # M1 PYR 
    # currently STN-like placeholder dynamics
    dV6dt = (-Il6 - Ik6 - Ina6 - It6 - Ica6 - Iahp6 - Ithpy - Ipypy - Ifipy + Iappctx6) / p_m1_pyr["Cm"]
    dN6dt = 0.75 * (n6 - N6) / tn6
    dH6dt = 0.75 * (h6 - H6) / th6
    dR6dt = 0.2 * (r6 - R6) / tr6
    dCA6dt = 3.75e-5 * (-Ica6 - It6 - p_m1_pyr["kca"] * CA6)
    dC6dt = 0.08 * (c6 - C6) / tc6

    # M1 FSI
    # currently STN-like placeholder dynamics
    dV7dt = (-Il7 - Ik7 - Ina7 - It7 - Ica7 - Iahp7 - Ipyfi - Ififi - Ithfi + Iappctx7) / p_m1_fsi["Cm"]
    dN7dt = 0.75 * (n7 - N7) / tn7
    dH7dt = 0.75 * (h7 - H7) / th7
    dR7dt = 0.2 * (r7 - R7) / tr7
    dCA7dt = 3.75e-5 * (-Ica7 - It7 - p_m1_fsi["kca"] * CA7)
    dC7dt = 0.08 * (c7 - C7) / tc7

    # S1 PYR
    # currently STN-like placeholder dynamics
    dV8dt = (-Il8 - Ik8 - Ina8 - It8 - Ica8 - Iahp8 + Iappctx8) / p_s1_pyr["Cm"]
    dN8dt = 0.75 * (n8 - N8) / tn8
    dH8dt = 0.75 * (h8 - H8) / th8
    dR8dt = 0.2 * (r8 - R8) / tr8
    dCA8dt = 3.75e-5 * (-Ica8 - It8 - p_s1_pyr["kca"] * CA8)
    dC8dt = 0.08 * (c8 - C8) / tc8

    # S1 FSI 
    # currently STN-like placeholder dynamics
    dV9dt = (-Il9 - Ik9 - Ina9 - It9 - Ica9 - Iahp9 + Iappctx9) / p_s1_fsi["Cm"]
    dN9dt = 0.75 * (n9 - N9) / tn9
    dH9dt = 0.75 * (h9 - H9) / th9
    dR9dt = 0.2 * (r9 - R9) / tr9
    dCA9dt = 3.75e-5 * (-Ica9 - It9 - p_s1_fsi["kca"] * CA9)
    dC9dt = 0.08 * (c9 - C9) / tc9

    # SNc 
    dV10dt = (-Il10 - Ina10 - Ikdr10 - Isk10 - Ica10 + Iappsnc) / p_snc["Cm"]
    dM10nadt = 1.9651 * jnp.exp(1.7127 * V10) * (1.0 - M10_na) - 0.0424 * jnp.exp(-1.5581 * V10) * M10_na
    dH10nadt = 9.566e-5 * jnp.exp(-2.4317 * V10) * (1.0 - H10_na) - 0.5296 * jnp.exp(1.1868 * V10) * H10_na
    dM10kdt = (m10 - M10_k) / tm10
    dM10cadt = (m10_ca - M10_ca) / tm10_ca
    dCA10dt = -1e-5 * Ica10 - (CA10 - 1.88e-4) / 20.0

    return {
        "V1_th": dV1dt,
        "H1_th": dH1dt,
        "R1_th": dR1dt,
        "S1_th": dS1dt,

        "V2_stn": dV2dt,
        "N2_stn": dN2dt,
        "H2_stn": dH2dt,
        "R2_stn": dR2dt,
        "C2_stn": dC2dt,
        "CA2_stn": dCA2dt,
        "S2_stn": dS2dt,
        "Z2_stn": dZ2dt,

        "V3_gpe": dV3dt,
        "N3_gpe": dN3dt,
        "H3_gpe": dH3dt,
        "R3_gpe": dR3dt,
        "CA3_gpe": dCA3dt,
        "S3_gpe": dS3dt,

        "W": dWdt,
        "x_pre": dx_pre_dt,
        "x_post": dx_post_dt,

        "V4_gpi": dV4dt,
        "N4_gpi": dN4dt,
        "H4_gpi": dH4dt,
        "R4_gpi": dR4dt,
        "CA4_gpi": dCA4dt,
        "S4_gpi": dS4dt,
        "Z4_gpi": dZ4dt,

        "V5_dstr": dV5ddt,
        "m5_dstr": dm5ddt,
        "h5_dstr": dh5ddt,
        "n5_dstr": dn5ddt,
        "p5_dstr": dp5ddt,
        "S5_dstr": dS5ddt,
        "S52_dstr": dS52ddt,
        "Z52_dstr": dZ52ddt,

        "V5_istr": dV5idt,
        "m5_istr": dm5idt,
        "h5_istr": dh5idt,
        "n5_istr": dn5idt,
        "p5_istr": dp5idt,
        "S5_istr": dS5idt,
        "S52_istr": dS52idt,
        "Z52_istr": dZ52idt,

        "V6_ctx": dV6dt,
        "N6_ctx": dN6dt,
        "H6_ctx": dH6dt,
        "R6_ctx": dR6dt,
        "C6_ctx": dC6dt,
        "CA6_ctx": dCA6dt,
        "S6_ctx": dS6dt,

        "V7_ctx": dV7dt,
        "N7_ctx": dN7dt,
        "H7_ctx": dH7dt,
        "R7_ctx": dR7dt,
        "C7_ctx": dC7dt,
        "CA7_ctx": dCA7dt,
        "S7_ctx": dS7dt,

        "V8_ctx": dV8dt,
        "N8_ctx": dN8dt,
        "H8_ctx": dH8dt,
        "R8_ctx": dR8dt,
        "C8_ctx": dC8dt,
        "CA8_ctx": dCA8dt,

        "V9_ctx": dV9dt,
        "N9_ctx": dN9dt,
        "H9_ctx": dH9dt,
        "R9_ctx": dR9dt,
        "C9_ctx": dC9dt,
        "CA9_ctx": dCA9dt,

        "V10_snc": dV10dt,
        "M10_na_snc": dM10nadt,
        "H10_na_snc": dH10nadt,
        "M10_ca_snc": dM10cadt,
        "M10_k_snc": dM10kdt,
        "CA10_snc": dCA10dt,
    }

#%% inital state
def make_initial_state(params, key):

    sizes = params["sizes"]
    plast = params["plasticity"]["gpe_to_stn_stdp"]

    n_th = sizes["th"]
    n_stn = sizes["stn"]
    n_gpe = sizes["gpe"]
    n_gpi = sizes["gpi"]
    n_dstr = sizes["dstr"]
    n_istr = sizes["istr"]
    n_ctx_fsi = sizes["ctx_fsi"]
    n_ctx_pyr = sizes["ctx_pyr"]
    n_snc = sizes["snc"]

    (
        key_v1, key_v2, key_v3, key_v4,
        key_v5d, key_v5i, key_v6, key_v7,
        key_v8, key_v9, key_v10,
        key_w
    ) = jax.random.split(key, 12)

    # baseline voltages
    V1_init = -62.0
    V2_init = -62.0
    V3_init = -62.0
    V4_init = -62.0
    V5d_init = -60.0
    V5i_init = -60.0
    V6_init = -62.0
    V7_init = -62.0
    V8_init = -62.0
    V9_init = -62.0
    V10_init = -50.0

    # noise amplitude in mV
    sigma_init = 2.0

    # randomised initial voltages
    V1_0 = V1_init + sigma_init * jax.random.normal(key_v1, (n_th,))
    V2_0 = V2_init + sigma_init * jax.random.normal(key_v2, (n_stn,))
    V3_0 = V3_init + sigma_init * jax.random.normal(key_v3, (n_gpe,))
    V4_0 = V4_init + sigma_init * jax.random.normal(key_v4, (n_gpi,))
    V5d_0 = V5d_init + sigma_init * jax.random.normal(key_v5d, (n_dstr,))
    V5i_0 = V5i_init + sigma_init * jax.random.normal(key_v5i, (n_istr,))
    V6_0 = V6_init + sigma_init * jax.random.normal(key_v6, (n_ctx_pyr,))
    V7_0 = V7_init + sigma_init * jax.random.normal(key_v7, (n_ctx_fsi,))
    V8_0 = V8_init + sigma_init * jax.random.normal(key_v8, (n_ctx_pyr,))
    V9_0 = V9_init + sigma_init * jax.random.normal(key_v9, (n_ctx_fsi,))
    V10_0 = V10_init + sigma_init * jax.random.normal(key_v10, (n_snc,))

    # STDP initial weights
    W0 = jax.random.uniform(
        key_w,
        (n_stn, n_gpe),
        minval=plast["W_init_min"],
        maxval=plast["W_init_max"],
    )

    y0 = {
        # TH
        "V1_th": V1_0,
        "H1_th": th_hinf(V1_0),
        "R1_th": th_rinf(V1_0),
        "S1_th": jnp.zeros((n_th,)),

        # STN
        "V2_stn": V2_0,
        "N2_stn": stn_ninf(V2_0),
        "H2_stn": stn_hinf(V2_0),
        "R2_stn": stn_rinf(V2_0),
        "C2_stn": stn_cinf(V2_0),
        "CA2_stn": jnp.full((n_stn,), 0.1),
        "S2_stn": jnp.zeros((n_stn,)),
        "Z2_stn": jnp.zeros((n_stn,)),

        # GPe
        "V3_gpe": V3_0,
        "N3_gpe": gpe_ninf(V3_0),
        "H3_gpe": gpe_hinf(V3_0),
        "R3_gpe": gpe_rinf(V3_0),
        "CA3_gpe": jnp.full((n_gpe,), 0.1),
        "S3_gpe": jnp.zeros((n_gpe,)),

        # STDP state
        "W": W0,
        "x_pre": jnp.zeros((n_gpe,)),
        "x_post": jnp.zeros((n_stn,)),

        # GPi
        "V4_gpi": V4_0,
        "N4_gpi": gpe_ninf(V4_0),
        "H4_gpi": gpe_hinf(V4_0),
        "R4_gpi": gpe_rinf(V4_0),
        "CA4_gpi": jnp.full((n_gpi,), 0.1),
        "S4_gpi": jnp.zeros((n_gpi,)),
        "Z4_gpi": jnp.zeros((n_gpi,)),

        # direct striatum
        "V5_dstr": V5d_0,
        "m5_dstr": str_alpham(V5d_0) / (str_alpham(V5d_0) + str_betam(V5d_0)),
        "h5_dstr": str_alphah(V5d_0) / (str_alphah(V5d_0) + str_betah(V5d_0)),
        "n5_dstr": str_alphan(V5d_0) / (str_alphan(V5d_0) + str_betan(V5d_0)),
        "p5_dstr": str_alphap(V5d_0) / (str_alphap(V5d_0) + str_betap(V5d_0)),
        "S5_dstr": jnp.full((n_dstr,), 0.1),
        "S52_dstr": jnp.zeros((n_dstr,)),
        "Z52_dstr": jnp.zeros((n_dstr,)),

        # indirect striatum
        "V5_istr": V5i_0,
        "m5_istr": str_alpham(V5i_0) / (str_alpham(V5i_0) + str_betam(V5i_0)),
        "h5_istr": str_alphah(V5i_0) / (str_alphah(V5i_0) + str_betah(V5i_0)),
        "n5_istr": str_alphan(V5i_0) / (str_alphan(V5i_0) + str_betan(V5i_0)),
        "p5_istr": str_alphap(V5i_0) / (str_alphap(V5i_0) + str_betap(V5i_0)),
        "S5_istr": jnp.full((n_istr,), 0.1),
        "S52_istr": jnp.zeros((n_istr,)),
        "Z52_istr": jnp.zeros((n_istr,)),

        # M1 cortex PYR
        "V6_ctx": V6_0,
        "N6_ctx": stn_ninf(V6_0),
        "H6_ctx": stn_hinf(V6_0),
        "R6_ctx": stn_rinf(V6_0),
        "C6_ctx": stn_cinf(V6_0),
        "CA6_ctx": jnp.full((n_ctx_pyr,), 0.1),
        "S6_ctx": jnp.zeros((n_ctx_pyr,)),

        # M1 cortex FSI
        "V7_ctx": V7_0,
        "N7_ctx": stn_ninf(V7_0),
        "H7_ctx": stn_hinf(V7_0),
        "R7_ctx": stn_rinf(V7_0),
        "C7_ctx": stn_cinf(V7_0),
        "CA7_ctx": jnp.full((n_ctx_fsi,), 0.1),
        "S7_ctx": jnp.zeros((n_ctx_fsi,)),

        # S1 cortex PYR
        "V8_ctx": V8_0,
        "N8_ctx": stn_ninf(V8_0),
        "H8_ctx": stn_hinf(V8_0),
        "R8_ctx": stn_rinf(V8_0),
        "C8_ctx": stn_cinf(V8_0),
        "CA8_ctx": jnp.full((n_ctx_pyr,), 0.1),

        # S1 cortex FSI
        "V9_ctx": V9_0,
        "N9_ctx": stn_ninf(V9_0),
        "H9_ctx": stn_hinf(V9_0),
        "R9_ctx": stn_rinf(V9_0),
        "C9_ctx": stn_cinf(V9_0),
        "CA9_ctx": jnp.full((n_ctx_fsi,), 0.1),

        # SNc
        "V10_snc": V10_0,
        "M10_na_snc": jnp.full((n_snc,), 0.0952),
        "H10_na_snc": jnp.full((n_snc,), 0.1848),
        "M10_ca_snc": jnp.full((n_snc,), 0.006271),
        "M10_k_snc": jnp.full((n_snc,), 0.0932),
        "CA10_snc": jnp.full((n_snc,), 1.88e-4),
    }

    return y0

#%% chunked Euler solver
def run_chunk_euler_scan(y0, params, t0, dt, chunk_length, dt_save):
    n_steps = int(round(chunk_length / dt))
    if not np.isclose(n_steps * dt, chunk_length):
        raise ValueError("chunk_length must be an integer multiple of dt.")

    save_every = int(round(dt_save / dt))
    if not np.isclose(save_every * dt, dt_save):
        raise ValueError("dt_save must be an integer multiple of dt.")

    t_eval = t0 + dt * jnp.arange(n_steps)      # times used in bg_rhs
    ts_out = t_eval + dt                        # times of returned y_next

    def euler_step(y, t):
        dy = bg_rhs(t, y, params)
        y_next = jax.tree_util.tree_map(lambda a, da: a + dt * da, y, dy)
        return y_next, y_next

    y_final, ys = jax.lax.scan(euler_step, y0, t_eval)

    ts_out = ts_out[::save_every]
    ys = jax.tree_util.tree_map(lambda a: a[::save_every], ys)

    return y_final, (ts_out, ys)


run_chunk_euler_scan = jax.jit(
    run_chunk_euler_scan,
    static_argnames=("dt", "chunk_length", "dt_save"),
)


def simulate_last_chunk_euler(y0, params, tmax, dt=0.1, dt_save=1.0, chunk_length=1000.0):
    n_chunks_float = tmax / chunk_length
    if not np.isclose(n_chunks_float, round(n_chunks_float)):
        raise ValueError("For this version, tmax must be an integer multiple of chunk_length.")

    n_chunks = int(round(n_chunks_float))
    if n_chunks < 1:
        raise ValueError("Need at least one chunk.")

    def advance_one_chunk(i, y):
        t0 = i * chunk_length
        y_next, _ = run_chunk_euler_scan(y, params, t0, dt, chunk_length, dt_save)
        return y_next

    if n_chunks > 1:
        y_before_last = jax.lax.fori_loop(0, n_chunks - 1, advance_one_chunk, y0)
    else:
        y_before_last = y0

    t0_last = (n_chunks - 1) * chunk_length
    y_final, (last_ts, last_ys) = run_chunk_euler_scan(
        y_before_last, params, t0_last, dt, chunk_length, dt_save
    )

    return {
        "ts": last_ts,
        "V1_th": last_ys["V1_th"],
        "V2_stn": last_ys["V2_stn"],
        "V3_gpe": last_ys["V3_gpe"],
        "V4_gpi": last_ys["V4_gpi"],
        "V5_dstr": last_ys["V5_dstr"],
        "V5_istr": last_ys["V5_istr"],
        "V6_ctx": last_ys["V6_ctx"],
        "V7_ctx": last_ys["V7_ctx"],
        "V8_ctx": last_ys["V8_ctx"],
        "V9_ctx": last_ys["V9_ctx"],
        "V10_snc": last_ys["V10_snc"],
        "W": last_ys["W"],
    }


#%% run simulation
tmax = 1000.0
chunk_size = 100.0      # 1 second per chunk
dt0 = 0.01
dt_save = 1           

# 1. choose condition
cfg = make_condition_config(
    pd=1,        # healthy
    stim=0,      # DBS off
    freq=130.0,
    tmax=tmax,
    dt=dt0,
)

# 2. build parameter tree
params = make_params(
    cfg=cfg,
    key0=jax.random.PRNGKey(0),
    n=4,
)

# 3. define initial state
y0 = make_initial_state(
    params=params,
    key=jax.random.PRNGKey(0)
)

# 4. run simulation
res = simulate_last_chunk_euler(
    y0, 
    params, 
    tmax, 
    dt0, 
    dt_save, 
    chunk_size)


#%%
# plot to check
ts = res["ts"]
V1 = res["V1_th"]
V2 = res["V2_stn"]
V3 = res["V3_gpe"]
V4 = res["V4_gpi"]
V5d = res["V5_dstr"]
V5i = res["V5_istr"]
V6 = res["V6_ctx"]
V7 = res["V7_ctx"]
V8 = res["V8_ctx"]
V9 = res["V9_ctx"]
V10 = res["V10_snc"]
W = res["W"]

plt.plot(ts, V1[:,3])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("TH")
plt.show()

# plot to check
plt.plot(ts, V2[:,2])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("STN")
plt.show()

# plot to check
plt.plot(ts, V3[:,3])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("GPe")
plt.show()

# plot to check
plt.plot(ts, V4[:,1])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("GPi")
plt.show()

# plot to check
plt.plot(ts, V5d[:,0])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("direct Striatum")
plt.show()

# plot to check
plt.plot(ts, V5i[:,1])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("indirect Striatum")
plt.show()

# plot to check
plt.plot(ts, V6[:,3])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("Cortex M1 (PYR)")
plt.show()

# plot to check
plt.plot(ts, V7[:,1])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("Cortex M1 (FSI)")
plt.show()

# plot to check
plt.plot(ts, V8[:,2])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("Cortex S1 (PYR)")
plt.show()

# plot to check
plt.plot(ts, V9[:,3])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("Cortex S1 (FSI)")
plt.show()

# plot to check
plt.plot(ts, V10[:,0])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("SNc")
plt.show()

# plot to check
plt.plot(ts, W[:,1])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("W")
plt.show()


# %% model validation
# set up
population_order = [
    "GPe", "STN", "GPi", "TH",
    "PYR M1", "FSI M1", "PYR S1", "FSI S1",
    "dStr", "iStr", "SNc"
]

population_voltages = {
        "TH": V1,
        "STN": V2,
        "GPe": V3,
        "GPi": V4,
        "dStr": V5d,
        "iStr": V5i,
        "PYR M1": V6,
        "FSI M1": V7,
        "PYR S1": V8,
        "FSI S1": V9,
        "SNc":V10,
    }

# 1. mean Hz rate
results = compute_metrics_all_populations(
    population_voltages=population_voltages,
    dt_ms=1.0,
    spike_height_map={
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
    },
    refractory_ms=2.0,
)

mean_rates = {pop: res["mean_rate_hz"] for pop, res in results.items()}

for pop, rate in mean_rates.items():
    print(f"{pop}: {rate:.3f} Hz")

# plot
def plot_population_boxplots(results, population_order=None):
    if population_order is None:
        population_order = list(results.keys())

    labels = [pop for pop in population_order if pop in results]
    data = [results[pop]["rates_hz"] for pop in labels]

    plt.figure(figsize=(8, 6))
    plt.boxplot(
        data,
        vert=False,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='*', markeredgecolor='black', markersize=7),
        flierprops=dict(marker='+', markeredgecolor='black', markersize=6),
    )
    plt.xlabel("Rate (Hz)")
    plt.ylabel("Population")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_population_boxplots(
    results,
    population_order=["GPe", "STN", "GPi", "TH", "PYR M1", "FSI M1", "PYR S1", "FSI S1", "dStr", "iStr"],
)
#%%
# 2. ISI CV

irregularity_results = compute_irregularity_all_populations(
    population_voltages=population_voltages,
    dt_ms=1.0,
    spike_height_map={
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
    },
    refractory_ms=2.0,
    min_spikes_for_cv=2,
)


def plot_irregularity_boxplots(
    results,
    population_order=None,
    figsize=(8, 6),
    title="Irregularity by population (CV_ISI)",
    xlabel="CV of ISI",
    ylabel="Population",
    xlim=None,
):
    if population_order is None:
        population_order = list(results.keys())

    labels = []
    data = []

    for pop in population_order:
        if pop not in results:
            continue
        vals = results[pop]["cv_isi"]
        vals = vals[np.isfinite(vals)]
        labels.append(pop)
        data.append(vals)

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='*', markeredgecolor='black', markersize=7),
        medianprops=dict(color='black', linewidth=1.8),
        whiskerprops=dict(linewidth=1.6),
        capprops=dict(linewidth=1.6),
        boxprops=dict(linewidth=1.6),
        flierprops=dict(marker='+', markeredgecolor='black', markersize=6),
    )

    colors = [
        "#6baed6", "#74c476", "#9ecae1", "#fdd835",
        "#fdae6b", "#9edae5", "#c7e9c0", "#fcbba1"
    ]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.3)

    if xlim is not None:
        ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


plot_irregularity_boxplots(
    irregularity_results,
    population_order=["GPe", "STN", "GPi", "TH", "PYR", "FSI", "dStr", "iStr"],
    xlim=(0, 2.5),
)

# %%
# 3. PSD GPi
# extract spike times from all neurons in nucleus
gpi_spike_times = extract_population_spike_times(V4, dt_ms=1.0, spike_height=0.0, refractory_ms=2.0)

# calculate rate by computing average spikes per bin
t_rate, gpi_rate = population_rate_from_spike_times(
    gpi_spike_times,
    tmax_ms= tmax,
    bin_ms= 1.0,
    n_neurons=4
)

# smoothed rate
gpi_rate_smooth = smooth_rate(gpi_rate, sigma_ms=2.0, bin_ms=1.0)

# Welch PSD
freqs, psd = welch_psd(
    gpi_rate_smooth, #check whether rate smoothed or not is better
    dt_ms=1.0,
    nperseg=512,
    noverlap=256
)
# plotted
plt.figure(figsize=(6,4))
plt.plot(freqs, psd)
plt.xlim(0, 50)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("GPi population-rate PSD (Welch)")
plt.show()

plt.figure(figsize=(8,3))
plt.plot(t_rate, gpi_rate, label="raw population rate")
plt.plot(t_rate, gpi_rate_smooth, label="smoothed population rate")
plt.xlabel("Time (ms)")
plt.ylabel("Rate (Hz)")
plt.title("GPi population rate")
plt.legend()
plt.show()








# %%
