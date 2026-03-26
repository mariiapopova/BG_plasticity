import numpy as np
import jax
from jax import jit
from jax import random
import jax.numpy as jnp

# gating variable functions
#STN
def stn_taur(V):
    return 7.1+17.5/(1+jnp.exp(-(V-68)/-2.2))

def stn_taun(V):
    return 1+100/(1+jnp.exp(-(V+80)/-26))

def stn_tauh(V):
    return 1+500/(1+jnp.exp(-(V+57)/-3))

def stn_tauc(V):
    return 1+10/(1+jnp.exp((V+80)/26))

def stn_rinf(V):
    return 1/(1+jnp.exp((V+67)/2))

def stn_ninf(V):
    return 1/(1+jnp.exp(-(V+32)/8.0))

def stn_minf(V):
    return 1/(1+jnp.exp(-(V+30)/15))

def stn_hinf(V):
    return 1/(1+jnp.exp((V+39)/3.1))

def stn_cinf(V):
    return 1/(1+jnp.exp(-(V+20)/8))

def stn_binf(R):
    return 1/(1+jnp.exp(-(R-0.4)/0.1))-1/(1+jnp.exp(0.4/0.1))

def stn_ainf(V):
    return 1/(1+jnp.exp(-(V+63)/7.8))

# GPe & GPi
def gpe_ainf(V):
    return 1/(1+jnp.exp(-(V+57)/2))

def gpe_hinf(V):
    return 1/(1+jnp.exp((V+58)/12))

def gpe_minf(V):
    return 1/(1+jnp.exp(-(V+37)/10))

def gpe_ninf(V):
    return 1/(1+jnp.exp(-(V+50)/14))

def gpe_rinf(V):
    return 1/(1+jnp.exp((V+70)/2))

def gpe_sinf(V):
    return 1/(1+jnp.exp(-(V+35)/2))

def gpe_tauh(V):
    return 0.05+0.27/(1+jnp.exp(-(V+40)/-12))

def gpe_taun(V):
    return 0.05+0.27/(1+jnp.exp(-(V+40)/-12))

#TH
def th_taur(V):
    return 0.15*(28 + jnp.exp(-(V+25)/10.5))

def th_tauh(V):
    a = 0.128*jnp.exp(-(V+46)/18)
    b = 4/(1+jnp.exp(-(V+23)/5))
    return 1/(a+b)

def th_rinf(V):
    return 1/(1+jnp.exp((V+84)/4))

def th_pinf(V):
    return 1/(1+jnp.exp(-(V+60)/6.2))

def th_minf(V):
    return 1/(1+jnp.exp(-(V+37)/7))

def th_hinf(V):
    return 1/(1+jnp.exp((V+41)/4))

# striatum
def str_Ggaba(V):
    return 2.0 * (1.0 + jnp.tanh(V / 4.0))

def str_betap(V):
    return (-3.209e-4 * (30.0 + V)) / (1.0 - jnp.exp((30.0 + V) / 9.0))

def str_betan(V):
    return 0.5 * jnp.exp((-57.0 - V) / 40.0)

def str_betam(V):
    return 0.28 * (V + 27.0) / (jnp.exp((27.0 + V) / 5.0) - 1.0)

def str_betah(V):
    return 4.0 / (1.0 + jnp.exp((-27.0 - V) / 5.0))

def str_alphap(V):
    return (3.209e-4 * (30.0 + V)) / (1.0 - jnp.exp((-30.0 - V) / 9.0))

def str_alphan(V):
    return (0.032 * (52.0 + V)) / (1.0 - jnp.exp((-52.0 - V) / 5.0))

def str_alpham(V):
    return (0.32 * (54.0 + V)) / (1.0 - jnp.exp((-54.0 - V) / 4.0))

def str_alphah(V):
    return 0.128 * jnp.exp((-50.0 - V) / 18.0)

# cortex
# Stable version of x / (exp(x/y) - 1)
def vtrap(x, y):
    small = jnp.abs(x / y) < 1e-6
    return jnp.where(small,y * (1 - (x / y) / 2.0),x / (jnp.exp(x / y) - 1.0)) 

def ctx_minf(V, vtraub):
    v2 = V - vtraub # convert to traub convention
    a = 0.32 * vtrap(13.0 - v2, 4.0)
    b = 0.28 * vtrap(v2 - 40.0, 5.0)
    return a/(a+b)

def ctx_taum(V, vtraub):
    v2 = V - vtraub 
    a = 0.32 * vtrap(13.0 - v2, 4.0)
    b = 0.28 * vtrap(v2 - 40.0, 5.0)
    return 1 / (a + b) # without Q10 factor

def ctx_ninf(V, vtraub):
    v2 = V - vtraub 
    a = 0.032 * vtrap(15-v2, 5)
    b = 0.5 * jnp.exp((10-v2)/40)
    return a / (a + b)

def ctx_taun(V, vtraub):
    v2 = V - vtraub 
    a = 0.032 * vtrap(15-v2, 5)
    b = 0.5 * jnp.exp((10-v2)/40)
    return 1 / (a + b)

def ctx_hinf(V, vtraub):
    v2 = V - vtraub 
    a = 0.128 * jnp.exp((17-v2)/18)
    b = 4 / (1 + jnp.exp((40-v2)/5))
    return a / (a + b)

def ctx_tauh(V, vtraub):
    v2 = V - vtraub 
    a = 0.128 * jnp.exp((17-v2)/18)
    b = 4 / (1 + jnp.exp((40-v2)/5))
    return 1 / (a + b)

def ctx_minf_m(V):
    return 1 / (1 + jnp.exp(-(V+35)/10))

def ctx_taum_m(V):
    tau_max = 1000
    celsius = 36
    tadj = 2.3 ** ((celsius-36)/10)
    tau_peak = tau_max / tadj
    return tau_peak / (3.3 * jnp.exp((V+35)/20) + 
                   jnp.exp(-(V+35)/20)) 

def ctx_minf_ca(V):
    shift = 2 # voltage shift
    Vm = V + shift
    return 1.0 / (1 + jnp.exp(-(Vm+57)/6.2))

def ctx_hinf_ca(V):
    shift = 2 
    Vm = V + shift
    return 1.0 / (1 + jnp.exp((Vm+81)/4.0))

def ctx_tauh_ca(V):
    shift = 2 
    Vm = V + shift
    phi_h = 3 ** ((36-24)/10) # temperature factor
    return (30.8 + (211.4 + jnp.exp((Vm+113.2)/5)) / (1 + jnp.exp((Vm+84)/3.2)))/phi_h # with temperature factor add: / phi_h

def ca_drive(ica, FARADAY, depth):
    drive = -(10000.0 * ica) / (2.0 * FARADAY * depth)
    # clamp so it cannot "pump inward" when ica is outward
    return jnp.maximum(drive, 0.0)

# synapses
# def Hinf(V, theta):
#     return 1/(1+jnp.exp(-(V - theta)/2))

def Hinf(V, theta):
    return 1/(1+jnp.exp(-(V - theta + 57)/2))


# creating a ring connectivity matrix
def w_matrix(n, k):
    # indices of all neurons
    pre = jnp.arange(n)

    # for each neuron the next k targets
    offsets = jnp.arange(1, k+1)[:, None]          # (k,1)
    targets = (pre + offsets) % n                  # (k,n)

    # initialize matrix
    W = jnp.zeros((n, n), dtype=jnp.float32)

    # replace with ones
    pre_idx = jnp.tile(pre, k)                     # (k*n,)
    post_idx = targets.flatten()                   # (k*n,)

    W = W.at[post_idx, pre_idx].set(1.0)

    return W


def w_matrix_random(key, n, p, k):
    W = jnp.zeros((p, n), dtype=jnp.float32)

    def connect_one_target(i, key):
        idx = jax.random.choice(key, n, shape=(k,), replace=False)
        return W[i].at[idx].set(1.0)

    keys = jax.random.split(key, p)

    W = jax.vmap(connect_one_target)(jnp.arange(p), keys)

    return W

def w_matrix_divergent(key, n, p, k):
    W = jnp.zeros((p, n), dtype=jnp.float32)

    def connect_one_source(i, key):
        idx = jax.random.choice(key, p, shape=(k,), replace=False)
        return W.at[idx, i].set(1.0)

    keys = jax.random.split(key, n)

    W = jax.vmap(connect_one_source)(jnp.arange(n), keys)

    # vmap gives (n, p, n) → we need to combine updates
    W = jnp.sum(W, axis=0)

    return W

# dopamine scaling from CTX to Str
def cD1(DA, AD1=10.0, lambda_str=7.5):
    return AD1 / (1.0 + jnp.exp(-lambda_str * (DA - 1.0)))

# STDP threshold 
def spike_event(V, threshold):
    return (V > threshold).astype(V.dtype)