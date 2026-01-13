
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
import diffrax as diffrax
import matplotlib.pyplot as plt

print(jax.devices())

#%%
# define neuron parameters 
# parameters in a dict (PyTree)
params = {
    "Cm": 1.0,
    "gl": 0.1,  "El": -65.0,
    "gna": 120.0, "Ena": 55.0,
    "gk": 30.0,  "Ek": -80.0,
    "gt": 0.5,
    "gca": 0.15,
    "gahp": 10.0,
    "Eca": 0.0,
    "k1": 10.0,
    "kca": 15.0,
}

# gating variable functions
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


# defining the functions for ODEterm
def gpe_rhs(t, y, args):

    V, N3, H3, R3, CA3 = y

    params = args

    Cm   = params["Cm"]
    gl   = params["gl"];  El   = params["El"]
    gna  = params["gna"]; Ena  = params["Ena"]
    gk   = params["gk"];  Ek   = params["Ek"]
    gt   = params["gt"];  Eca  = params["Eca"]
    gca  = params["gca"]
    gahp = params["gahp"]
    k1   = params["k1"]
    kca  = params["kca"]

    # gating steady states & taus from V
    m3  = gpe_minf(V)
    n3  = gpe_ninf(V)
    h3  = gpe_hinf(V)
    a3  = gpe_ainf(V)
    s3  = gpe_sinf(V)
    r3  = gpe_rinf(V)
    tn3 = gpe_taun(V)
    th3 = gpe_tauh(V)
    tr3 = 30.0

    # currents
    Il   = gl  * (V - El)
    Ik3  = gk  * (N3**4)     * (V - Ek)
    Ina3 = gna * (m3**3) * H3   * (V - Ena)
    It3  = gt  * (a3**3) * R3   * (V - Eca)
    Ica3 = gca * (s3**2)       * (V - Eca)
    Iahp3 = gahp * (V - Ek) * (CA3 / (CA3 + k1))

    Iappgpe = 9.5

    dVdt   = (-Il - Ik3 - Ina3 - It3 - Ica3 - Iahp3 + Iappgpe) / Cm
    dN3dt  = 0.1  * (n3 - N3) / tn3
    dH3dt  = 0.05 * (h3 - H3) / th3
    dR3dt  = 1.0  * (r3 - R3) / tr3
    dCA3dt = 1e-4 * (-Ica3 - It3 - kca * CA3)

    return jnp.array([dVdt, dN3dt, dH3dt, dR3dt, dCA3dt])

# time scale
tmax = 10000.0
dt = 0.1
ts = jnp.arange(0.0, tmax, dt)

# initial gating values from v0
v0 = -60
V_init  = v0
N_init  = gpe_ninf(V_init)
H_init  = gpe_hinf(V_init)
R_init  = gpe_rinf(V_init)
CA_init = 0.0

y0 = jnp.array([V_init, N_init, H_init, R_init, CA_init])


@jax.jit
def simulate_gpe(y0, params):

    term = diffrax.ODETerm(gpe_rhs)
    solver = diffrax.Tsit5()

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=ts[0],
        t1=tmax,
        dt0=dt,                         # initial step size
        max_steps=1000000,
        y0=y0,
        args=params,
        saveat=diffrax.SaveAt(ts=ts),   # save at our time grid 
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
        progress_meter=diffrax.TextProgressMeter(minimum_increase=0.05)
    )


    # sol.ys has shape (len(ts), 5)
    return ts, sol.ys

#run simulation
ts, ys = simulate_gpe(y0, params)
V = ys[:, 0]

#%%
# plot to check
plt.plot(ts, V)
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()


# %%
