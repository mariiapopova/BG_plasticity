
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

print(jax.devices())

#%%
# define neuron parameters 
# parameters in a dict (PyTree)
params = {
    # membrane params
    # in order of STN, GPe
    "Cm": 1.0,
    "gl": jnp.array([2.25, 0.1]),  "El": jnp.array([-60.0, -65.0]),
    "gna": jnp.array([37, 120]), "Ena": jnp.array([55, 55]),
    "gk": jnp.array([45, 30]),  "Ek": jnp.array([-80, -80]),
    "gt": jnp.array([0.5, 0.5]),
    "gca": jnp.array([2, 0.15]), "Eca": jnp.array([140, 120]),
    "gahp": jnp.array([20, 10]),
    "k1": jnp.array([15, 10]),
    "kca": jnp.array([22.5, 15]),
    
    # synapse params
    # in order of Igesn,Isnge
    "A": jnp.array([0.2, 0.2]),
    "B": jnp.array([0.1, 0.04]),
    "the": jnp.array([30, 20]),
    "gsyn": jnp.array([1, 0.3]),
    "Esyn": jnp.array([-85, 0]),
    "tau": 5, "gpeak1": 0.3, "gpeak": 0.43,

}


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


# GPe
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

# synapses
def Hinf(V):
    return 1/(1+jnp.exp(-(V+57)/2))

# defining the functions for ODEterm
def gpe_rhs(t, y, args):

    V2, N2, H2, R2, C2, CA2, S2, \
    V3, N3, H3, R3, CA3, S3 = y

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
    A    = params["A"]
    B    = params["B"]
    the  = params["the"]
    gsyn = params["gsyn"]
    Esyn = params["Esyn"]

    # gating steady states & taus from V
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


    # ion currents stn
    Il2   = gl[0]   * (V2 - El[0])
    Ik2   = gk[0]   * (N2**4) * (V2 - Ek[0])
    Ina2  = gna[0]  * (m2**3) * H2 * (V2 - Ena[0])
    It2   = gt[0]   * (a2**3) * (b2**2) * (V2 - Eca[0])   
    Ica2  = gca[0]  * (c2**2)           * (V2 - Eca[0])
    Iahp2 = gahp[0] * (V2 - Ek[0]) * (CA2 / (CA2 + k1[0]))

    # applied current stn    
    Iappstn = 35.0

    # synaptic current from GPe to STN
    Igesn=0.5*(gsyn[0]*(V2-Esyn[0])*(A[0]*S2))

    # differential equations
    dV2dt   = (-Il2 - Ik2 - Ina2 - It2 - Ica2 - Iahp2 - Igesn + Iappstn) / Cm
    dN2dt   = 0.75 * (n2 - N2) / tn2
    dH2dt   = 0.75 * (h2 - H2) / th2
    dR2dt   = 0.2 * (r2 - R2) / tr2
    dCA2dt  = 3.75 * 1e-5 * (-Ica2 - It2 - kca[1] * CA2)
    dC2dt   = 0.08 * (c2 - C2) / tc2
    dS2dt   = A[0]*(1-S2)*Hinf(V2-the[0])-B[0]*S2

    # currents gpe
    Il   = gl[1]  * (V3 - El[1])
    Ik3  = gk[1]  * (N3**4)     * (V3 - Ek[1])
    Ina3 = gna[1] * (m3**3) * H3   * (V3 - Ena[1])
    It3  = gt[1]  * (a3**3) * R3   * (V3 - Eca[1])
    Ica3 = gca[1] * (s3**2)       * (V3 - Eca[1])
    Iahp3 = gahp[1] * (V3 - Ek[1]) * (CA3 / (CA3 + k1[1]))

    # applied current gpe
    Iappgpe = 9.5

    # synaptic current from STN to GPe
    Isnge=0.5*(gsyn[1]*(V3-Esyn[1])*(A[1]*S3))

    # differential equations
    dV3dt  = (-Il - Ik3 - Ina3 - It3 - Ica3 - Iahp3 - Isnge + Iappgpe) / Cm
    dN3dt  = 0.1  * (n3 - N3) / tn3
    dH3dt  = 0.05 * (h3 - H3) / th3
    dR3dt  = 1.0  * (r3 - R3) / tr3
    dCA3dt = 1e-4 * (-Ica3 - It3 - kca[1] * CA3)
    dS3dt  = A[1]*(1-S3)*Hinf(V3-the[1])-B[1]*S3

    return jnp.array([dV2dt, dN2dt, dH2dt, dR2dt, dCA2dt,dC2dt, dS2dt,
                      dV3dt, dN3dt, dH3dt, dR3dt, dCA3dt, dS3dt])


# initial values and params
V2_init = -60
V3_init = -60

N2_init  = stn_ninf(V2_init)
H2_init  = stn_hinf(V2_init)
R2_init  = stn_rinf(V2_init)
CA2_init = 0.1
C2_init  = stn_cinf(V2_init)
S2_init  = 0.1 

N3_init  = stn_ninf(V3_init)
H3_init  = stn_hinf(V3_init)
R3_init  = stn_rinf(V3_init)
CA3_init = 0.1
S3_init  = 0.1 

y0 = jnp.array([V2_init, N2_init, H2_init, R2_init, CA2_init, C2_init, S2_init,
                V3_init, N3_init, H3_init, R3_init, CA3_init, S3_init])



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


    # sol.ys has shape (len(ts), 13)
    return ts, sol.ys

#%% 
# time scale
tmax = 100000.0
dt = 0.1
ts = jnp.arange(0.0, tmax, dt)

#run
ts, ys = simulate_gpe(y0, params)
V2 = ys[:, 0]
V3 = ys[:, 7]

#%%
# plot to check
plt.plot(ts, V2)
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()

# plot to check
plt.plot(ts, V3)
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()

# %%
