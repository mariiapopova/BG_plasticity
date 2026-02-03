
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

#jax.config.update("jax_enable_x64", True)
print(jax.devices())

#%%
# define neuron parameters 
#number of neurons per nucleus
n_th = 4
n_stn = 4
n_gpe = 4
n_gpi = 4

# parameters in a dict (PyTree)
params = {
    # membrane params
    # in order of TH, STN, GPe, GPi
    "Cm": 1.0,
    "gl": jnp.array([0.05, 2.25, 0.1, 0.1]),  "El": jnp.array([-70, -60.0, -65.0, -67.0]),
    "gna": jnp.array([3, 37, 120, 120]), "Ena": jnp.array([50, 55, 55, 45]),
    "gk": jnp.array([5, 45, 30, 30]),  "Ek": jnp.array([-75, -80, -80, -95]),
    "gt": jnp.array([5, 0.5, 0.5, 0.5]), "Et":0,
    "gca": jnp.array([0, 2, 0.15, 0.15]), "Eca": jnp.array([0, 140, 120, 120]),
    "gahp": jnp.array([0, 10, 10, 10]),
    "k1": jnp.array([0, 20, 10, 10]),
    "kca": jnp.array([0, 15, 15, 15]),
    
    # synapse params (Rubin, 2004)
    # in order of Igith, Igesn,Isnge, Igegi, Isngi, Igege
    "A": jnp.array([2.0 , 3.0 , 2.0, 3.0, 2.0, 3.0]),
    "B": jnp.array([0.08, 0.1, 0.04, 0.1, 0.08, 0.1]),
    "the": jnp.array([20, 30, 20, 30, 0, 30]),
    "gsyn": jnp.array([0.06, 0.9, 0.3, 1, 0.3, 1]),
    "Esyn": jnp.array([-85, -100, 0, -100, 0, -100]),
    "tau": 5, "gpeak1": 0.3, "gpeak": 0.43,

    # connectivity matrix
    # 1 : 1 connectivity
    "w_gpe_stn": jnp.eye(n_stn, n_gpe),
    "w_stn_gpe": jnp.eye(n_gpe, n_stn),
    "w_gpi_th":  jnp.eye(n_th, n_gpi),
    "w_gpe_gpi": jnp.eye(n_gpi, n_gpe),
    "w_stn_gpi": jnp.eye(n_gpi, n_stn),
    # 2 : 1 connectivity for self-inhibtion
    "w_gpe_gpe": jnp.array([
        [0,1,1,0],
        [0,0,1,1],
        [1,0,0,1],
        [1,1,0,0],], dtype=jnp.float32)
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

# synapses
def Hinf(V, theta):
    return 1/(1+jnp.exp(-(V - theta)/2))

# defining the functions for ODEterm
def gpe_rhs(t, y, args):
    params = args

    # TH
    V1  = y["V1_th"]
    H1  = y["H1_th"]
    R1  = y["R1_th"]
    S1  = y["S1_th"]

    # STN
    V2  = y["V2_stn"]   
    N2  = y["N2_stn"]
    H2  = y["H2_stn"]
    R2  = y["R2_stn"]
    C2  = y["C2_stn"]
    CA2 = y["CA2_stn"]
    S2  = y["S2_stn"]

    # GPe
    V3  = y["V3_gpe"]   
    N3  = y["N3_gpe"]
    H3  = y["H3_gpe"]
    R3  = y["R3_gpe"]
    CA3 = y["CA3_gpe"]
    S3_1  = y["S3_1_gpe"]
    S3_2  = y["S3_2_gpe"]

    #GPi
    V4  = y["V4_gpi"]   
    N4  = y["N4_gpi"]
    H4  = y["H4_gpi"]
    R4  = y["R4_gpi"]
    CA4 = y["CA4_gpi"]
    S4_1  = y["S4_1_gpi"]   
    S4_2  = y["S4_2_gpi"]

    # STR direct and indirect
    V5d  = y["V5d_str"]   
    V5i  = y["V5i_str"]  
    N5  = y["N2_stn"]
    H5  = y["H2_stn"]
    R5  = y["R2_stn"]
    C5  = y["C2_stn"]
    CA5 = y["CA2_stn"]
    S5  = y["S2_stn"]

    Cm   = params["Cm"]
    gl   = params["gl"];  El   = params["El"]
    gna  = params["gna"]; Ena  = params["Ena"]
    gk   = params["gk"];  Ek   = params["Ek"]
    gt   = params["gt"];  Et  = params["Et"]
    gca  = params["gca"]; Eca  = params["Eca"]
    gca  = params["gca"]
    gahp = params["gahp"]
    k1   = params["k1"]
    kca  = params["kca"]
    A    = params["A"]
    B    = params["B"]
    the  = params["the"]
    gsyn = params["gsyn"]
    Esyn = params["Esyn"]
    w_gpe_stn = params["w_gpe_stn"]
    w_stn_gpe = params["w_stn_gpe"] 
    w_gpi_th = params["w_gpi_th"]
    w_gpe_gpi = params["w_gpe_gpi"]
    w_stn_gpi = params["w_stn_gpi"]
    w_gpe_gpe = params["w_gpe_gpe"] 


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

    #thalamic cell currents
    Il1  = gl[0]  * (V1 - El[0])
    Ina1 = gna[0] * (m1**3) * H1 * (V1 - Ena[0])
    Ik1  = gk[0]  * ((0.75 * (1 - H1))**4) * (V1 - Ek[0])  # misspelled in So paper
    It1  = gt[0]  * (p1**2) * R1 * (V1 - Et)

    # as Istim from og script (change later to input from motor cortex)
    Iapp_th = 1.0

    # ion currents stn
    Il2   = gl[1]   * (V2 - El[1])
    Ik2   = gk[1]   * (N2**4) * (V2 - Ek[1])
    Ina2  = gna[1]  * (m2**3) * H2 * (V2 - Ena[1])
    It2   = gt[1]   * (a2**3) * (b2**2) * (V2 - Eca[1])   
    Ica2  = gca[1]  * (c2**2)           * (V2 - Eca[1])
    Iahp2 = gahp[1] * (V2 - Ek[1]) * (CA2 / (CA2 + k1[1]))

    # applied current stn    
    Iappstn = 25.0

    # currents gpe
    Il3  = gl[2]  * (V3 - El[2])
    Ik3  = gk[2]  * (N3**4)     * (V3 - Ek[2])
    Ina3 = gna[2] * (m3**3) * H3   * (V3 - Ena[2])
    It3  = gt[2]  * (a3**3) * R3   * (V3 - Eca[2])
    Ica3 = gca[2] * (s3**2)       * (V3 - Eca[2])
    Iahp3 = gahp[2] * (V3 - Ek[2]) * (CA3 / (CA3 + k1[2]))

    # applied current gpe
    Iappgpe = 2.0

    # currents gpi
    Il4  = gl[3]  * (V4 - El[3])
    Ik4  = gk[3]  * (N4**4)     * (V4 - Ek[3])
    Ina4 = gna[3] * (m4**3) * H4   * (V4 - Ena[3])
    It4  = gt[3]  * (a4**3) * R4   * (V4 - Eca[3])
    Ica4 = gca[3] * (s4**2)       * (V4 - Eca[3])
    Iahp4 = gahp[3] * (V4 - Ek[3]) * (CA4 / (CA4 + k1[3]))

    # applied current gpi
    Iappgpi = 5.0


    # synapses with connectivity matrices

    # presynaptic activation
    H_gpe = Hinf(V3, theta=30.0)
    H_stn = Hinf(V2, theta=20.0)
    H_gpi = Hinf(V4, theta=20.0)

    # GPe to STN: 1 GPe to 1 STN
    drive_stn = w_gpe_stn @ H_gpe
    # STN to GPe: 1 STN to 1 GPe
    drive_gpe1 = w_stn_gpe @ H_stn
    # GPe - GPi: 1 GPe to 1 GPi
    drive_gpi1 = w_gpe_gpi @ H_gpe
    # STN to GPi: 1 STN to 1 GPi
    drive_gpi2 = w_stn_gpi @ H_stn
    # GPi - TH: 1 GPe to 1 TH
    drive_th = w_gpi_th @ H_gpi
    # GPe - GPe: 1 GPe to 1 GPe
    drive_gpe2 = w_gpe_gpe @ H_gpe


    # differential equations synapses
    dS1dt = A[0] * (1 - S1) * drive_th - B[0] * S1
    dS2dt = A[1] * (1 - S2) * drive_stn - B[1] * S2
    dS31dt = A[2] * (1 - S3_1) * drive_gpe1  - B[2] * S3_1
    dS32dt = A[5] * (1 - S3_2) * drive_gpe2  - B[5] * S3_2
    dS41dt = A[3] * (1 - S4_1) * drive_gpi1  - B[3] * S4_1
    dS42dt = A[4] * (1 - S4_2) * drive_gpi2  - B[4] * S4_2

    # synaptic currents using those gating variables
    Igesn = 0.5 * (gsyn[1] * (V2 - Esyn[1]) *  S2)
    Isnge = 0.5 * (gsyn[2] * (V3 - Esyn[2]) *  S3_1)
    Isngi = 0.5 * (gsyn[3] * (V4 - Esyn[3]) *  S4_2)
    Igegi = 0.5 * (gsyn[4] * (V4 - Esyn[4]) *  S4_1)
    Igith = 0.5 * (gsyn[0] * (V1 - Esyn[0]) *  S1)
    Igege = 0.5 * (gsyn[5] * (V3 - Esyn[5]) *  S3_2)


    # differential equations th
    dV1dt = (-Il1 - Ik1 - Ina1 - It1 - Igith + Iapp_th) / Cm
    dH1dt   = (h1 - H1) / th1
    dR1dt   = (r1 - R1) / tr1

    # differential equations stn
    dV2dt   = (-Il2 - Ik2 - Ina2 - It2 - Ica2 - Iahp2 - Igesn + Iappstn) / Cm
    dN2dt   = 0.75 * (n2 - N2) / tn2
    dH2dt   = 0.75 * (h2 - H2) / th2
    dR2dt   = 0.2 * (r2 - R2) / tr2
    dCA2dt  = 3.75 * 1e-5 * (-Ica2 - It2 - kca[0] * CA2)
    dC2dt   = 0.08 * (c2 - C2) / tc2

    # differential equations gpe
    dV3dt  = (-Il3 - Ik3 - Ina3 - It3 - Ica3 - Iahp3 - Isnge - Igege + Iappgpe) / Cm
    dN3dt  = 0.1  * (n3 - N3) / tn3
    dH3dt  = 0.05 * (h3 - H3) / th3
    dR3dt  = 1.0  * (r3 - R3) / tr3
    dCA3dt = 1e-4 * (-Ica3 - It3 - kca[1] * CA3)

    # differential equations gpi
    dV4dt  = (-Il4 - Ik4 - Ina4 - It4 - Ica4 - Iahp4 - Igegi - Isngi + Iappgpi) / Cm
    dN4dt  = 0.1  * (n4 - N4) / tn4
    dH4dt  = 0.05 * (h4 - H4) / th4
    dR4dt  = 1.0  * (r4 - R4) / tr4
    dCA4dt = 1e-4 * (-Ica4 - It4 - kca[2] * CA4)
    

    return {
        "V1_th":  dV1dt,
        "H1_th":  dH1dt,
        "R1_th":  dR1dt,
        "S1_th":  dS1dt,

        "V2_stn":  dV2dt,
        "N2_stn":  dN2dt,
        "H2_stn":  dH2dt,
        "R2_stn":  dR2dt,
        "C2_stn":  dC2dt,
        "CA2_stn": dCA2dt,
        "S2_stn":  dS2dt,

        "V3_gpe":  dV3dt,
        "N3_gpe":  dN3dt,
        "H3_gpe":  dH3dt,
        "R3_gpe":  dR3dt,
        "CA3_gpe": dCA3dt,
        "S3_1_gpe":  dS31dt,
        "S3_2_gpe":  dS32dt,

        "V4_gpi":  dV4dt,
        "N4_gpi":  dN4dt,
        "H4_gpi":  dH4dt,
        "R4_gpi":  dR4dt,
        "CA4_gpi": dCA4dt,
        "S4_1_gpi":  dS41dt,
        "S4_2_gpi":  dS42dt,
    }


# initial values and params
V1_init = -90
V2_init = -60
V3_init = -60
V4_init = -60

y0 = {
    "V1_th":  jnp.full((n_th,), V1_init),
    "H1_th":  th_hinf(V1_init)  * jnp.ones((n_th,)),
    "R1_th":  th_rinf(V1_init)  * jnp.ones((n_th,)),
    "S1_th":  jnp.full((n_th,), 0.1),

    "V2_stn":  jnp.full((n_stn,), V2_init),
    "N2_stn":  stn_ninf(V2_init)  * jnp.ones((n_stn,)),
    "H2_stn":  stn_hinf(V2_init)  * jnp.ones((n_stn,)),
    "R2_stn":  stn_rinf(V2_init)  * jnp.ones((n_stn,)),
    "C2_stn":  stn_cinf(V2_init)  * jnp.ones((n_stn,)),
    "CA2_stn": jnp.full((n_stn,), 0.1),
    "S2_stn":  jnp.full((n_stn,), 0.1),

    "V3_gpe":  jnp.full((n_gpe,), V3_init),
    "N3_gpe":  gpe_ninf(V3_init)  * jnp.ones((n_gpe,)),  
    "H3_gpe":  gpe_hinf(V3_init)  * jnp.ones((n_gpe,)),
    "R3_gpe":  gpe_rinf(V3_init)  * jnp.ones((n_gpe,)),
    "CA3_gpe": jnp.full((n_gpe,), 0.1),
    "S3_1_gpe":  jnp.full((n_gpe,), 0.1),
    "S3_2_gpe":  jnp.full((n_gpe,), 0.1),

    "V4_gpi":  jnp.full((n_gpi,), V4_init),
    "N4_gpi":  gpe_ninf(V4_init)  * jnp.ones((n_gpi,)),  
    "H4_gpi":  gpe_hinf(V4_init)  * jnp.ones((n_gpi,)),
    "R4_gpi":  gpe_rinf(V4_init)  * jnp.ones((n_gpi,)),
    "CA4_gpi": jnp.full((n_gpi,), 0.1),
    "S4_1_gpi":  jnp.full((n_gpi,), 0.1),
    "S4_2_gpi":  jnp.full((n_gpi,), 0.1),
}
#%% chunked diffrax solver version

def run_chunk(y0, params, t0, t1, dt0, dt_save ):
    ts = jnp.arange(t0, t1 + 1e-9, dt_save)

    term = diffrax.ODETerm(gpe_rhs)
    solver = diffrax.Tsit5()

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,                        # internal dt = dt used by step solver                    
        max_steps=1000000,
        y0=y0,
        args=params,
        saveat=diffrax.SaveAt(ts=ts),   # external dt = saved values
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7)
    )

    # saving last state of chunk 
    yT = jax.tree.map(lambda a: a[-1], sol.ys)  
    return ts, sol.ys, yT


#run_chunk_jit = jax.jit(run_chunk, static_argnames=("dt0", "dt_save"))


def simulate_chunked(y0, params, tmax, chunk_size, dt0=0.1, dt_save=1.0):

    # starting variables
    t0 = 0.0
    y = y0

    # what we want to save, e.g. only V2 (of stn)
    # saved at ts (time scale with dt_save)
    all_ts = []
    all_V1 = []
    all_V2 = []
    all_V3 = []
    all_V4 = []

    # looping across chunks 
    # using jit compilation inside each chunk (run_chunk)
    while t0 < tmax:
        t1 = jnp.minimum(t0 + chunk_size, tmax)
        ts, ys, y = run_chunk(y, params, t0, t1, dt0, dt_save)

        all_ts.append(ts)
        all_V1.append(ys["V1_th"])
        all_V2.append(ys["V2_stn"])   # shape (len(ts), n_th)
        all_V3.append(ys["V3_gpe"])
        all_V4.append(ys["V4_gpi"])
        t0 = float(t1)

    return jnp.concatenate(all_ts), jnp.concatenate(all_V1), jnp.concatenate(all_V2), jnp.concatenate(all_V3), jnp.concatenate(all_V4)



#%% run simulation
tmax = 1000.0
chunk_size = 100.0      # 1 second per chunk
dt0 = 0.1
dt_save = 1.0            # save every 1 ms

ts, V1, V2, V3, V4 = simulate_chunked(y0, params, tmax, chunk_size, dt0, dt_save)

#%%
# plot to check
plt.plot(ts, V1[:,3])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()

# plot to check
plt.plot(ts, V2[:,2])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()

# plot to check
plt.plot(ts, V3[:,3])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()


# plot to check
plt.plot(ts, V4[:,1])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()
# %%
