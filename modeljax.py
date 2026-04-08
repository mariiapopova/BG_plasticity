
#%%
# choose device to do calculations on
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["JAX_LOG_COMPILES"] = "1"

import numpy as np
import scipy
import jax
from jax import jit
from jax import random
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
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
    #pd switch
    "pd": 0,
    # membrane params
    # in order of TH, STN, GPe, GPi
    "Cm": 1.0,
    "gl": jnp.array([0.05, 2.25, 0.1, 0.1]),  "El": jnp.array([-70, -60.0, -65.0, -65.0]),
    "gna": jnp.array([3, 37, 120, 120]), "Ena": jnp.array([50, 55, 55, 55]),
    "gk": jnp.array([5, 45, 30, 30]),  "Ek": jnp.array([-75, -80, -80, -80]),
    "gt": jnp.array([5, 0.5, 0.5, 0.5]), "Et":0,
    "gca": jnp.array([0, 2, 0.15, 0.15]), "Eca": jnp.array([0, 140, 120, 120]),
    "gahp": jnp.array([0, 20, 10, 10]),
    "k1": jnp.array([0, 15, 10, 10]),
    "kca": jnp.array([0, 22.5, 15, 15]),
    
    # synapse params (Rubin, 2004)
    # in order of Igith, Igesn,Isnge, Igege, Igegi, Isngi
    "A": jnp.array([2.0 , 2.0 , 3.0, 2.0, 2.0, 3.0]),
    "B": jnp.array([0.04, 0.04, 0.1, 0.04, 0.04, 0.1]),
    "the": jnp.array([20, 20, 30, 20, 20, 30]),
    "gsyn": jnp.array([0.08, 1, 0.3, 1, 1, 0.3]),
    "Esyn": jnp.array([-85, -85, 0, -85, -85, 0]),
    "tau": 5, "gpeak1": 0.3, "gpeak": 0.43, #where does this go

    # connectivity matrix
    # 1 : 1 connectivity
    #"w_gpe_stn": jnp.eye(n_stn, n_gpe),
    #"w_stn_gpe": jnp.eye(n_gpe, n_stn),
    "w_gpi_th":  jnp.eye(n_th, n_gpi),
    #"w_gpe_gpi": jnp.eye(n_gpi, n_gpe),
    #"w_stn_gpi": jnp.eye(n_gpi, n_stn),
    # 2 : 1 connectivity for self-inhibtion
    "w_stn_gpe": jnp.array([
        [1,1,0,0],
        [0,1,1,0],
        [0,0,1,1],
        [1,0,0,1],], dtype=jnp.float64),
    "w_stn_gpi": jnp.array([
        [1,1,0,0],
        [0,1,1,0],
        [0,0,1,1],
        [1,0,0,1],], dtype=jnp.float64),
    "w_gpe_stn": jnp.array([
        [1,0,0,1],
        [1,1,0,0],
        [0,1,1,0],
        [0,0,1,1],], dtype=jnp.float64),
    "w_gpe_gpi": jnp.array([
        [0,0,1,1],
        [1,0,0,1],
        [1,1,0,0],
        [0,1,1,0],], dtype=jnp.float64),
    "w_gpe_gpe": jnp.array([
        [0,0,1,1],
        [1,0,0,1],
        [1,1,0,0],
        [0,1,1,0],], dtype=jnp.float64)
}

params_pd = {
    #pd switch
    "pd": 1,
    # membrane params
    # in order of TH, STN, GPe, GPi
    "Cm": 1.0,
    "gl": jnp.array([0.05, 2.25, 0.1, 0.1]),  "El": jnp.array([-70, -60.0, -65.0, -65.0]),
    "gna": jnp.array([3, 37, 120, 120]), "Ena": jnp.array([50, 55, 55, 55]),
    "gk": jnp.array([5, 45, 30, 30]),  "Ek": jnp.array([-75, -80, -80, -80]),
    "gt": jnp.array([5, 0.5, 0.5, 0.5]), "Et":0,
    "gca": jnp.array([0, 2, 0.15, 0.15]), "Eca": jnp.array([0, 140, 120, 120]),
    "gahp": jnp.array([0, 20, 10, 10]),
    "k1": jnp.array([0, 15, 10, 10]),
    "kca": jnp.array([0, 22.5, 15, 15]),
    
    # synapse params (Rubin, 2004)
    # in order of Igith, Igesn,Isnge, Igege, Igegi, Isngi
    "A": jnp.array([2.0 , 2.0 , 3.0, 2.0, 2.0, 3.0]),
    "B": jnp.array([0.04, 0.04, 0.1, 0.04, 0.04, 0.1]),
    "the": jnp.array([20, 20, 30, 20, 20, 30]),
    "gsyn": jnp.array([0.08, 1, 0.3, 1, 1, 0.3]),
    "Esyn": jnp.array([-85, -85, 0, -85, -85, 0]),
    "tau": 5, "gpeak1": 0.3, "gpeak": 0.43, #where does this go

    # connectivity matrix
    # 1 : 1 connectivity
    #"w_gpe_stn": jnp.eye(n_stn, n_gpe),
    #"w_stn_gpe": jnp.eye(n_gpe, n_stn),
    "w_gpi_th":  jnp.eye(n_th, n_gpi),
    #"w_gpe_gpi": jnp.eye(n_gpi, n_gpe),
    #"w_stn_gpi": jnp.eye(n_gpi, n_stn),
    # 2 : 1 connectivity for self-inhibtion
    "w_stn_gpe": jnp.array([
        [1,1,0,0],
        [0,1,1,0],
        [0,0,1,1],
        [1,0,0,1],], dtype=jnp.float64),
    "w_stn_gpi": jnp.array([
        [1,1,0,0],
        [0,1,1,0],
        [0,0,1,1],
        [1,0,0,1],], dtype=jnp.float64),
    "w_gpe_stn": jnp.array([
        [1,0,0,1],
        [1,1,0,0],
        [0,1,1,0],
        [0,0,1,1],], dtype=jnp.float64),
    "w_gpe_gpi": jnp.array([
        [0,0,1,1],
        [1,0,0,1],
        [1,1,0,0],
        [0,1,1,0],], dtype=jnp.float64),
    "w_gpe_gpe": jnp.array([
        [0,0,1,1],
        [1,0,0,1],
        [1,1,0,0],
        [0,1,1,0],], dtype=jnp.float64)
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
    return 1/(1+jnp.exp(-(V - theta + 57)/2))

# def Hinf1(V):
#     return 1/(1+jnp.exp(-(V + 57)/2))

def stdp_derivatives(W, x_pre, x_post, H_pre, H_post):

    tau_pre  = 12.0
    tau_post = 27.5
    A_plus  = 0.002
    A_minus = 0.002 * 1.1

    dx_pre  = -x_pre/tau_pre  + H_pre
    dx_post = -x_post/tau_post + H_post

    dW = (
        A_plus  * jnp.outer(H_post, x_pre)
        - A_minus * jnp.outer(x_post, H_pre)
    )

    return dW, dx_pre, dx_post

# defining the functions for ODEterm
@jax.jit
def gpe_rhs(t, y, args):
    params = args
    # TH
    V1  = y["V1_th"]
    H1  = y["H1_th"]
    R1  = y["R1_th"]

    # STN
    V2  = y["V2_stn"]   
    N2  = y["N2_stn"]
    H2  = y["H2_stn"]
    R2  = y["R2_stn"] #seems like we don't need this
    C2  = y["C2_stn"]
    CA2 = y["CA2_stn"]
    S2  = y["S2_stn"]
    Z2  = y["Z2_stn"]

    # GPe
    V3  = y["V3_gpe"]   
    N3  = y["N3_gpe"]
    H3  = y["H3_gpe"]
    R3  = y["R3_gpe"]
    CA3 = y["CA3_gpe"]
    S3  = y["S3_gpe"]

    #GPi
    V4  = y["V4_gpi"]   
    N4  = y["N4_gpi"]
    H4  = y["H4_gpi"]
    R4  = y["R4_gpi"]
    CA4 = y["CA4_gpi"]
    S4  = y["S4_gpi"]   
    Z4  = y["Z4_gpi"]

    # # STR direct and indirect
    # V5d  = y["V5d_str"]   
    # V5i  = y["V5i_str"]  
    # N5  = y["N2_stn"]
    # H5  = y["H2_stn"]
    # R5  = y["R2_stn"]
    # C5  = y["C2_stn"]
    # CA5 = y["CA2_stn"]
    # S5  = y["S2_stn"]

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
    gpeak = params["gpeak"]
    gpeak1 = params["gpeak1"]
    tau = params["tau"]

    pd = params["pd"]


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
    Iapp_th = 1.7

    # ion currents stn
    Il2   = gl[1]   * (V2 - El[1])
    Ik2   = gk[1]   * (N2**4) * (V2 - Ek[1])
    Ina2  = gna[1]  * (m2**3) * H2 * (V2 - Ena[1])
    It2   = gt[1]   * (a2**3) * (b2**2) * (V2 - Eca[1])   
    Ica2  = gca[1]  * (c2**2)           * (V2 - Eca[1])
    Iahp2 = gahp[1] * (V2 - Ek[1]) * (CA2 / (CA2 + k1[1]))

    # applied current stn    
    Iappstn = 35.0 #35

    # currents gpe
    Il3  = gl[2]  * (V3 - El[2])
    Ik3  = gk[2]  * (N3**4)     * (V3 - Ek[2])
    Ina3 = gna[2] * (m3**3) * H3   * (V3 - Ena[2])
    It3  = gt[2]  * (a3**3) * R3   * (V3 - Eca[2])
    Ica3 = gca[2] * (s3**2)       * (V3 - Eca[2])
    Iahp3 = gahp[2] * (V3 - Ek[2]) * (CA3 / (CA3 + k1[2]))

    # applied current gpe
    Iappgpe = 15-6*pd #15

    # currents gpi
    Il4  = gl[3]  * (V4 - El[3])
    Ik4  = gk[3]  * (N4**4)     * (V4 - Ek[3])
    Ina4 = gna[3] * (m4**3) * H4   * (V4 - Ena[3])
    It4  = gt[3]  * (a4**3) * R4   * (V4 - Eca[3])
    Ica4 = gca[3] * (s4**2)       * (V4 - Eca[3])
    Iahp4 = gahp[3] * (V4 - Ek[3]) * (CA4 / (CA4 + k1[3]))

    # applied current gpi
    Iappgpi = 15 #15


    # synapses with connectivity matrices

    # presynaptic activation
    #H_gpe = Hinf1(V3-20) #why not 57 everywhere
    H_gpe = Hinf(V3, theta=20.0) #why not 57 everywhere
    H_stn = Hinf(V2, theta=30.0)
    H_gpi = Hinf(V4, theta=20.0)

    # # GPe to STN: 1 GPe to 1 STN
    # drive_stn = w_gpe_stn @ H_gpe
    # # STN to GPe: 1 STN to 1 GPe
    #drive_gpe = w_stn_gpe @ H_stn
    # # GPe - GPi: 1 GPe to 1 GPi
    # drive_gpi1 = w_gpe_gpi @ H_gpe
    # # STN to GPi: 1 STN to 1 GPi
    # drive_gpi2 = w_stn_gpi @ H_stn
    # # GPi - TH: 1 GPe to 1 TH
    # drive_th = w_gpi_th @ H_gpi
    # # GPe - GPe: 1 GPe to 1 GPe
    # drive_gpe2 = w_gpe_gpe @ H_gpe


    # differential equations synapses
    u = gpeak1 / (tau * jnp.exp(-1.0)) * H_gpi
    dS4dt = Z4
    dZ4dt = u - (2.0 / tau) * Z4 - (1.0 / tau**2) * S4
    #dS1dt = A[0] * (1 - S1) * H_gpi - B[0] * S1
    dS3dt = A[1] * (1 - S3) * H_gpe - B[1] * S3

    u1 = gpeak / (tau * jnp.exp(-1.0)) * H_stn
    dS2dt = Z2
    dZ2dt = u1 - (2.0 / tau) * Z2 - (1.0 / tau**2) * S2
    #dS31dt = A[2] * (1 - S3_1) * H_stn  - B[2] * S3_1
    # dS32dt = A[3] * (1 - S3_2) * H_gpe  - B[3] * S3_2
    # dS41dt = A[4] * (1 - S4_1) * H_gpe  - B[4] * S4_1
    # dS42dt = A[5] * (1 - S4_2) * H_stn - B[5] * S4_2

    #dw_gpe_stn, dx_pre, dx_post = stdp_derivatives(w_gpe_stn, x_pre, x_post, H_pre, H_post)
    
    # synaptic currents using those gating variables
    Igith = 1.4 *  (gsyn[0] * (V1 - Esyn[0]) * (w_gpi_th @ S4))
    Igesn = 0.5 * (gsyn[1] * (V2 - Esyn[1]) * (w_gpe_stn @ S3))
    Isnge = 0.5 *  (gsyn[2] * (V3 - Esyn[2]) * ( w_stn_gpe  @ S2))
    Igege = 0.5 *  (gsyn[3] * (V3 - Esyn[3]) *  (w_gpe_gpe @ S3))
    Igegi = 0.5 *   (gsyn[4] * (V4 - Esyn[4]) *  (w_gpe_gpi @ S3))
    Isngi = 0.5 *   (gsyn[5] * (V4 - Esyn[5]) *  (w_stn_gpi  @S2))
    


    # differential equations th
    dV1dt = (-Il1 - Ik1 - Ina1 - It1 - Igith + Iapp_th) / Cm
    dH1dt   = (h1 - H1) / th1
    dR1dt   = (r1 - R1) / tr1

    # differential equations stn
    dV2dt   = (-Il2 - Ik2 - Ina2 - It2 - Ica2 - Iahp2 - Igesn + Iappstn) / Cm
    dN2dt   = 0.75 * (n2 - N2) / tn2
    dH2dt   = 0.75 * (h2 - H2) / th2
    dR2dt   = 0.2 * (r2 - R2) / tr2
    dCA2dt  = 3.75 * 1e-5 * (-Ica2 - It2 - kca[1] * CA2)
    dC2dt   = 0.08 * (c2 - C2) / tc2

    # differential equations gpe
    dV3dt  = (-Il3 - Ik3 - Ina3 - It3 - Ica3 - Iahp3 - Isnge - Igege + Iappgpe) / Cm
    dN3dt  = 0.1  * (n3 - N3) / tn3
    dH3dt  = 0.05 * (h3 - H3) / th3
    dR3dt  = 1.0  * (r3 - R3) / tr3
    dCA3dt = 1e-4 * (-Ica3 - It3 - kca[2] * CA3)

    # differential equations gpi
    dV4dt  = (-Il4 - Ik4 - Ina4 - It4 - Ica4 - Iahp4 - Igegi - Isngi + Iappgpi) / Cm
    dN4dt  = 0.1  * (n4 - N4) / tn4
    dH4dt  = 0.05 * (h4 - H4) / th4
    dR4dt  = 1.0  * (r4 - R4) / tr4
    dCA4dt = 1e-4 * (-Ica4 - It4 - kca[3] * CA4)
    

    return {
        "V1_th":  dV1dt,
        "H1_th":  dH1dt,
        "R1_th":  dR1dt,

        "V2_stn":  dV2dt,
        "N2_stn":  dN2dt,
        "H2_stn":  dH2dt,
        "R2_stn":  dR2dt,
        "C2_stn":  dC2dt,
        "CA2_stn": dCA2dt,
        "S2_stn":  dS2dt,
        "Z2_stn":  dZ2dt,

        "V3_gpe":  dV3dt,
        "N3_gpe":  dN3dt,
        "H3_gpe":  dH3dt,
        "R3_gpe":  dR3dt,
        "CA3_gpe": dCA3dt,
        "S3_gpe":  dS3dt,

        "V4_gpi":  dV4dt,
        "N4_gpi":  dN4dt,
        "H4_gpi":  dH4dt,
        "R4_gpi":  dR4dt,
        "CA4_gpi": dCA4dt,
        "S4_gpi":  dS4dt,
        "Z4_gpi":  dZ4dt,
    }


# initial values and params
V1_init = -62
V2_init = -62
V3_init = -62
V4_init = -62

y0 = {
    "V1_th":  jnp.full((n_th,), V1_init),
    "H1_th":  th_hinf(V1_init)  * jnp.ones((n_th,)),
    "R1_th":  th_rinf(V1_init)  * jnp.ones((n_th,)),

    "V2_stn":  jnp.full((n_stn,), V2_init),
    "N2_stn":  stn_ninf(V2_init)  * jnp.ones((n_stn,)),
    "H2_stn":  stn_hinf(V2_init)  * jnp.ones((n_stn,)),
    "R2_stn":  stn_rinf(V2_init)  * jnp.ones((n_stn,)),
    "C2_stn":  stn_cinf(V2_init)  * jnp.ones((n_stn,)),
    "CA2_stn": jnp.full((n_stn,), 0.1),
    "S2_stn":  jnp.zeros((n_stn,),),
    "Z2_stn":  jnp.zeros((n_stn,),),

    "V3_gpe":  jnp.full((n_gpe,), V3_init),
    "N3_gpe":  gpe_ninf(V3_init)  * jnp.ones((n_gpe,)),  
    "H3_gpe":  gpe_hinf(V3_init)  * jnp.ones((n_gpe,)),
    "R3_gpe":  gpe_rinf(V3_init)  * jnp.ones((n_gpe,)),
    "CA3_gpe": jnp.full((n_gpe,), 0.1),
    "S3_gpe":  jnp.zeros((n_gpe,),),

    "V4_gpi":  jnp.full((n_gpi,), V4_init),
    "N4_gpi":  gpe_ninf(V4_init)  * jnp.ones((n_gpi,)),  
    "H4_gpi":  gpe_hinf(V4_init)  * jnp.ones((n_gpi,)),
    "R4_gpi":  gpe_rinf(V4_init)  * jnp.ones((n_gpi,)),
    "CA4_gpi": jnp.full((n_gpi,), 0.1),
    "S4_gpi":  jnp.zeros((n_gpi,),),
    "Z4_gpi":  jnp.zeros((n_gpi,),),
}

#%%
@jax.jit
def run_chunk(y0, params, t0, t1, dt0, ts):
    """
    Solve ODE from t0 -> t1 starting from y0.
    ts must be a concrete array of times (not tracers).
    """
    term = diffrax.ODETerm(gpe_rhs)
    solver = diffrax.Tsit5()

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=params,
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=1_000_000,
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6, dtmin = 0.001, force_dtmin=True)
    )
    
    # last state of chunk
    yT = jax.tree.map(lambda a: a[-1], sol.ys)
    return sol.ys, yT

def simulate_chunked(
    y0, params, tmax, chunk_size, dt0=0.1, dt_save=1
):
    """
    Simulate ODE in chunks using Diffrax, keeping only the last chunk.
    Fully JAX-compatible (jit + scan safe).
    """

    # --- static values ---
    n_chunks = int(tmax // chunk_size)
    n_steps = int(chunk_size // dt_save)

    # --- template ts for shape inference ---
    ts_template = dt_save * jnp.arange(n_steps)

    # --- get shapes (no real compute) ---
    ys_shape, yT_shape = jax.eval_shape(
        lambda y: run_chunk(y, params, 0.0, chunk_size, dt0, ts_template),
        y0,
    )

    # --- initialize carry ---
    init_ts = jnp.zeros((n_steps,))
    init_ys = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), ys_shape)

    def step(carry, i):
        y, t0, last_ts, last_ys = carry

        # fixed chunk end (no Python min!)
        t1 = t0 + chunk_size

        # fixed ts (same shape every iteration)
        ts = t0 + dt_save * jnp.arange(n_steps)

        # run solver
        ys, y_next = run_chunk(y, params, t0, t1, dt0, ts)

        # detect last iteration
        is_last = (i == n_chunks - 1)

        # update only on last chunk
        last_ts = jax.lax.select(is_last, ts, last_ts)
        last_ys = jax.tree.map(
            lambda new, old: jax.lax.select(is_last, new, old),
            ys,
            last_ys,
        )

        return (y_next, t1, last_ts, last_ys), None

    # --- run scan ---
    (y_final, t_final, last_ts, last_ys), _ = jax.lax.scan(
        step,
        (y0, 0.0, init_ts, init_ys),
        xs=jnp.arange(n_chunks),
    )

    # --- return last chunk ---
    return (
        last_ts,
        last_ys["V1_th"],
        last_ys["V2_stn"],
        last_ys["V3_gpe"],
        last_ys["V4_gpi"],
    )

# def simulate_chunked(y0, params, tmax, chunk_size, dt0=0.1, dt_save=1):
#     """
#     Simulate in chunks, generating a concrete ts for each chunk
#     to avoid arange tracer issues.
#     """
#     t0 = 0.0
#     y = y0

#     all_ts = []
#     all_V1 = []
#     all_V2 = []
#     all_V3 = []
#     all_V4 = []

#     while t0 < tmax:
#         t1 = min(t0 + chunk_size, tmax)

#         # ---- Compute concrete ts BEFORE calling run_chunk ----
#         n_steps = int(jnp.ceil((t1 - t0)/dt_save))
#         ts = t0 + dt_save * jnp.arange(n_steps)  # <--- concrete stop, no tracers

#         # run chunk (JIT)
#         ys, y = run_chunk(y, params, t0, t1, dt0, ts)

#         # collect
#         all_ts.append(ts)
#         all_V1.append(ys["V1_th"])
#         all_V2.append(ys["V2_stn"])
#         all_V3.append(ys["V3_gpe"])
#         all_V4.append(ys["V4_gpi"])

#         # next chunk
#         t0 = float(t1)  # make sure t0 is python float

#     # concatenate all chunks
#     all_ts = jnp.concatenate(all_ts)
#     all_V1 = jnp.concatenate(all_V1)
#     all_V2 = jnp.concatenate(all_V2)
#     all_V3 = jnp.concatenate(all_V3)
#     all_V4 = jnp.concatenate(all_V4)

#     return all_ts, all_V1, all_V2, all_V3, all_V4

#%% PERFECT EULER
# def run_chunk_euler_scan(y0, params, t0, dt,chunk_length):
#     n_steps =chunk_length//dt
#     ts = t0 + dt * jnp.arange(n_steps)
#     print("Compile chunk")

#     def euler_step(y, t):
#         dy = gpe_rhs(t, y, params)
#         y_next = jax.tree.map(lambda y, dy: y + dt * dy, y, dy)
#         return y_next, y_next

#     y_final, ys = jax.lax.scan(euler_step, y0, ts)
#     return y_final, (ts, ys)


# # ✅ jit compile for speed
# run_chunk_euler_scan = jax.jit(run_chunk_euler_scan,static_argnames=('dt','chunk_length'))

#DIFFRAX Euler-worse than mine
# def run_chunk_euler_diffrax_fixed(y0, params, t0, dt, chunk_size, dt_save):
#     term = diffrax.ODETerm(gpe_rhs)
#     solver = diffrax.Euler()

#     # number of saved points as static integer
#     n_save = int(chunk_size // dt_save)
#     ts = t0 + dt_save * jnp.arange(n_save)  # traced t0 is fine here

#     saveat = diffrax.SaveAt(ts=ts)

#     sol = diffrax.diffeqsolve(
#         term,
#         solver,
#         t0=t0,
#         t1=t0 + chunk_size,
#         dt0=dt,
#         y0=y0,
#         args=params,
#         saveat=saveat,
#         max_steps=int(1e8),
#     )

#     print(sol.stats)   # total solver steps

#     yT = jax.tree.map(lambda a: a[-1], sol.ys)
#     return yT, (ts, sol.ys)

# run_chunk_euler_diffrax = jax.jit(run_chunk_euler_diffrax_fixed, static_argnames=("dt_save", "chunk_size"))

# def simulate_last_chunk_euler(
#     y0, params, tmax, dt=0.1, dt_save=1, chunk_length=1000
# ):
#     n_chunks = int(tmax // chunk_length)

#     # --- get shapes for initialization (no real compute) ---
#     _, (ts_shape, ys_shape) = jax.eval_shape(
#         lambda y: run_chunk_euler_diffrax_fixed(y, params, 0.0, dt, chunk_length, dt_save),
#         y0,
#     )

#     init_ts = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), ts_shape)
#     init_ys = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), ys_shape)

#     def step(carry, i):
#         y, t0, last_ts, last_ys = carry

#         y_next, (ts, ys) = run_chunk_euler_diffrax_fixed(
#             y, params, t0, dt=dt, chunk_size=chunk_length, dt_save=dt_save
#         )

#         # detect last iteration
#         is_last = (i == n_chunks - 1)

#         # update only on last step
#         last_ts = jax.lax.select(is_last, ts, last_ts)
#         last_ys = jax.tree.map(
#             lambda new, old: jax.lax.select(is_last, new, old),
#             ys,
#             last_ys,
#         )

#         return (y_next, t0 + chunk_length, last_ts, last_ys), None

#     (y_final, t_final, last_ts, last_ys), _ = jax.lax.scan(
#         step,
#         (y0, 0.0, init_ts, init_ys),
#         xs=jnp.arange(n_chunks),
#     )

#     return (
#         last_ts,
#         last_ys["V1_th"],
#         last_ys["V2_stn"],
#         last_ys["V3_gpe"],
#         last_ys["V4_gpi"],
#     )
#%% run simulation
# tmax = 1000.0
# chunk_size = 1000.0      # 1 second per chunk
# dt0 = 0.01
# dt_save = 1           # save every 1 ms

# ts, V1, V2, V3, V4 = simulate_last_chunk_euler(y0, params, tmax, dt0, dt_save, chunk_size)
# ts_pd, V1_pd, V2_pd, V3_pd, V4_pd = simulate_last_chunk_euler(y0, params_pd, tmax, dt0,dt_save,chunk_size)

# ts, V1, V2, V3, V4 = simulate_last_chunk_euler(y0, params, tmax, dt0, dt_save, chunk_size)
# ts_pd, V1_pd, V2_pd, V3_pd, V4_pd = simulate_last_chunk_euler(y0, params_pd, tmax, dt0, dt_save, chunk_size)

# ts, V1, V2, V3, V4 = simulate_chunked(y0, params, tmax, chunk_size, dt0, dt_save)
# ts_pd, V1_pd, V2_pd, V3_pd, V4_pd = simulate_chunked(y0, params_pd, tmax, chunk_size, dt0, dt_save)

#%%
tmax = 100.0
chunk_size = 100.0      # 1 second per chunk
dt0 = 0.01
dt_save = 1           # save every 1 ms

ts, V1, V2, V3, V4 = simulate_chunked(y0, params, tmax, chunk_size, dt0, dt_save)
ts_pd, V1_pd, V2_pd, V3_pd, V4_pd = simulate_chunked(y0, params_pd, tmax, chunk_size, dt0, dt_save)
# plot to check
plt.plot(ts, V1[:,0])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()

# plot to check
plt.plot(ts, V2[:,0])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()

# plot to check
plt.plot(ts, V3[:,0])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()


# plot to check
plt.plot(ts, V4[:,0])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.show()
# %%
def findfreq(sig): #in Hz
    #delta = 1e-5 #remember that only with dt 0.01 and t 1000 ms
    val = sig
    h=scipy.signal.detrend(val)
    peaks=scipy.signal.find_peaks(h,height=23)[0]
    #fr=1/(np.mean(np.diff(peaks))*delta)
    fr=np.size(peaks)
    if np.size(peaks)<1:
        fr=0
           
    return fr

print("STN FR:", findfreq(V2[:,2]))
print("GPe FR:", findfreq(V3[:,3]))
print("GPi FR:", findfreq(V4[:,1]))

#%% fourier transform

# sig2 = V2[:,0]
# fft_res2 = np.abs(np.fft.rfft(np.fft.ifftshift(sig2-np.mean(sig2))))
# ff_freqs2 = np.fft.rfftfreq(n= 10000,d = 1e-4)

# sig2_pd = V2_pd[:,0]
# fft_res2_pd = np.abs(np.fft.rfft(np.fft.ifftshift(sig2_pd-np.mean(sig2_pd))))
# ff_freqs2_pd = np.fft.rfftfreq(n= 10000,d = 1e-4)

# sig3 = V3[:,0]
# fft_res3 = np.abs(np.fft.rfft(np.fft.ifftshift(sig3-np.mean(sig3))))
# ff_freqs3 = np.fft.rfftfreq(n= 10000,d = 1e-4)

# sig3_pd = V3_pd[:,0]
# fft_res3_pd = np.abs(np.fft.rfft(np.fft.ifftshift(sig3_pd-np.mean(sig3_pd))))
# ff_freqs3_pd = np.fft.rfftfreq(n= 10000,d = 1e-4)

# sig4 = V4[:,0]
# fft_res4 = np.abs(np.fft.rfft(np.fft.ifftshift(sig4-np.mean(sig4))))
# ff_freqs4 = np.fft.rfftfreq(n= 10000,d = 1e-4)

# sig4_pd = V4_pd[:,0]
# fft_res4_pd = np.abs(np.fft.rfft(np.fft.ifftshift(sig4_pd-np.mean(sig4_pd))))
# ff_freqs4_pd = np.fft.rfftfreq(n= 10000,d = 1e-4)

# plt.figure(figsize=(15,8))
# plt.plot(ff_freqs2[:50],fft_res2[:50],label='Healthy')
# plt.plot(ff_freqs2_pd[:50],fft_res2_pd[:50],label='PD')
# plt.ylabel('Amplitude [a.u.]')
# plt.xlabel('Frequency [Hz]')
# plt.legend()
# plt.title('Mean STN psd')
# plt.show()

# plt.figure(figsize=(15,8))
# plt.plot(ff_freqs3[:50],fft_res3[:50],label='Healthy')
# plt.plot(ff_freqs3_pd[:50],fft_res3_pd[:50],label='PD')
# plt.ylabel('Amplitude [a.u.]')
# plt.xlabel('Frequency [Hz]')
# plt.legend()
# plt.title('Mean GPe psd')
# plt.show()

# plt.figure(figsize=(15,8))
# plt.plot(ff_freqs4[:50],fft_res4[:50],label='Healthy')
# plt.plot(ff_freqs4_pd[:50],fft_res4_pd[:50],label='PD')
# plt.ylabel('Amplitude [a.u.]')
# plt.xlabel('Frequency [Hz]')
# plt.legend()
# plt.title('Mean GPi psd')
# plt.show()