
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
import matplotlib.mlab as mlab

#jax.config.update("jax_enable_x64", True)
print(jax.devices())

#%%
# define neuron parameters 
#number of neurons per nucleus
n_th = 4
n_stn = 4
n_gpe = 4
n_gpi = 4
n_dstr = 4
n_istr = 4
n_ctx_fsi = 2
n_ctx_pyr = 20

# pd (pd = 1) or healthy (pd = 0)
pd = 0

DA = 0.9 # healthy
#DA = 0.1  # pd

# dopamine scaling from CTX to Str
def cD1(DA, AD1=10.0, lambda_str=7.5):
    return AD1 / (1.0 + jnp.exp(-lambda_str * (DA - 1.0)))

# STDP threshold 
def spike_event(V, threshold):
    return (V > threshold).astype(V.dtype)


# parameters in a dict (PyTree)
params = {
    # membrane params
    # in order of TH, STN, GPe, GPi, Str, CTX (PYR), CTX (FSI) (1e-5, 0.00015)
    "Cm": 1.0,
    "gl": jnp.array([0.05, 2.25, 0.1, 0.1, 0.1, 0.01, 0.15]),  "El": jnp.array([-70, -60, -65, -65, -67, -85, -70]),
    "gna": jnp.array([3, 37, 120, 120, 100]), "Ena": jnp.array([50, 55, 55, 55, 50, 50, 50]),
    "gk": jnp.array([5, 45, 30, 30, 80]),  "Ek": jnp.array([-75, -80, -80, -80, -100, -100, -100]),
    "gt": jnp.array([5, 0.5, 0.5, 0.5]), "Et":0,
    "gca": jnp.array([0, 2, 0.15, 0.15]), "Eca": jnp.array([0, 140, 120, 120, 120]),
    "gahp": jnp.array([0, 20, 10, 10]),
    "k1": jnp.array([0, 15, 10, 10]),
    "kca": jnp.array([0, 22.5, 15, 15]),
    "gm": 1,"Em": -100, # for striatum muscarinic current

    # specific cortex params
    # in order of CTX (PYR), CTX (FSI)
    #"diam": jnp.array([96, 67]), "L": jnp.array([96, 67]), "Ra": 100, "nseg": 1, # geometry
    "vtraub": jnp.array([-55, -55]), "gnabar": jnp.array([50, 50]), "gkbar": jnp.array([5, 10]), # mchh2
    # just CTX (PYR)
    "gkbar_m": 0.03, # m current
    "depth": 1, "tau_r": 5, "cainf": 2.4e-4, "gcabar": 0.4, # calcium dynamics
    "cao": 2 , "celsius": 36, "R": 8.314462618, "FARADAY": 96485.3321, # calcium dynamics

    
    # synapse params (Rubin, 2004)
    # in order of Igith, Igesn, Isnge, Igege, Igegi, Isngi
    "A": jnp.array([2.0 , 2.0 , 3.0, 2.0, 2.0, 3.0]),
    "B": jnp.array([0.04, 0.04, 0.1, 0.04, 0.04, 0.1]),
    "the": jnp.array([20, 20, 30, 20, 20, 30]),
    "gsyn": jnp.array([0.08, 1, 0.3, 1, 1, 0.3]),
    "Esyn": jnp.array([-85, -85, 0, -85, -85, 0]),
    "tau": 5, "gpeak1": 0.3, "gpeak": 0.43, #parameters for second-order alpha synapse

    # synapse params (Karamavelu, 2004)
    # in order of Istrstr, Istrge, Istrgi 
    "gsynstr": jnp.array([0.8, 0.5, 0.5]),
    "Esynstr": jnp.array([-80, -85, -85]),
    "ggaba": 0.1, "tau_i": 13,

    # synapse params ctx (Santiniello, 2019)
    # in order of Ipyr, Ifsi
    "alphactx": jnp.array([0.55 , 2.5]),
    "gsynctx": jnp.array([0.27, 0.36, 0.1, 0.29]), # in order of Ipypy, Ipyfsi, Ififi, Ifipy
    "Esynctx": jnp.array([0, -80]),
    "tau_ctx": jnp.array([5.26 , 5.56]),

    # stdp params
    "tau_pre": 12.0,
    "tau_post": 27.5,

    # connectivity matrix
    # 1 : 1 connectivity
    #"w_gpe_stn": jnp.eye(n_stn, n_gpe),
    #"w_stn_gpe": jnp.eye(n_gpe, n_stn),
    "w_gpi_th":  jnp.eye(n_th, n_gpi),
    #"w_gpe_gpi": jnp.eye(n_gpi, n_gpe),
    #"w_stn_gpi": jnp.eye(n_gpi, n_stn),
    "w_istr_gpe": jnp.eye(n_gpe, n_istr),
    "w_dstr_gpi": jnp.eye(n_gpi, n_dstr),
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
    #"w_gpe_stn": jnp.array([
    #    [1,0,0,1],
    #    [1,1,0,0],
    #    [0,1,1,0],
    #    [0,0,1,1],], dtype=jnp.float64),
    "w_gpe_gpi": jnp.array([
        [0,0,1,1],
        [1,0,0,1],
        [1,1,0,0],
        [0,1,1,0],], dtype=jnp.float64),
    "w_gpe_gpe": jnp.array([
        [0,0,1,1],
        [1,0,0,1],
        [1,1,0,0],
        [0,1,1,0],], dtype=jnp.float64),
    # 3 : 1 connectivity in direct striatum
    "w_dstr": jnp.array([
        [0,1,1,1],
        [1,0,1,1],
        [1,1,0,1],
        [1,1,1,0],], dtype=jnp.float32),
    # 4 : 1 connectivity in indirect striatum
    "w_istr": jnp.ones((n_istr, n_istr), dtype=jnp.float32),
    
    # all-to-all connectivity in FSI-FSI CTX
    "w_fsi": jnp.ones((n_ctx_fsi, n_ctx_fsi), dtype=jnp.float32) - jnp.eye(n_ctx_fsi, dtype=jnp.float32),
    # all-to-five connectivity in PYR-PYR CTX
    "w_pyr": w_matrix(n = n_ctx_pyr, k = 5),
    # all-to-all connectivity in PYR-FSI CTX
    "w_pyr_fsi": jnp.ones((n_ctx_fsi, n_ctx_pyr), dtype=jnp.float32),
    # all-to-all connectivity in FSI-PYR CTX
    "w_fsi_pyr": jnp.ones((n_ctx_pyr, n_ctx_fsi), dtype=jnp.float32),
    # all-to-all in PYR CTX - Str
    "w_pyr_str":  jnp.ones((n_dstr, n_ctx_pyr), dtype=jnp.float32) * cD1(DA),
}

# defining the functions for ODEterm
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
    R2  = y["R2_stn"] 
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


    # STR direct and indirect
    V5d  = y["V5_dstr"]
    m5d  = y["m5_dstr"]
    h5d  = y["h5_dstr"]
    n5d  = y["n5_dstr"]
    p5d  = y["p5_dstr"]
    S5d  = y["S5_dstr"]
    S5d_2 = y["S52_dstr"]
    Z5d_2 = y["Z52_dstr"]

    V5i  = y["V5_istr"] 
    m5i  = y["m5_istr"]
    h5i  = y["h5_istr"]
    n5i  = y["n5_istr"]
    p5i  = y["p5_istr"]
    S5i  = y["S5_istr"]
    S5i_2 = y["S52_istr"]
    Z5i_2 = y["Z52_istr"]

    # CTX PYR & FSI
    # PYR
    V6 = y["V6_ctx"]
    M6 = y["M6_ctx"]
    N6 = y["N6_ctx"]
    H6 = y["H6_ctx"]
    MM6 = y["MM6_ctx"]
    Hca6 = y["Hca6_ctx"]
    cai6 = y["cai6_ctx"]
    S6 = y["S6_ctx"]
    
    #FSI
    V7 = y["V7_ctx"]
    M7 = y["M7_ctx"]
    N7 = y["N7_ctx"]
    H7 = y["H7_ctx"]
    S7 = y["S7_ctx"]

    # STDP state variables
    x_pre = y["x_pre"]
    x_post = y["x_post"]
    W = y["W"]


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
    gm   = params["gm"];  Em   = params["Em"]
    A    = params["A"]
    B    = params["B"]
    the  = params["the"]
    gsyn = params["gsyn"]
    Esyn = params["Esyn"]
    gsynstr = params["gsynstr"]
    Esynstr = params["Esynstr"]
    ggaba = params["ggaba"]; tau_i = params["tau_i"]

    gpeak = params["gpeak"]; tau = params["tau"]
    gpeak1 = params["gpeak1"]
    #diam = params["diam"]; L = params["L"]; Ra = params["Ra"];nseg = params["nseg"]
    vtraub = params["vtraub"];gnabar = params["gnabar"]; gkbar = params["gkbar"]
    gkbar_m = params["gkbar_m"]
    depth = params["depth"]; tau_r = params["tau_r"]; cainf = params["cainf"]; gcabar = params["gcabar"]
    cao = params["cao"];  celsius = params["celsius"]; R = params["R"]; FARADAY  =params["FARADAY"]

    alphactx = params["alphactx"];  gsynctx = params["gsynctx"]; Esynctx = params["Esynctx"]; tau_ctx = params["tau_ctx"]

    tau_pre = params["tau_pre"]
    tau_post = params["tau_post"]

    #w_gpe_stn = params["w_gpe_stn"]
    w_stn_gpe = params["w_stn_gpe"] 
    w_gpi_th = params["w_gpi_th"]
    w_gpe_gpi = params["w_gpe_gpi"]
    w_stn_gpi = params["w_stn_gpi"]
    w_gpe_gpe = params["w_gpe_gpe"]
    w_istr = params["w_istr"]
    w_dstr = params["w_dstr"]
    w_dstr_gpi = params["w_dstr_gpi"]
    w_istr_gpe = params["w_istr_gpe"]
    w_pyr = params["w_pyr"]
    w_fsi = params["w_fsi"]
    w_pyr_fsi = params["w_pyr_fsi"]
    w_fsi_pyr = params["w_fsi_pyr"]
    w_pyr_str = params["w_pyr_str"]


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

    m6 = ctx_minf(V6, vtraub[0])
    n6 = ctx_ninf(V6, vtraub[0])
    h6 = ctx_hinf(V6, vtraub[0])
    tm6 = ctx_taum(V6, vtraub[0])
    tn6 = ctx_taun(V6, vtraub[0])
    th6 = ctx_tauh(V6, vtraub[0])
    m7 = ctx_minf(V7, vtraub[1])
    n7 = ctx_ninf(V7, vtraub[1])
    h7 = ctx_hinf(V7, vtraub[1])
    tm7 = ctx_taum(V7, vtraub[1])
    tn7 = ctx_taun(V7, vtraub[1])
    th7 = ctx_tauh(V7, vtraub[1])

    mm6 = ctx_minf_m(V6)
    tmm6 = ctx_taum_m(V6)
    mca6 = ctx_minf_ca(V6)
    hca6 = ctx_hinf_ca(V6)
    thca6 = ctx_tauh_ca(V6)



    # thalamic currents
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
    Ica2  = gca[1]  * (c2**2) * (V2 - Eca[1])
    Iahp2 = gahp[1] * (V2 - Ek[1]) * (CA2 / (CA2 + k1[1]))

    # applied current stn    
    Iappstn = 35.0

    # currents gpe
    Il3  = gl[2]  * (V3 - El[2])
    Ik3  = gk[2]  * (N3**4) * (V3 - Ek[2])
    Ina3 = gna[2] * (m3**3) * H3 * (V3 - Ena[2])
    It3  = gt[2]  * (a3**3) * R3 * (V3 - Eca[2])
    Ica3 = gca[2] * (s3**2) * (V3 - Eca[2])
    Iahp3 = gahp[2] * (V3 - Ek[2]) * (CA3 / (CA3 + k1[2]))

    # applied current gpe
    Iappgpe = 15.0

    # currents gpi
    Il4  = gl[3]  * (V4 - El[3])
    Ik4  = gk[3]  * (N4**4) * (V4 - Ek[3])
    Ina4 = gna[3] * (m4**3) * H4 * (V4 - Ena[3])
    It4  = gt[3]  * (a4**3) * R4 * (V4 - Eca[3])
    Ica4 = gca[3] * (s4**2) * (V4 - Eca[3])
    Iahp4 = gahp[3] * (V4 - Ek[3]) * (CA4 / (CA4 + k1[3]))

    # applied current gpi
    Iappgpi = 15.0

    # currents str (direct and indirect)
    Ina5d = gna[4] * (m5d**3) * h5d * (V5d - Ena[4])
    Ik5d =  gk[4]  * (n5d**4) * (V5d - Ek[4])
    Il5d =  gl[4]  * (V5d - El[4]) 
    Im5d = (2.6 - 1.1 * pd) * gm * p5d * (V5d - Em)

    Ina5i = gna[4] * (m5i**3) * h5i * (V5i - Ena[4])
    Ik5i =  gk[4]  * (n5i**4) * (V5i - Ek[4])
    Il5i =  gl[4]  * (V5i - El[4]) 
    Im5i = (2.6 - 1.1 * pd) * gm * p5i * (V5i - Em)

    # currents cortex
    # CTX PYR
    Il6 = gl[5] * (V6 - El[5])
    Ina6 = gnabar[0] * (M6**3) * H6 * (V6 - Ena[5])
    Ik6 =  gkbar[0]  * (N6**4) * (V6 - Ek[5])
    Im6 = gkbar_m * MM6 * (V6 - Ek[5])
    carev = 1e3 * (R *(celsius+273.15)) / (2 * FARADAY) * jnp.log (cao/cai6)  # calcium dynamics
    Ica6 = gcabar * (mca6**2) * Hca6 * (V6 - carev)   # calcium dynamics
   # applied current 
    Iappctx6 = 5

    # CTX FSI
    Il7 = gl[6] * (V7 - El[6])
    Ina7 = gnabar[1] * (M7**3) * H7 * (V7 - Ena[6])
    Ik7 =  gkbar[1]  * (N7**4) * (V7 - Ek[6])

   # applied current 
    Iappctx7 = 3

    # synapses with connectivity matrices

    # presynaptic activation
    H_gpe = Hinf(V3, theta=30.0)
    H_stn = Hinf(V2, theta=20.0)
    H_gpi = Hinf(V4, theta=20.0)
    H_istr = Hinf(V5i, theta=20.0)
    H_dstr = Hinf(V5d, theta=20.0)

    # differential equations synapses
    # STN synapses (2nd order alpha synapses)
    u2 = gpeak / (tau * jnp.exp(-1.0)) * H_stn
    dS2dt = Z2
    dZ2dt = u2 - (2.0 / tau) * Z2 - (1.0 / tau**2) * S2

    # GPi synaptic currents (2nd order alpha synapses)
    u4 = gpeak1 / (tau * jnp.exp(-1.0)) * H_gpi
    dS4dt = Z4
    dZ4dt = u4 - (2.0 / tau) * Z4 - (1.0 / tau**2) * S4

    # GPe synaptic currents 
    dS3dt = A[1] * (1 - S3) * H_gpe - B[1] * S3

    # GPe-STN STDP
    #spike indicators
    pre_spike  = spike_event(V3, -20)
    post_spike = spike_event(V2, -20)
    # STDP traces
    dx_pre_dt  = -x_pre/tau_pre  + pre_spike
    dx_post_dt = -x_post/tau_post + post_spike

    # STDP weight rule
    A_plus  = 0.002
    A_minus = 0.002 * 1.1

    dWdt = (
        A_plus  * jnp.outer(post_spike, x_pre)
        - A_minus * jnp.outer(x_post, pre_spike)
    )

    # str synaptic currents (2nd order alpha synapses)
    #str to gpe
    u3 = gpeak1 / (tau * jnp.exp(-1.0)) * H_istr
    dS52idt = Z5i_2
    dZ52idt = u3 - (2.0 / tau) * Z5i_2 - (1.0 / tau**2) * S5i_2

    #str to gpi
    u5 = gpeak1 / (tau * jnp.exp(-1.0)) * H_dstr
    dS52ddt = Z5d_2
    dZ52ddt = u5 - (2.0 / tau) * Z5d_2 - (1.0 / tau**2) * S5d_2  

    # recurrent gaba currents
    dS5idt = (w_istr @ str_Ggaba(V5i)) * (1.0 - S5i) - (S5i /tau_i)
    dS5ddt = (w_dstr @ str_Ggaba(V5d)) * (1.0 -S5d) - (S5d /tau_i)

    # CTX
    # PYR AMPA synapses (Santiniello, 2019)
    dS6dt = alphactx[0] * (1 + jnp.tanh((V6)/ 4))*(1.0 - S6) - (S6 / tau_ctx[0]) 
    # FSI GABA synapses
    dS7dt = alphactx[1] * (1 + jnp.tanh((V7)/ 4))*(1.0 - S7) - (S7 / tau_ctx[1]) 

    # synaptic currents using those gating variables
    Igith = 1.4 *  (gsyn[0] * (V1 - Esyn[0]) * (w_gpi_th @ S4))
    Igesn = 0.5 *  (gsyn[1] * (V2 - Esyn[1]) * (W @ S3))
    Isnge = 0.5 *  (gsyn[2] * (V3 - Esyn[2]) * ( w_stn_gpe  @ S2))
    Igege = 0.5 *  (gsyn[3] * (V3 - Esyn[3]) *  (w_gpe_gpe @ S3))
    Igegi = 0.5 *  (gsyn[4] * (V4 - Esyn[4]) *  (w_gpe_gpi @ S3))
    Isngi = 0.5 *  (gsyn[5] * (V4 - Esyn[5]) *  (w_stn_gpi  @ S2))
    Istrge = gsynstr[0] * (V3 - Esynstr[0]) * (w_istr_gpe  @ S5i_2) # from in FOG: Istrge = 0.1 * ... - why?
    Istrgi = gsynstr[1] * (V4 - Esynstr[1]) * (w_dstr_gpi  @ S5d_2)
    Istrd = (ggaba/3) * (V5d - Esynstr[0]) * S5d
    Istri = (ggaba/4) * (V5i - Esynstr[1]) * S5i
    Ipypy = gsynctx[0] * (V6 - Esynctx[0]) * (w_pyr  @ S6) / 5 # change to normalizing by number of presynaptic input neurons
    Ipyfi = gsynctx[1] * (V7 - Esynctx[1]) * (w_pyr_fsi  @ S6) / 20
    Ififi = gsynctx[2] * (V7 - Esynctx[1]) * (w_fsi  @ S7) / 2
    Ifipy = gsynctx[3] * (V6 - Esynctx[0]) * (w_fsi_pyr  @ S7) /2
    Ipystr = gsynctx[0] * (V5d - Esynctx[0]) * (w_pyr_str  @ S6) / 20 # check for valid parameters here 


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
    dV3dt  = (-Il3 - Ik3 - Ina3 - It3 - Ica3 - Iahp3 - Isnge - Igege - Istrge + Iappgpe) / Cm
    dN3dt  = 0.1  * (n3 - N3) / tn3
    dH3dt  = 0.05 * (h3 - H3) / th3
    dR3dt  = 1.0  * (r3 - R3) / tr3
    dCA3dt = 1e-4 * (-Ica3 - It3 - kca[2] * CA3)

    # differential equations gpi
    dV4dt  = (-Il4 - Ik4 - Ina4 - It4 - Ica4 - Iahp4 - Igegi - Isngi - Istrgi + Iappgpi) / Cm
    dN4dt  = 0.1  * (n4 - N4) / tn4
    dH4dt  = 0.05 * (h4 - H4) / th4
    dR4dt  = 1.0  * (r4 - R4) / tr4
    dCA4dt = 1e-4 * (-Ica4 - It4 - kca[2] * CA4)

    # differential equations str
    dV5ddt = (-Il5d - Ik5d - Ina5d - Im5d - Istrd - Ipystr) / Cm
    dm5ddt = str_alpham(V5d) * (1 - m5d) - str_betam(V5d) * m5d
    dh5ddt = str_alphah(V5d) * (1 - h5d) - str_betah(V5d) * h5d
    dn5ddt = str_alphan(V5d) * (1 - n5d) - str_betan(V5d) * n5d
    dp5ddt = str_alphap(V5d) * (1 - p5d) - str_betap(V5d) * p5d

    dV5idt = (-Il5i - Ik5i - Ina5i - Im5i - Istri) / Cm
    dm5idt = str_alpham(V5i) * (1 - m5i) - str_betam(V5i) * m5i
    dh5idt = str_alphah(V5i) * (1 - h5i) - str_betah(V5i) * h5i
    dn5idt = str_alphan(V5i) * (1 - n5i) - str_betan(V5i) * n5i
    dp5idt = str_alphap(V5i) * (1 - p5i) - str_betap(V5i) * p5i

    # differential equations CTX
    dV6dt = (-Il6 - Ik6 - Ina6 - Im6 - Ica6 - Ipypy - Ifipy + Iappctx6) / Cm
    dM6dt = (m6 - M6) / tm6
    dN6dt = (n6 - N6) / tn6
    dH6dt = (h6 - H6) / th6
    dMM6dt = (mm6 - MM6) / tmm6
    dHca6dt = (hca6 -Hca6) / thca6
    dcai6dt = ca_drive(Ica6, FARADAY, depth) + (cainf - cai6) / tau_r
    

    dV7dt = (-Il7 - Ik7 - Ina7 + Iappctx7) / Cm  # took out: Ipyfi & Ififi (too strong?)
    dM7dt = (m7 - M7) / tm7
    dN7dt = (n7 - N7) / tn7
    dH7dt = (h7 - H7) / th7

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
        "S3_gpe": dS3dt,
        "W": dWdt,
        "x_pre": dx_pre_dt,
        "x_post": dx_post_dt,

        "V4_gpi":  dV4dt,
        "N4_gpi":  dN4dt,
        "H4_gpi":  dH4dt,
        "R4_gpi":  dR4dt,
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
        "M6_ctx": dM6dt,
        "N6_ctx": dN6dt,
        "H6_ctx": dH6dt,
        "MM6_ctx": dMM6dt,
        "Hca6_ctx": dHca6dt,
        "cai6_ctx": dcai6dt,
        "S6_ctx": dS6dt,

        "V7_ctx": dV7dt,
        "M7_ctx": dM7dt,
        "N7_ctx": dN7dt,
        "H7_ctx": dH7dt,
        "S7_ctx": dS7dt,
    }



# initial values and params
key = jax.random.PRNGKey(0)
(
    key_v1, key_v2, key_v3, key_v4,
    key_v5d, key_v5i, key_v6, key_v7,
    key_w
) = jax.random.split(key, 9)

V1_init = -62
V2_init = -62
V3_init = -62
V4_init = -62
V5d_init = -60
V5i_init = -60
V6_init = -60
V7_init = -60
vtraub = params["vtraub"]

# noise amplitude in mV
sigma_init = 2.0

V1_0 = V1_init + sigma_init * jax.random.normal(key_v1, (n_th,))
V2_0 = V2_init + sigma_init * jax.random.normal(key_v2, (n_stn,))
V3_0 = V3_init + sigma_init * jax.random.normal(key_v3, (n_gpe,))
V4_0 = V4_init + sigma_init * jax.random.normal(key_v4, (n_gpi,))
V5d_0 = V5d_init + sigma_init * jax.random.normal(key_v5d, (n_dstr,))
V5i_0 = V5i_init + sigma_init * jax.random.normal(key_v5i, (n_istr,))
V6_0 = V6_init + sigma_init * jax.random.normal(key_v6, (n_ctx_pyr,))
V7_0 = V7_init + sigma_init * jax.random.normal(key_v7, (n_ctx_fsi,))

y0 = {
    "V1_th":  V1_0,
    "H1_th":  th_hinf(V1_0),
    "R1_th":  th_rinf(V1_0),

    "V2_stn":  V2_0,
    "N2_stn":  stn_ninf(V2_0),
    "H2_stn":  stn_hinf(V2_0),
    "R2_stn":  stn_rinf(V2_0),
    "C2_stn":  stn_cinf(V2_0),
    "CA2_stn": jnp.full((n_stn,), 0.1),
    "S2_stn":  jnp.zeros((n_stn,)),
    "Z2_stn":  jnp.zeros((n_stn,)),

    "V3_gpe":  V3_0,
    "N3_gpe":  gpe_ninf(V3_0),
    "H3_gpe":  gpe_hinf(V3_0),
    "R3_gpe":  gpe_rinf(V3_0),
    "CA3_gpe": jnp.full((n_gpe,), 0.1),
    "S3_gpe":  jnp.zeros((n_gpe,)),

    "W": jax.random.uniform(key_w, (n_stn, n_gpe), minval=0.05, maxval=0.20),
    "x_pre": jnp.zeros((n_gpe,)),
    "x_post": jnp.zeros((n_stn,)),

    "V4_gpi":  V4_0,
    "N4_gpi":  gpe_ninf(V4_0),
    "H4_gpi":  gpe_hinf(V4_0),
    "R4_gpi":  gpe_rinf(V4_0),
    "CA4_gpi": jnp.full((n_gpi,), 0.1),
    "S4_gpi":  jnp.zeros((n_gpi,)),
    "Z4_gpi":  jnp.zeros((n_gpi,)),

    "V5_dstr":  V5d_0,
    "m5_dstr":  str_alpham(V5d_0) / (str_alpham(V5d_0) + str_betam(V5d_0)),
    "h5_dstr":  str_alphah(V5d_0) / (str_alphah(V5d_0) + str_betah(V5d_0)),
    "n5_dstr":  str_alphan(V5d_0) / (str_alphan(V5d_0) + str_betan(V5d_0)),
    "p5_dstr":  str_alphap(V5d_0) / (str_alphap(V5d_0) + str_betap(V5d_0)),
    "S5_dstr":  jnp.full((n_dstr,), 0.1),
    "S52_dstr": jnp.zeros((n_dstr,)),
    "Z52_dstr": jnp.zeros((n_dstr,)),

    "V5_istr":  V5i_0,
    "m5_istr":  str_alpham(V5i_0) / (str_alpham(V5i_0) + str_betam(V5i_0)),
    "h5_istr":  str_alphah(V5i_0) / (str_alphah(V5i_0) + str_betah(V5i_0)),
    "n5_istr":  str_alphan(V5i_0) / (str_alphan(V5i_0) + str_betan(V5i_0)),
    "p5_istr":  str_alphap(V5i_0) / (str_alphap(V5i_0) + str_betap(V5i_0)),
    "S5_istr":  jnp.full((n_istr,), 0.1),
    "S52_istr": jnp.zeros((n_istr,)),
    "Z52_istr": jnp.zeros((n_istr,)),

    "V6_ctx":  V6_0,
    "N6_ctx":  ctx_ninf(V6_0, vtraub[0]),
    "H6_ctx":  ctx_hinf(V6_0, vtraub[0]),
    "M6_ctx":  ctx_minf(V6_0, vtraub[0]),
    "MM6_ctx": ctx_minf_m(V6_0),
    "Hca6_ctx": ctx_hinf_ca(V6_0),
    "cai6_ctx": 2.4e-4 * jnp.ones((n_ctx_pyr,)),
    "S6_ctx":  jnp.zeros((n_ctx_pyr,)),

    "V7_ctx":  V7_0,
    "M7_ctx":  ctx_minf(V7_0, vtraub[1]),
    "H7_ctx":  ctx_hinf(V7_0, vtraub[1]),
    "N7_ctx":  ctx_ninf(V7_0, vtraub[1]),
    "S7_ctx":  jnp.zeros((n_ctx_fsi,)),
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
    all_V5d= []
    all_V5i= []
    all_V6 = []
    all_V7 = []
    all_W = []

    # looping across chunks 
    # using jit compilation inside each chunk (run_chunk)
    while t0 < tmax:
        t1 = min(t0 + chunk_size, tmax)
        ts, ys, y = run_chunk(y, params, t0, t1, dt0, dt_save)

        all_ts.append(ts)
        all_V1.append(ys["V1_th"])  # shape (len(ts), n_th)
        all_V2.append(ys["V2_stn"])   
        all_V3.append(ys["V3_gpe"])
        all_V4.append(ys["V4_gpi"])
        all_V5d.append(ys["V5_dstr"])
        all_V5i.append(ys["V5_istr"])
        all_V6.append(ys["V6_ctx"])
        all_V7.append(ys["V7_ctx"])
        all_W.append(ys["W"])

        t0 = float(t1)

    return (
        jnp.concatenate(all_ts),
        jnp.concatenate(all_V1),
        jnp.concatenate(all_V2),
        jnp.concatenate(all_V3),
        jnp.concatenate(all_V4),
        jnp.concatenate(all_V5d),
        jnp.concatenate(all_V5i),
        jnp.concatenate(all_V6),
        jnp.concatenate(all_V7),
        jnp.concatenate(all_W)
    )



#%% run simulation
tmax = 1000.0
chunk_size = 100.0      # 1 second per chunk
dt0 = 0.1
dt_save = 1.0            # save every 1 ms

ts, V1, V2, V3, V4, V5d, V5i, V6, V7, W = simulate_chunked(y0, params, tmax, chunk_size, dt0, dt_save)

population_voltages = {
    "TH": V1,
    "STN": V2,
    "GPe": V3,
    "GPi": V4,
    "dStr": V5d,
    "iStr": V5i,
    "PYR": V6,
    "FSI": V7,
}

#%%
# plot to check
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
plt.title("Cortex (PYR)")
plt.show()

# plot to check
plt.plot(ts, V7[:,1])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("Cortex (FSI)")
plt.show()

# plot to check
plt.plot(ts, W[:,1])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("W")
plt.show()

# %% model validation
# 1. mean Hz rate
results = compute_metrics_all_populations(
    population_voltages=population_voltages,
    dt_ms=1.0,
    spike_height_map={
        "GPe": -20.0,
        "GPi": -20.0,
        "TH": -20.0,
        "STN": -20.0,
        "PYR": 0.0,
        "FSI": -20.0,
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
    population_order=["GPe", "STN", "GPi", "TH", "PYR", "FSI", "dStr", "iStr"],
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
        "PYR": 0.0,
        "FSI": -20.0,
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
    n_neurons=n_gpi
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
