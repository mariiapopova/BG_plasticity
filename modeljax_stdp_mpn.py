
#%%
# choose device to do calculations on
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.chdir(r"/home/mpopova/projects/vCR/BG_plasticity")

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
from dbssyn import *

jax.config.update("jax_enable_x64", True)
print(jax.devices())

#%%
# define neuron parameters 
#number of neurons per nucleus
default_n = 4
n_th = default_n
n_stn = default_n
n_gpe = default_n
n_gpi = default_n
n_dstr = default_n
n_istr = default_n
n_ctx_fsi = default_n * 2
n_ctx_pyr = default_n * 20 

#%% parameters
# parameters in a dict (PyTree)
base_params = {
    "pd": 0,
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

    # synapse params (Rubin, 2004)
    # in order of Igith, Igesn, Isnge, Igege, Igegi, Isngi, igestri, Igestrd
    "A": jnp.array([2.0 , 2.0 , 3.0, 2.0, 2.0, 3.0]),
    "B": jnp.array([0.04, 0.04, 0.1, 0.04, 0.04, 0.1]),
    "the": jnp.array([20, 20, 30, 20, 20, 30]),
    "gsyn": jnp.array([0.08, 1, 0.3, 1, 1, 0.3, 1, 1]),
    "Esyn": jnp.array([-85, -85, 0, -85, -85, 0, -85, -85]),
    "tau": 5, "gpeak1": 0.3, "gpeak": 0.43, #parameters for second-order alpha synapse

    # synapse params (Karamavelu, 2004)
    # in order of Istrstr, Istrge, Istrgi 
    "gsynstr": jnp.array([0.8, 0.5, 0.5]),
    "Esynstr": jnp.array([-80, -85, -85]),
    "ggaba": 0.1, "tau_i": 13,

    # synapse params ctx (Santiniello, 2019)
    # in order of Ipyr, Ifsi
    "gsynctx": jnp.array([0.3, 0.3, 1, 1, 0.09, 0.09, 0.3, 0.1, 0.3, 0.3, 0.3, 0.3]),
    "Esynctx": jnp.array([0, -80]),

    #thalamic synapses
    "gsynth": jnp.array([0.3]),
    "Esynth": jnp.array([0]),

    # stdp params
    "tau_pre": 12.0,
    "tau_post": 27.5,
    "tau2_pre": 2,
    "tau2_post": 4,

    # connectivity matrix
    # 1 : 1 connectivity
    #"w_gpe_stn": jnp.eye(n_stn, n_gpe),
    #"w_stn_gpe": jnp.eye(n_gpe, n_stn),
    "w_gpi_th":  jnp.eye(n_th, n_gpi),
    #"w_gpe_gpi": jnp.eye(n_gpi, n_gpe),
    #"w_stn_gpi": jnp.eye(n_gpi, n_stn),
    "w_istr_gpe": jnp.eye(n_gpe, n_istr),
    "w_dstr_gpi": jnp.eye(n_gpi, n_dstr),
    "w_gpe_istr": jnp.eye(n_istr, n_gpe),
    "w_gpe_dstr": jnp.eye(n_dstr, n_gpe),
    # 2 : 1 connectivity for self-inhibtion
    # "w_stn_gpe": jnp.array([
    #     [1,1,0,0],
    #     [0,1,1,0],
    #     [0,0,1,1],
    #     [1,0,0,1],], dtype=jnp.float64),
    # "w_stn_gpi": jnp.array([
    #     [1,1,0,0],
    #     [0,1,1,0],
    #     [0,0,1,1],
    #     [1,0,0,1],], dtype=jnp.float64),
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
    # "w_gpe_gpe": jnp.array([
    #     [0,0,1,1],
    #     [1,0,0,1],
    #     [1,1,0,0],
    #     [0,1,1,0],], dtype=jnp.float64),
    # 3 : 1 connectivity in direct striatum
    "w_dstr": jnp.array([
        [0,1,1,1],
        [1,0,1,1],
        [1,1,0,1],
        [1,1,1,0],], dtype=jnp.float32),
    # 4 : 1 connectivity in indirect striatum
    "w_istr": jnp.ones((n_istr, n_istr), dtype=jnp.float32),
    
    # all-to-all connectivity in FSI-FSI CTX
    "w_fsi": jnp.ones((n_ctx_fsi, n_ctx_fsi), dtype=jnp.float64) - jnp.eye(n_ctx_fsi, dtype=jnp.float64),
    # all-to-five connectivity in PYR-PYR CTX
    "w_pyr": w_matrix_divergent(jax.random.PRNGKey(0), n = n_ctx_pyr,p= n_ctx_pyr, k = 5),#w_matrix(n = n_ctx_pyr, k = 1),
    # all-to-all connectivity in PYR-FSI CTX
    "w_pyr_fsi": jnp.ones((n_ctx_fsi, n_ctx_pyr), dtype=jnp.float64),
    # all-to-all connectivity in FSI-PYR CTX
    "w_fsi_pyr": jnp.ones((n_ctx_pyr, n_ctx_fsi), dtype=jnp.float64),
    # all-to-all in PYR CTX - Str
    "w_pyr_str": w_matrix_random(jax.random.PRNGKey(0), n = n_ctx_pyr,p= n_dstr, k = 5),
    "w_pyr_stn": w_matrix_random(jax.random.PRNGKey(0), n = n_ctx_pyr,p= n_stn, k = 5),
    "w_pyr_th": w_matrix_random(jax.random.PRNGKey(0), n = n_ctx_pyr,p= n_th, k = 4),
    #"w_pyr_str":  jnp.ones((n_dstr, n_ctx_pyr), dtype=jnp.float32) * cD1(DA),
    "w_th_pyr": w_matrix_divergent(jax.random.PRNGKey(0), n = n_th,p= n_ctx_pyr, k = 6),
    "w_th_fsi": w_matrix_divergent(jax.random.PRNGKey(0), n = n_th,p= n_ctx_fsi, k = 2),
}

#%% main function
# defining the functions for ODEterm
def gpe_rhs(t, y, args):
    print("Compile gpe_rhs")

#%% initialization gpe_rhs
    params = args

    # TH
    S1  = y["S1_th"]
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

    # M1 CTX PYR & FSI
    # PYR
    V6 = y["V6_ctx"]
    N6  = y["N6_ctx"]
    H6  = y["H6_ctx"]
    R6  = y["R6_ctx"] 
    C6  = y["C6_ctx"]
    CA6 = y["CA6_ctx"]
    S6 = y["S6_ctx"]
    
    #FSI
    V7 = y["V7_ctx"]
    N7  = y["N7_ctx"]
    H7  = y["H7_ctx"]
    R7  = y["R7_ctx"] 
    C7  = y["C7_ctx"]
    CA7 = y["CA7_ctx"]
    S7 = y["S7_ctx"]


    # STDP state variables
    x_pre = y["x_pre"]
    x_post = y["x_post"]
    W = y["W"]
    x1_pre = y["x1_pre"]
    x1_post = y["x1_post"]
    W1 = y["W1"]
    x2_pre = y["x2_pre"]
    x2_post = y["x2_post"]
    W2 = y["W2"]

    pd   = params["pd"]
    Idbs_current = params["Idbs_current"]
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
 
    gsynctx = params["gsynctx"]; Esynctx = params["Esynctx"]; 

    gsynth = params["gsynth"]; Esynth = params["Esynth"]

    tau_pre = params["tau_pre"]
    tau_post = params["tau_post"]
    tau2_pre = params["tau2_pre"]
    tau2_post = params["tau2_post"]

    w_gpe_stn = params["w_gpe_stn"]
    #w_stn_gpe = params["w_stn_gpe"] 
    w_gpi_th = params["w_gpi_th"]
    w_gpe_gpi = params["w_gpe_gpi"]
    w_gpe_istr = params["w_gpe_istr"]
    w_gpe_dstr = params["w_gpe_dstr"]
    #w_stn_gpi = params["w_stn_gpi"]
    #w_gpe_gpe = params["w_gpe_gpe"]
    w_istr = params["w_istr"]
    w_dstr = params["w_dstr"]
    w_dstr_gpi = params["w_dstr_gpi"]
    w_istr_gpe = params["w_istr_gpe"]
    w_pyr = params["w_pyr"]
    w_fsi = params["w_fsi"]
    w_pyr_fsi = params["w_pyr_fsi"]
    w_fsi_pyr = params["w_fsi_pyr"]
    w_pyr_str = params["w_pyr_str"]
    w_pyr_stn = params["w_pyr_stn"]
    w_pyr_th = params["w_pyr_th"]
    w_th_pyr = params["w_th_pyr"] 
    w_th_fsi = params["w_th_fsi"] 


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


#%% currents gpe_rhs
    
    t_idx = jnp.int32(t / dt0)
    Idbs = Idbs_current[t_idx]

    # thalamic currents
    Il1  = gl[0]  * (V1 - El[0])
    Ina1 = gna[0] * (m1**3) * H1 * (V1 - Ena[0])
    Ik1  = gk[0]  * ((0.75 * (1 - H1))**4) * (V1 - Ek[0])  # misspelled in So paper
    It1  = gt[0]  * (p1**2) * R1 * (V1 - Et)

    # as Istim from og script (change later to input from motor cortex)
    Iapp_th = 0

    # ion currents stn
    Il2   = gl[1]   * (V2 - El[1])
    Ik2   = gk[1]   * (N2**4) * (V2 - Ek[1])
    Ina2  = gna[1]  * (m2**3) * H2 * (V2 - Ena[1])
    It2   = gt[1]   * (a2**3) * (b2**2) * (V2 - Eca[1])   
    Ica2  = gca[1]  * (c2**2) * (V2 - Eca[1])
    Iahp2 = gahp[1] * (V2 - Ek[1]) * (CA2 / (CA2 + k1[1]))

    # applied current stn    
    Iappstn = 37 #20

    # currents gpe
    Il3  = gl[2]  * (V3 - El[2])
    Ik3  = gk[2]  * (N3**4) * (V3 - Ek[2])
    Ina3 = gna[2] * (m3**3) * H3 * (V3 - Ena[2])
    It3  = gt[2]  * (a3**3) * R3 * (V3 - Eca[2])
    Ica3 = gca[2] * (s3**2) * (V3 - Eca[2])
    Iahp3 = gahp[2] * (V3 - Ek[2]) * (CA3 / (CA3 + k1[2]))

    # applied current gpe
    Iappgpe = 7#5 #+ 2*pd

    # currents gpi
    Il4  = gl[3]  * (V4 - El[3])
    Ik4  = gk[3]  * (N4**4) * (V4 - Ek[3])
    Ina4 = gna[3] * (m4**3) * H4 * (V4 - Ena[3])
    It4  = gt[3]  * (a4**3) * R4 * (V4 - Eca[3])
    Ica4 = gca[3] * (s4**2) * (V4 - Eca[3])
    Iahp4 = gahp[3] * (V4 - Ek[3]) * (CA4 / (CA4 + k1[3]))

    # applied current gpi
    Iappgpi = 15.5#10 #+ 2*pd

    # currents str (direct and indirect)
    Ina5d = gna[4] * (m5d**3) * h5d * (V5d - Ena[4])
    Ik5d =  gk[4]  * (n5d**4) * (V5d - Ek[4])
    Il5d =  gl[4]  * (V5d - El[4]) 
    Im5d = (2.6 - 0.2 *pd) * gm * p5d * (V5d - Em)
    Iappstrd = 0.4

    Ina5i = gna[4] * (m5i**3) * h5i * (V5i - Ena[4])
    Ik5i =  gk[4]  * (n5i**4) * (V5i - Ek[4])
    Il5i =  gl[4]  * (V5i - El[4]) 
    Im5i = (2.6 - 0.2 *pd) * gm * p5i * (V5i - Em)
    Iappstri = 0.6

    #currents cortex - like stn
    # CTX PYR
    Il6   = gl[1]   * (V6 - El[1])
    Ik6   = gk[1]   * (N6**4) * (V6 - Ek[1])
    Ina6  = gna[1]  * (m6**3) * H6 * (V6 - Ena[1])
    It6  = gt[1]   * (a6**3) * (b6**2) * (V6 - Eca[1])   
    Ica6  = gca[1]  * (c6**2) * (V6 - Eca[1])
    Iahp6 = gahp[1] * (V6 - Ek[1]) * (CA6 / (CA6 + k1[1]))
    # applied current 
    Iappctx6 = 10

    # CTX FSI
    Il7   = gl[1]   * (V7 - El[1])
    Ik7   = gk[1]   * (N7**4) * (V7 - Ek[1])
    Ina7  = gna[1]  * (m7**3) * H7 * (V7 - Ena[1])
    It7  = gt[1]   * (a7**3) * (b7**2) * (V7 - Eca[1])   
    Ica7  = gca[1]  * (c7**2) * (V7 - Eca[1])
    Iahp7 = gahp[1] * (V7 - Ek[1]) * (CA7 / (CA7 + k1[1]))
    # applied current 
    Iappctx7 = 30


    # synapses with connectivity matrices

    # presynaptic activation
    H_th = Hinf(V1, theta=-20.0)
    H_gpe = Hinf(V3, theta=20.0)
    H_stn = Hinf(V2, theta=30.0)
    H_gpi = Hinf(V4, theta=20.0)
    H_istr = Hinf(V5i, theta=20.0)
    H_dstr = Hinf(V5d, theta=20.0)
    H_ctxpyr = Hinf(V6, theta=30.0)
    H_ctxfi = Hinf(V7, theta=30.0)

    # differential equations synapses
    dS1dt = A[0] * (1 - S1) * H_th - B[0] * S1 #like rubin and terman gpi

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

    # STN-GPe STDP
    #spike indicators
    pre_spike = H_stn
    post_spike = H_gpe

    #STN-GPI
    pre_spike1 = H_stn
    post_spike1 = H_gpi

    # #GPe-GPe
    pre_spike2 = H_gpe
    post_spike2 = H_gpe

    # STDP traces
    dx_pre_dt  = -x_pre/tau_pre  + pre_spike
    dx_post_dt = -x_post/tau_post + post_spike

    dx1_pre_dt  = -x1_pre/tau_pre  + pre_spike1
    dx1_post_dt = -x1_post/tau_post + post_spike1

    dx2_pre_dt  = -x2_pre/tau2_pre  + pre_spike2
    dx2_post_dt = -x2_post/tau2_post + x2_pre

    # STDP weight rule
    A_plus  = 0.002
    A_minus = 0.002 * 1.1

    A2_plus = 1
    A2_minus = 5/8

    dWdt_raw = (
        A_plus  * jnp.outer(post_spike, x_pre)
        - A_minus * jnp.outer(x_post, pre_spike)
    )

    W_proj = jnp.clip(W + dWdt_raw, 0.0, 1.0)
    dWdt = W_proj - W 

    dW1dt_raw = (
        A_plus  * jnp.outer(post_spike1, x1_pre)
        - A_minus * jnp.outer(x1_post, pre_spike1)
    )

    W1_proj = jnp.clip(W1 + dW1dt_raw, 0.0, 1.0)
    dW1dt = W1_proj - W1 

    dW2dt_raw = (
        -A2_plus * jnp.outer(post_spike2, x2_pre)
        + A2_minus * jnp.outer(x2_post, pre_spike2)
    )

    dW2dt_raw = 0.004*dW2dt_raw

    W2_proj = jnp.clip(W2 +dW2dt_raw, 0.0001, 0.6)
    dW2dt = W2_proj - W2

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
    dS5idt = (str_Ggaba(V5i) * (1.0 - S5i)) - (S5i /tau_i)
    dS5ddt = (str_Ggaba(V5d) * (1.0 -S5d)) - (S5d /tau_i)

    # CTX
    dS6dt = A[2] * (1 - S6) * H_ctxpyr - B[2] * S6 #like rubin and terman gpi
    dS7dt = A[2] * (1 - S7) * H_ctxfi - B[2] * S7 #like rubin and terman gpi

    # synaptic currents using those gating variables
    Igith = 1.4 *  (gsyn[0] * (V1 - Esyn[0]) * (w_gpi_th @ S4))
    Igesn = 0.5 *  ((gsyn[1]-0.6*pd) * (V2 - Esyn[1]) * (w_gpe_stn @ S3))
    Isnge =  0.25*(gsyn[2] * (V3 - Esyn[2]) * ( W  @ S2))
    #Igege = 0.5 *  (gsyn[3] * (V3 - Esyn[3]) *  (w_gpe_gpe @ S3))
    Igege = 0.25 *  ((0.8*pd+gsyn[3]) * (V3 - Esyn[3]) *  (W2 @ S3))
    #Igege=0.5*((gsyn[2]+0.8*pd)*(V3-Esyn[2])*(w_1fin*S3_1+w_2fin*S3_2))
    Igegi = 0.5 *  (gsyn[4] * (V4 - Esyn[4]) *  (w_gpe_gpi @ S3))
    #Isngi = 0.5 *  (gsyn[5] * (V4 - Esyn[5]) *  (w_stn_gpi  @ S2))
    Isngi =  0.25*((gsyn[5]+0.6*pd) * (V4 - Esyn[5]) *  (W1  @ S2))
    Istrge = gsynstr[1] * (V3 - Esynstr[1]) * (w_istr_gpe  @ S5i_2) # from in FOG: Istrge = 0.1 * ... - why?
    Istrgi = gsynstr[1] * (V4 - Esynstr[1]) * (w_dstr_gpi  @ S5d_2)
    Istrd = (ggaba/3) * (V5d - Esynstr[0]) * (w_dstr @ S5d)
    Istri = (ggaba/4) * (V5i - Esynstr[0]) * (w_istr @ S5i)
    Ipypy = 0.2*gsynctx[0] * (V6 - Esynctx[0]) * (w_pyr  @ S6) # change to normalizing by number of presynaptic input neurons
    Ipyfi = (1/n_ctx_pyr)*gsynctx[1] * (V7 - Esynctx[0]) * (w_pyr_fsi  @ S6) 
    Ififi = (1/n_ctx_fsi)*gsynctx[2] * (V7 - Esynctx[1]) * (w_fsi  @ S7) 
    Ifipy = (1/n_ctx_fsi)*gsynctx[3] * (V6 - Esynctx[1]) * (w_fsi_pyr  @ S7) 
    Ipystrd = 0.2*(5*gsynctx[5] -0.3*pd) * (V5d - Esynctx[0]) * (w_pyr_str  @ S6)  # check for valid parameters here 
    Ipystri = 0.2*(5*gsynctx[4]) * (V5i - Esynctx[0]) * (w_pyr_str  @ S6)  # check for valid parameters here 
    Ipysn = 0.2*gsynctx[6] * (V2 - Esynctx[0]) * (w_pyr_stn  @ S6)  # check for valid parameters here 
    Ipyth = 0.25*gsynctx[7] * (V1 - Esynctx[0]) * (w_pyr_th  @ S6)  # check for valid parameters here 
    Ithpy = gsynth[0] * (V6 - Esynth[0]) * (w_th_pyr  @ S1)  # check for valid parameters here 
    Ithfi = gsynth[0] * (V7 - Esynth[0]) * (w_th_fsi  @ S1)  # check for valid parameters here 
    Igestri = (gsyn[6] * (V5i - Esyn[6]) *  (w_gpe_istr @ S3))
    Igestrd = (gsyn[7] * (V5d - Esyn[7]) *  (w_gpe_dstr @ S3))

#%% dif equation updates gpe_rhs
    # differential equations th
    dV1dt = (-Il1 - Ik1 - Ina1 - It1 - Igith - Ipyth + Iapp_th) / Cm
    dH1dt   = (h1 - H1) / th1
    dR1dt   = (r1 - R1) / tr1

    # differential equations stn
    dV2dt   = (-Il2 - Ik2 - Ina2 - It2 - Ica2 - Iahp2 - Igesn - Ipysn + Iappstn + 15*Idbs) / Cm
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
    dV5ddt = (-Il5d - Ik5d - Ina5d - Im5d - Istrd - Ipystrd - Igestrd +Iappstrd) / Cm
    dm5ddt = str_alpham(V5d) * (1 - m5d) - str_betam(V5d) * m5d
    dh5ddt = str_alphah(V5d) * (1 - h5d) - str_betah(V5d) * h5d
    dn5ddt = str_alphan(V5d) * (1 - n5d) - str_betan(V5d) * n5d
    dp5ddt = str_alphap(V5d) * (1 - p5d) - str_betap(V5d) * p5d

    dV5idt = (-Il5i - Ik5i - Ina5i - Im5i - Istri - Ipystri - Igestri +Iappstri) / Cm
    dm5idt = str_alpham(V5i) * (1 - m5i) - str_betam(V5i) * m5i
    dh5idt = str_alphah(V5i) * (1 - h5i) - str_betah(V5i) * h5i
    dn5idt = str_alphan(V5i) * (1 - n5i) - str_betan(V5i) * n5i
    dp5idt = str_alphap(V5i) * (1 - p5i) - str_betap(V5i) * p5i

    # differential equations CTX
    dV6dt   = (-Il6 - Ik6 - Ina6 - It6 - Ica6 - Iahp6 - Ipypy - Ifipy - Ithpy + Iappctx6) / Cm
    dN6dt   = 0.75 * (n6 - N6) / tn6
    dH6dt   = 0.75 * (h6 - H6) / th6
    dR6dt   = 0.2 * (r6 - R6) / tr6
    dCA6dt  = 3.75 * 1e-5 * (-Ica6 - It6 - kca[1] * CA6)
    dC6dt   = 0.08 * (c6 - C6) / tc6

    dV7dt   = (-Il7 - Ik7 - Ina7 - It7 - Ica7 - Iahp7 - Ipyfi - Ififi  - Ithfi + Iappctx7) / Cm
    dN7dt   = 0.75 * (n7 - N7) / tn7
    dH7dt   = 0.75 * (h7 - H7) / th7
    dR7dt   = 0.2 * (r7 - R7) / tr7
    dCA7dt  = 3.75 * 1e-5 * (-Ica7 - It7 - kca[1] * CA7)
    dC7dt   = 0.08 * (c7 - C7) / tc7

    return {
        "S1_th":  dS1dt,
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
        "W2": dW2dt,
        "x2_pre": dx2_pre_dt,
        "x2_post": dx2_post_dt,

        "V4_gpi":  dV4dt,
        "N4_gpi":  dN4dt,
        "H4_gpi":  dH4dt,
        "R4_gpi":  dR4dt,
        "CA4_gpi": dCA4dt,
        "S4_gpi": dS4dt,
        "Z4_gpi": dZ4dt,
        "W1": dW1dt,
        "x1_pre": dx1_pre_dt,
        "x1_post": dx1_post_dt,

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
        "N6_ctx":  dN6dt,
        "H6_ctx":  dH6dt,
        "R6_ctx":  dR6dt,
        "C6_ctx":  dC6dt,
        "CA6_ctx": dCA6dt,
        "S6_ctx": dS6dt,

        "V7_ctx": dV7dt,
        "N7_ctx":  dN7dt,
        "H7_ctx":  dH7dt,
        "R7_ctx":  dR7dt,
        "C7_ctx":  dC7dt,
        "CA7_ctx": dCA7dt,
        "S7_ctx": dS7dt,

    }



# initial values and params
key = jax.random.PRNGKey(0)
(
    key_v1, key_v2, key_v3, key_v4,
    key_v5d, key_v5i, key_v6, key_v7,
    key_w, key_w1, key_w2
) = jax.random.split(key, 11)

V1_init = -62
V2_init = -62
V3_init = -62
V4_init = -62
V5d_init = -60
V5i_init = -60
V6_init = -62
V7_init = -62

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
    "S1_th":  jnp.zeros((n_th,)),
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

    "W": jax.random.uniform(key_w, (n_gpe, n_stn), minval=0.85, maxval=1),
    "x_pre": jnp.zeros((n_stn,)),
    "x_post": jnp.zeros((n_gpe,)),
    "W2": jax.random.uniform(key_w2, (n_gpe, n_gpe), minval=0.45, maxval=0.6),
    "x2_pre": jnp.zeros((n_gpe,)),
    "x2_post": jnp.zeros((n_gpe,)),

    "V4_gpi":  V4_0,
    "N4_gpi":  gpe_ninf(V4_0),
    "H4_gpi":  gpe_hinf(V4_0),
    "R4_gpi":  gpe_rinf(V4_0),
    "CA4_gpi": jnp.full((n_gpi,), 0.1),
    "S4_gpi":  jnp.zeros((n_gpi,)),
    "Z4_gpi":  jnp.zeros((n_gpi,)),

    "W1": jax.random.uniform(key_w1, (n_gpi, n_stn), minval=0.85, maxval=1),
    "x1_pre": jnp.zeros((n_stn,)),
    "x1_post": jnp.zeros((n_gpi,)),

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
    "N6_ctx":  stn_ninf(V6_0),
    "H6_ctx":  stn_hinf(V6_0),
    "R6_ctx":  stn_rinf(V6_0),
    "C6_ctx":  stn_cinf(V6_0),
    "CA6_ctx": jnp.full((n_ctx_pyr,), 0.1),
    "S6_ctx":  jnp.zeros((n_ctx_pyr,)),

    "V7_ctx":  V7_0,
    "N7_ctx":  stn_ninf(V7_0),
    "H7_ctx":  stn_hinf(V7_0),
    "R7_ctx":  stn_rinf(V7_0),
    "C7_ctx":  stn_cinf(V7_0),
    "CA7_ctx": jnp.full((n_ctx_fsi,), 0.1),
    "S7_ctx":  jnp.zeros((n_ctx_fsi,)),
}

#%% chunked Euler solver
def run_chunk_euler_scan(y0, params, t0, dt,chunk_length):
    n_steps =chunk_length//dt
    ts = t0 + dt * jnp.arange(n_steps)
    print("Compile chunk")

    def euler_step(y, t):
        dy = gpe_rhs(t, y, params)
        y_next = jax.tree.map(lambda y, dy: y + dt * dy, y, dy)
        return y_next, y_next

    y_final, ys = jax.lax.scan(euler_step, y0, ts)
    return y_final, (ts, ys)


run_chunk_euler_scan = jax.jit(run_chunk_euler_scan,static_argnames=('dt','chunk_length'))


def simulate_last_chunk_euler(
    y0, params, tmax, dt=0.1, dt_save=1, chunk_length=1000
):
    n_chunks = int(tmax // chunk_length)

    # --- get shapes for initialization (no real compute) ---
    _, (ts_shape, ys_shape) = jax.eval_shape(
        lambda y: run_chunk_euler_scan(y, params, 0.0, dt, chunk_length),
        y0,
    )

    init_ts = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), ts_shape)
    init_ys = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), ys_shape)

    def step(carry, i):
        y, t0, last_ts, last_ys = carry

        y_next, (ts, ys) = run_chunk_euler_scan(
            y, params, t0, dt=dt, chunk_length=chunk_length
        )

        # detect last iteration
        is_last = (i == n_chunks - 1)

        # update only on last step
        last_ts = jax.lax.select(is_last, ts, last_ts)
        last_ys = jax.tree.map(
            lambda new, old: jax.lax.select(is_last, new, old),
            ys,
            last_ys,
        )

        return (y_next, t0 + chunk_length, last_ts, last_ys), None

    (y_final, t_final, last_ts, last_ys), _ = jax.lax.scan(
        step,
        (y0, 0.0, init_ts, init_ys),
        xs=jnp.arange(n_chunks),
    )

    return (
        last_ts,
        last_ys["V1_th"],
        last_ys["V2_stn"],
        last_ys["V3_gpe"],
        last_ys["V4_gpi"],
        last_ys["V5_dstr"],
        last_ys["V5_istr"],
        last_ys["V6_ctx"],
        last_ys["V7_ctx"],
        last_ys["W"],
        last_ys["W1"],
        last_ys["W2"],
    )

#%% run simulation
tmax = 10_000.0
chunk_size = 1_000.0      # 1 second per chunk
dt0 = 0.01
dt_save = 1           # save every 1 ms

Idbs_current = dbssyn(
    f=130,
    tmax=tmax,
    dt=dt0,
    sw=0,
    shift=0
)

Idbs_current_thlow = dbssyn(
    f=100,
    tmax=tmax,
    dt=dt0,
    sw=1,
    shift=0
)

Idbs_current_thhigh = dbssyn(
    f=200,
    tmax=tmax,
    dt=dt0,
    sw=1,
    shift=0
)

Idbs_current_thinter = dbssyn(
    f=150,
    tmax=tmax,
    dt=dt0,
    sw=1,
    shift=0
)

params = {
    **base_params,
    "pd": 0,
    "dt": dt0,
    "Idbs_current": jnp.zeros(int(tmax / dt0))
}

params_pd = {
    **base_params,
    "pd": 1,
    "dt": dt0,
    "Idbs_current": jnp.zeros(int(tmax / dt0))
}

params_dbs = {
    **base_params,
    "pd": 1,
    "dt": dt0,
    "Idbs_current": jnp.array(Idbs_current),
}

params_dbsthh = {
    **base_params,
    "pd": 1,
    "dt": dt0,
    "Idbs_current": jnp.array(Idbs_current_thhigh),
}

params_dbsthl = {
    **base_params,
    "pd": 1,
    "dt": dt0,
    "Idbs_current": jnp.array(Idbs_current_thlow),
}

params_dbsthi = {
    **base_params,
    "pd": 1,
    "dt": dt0,
    "Idbs_current": jnp.array(Idbs_current_thinter),
}

# ts, V1, V2, V3, V4, V5d, V5i, V6, V7, W, W1, W2 = simulate_last_chunk_euler(y0, params, tmax, dt0, dt_save, chunk_size)
# ts_pd, V1_pd, V2_pd, V3_pd, V4_pd, V5d_pd, V5i_pd, V6_pd, V7_pd, W_pd, W1_pd, W2_pd = simulate_last_chunk_euler(y0, params_pd, tmax, dt0, dt_save, chunk_size)


# population_voltages = {
#     "TH": V1,
#     "STN": V2,
#     "GPe": V3,
#     "GPi": V4,
#     "dStr": V5d,
#     "iStr": V5i,
#     "PYR M1": V6,
#     "FSI M1": V7,
# }

# population_voltages_pd = {
#     "TH": V1_pd,
#     "STN": V2_pd,
#     "GPe": V3_pd,
#     "GPi": V4_pd,
#     "dStr": V5d_pd,
#     "iStr": V5i_pd,
#     "PYR M1": V6_pd,
#     "FSI M1": V7_pd,
# }

def run_single(params):
    return simulate_last_chunk_euler(
        y0, params, tmax, dt0, dt_save, chunk_size
    )

batched_run = jax.vmap(run_single)

params_batch = jax.tree.map(
    lambda *xs: jnp.stack(xs),
    params, params_pd, params_dbs, params_dbsthh, params_dbsthl, params_dbsthi
)

results = batched_run(params_batch)

population_voltages = {
    "TH": results[1][0],
    "STN": results[2][0],
    "GPe": results[3][0],
    "GPi": results[4][0],
    "dStr": results[5][0],
    "iStr": results[6][0],
    "PYR M1": results[7][0],
    "FSI M1": results[8][0],
}

population_voltages_pd = {
    "TH": results[1][1],
    "STN": results[2][1],
    "GPe": results[3][1],
    "GPi": results[4][1],
    "dStr": results[5][1],
    "iStr": results[6][1],
    "PYR M1": results[7][1],
    "FSI M1": results[8][1],
}

population_voltages_dbs = {
    "TH": results[1][2],
    "STN": results[2][2],
    "GPe": results[3][2],
    "GPi": results[4][2],
    "dStr": results[5][2],
    "iStr": results[6][2],
    "PYR M1": results[7][2],
    "FSI M1": results[8][2],
}

population_voltages_dbsthh = {
    "TH": results[1][3],
    "STN": results[2][3],
    "GPe": results[3][3],
    "GPi": results[4][3],
    "dStr": results[5][3],
    "iStr": results[6][3],
    "PYR M1": results[7][3],
    "FSI M1": results[8][3],
}

population_voltages_dbsthl = {
    "TH": results[1][4],
    "STN": results[2][4],
    "GPe": results[3][4],
    "GPi": results[4][4],
    "dStr": results[5][4],
    "iStr": results[6][4],
    "PYR M1": results[7][4],
    "FSI M1": results[8][4],
}

population_voltages_dbsthi = {
    "TH": results[1][5],
    "STN": results[2][5],
    "GPe": results[3][5],
    "GPi": results[4][5],
    "dStr": results[5][5],
    "iStr": results[6][5],
    "PYR M1": results[7][5],
    "FSI M1": results[8][5],
}

#%%
#plot to check = healthy
plt.plot(results[0][0], results[1][0][:,3])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("TH")
plt.show()

# plot to check
plt.plot(results[0][0], results[2][0][:,2])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("STN")
plt.show()

# plot to check
plt.plot(results[0][0], results[3][0][:,3])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("GPe")
plt.show()

# plot to check
plt.plot(results[0][0], results[4][0][:,1])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("GPi")
plt.show()

# plot to check
plt.plot(results[0][0], results[5][0][:,0])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("direct Striatum")
plt.show()

# plot to check
plt.plot(results[0][0], results[6][0][:,1])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("indirect Striatum")
plt.show()

# # plot to check
# plt.plot(ts, V6[:,3])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("Cortex M1 (PYR)")
# plt.show()

# # plot to check
# plt.plot(ts, V7[:,1])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("Cortex M1 (FSI)")
# plt.show()

# # plot to check
# plt.plot(ts, V8[:,3])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("Cortex S1 (PYR)")
# plt.show()

# # plot to check
# plt.plot(ts, V9[:,1])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("Cortex S1 (FSI)")
# plt.show()

# # plot to check
# plt.plot(ts, W[:,1])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("W, STN-GPe")
# plt.show()

# # plot to check
# plt.plot(ts, W1[:,1])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("W, STN-GPi")
# plt.show()

# # plot to check
# plt.plot(ts, W2[:,1])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("W, GPe-GPe")
# plt.show()

print("-----PD----")
#plot to check pd
plt.plot(results[0][0], results[1][1][:,3])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("TH")
plt.show()

# plot to check
plt.plot(results[0][0], results[2][1][:,2])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("STN")
plt.show()

# plot to check
plt.plot(results[0][0], results[3][1][:,3])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("GPe")
plt.show()

# plot to check
plt.plot(results[0][0], results[4][1][:,1])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("GPi")
plt.show()

# plot to check
plt.plot(results[0][0], results[5][1][:,0])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("direct Striatum")
plt.show()

# plot to check
plt.plot(results[0][0], results[6][1][:,1])
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.title("indirect Striatum")
plt.show()

# # plot to check
# plt.plot(ts, V6_pd[:,3])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("Cortex M1 (PYR)")
# plt.show()

# # plot to check
# plt.plot(ts, V7_pd[:,1])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("Cortex M1 (FSI)")
# plt.show()

# # plot to check
# plt.plot(ts, V8_pd[:,3])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("Cortex S1 (PYR)")
# plt.show()

# # plot to check
# plt.plot(ts, V9_pd[:,1])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("Cortex S1 (FSI)")
# plt.show()

# # plot to check
# plt.plot(ts, W_pd[:,1])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("W, STN-GPe")
# plt.show()

# # plot to check
# plt.plot(ts, W1_pd[:,1])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("W, STN-GPi")
# plt.show()

# # plot to check
# plt.plot(ts, W2_pd[:,1])
# plt.xlabel("t (ms)")
# plt.ylabel("V (mV)")
# plt.title("W, GPe-GPe")
# plt.show()

# %% model validation
pop_quest = population_voltages_dbs
volt_quest = results[4][0] #gpi

#1. mean Hz rate
results_met = compute_metrics_all_populations(
    population_voltages=pop_quest,
    dt_ms=dt0,
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

mean_rates = {pop: res["mean_rate_hz"] for pop, res in results_met.items()}

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
    results_met,
    population_order=["GPe", "STN", "GPi", "TH", "PYR M1", "FSI M1", "PYR S1", "FSI S1", "dStr", "iStr"],
)
# #%%
# 2. ISI CV

irregularity_results = compute_irregularity_all_populations(
    population_voltages=pop_quest,
    dt_ms=1.0,
    spike_height_map={
        "GPe": 0.0,
        "GPi": 0.0,
        "TH": -20.0,
        "STN": 0.0,
        "PYR M1": 0.0,
        "FSI M1": 0.0,
        "dStr": 0.0,
        "iStr": 0.0,
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
    population_order=["GPe", "STN", "GPi", "TH", "PYR M1", "FSI M1", "dStr", "iStr"],
    xlim=(0, 2.5),
)

# %%
# 3. PSD GPi
sig1 = results[4][0][:,0]
sig2 = results[4][1][:,0]
sig3 = results[4][2][:,0]
sig4 = results[4][3][:,0]
sig5 = results[4][4][:,0]
sig6 = results[4][5][:,0]
fft_res1 = np.abs(np.fft.rfft(np.fft.ifftshift(sig1-np.mean(sig1))))
ff_freqs1 = np.fft.rfftfreq(n= sig1.shape[0],d = 1e-5)
fft_res2 = np.abs(np.fft.rfft(np.fft.ifftshift(sig2-np.mean(sig2))))
ff_freqs2 = np.fft.rfftfreq(n= sig2.shape[0],d = 1e-5)
fft_res3 = np.abs(np.fft.rfft(np.fft.ifftshift(sig3-np.mean(sig3))))
ff_freqs3 = np.fft.rfftfreq(n= sig3.shape[0],d = 1e-5)
fft_res4 = np.abs(np.fft.rfft(np.fft.ifftshift(sig4-np.mean(sig4))))
ff_freqs4 = np.fft.rfftfreq(n= sig4.shape[0],d = 1e-5)
fft_res5 = np.abs(np.fft.rfft(np.fft.ifftshift(sig5-np.mean(sig5))))
ff_freqs5 = np.fft.rfftfreq(n= sig5.shape[0],d = 1e-5)
fft_res6 = np.abs(np.fft.rfft(np.fft.ifftshift(sig6-np.mean(sig6))))
ff_freqs6 = np.fft.rfftfreq(n= sig6.shape[0],d = 1e-5)

plt.figure()
plt.plot(ff_freqs1[10:30],fft_res1[10:30],label='Healthy')
plt.plot(ff_freqs2[10:30],fft_res2[10:30],label='PD')
plt.plot(ff_freqs3[10:30],fft_res3[10:30],label='DBS')
plt.plot(ff_freqs4[10:30],fft_res4[10:30],label='DBS theta high, 200 Hz')
plt.plot(ff_freqs5[10:30],fft_res5[10:30],label='DBS theta low, 100 Hz')
plt.plot(ff_freqs6[10:30],fft_res6[10:30],label='DBS theta inter, 150 Hz')
plt.legend()

# # %%