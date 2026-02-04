import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["JAX_LOG_COMPILES"] = "1"

#%%
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import equinox as eqx
from functools import partial
from diffrax import TextProgressMeter,ODETerm, Dopri5, SaveAt, diffeqsolve, Kvaerno5, PIDController, Tsit5
import matplotlib.pyplot as plt

class HHParams(eqx.Module):
    C: float
    gNa: float
    gK: float
    gL: float
    ENa: float
    EK: float
    EL: float
    I_ext: float

class RegionState(eqx.Module):
    V: jnp.ndarray
    m: jnp.ndarray
    h: jnp.ndarray
    n: jnp.ndarray

class SynapseParams(eqx.Module):
    g_syn: float
    E_syn: float
    V_th: float = -20.0
    k: float = 5.0

class NetworkState(eqx.Module):
    stn: RegionState
    gpi: RegionState


class NetworkParams(eqx.Module):
    stn: HHParams
    gpi: HHParams
    stn_to_gpi: SynapseParams

@jax.jit
def hh_gating_dynamics(V, m, h, n):
    alpha_m = 0.1 * (V + 40) / (1 - jnp.exp(-(V + 40) / 10))
    beta_m  = 4.0 * jnp.exp(-(V + 65) / 18)

    alpha_h = 0.07 * jnp.exp(-(V + 65) / 20)
    beta_h  = 1.0 / (1 + jnp.exp(-(V + 35) / 10))

    alpha_n = 0.01 * (V + 55) / (1 - jnp.exp(-(V + 55) / 10))
    beta_n  = 0.125 * jnp.exp(-(V + 65) / 80)

    dm = alpha_m * (1 - m) - beta_m * m
    dh = alpha_h * (1 - h) - beta_h * h
    dn = alpha_n * (1 - n) - beta_n * n

    return dm, dh, dn

@jax.jit
def syn_activation(V_pre, syn: SynapseParams):
    return 1.0 / (1.0 + jnp.exp(-(V_pre - syn.V_th) / syn.k))

@jax.jit
def hh_rhs_region(state: RegionState, p: HHParams):
    dm, dh, dn = hh_gating_dynamics(state.V, state.m, state.h, state.n)

    I_Na = p.gNa * state.m**3 * state.h * (state.V - p.ENa)
    I_K  = p.gK  * state.n**4 * (state.V - p.EK)
    I_L  = p.gL * (state.V - p.EL)

    dV = (p.I_ext - I_Na - I_K - I_L) / p.C

    return RegionState(V=dV, m=dm, h=dh, n=dn)


def network_rhs(t, state: NetworkState, params: NetworkParams):
    print("Tracing / compiling f")
    # Intrinsic dynamics
    dstn = hh_rhs_region(state.stn, params.stn)
    dgpi = hh_rhs_region(state.gpi, params.gpi)

    # Synaptic current STN → GPi
    S = syn_activation(state.stn.V, params.stn_to_gpi)
    I_syn = params.stn_to_gpi.g_syn * S * (state.gpi.V - params.stn_to_gpi.E_syn)

    # Apply synaptic current to GPi voltage
    dgpi = eqx.tree_at(
        lambda x: x.V,
        dgpi,
        dgpi.V - I_syn / params.gpi.C
    )

    return NetworkState(stn=dstn, gpi=dgpi)

init_region = RegionState(
    V=jnp.array(-65.0),
    m=jnp.array(0.05),
    h=jnp.array(0.6),
    n=jnp.array(0.32),
)

y0 = NetworkState(
    stn=init_region,
    gpi=init_region,
)

params = NetworkParams(
    stn=HHParams(
        C=1.0, gNa=120.0, gK=36.0, gL=0.3,
        ENa=50.0, EK=-77.0, EL=-54.4,
        I_ext=10.0,
    ),
    gpi=HHParams(
        C=1.0, gNa=100.0, gK=30.0, gL=0.3,
        ENa=50.0, EK=-77.0, EL=-54.4,
        I_ext=5.0,
    ),
    stn_to_gpi=SynapseParams(
        g_syn=0.5,
        E_syn=0.0,
    ),
)
#ts = jnp.linspace(0, 50.0, 1000)
#%%
network_rhs = jax.jit(network_rhs)
network_rhs(10,y0,params)
#%%
# Adaptive controller for stiff solver
tmax = 1000.0
dt = 0.1
ts = jnp.arange(0.0, tmax, dt)

#@jax.jit
def simulate_gpe(y0, params):

    term = ODETerm(network_rhs)
    solver = Tsit5()

    sol = diffeqsolve(
        term,
        solver,
        t0=ts[0],
        t1=tmax,
        dt0=dt,                         # initial step size
        max_steps=100000,
        y0=y0,
        args=params,
        saveat=SaveAt(ts=ts),   # save at our time grid 
        stepsize_controller=PIDController(rtol=1e-4, atol=1e-6),
        progress_meter=TextProgressMeter(minimum_increase=0.05)
    )


    # sol.ys has shape (len(ts), 5)
    return ts, sol.ys, sol.stats["num_steps"]

#run simulation
ts, ys, stats = simulate_gpe(y0, params)
V_stn_1s = ys.stn.V
V_gpi_1s = ys.gpi.V

plt.plot(V_stn_1s, label="STN")
plt.plot(V_gpi_1s, label="GPi")
plt.legend()
plt.title("STN–GPi HH Dynamics (1 s)")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.show()
#%%
#vmapped
tmax = 1000.0
dt = 0.1
ts = jnp.arange(0.0, tmax, dt)

term = ODETerm(network_rhs)
solver = Tsit5()
saveat = SaveAt(ts=ts)
stepsize_controller = PIDController(rtol=1e-4, atol=1e-6)

def simulate_gpe_one(y0, params):
    sol = diffeqsolve(
        term,
        solver,
        t0=ts[0],
        t1=tmax,
        dt0=dt,
        max_steps=100_000,
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    return sol.ys

ys = simulate_gpe_one(y0, params)

simulate_gpe_batch = jax.vmap(
    simulate_gpe_one,
    in_axes=(0, None),   # vmap over y0, shared params
)

simulate_gpe_batch = jax.jit(simulate_gpe_batch)

y0s = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None, ...], 7, axis=0), y0)

# vmap over the first axis
simulate_gpe_batch = jax.vmap(simulate_gpe_one, in_axes=(0, None))

ys_batch = simulate_gpe_batch(y0s, params)
# controller = PIDController(rtol=1e-5, atol=1e-7)

# # Function to run the simulation
# def run_network_100s():
#     return diffeqsolve(
#         ODETerm(network_rhs),
#         Tsit5(),
#         t0=0.0,
#         t1=100000.0,             # 1 s = 1000 ms
#         y0=y0,
#         args=params,
#         dt0=0.05,
#         saveat=SaveAt(steps=100),  # save every 10 steps
#         stepsize_controller=controller,
#         max_steps=50_000_000,
#     )

# # JIT-compile the full solve
# jit_run_network_1s = jax.jit(run_network_100s)

# # Run
# sol_1s = jit_run_network_1s()
# ys_1s = sol_1s.ys

# V_stn_1s = ys_1s.stn.V
# V_gpi_1s = ys_1s.gpi.V

# plt.plot(V_stn_1s, label="STN")
# plt.plot(V_gpi_1s, label="GPi")
# plt.legend()
# plt.title("STN–GPi HH Dynamics (1 s)")
# plt.xlabel("Time (ms)")
# plt.ylabel("Voltage (mV)")
# plt.show()
# ts = jnp.linspace(0.0, 50.0, 1000)
# controller = PIDController(rtol=1e-5, atol=1e-7)

# # sol = diffeqsolve(
# #     ODETerm(network_rhs),
# #     Kvaerno5(),
# #     t0=0.0,
# #     t1=100_000.0,       # 100 s in ms
# #     y0=y0,
# #     args=params,
# #     dt0=0.05,
# #     saveat=SaveAt(steps=100),  # save every 100 internal steps
# #     stepsize_controller=controller,
# #     max_steps=50_000_000,
# # )

# jit_run_network = jax.jit(lambda: diffeqsolve(
#     ODETerm(network_rhs),
#     Kvaerno5(),
#     t0=0.0,
#     t1=100000.0,
#     y0=y0,
#     args=params,
#     dt0=0.05,
#     saveat=SaveAt(steps=100),
#     stepsize_controller=controller,
#     max_steps=50_000_000,
# ))

# sol = jit_run_network()
# # # Define a function that performs the solve
# # def run_network(y0, params, t1):
# #     ts = jnp.linspace(0.0, t1, 1000)
# #     return diffeqsolve(
# #         ODETerm(network_rhs),
# #         Dopri5(),
# #         t0=0.0,
# #         t1=t1,
# #         dt0=0.01,
# #         y0=y0,
# #         args=params,
# #         saveat=SaveAt(ts=ts),
# #         max_steps=1_000_000,
# #     )

# # # JIT the solver
# # jit_run_network = jax.jit(run_network)

# # # Now call it
# # sol = jit_run_network(y0, params, 50.0)
# ys = sol.ys

# V_stn = ys.stn.V
# V_gpi = ys.gpi.V

# plt.plot(V_stn, label="STN")
# plt.plot(V_gpi, label="GPi")
# plt.legend()
# plt.title("STN–GPi Hodgkin–Huxley Dynamics")
# plt.xlabel("Time (ms)")
# plt.ylabel("Voltage (mV)")
# plt.show()
# #%%
# import jax
# import jax.numpy as jnp
# from diffrax import ODETerm, Dopri5, PIDController, SaveAt, diffeqsolve
# import matplotlib.pyplot as plt

# # Hodgkin-Huxley RHS (differential equations)
# def hh_rhs(t, state, args):
#     V, m, h, n = state
#     (C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_ext) = args

#     # Channel kinetics
#     alpha_m = 0.1 * (V + 40.0) / (1 - jnp.exp(-(V + 40.0) / 10.0))
#     beta_m  = 4.0 * jnp.exp(-(V + 65.0) / 18.0)
#     alpha_h = 0.07 * jnp.exp(-(V + 65.0) / 20.0)
#     beta_h  = 1.0 / (1.0 + jnp.exp(-(V + 35.0) / 10.0))
#     alpha_n = 0.01 * (V + 55.0) / (1 - jnp.exp(-(V + 55.0) / 10.0))
#     beta_n  = 0.125* jnp.exp(-(V + 65.0) / 80.0)

#     # Gating derivatives
#     dmdt = alpha_m * (1 - m) - beta_m * m
#     dhdt = alpha_h * (1 - h) - beta_h * h
#     dndt = alpha_n * (1 - n) - beta_n * n

#     # Currents
#     I_Na = g_Na * (m**3) * h * (V - E_Na)
#     I_K  = g_K  * (n**4)      * (V - E_K)
#     I_L  = g_L * (V - E_L)

#     # Voltage derivative
#     dVdt = (I_ext - I_Na - I_K - I_L) / C_m

#     return jnp.array([dVdt, dmdt, dhdt, dndt])

# # Initial conditions (e.g., resting state)
# V0 = -65.0
# m0 = 0.05
# h0 = 0.6
# n0 = 0.32
# y0 = jnp.array([V0, m0, h0, n0])

# # Parameters
# params = (
#     1.0,   # C_m
#     120.0, # g_Na
#     36.0,  # g_K
#     0.3,   # g_L
#     50.0,  # E_Na
#     -77.0, # E_K
#     -54.387,# E_L
#     10.0   # Applied current I_ext
# )

# # Set up the integrator
# term = ODETerm(hh_rhs)
# solver = Dopri5()
# ctrl   = PIDController(rtol=1e-6, atol=1e-7)
# save_at= SaveAt(ts=jnp.linspace(0, 1000.0, 10))

# # Solve
# sol = diffeqsolve(
#     term,
#     solver,
#     t0=0.0,
#     t1=1000.0,
#     dt0=0.01,
#     y0=y0,
#     args=params,
#     stepsize_controller=ctrl,
#     saveat=save_at,
# )

# # Extract and plot
# t = sol.ts
# V = sol.ys[:, 0]

# plt.plot(t, V)
# plt.title("Hodgkin–Huxley Voltage Trace")
# plt.xlabel("Time (ms)")
# plt.ylabel("Voltage (mV)")
# plt.show()