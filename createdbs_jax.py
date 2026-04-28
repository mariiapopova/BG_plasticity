import jax.numpy as jnp

# JAX-compatible DBS waveform
def createdbs_jax(freq, tmax, dt, pulse_width=0.3, amplitude=300.0):

    n_steps = int(round(tmax / dt)) + 1  # do we need +1?
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

# look up function for in the rhs 
def dbs_current(t, cfg):
    dt = cfg["dt"]
    Idbs = cfg["Idbs"]

    idx = jnp.clip(
        jnp.floor(t / dt).astype(jnp.int32),
        0,
        Idbs.shape[0] - 1
    )