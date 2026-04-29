from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def make_turner(tmax, dt, sw):
    n = int(tmax / dt)

    # sw = 0 → continuous DBS
    if sw == 0:
        return np.ones(n)

    # sw = 1 : regular burst (50% duty cycle blocks)
    elif sw == 1:
        block = int((100.0 / dt))  # 100 ms blocks
        turner = np.zeros(n)
        toggle = True
        i = 0
        while i < n:
            length = min(block, n - i)
            val = 1.0 if toggle else 0.0
            for k in range(length):
                turner[i + k] = val
            i += length
            toggle = not toggle
        return turner

    # sw = 2 : irregular burst ( 45 / 135 ms pattern)
    elif sw == 2:
        turner = np.zeros(n)
        toggle = True
        i = 0
        while i < n:
            if toggle:
                length = int(45.0 / dt)
                val = 1.0
            else:
                length = int(3 * 45.0 / dt)
                val = 0.0

            length = min(length, n - i)

            for k in range(length):
                turner[i + k] = val

            i += length
            toggle = not toggle

        return turner

    else:
        return np.ones(n)
    

@jit(nopython=True, cache=True)
def dbssyn(f, tmax, dt, sw):

    td = 0
    ti = 0
    tf = tmax + td
    t = np.arange(ti, tf, dt)

    # spike train
    sp = np.zeros(len(t))

    # DBS frequency
    T = round((1000.0 / f) / dt)
    ts = np.arange(1, int(tmax/dt), T).astype(np.int64)
    for i in range(len(ts)):
        if ts[i] < len(sp):
            sp[ts[i]] = 1.0 / dt

    # temporal modulation
    turner = make_turner(tmax, dt, sw)

    # synapse params
    taus = 1.7

    tauf = np.array([[670,17,326], [376,21,62]])
    taud = np.array([[138,671,329], [45,706,144]])
    U = np.array([[.09,.5,.29], [.016,.25,.32]])

    A = np.array([[.00025,.00025,.00025], [.00025,.00025,.00025]])
    A = 100 * A

    ie = np.ones(2)

    # weighting
    exper = np.array([45,38,17])
    inper = np.array([8,76,16])
    per = np.vstack((exper, inper))

    we = 2.5 * 200
    wi = 0.0

    weg = np.array([we, wi])
    wegst = np.vstack((weg, weg, weg)).T
    perfin = wegst * per

    A = A * perfin

    # state variables
    u = np.zeros(len(t))
    x = np.ones(len(t))
    I = np.zeros(len(t))

    PSC = np.zeros((2, A.shape[1], len(t)))

    # synaptic loop
    for q in range(len(ie)):
        w = 1.0 if q == 0 else -1.0

        for p in range(A.shape[1]):

            # reset states per synapse
            u[:] = 0.0
            x[:] = 1.0
            I[:] = 0.0

            for i in range(td, len(t)-1):

                drive = sp[i-td] * turner[i-td]

                u[i+1] = u[i] + dt * (-(u[i]/tauf[q,p]) + U[q,p]*(1-u[i])*drive)
                x[i+1] = x[i] + dt * ((1/taud[q,p])*(1-x[i]) - u[i+1]*x[i]*drive)
                I[i+1] = I[i] + dt * ((-1/taus)*I[i] + A[q,p]*u[i+1]*x[i]*drive)

            PSC[q,p,:] = w * I

    PSC_exc = np.sum(PSC[0,:,:], axis=0)
    PSC_inh = np.sum(PSC[1,:,:], axis=0)

    return PSC_exc + PSC_inh

