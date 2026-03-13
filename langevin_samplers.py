
import numpy as np
import matplotlib.pyplot as plt 
import tqdm

#one single verlet step
def verlet_step(x, v, gradU, delta):
    
    v = v - 0.5 * delta * gradU(x)
    x = x + delta * v
    v = v - 0.5 * delta * gradU(x)

    return x, v

def hamiltonian_drift(x, v, gradU, delta, eta, K, T):
    
        X = np.zeros((K, len(x))) 
        V = np.zeros((K, len(v)))
        
        #K verlet steps 
        for i in range(K):
            x,v = verlet_step(x,v, gradU, delta)
            X[i] = x
            V[i] = v
        
        #velocity refresh
        v = eta * v + np.sqrt(1 - eta**2) * np.sqrt(T) * np.random.randn(*v.shape)
        
        return x,v,X,V
        

def sample_run(x0, v0, gradU, N, delta, K, eta, T):
    X = np.zeros((N * K, len(x0)))
    V = np.zeros((N * K, len(v0)))
    x,v = x0.copy(), v0.copy()
    
    for i in range(N):
        # deterministic Hamiltonian evolution
        x, v, X_h, V_h = hamiltonian_drift(x, v, gradU, delta, eta, K, T)
        
        # store intermediate positions
        X[i*K:(i+1)*K, :] = X_h 
        V[i*K:(i+1)*K, :] = V_h 
    return X,V

def sample_run_streaming_mean(x0, v0, gradU, N, delta, K, eta, T, burn=0, thin=1, dtype=np.float32):
    x = x0.astype(dtype).copy()
    v = v0.astype(dtype).copy()

    mean_x = np.zeros_like(x, dtype=dtype)
    count = 0

    for i in range(N):
        # deterministic Hamiltonian evolution
        x, v, _, _ = hamiltonian_drift(x, v, gradU, delta, eta, K, T)

        if i < burn:
            continue
        if ((i - burn) % thin) != 0:
            continue

        mean_x += x
        count += 1

    mean_x /= max(count, 1)
    return mean_x


def run_od_langevin(x0, v0, gradU, N, delta=0.09, K=1, eta=0, T=1):
    return sample_run(x0, v0, gradU, N, delta, K, eta, T)

def run_hmc(x0, v0, gradU, N, delta = 0.09, K= 10, eta = 0, T=1):
    return sample_run(x0, v0, gradU, N, delta, K, eta, T)

def run_k_langevin(x0, v0, gradU, N, delta=0.09, K=1, eta=None, T=1):
    if eta is None:
        eta = np.exp(-0.5 * delta)
    return sample_run(x0, v0, gradU, N, delta, K, eta,T)

def run_od_langevin_streaming(x0, v0, gradU, N, delta=0.09, K=1, eta=0, T=1, burn=0, thin=1):
    return sample_run_streaming_mean(x0, v0, gradU, N, delta, K, eta, T, burn, thin)


def several_runs(x0, v0, gradU, sampler, n, N, delta, K, eta, T=1):
    """
    Runs an ensemble of independent trajectories.

    Returns:
        X_ensemble : array of shape (n, N*K, d)
        V_ensemble : array of shape (n, N*K, d)
    """
    d = len(x0)

    X_ensemble = np.zeros((n, N * K, d))
    V_ensemble = np.zeros((n, N * K, d))

    for i in range(n):
        v0_i = np.random.randn(d)

        X, V = sampler(x0,v0_i,gradU,N,delta,K,eta,T)

        X_ensemble[i] = X
        V_ensemble[i] = V

    return X_ensemble, V_ensemble

def annealed_run(x0, v0, gradU, N, delta, K, eta, T_schedule,
                 store_every=1, store_inner=False, freeze_steps=0):
    """
    Annealed version of your kinetic / splitting chain.

    Args:
        x0, v0: initial position/velocity (vectors)
        gradU: gradient function on vectors
        N: number of outer iterations
        delta, K, eta: your usual parameters
        T_schedule: function i -> T_i
        store_every: store every 'store_every' outer iterations
        store_inner: if True, store all K intermediate Verlet positions too
        freeze_steps: after annealing, run extra steps with T=0 (purely deterministic refresh if eta=1, or
                      velocity refresh without noise if eta<1 but T=0)

    Returns:
        xs: stored positions (list -> array)
        Ts: stored temperatures aligned with xs
        x_last, v_last: final state
    """
    x = x0.copy()
    v = v0.copy()

    xs = []
    Ts = []

    for i in range(N):
        T = float(T_schedule(i))

        # Your updated drift should accept T and use it in the refresh noise
        x, v, X_h, V_h = hamiltonian_drift(x, v, gradU, delta, eta, K, T=T)

        if (i % store_every) == 0:
            if store_inner:
                # store all K intermediate positions
                xs.append(X_h.copy())   # shape (K, d)
                Ts.append(T)
            else:
                xs.append(x.copy())     # shape (d,)
                Ts.append(T)

    # Optional freeze / polishing at T=0
    for j in range(freeze_steps):
        x, v, _, _ = hamiltonian_drift(x, v, gradU, delta, eta, K, T=0.0)

    xs = np.array(xs)
    Ts = np.array(Ts)

    return xs, Ts, x, v

def annealed_run_streaming_mean(x0, v0, gradU, N, delta, K, eta, T_schedule,
                                burn=0, thin=1, dtype=np.float32, freeze_steps=0):
    x = x0.astype(dtype).copy()
    v = v0.astype(dtype).copy()

    mean_x = np.zeros_like(x, dtype=dtype)
    count = 0

    for i in range(N):
        T = float(T_schedule(i))
        x, v, _, _ = hamiltonian_drift(x, v, gradU, delta, eta, K, T=T)

        if i < burn:
            continue
        if ((i - burn) % thin) != 0:
            continue

        mean_x += x
        count += 1

    mean_x /= max(count, 1)

    for j in range(freeze_steps):
        x, v, _, _ = hamiltonian_drift(x, v, gradU, delta, eta, K, T=0.0)

    return mean_x, x, v