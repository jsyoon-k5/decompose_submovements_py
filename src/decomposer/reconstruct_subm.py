import numpy as np

def reconstruct_min_jerk(movements, time_len, time_interval):
    """
    Reconstruct minimum jerk trajectory (2d)

    Parameters
    ------------
    movements:
        N x M array, each row refers to submovement parameters
    time_len:
        integer that refers to the length of data samples
    time_interval:
        1 / sampling rate
    """
    times = np.arange(time_len) * time_interval
    recon = np.zeros((time_len, 2))
    n_submovements = movements.shape[0]

    for k in range(n_submovements):
        recon += calc_min_jerk_recon(movements[k], times)

    return recon


def calc_min_jerk_recon(parameters, times):
    N_PARAMS_PER_SUBMOVEMENT = 2
    n_dim_movement = len(parameters) - N_PARAMS_PER_SUBMOVEMENT

    # Initialize v_pred with NaN
    v_pred = np.full((len(times), n_dim_movement), np.nan)

    # Extract parameters
    t0 = parameters[0]
    dur = parameters[1]
    amp = parameters[N_PARAMS_PER_SUBMOVEMENT:].reshape(-1, 1)  # Convert to column vector

    # Find range of times within the submovement
    this_rng = np.where((times >= t0) & (times < t0 + dur))[0]
    t = times[this_rng]
    
    # Compute normalized time
    nt = (t - t0) / dur

    # Compute velocity prediction for the submovement
    shape = -60 * nt**3 + 30 * nt**4 + 30 * nt**2
    v_pred[this_rng, :] = (amp @ (1 / dur * shape).reshape(1, -1)).T
    v_pred = np.nan_to_num(v_pred)

    return v_pred

