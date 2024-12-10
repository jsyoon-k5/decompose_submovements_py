import numpy as np

def min_jerk_cost_fn(parameters, times, v, tv):
    """
    Compute the cost, gradient, predicted velocities, and M matrix
    for the minimum jerk model.

    Parameters
    ----------
    parameters : np.ndarray
        Array of submovement parameters, flattened as [t0, D, t0, D, ...].
    times : np.ndarray
        Time vector.
    v : np.ndarray
        N x 2 Observed velocities.
    tv : np.ndarray
        Tangential velocity.

    Returns
    -------
    cost : float
        Total cost.
    grad : np.ndarray
        Gradient of the cost with respect to parameters.
    v_pred : np.ndarray
        Predicted velocities.
    M : np.ndarray
        Matrix of coefficients for the submovement weights.
    """
    N_PARAMS_PER_SUBMOVEMENT = 2
    params = parameters.reshape(-1, N_PARAMS_PER_SUBMOVEMENT)
    n_movements = params.shape[0]

    # Initialize F matrix
    # Each column is movement
    F = np.zeros((len(times), n_movements))

    # Populate F using MJxy for each submovement
    for n in range(n_movements):
        t0 = params[n][0]
        dur = params[n][1]
        time_indices = np.where((times >= t0) & (times < t0 + dur))[0]
        t = times[time_indices]
        Bx, *_ = MJxy(t0, dur, 1, 1, t, return_type=0)
        F[time_indices, n] = Bx     # Bx and By are identical in this output.

    # Compute the ridge regularization matrix and M
    ridge = np.eye(n_movements) * 0.5
    # M = np.linalg.solve(F.T @ F + ridge, F.T @ v)
    M = np.linalg.pinv(F.T @ F + ridge) @ F.T @ v   # n_movements x 2 -> Amplitude for min_jerk

    # Flatten the full parameters ([parameters, M])
    full_params = np.hstack([params, M]).flatten()

    cost, full_grad, v_pred = calculateerrorMJxy(full_params, times, v, tv)
    grad_inds = []
    for k in range(n_movements):
        grad_inds.extend([4 * k, 4 * k + 1])  # Extract indices for t0 and D
    grad = full_grad[grad_inds]

    return cost, grad, v_pred, M


def calculateerrorMJxy(parameters, times, vel, tangvel, time_interval=None):
    """
    Calculate the error between the predicted and actual profile (in 2D).

    Parameters
    ----------
    parameters : np.ndarray
        Array of parameters (length 4 * n_submovements).
        Each submovement has T0 (onset), D (duration), Dx (x amplitude), Dy (y amplitude).
    times : np.ndarray
        1D array of time points of recorded movement.
    vel : np.ndarray
        N x 2 array of x and y velocities.
    tangvel : np.ndarray
        1D array of tangential velocities (sqrt(vel[:,0]**2 + vel[:,1]**2)).
    time_interval : float, optional
        Time step between evaluations, default is computed from `times`.

    Returns
    -------
    epsilon : float
        Total error between predicted and actual profiles.
    grad : np.ndarray, optional
        Gradient of the error with respect to parameters.
    sumpredicted : np.ndarray
        Predicted 2D trajectory.
    """
    N_PARAMS_PER_SUBMOVEMENT = 4

    if time_interval is None:
        time_interval = np.mean(np.diff(times))

    params = parameters.reshape(-1, N_PARAMS_PER_SUBMOVEMENT)
    n_submovements = params.shape[0]

    trajectoryx = vel[:, 0]
    trajectoryy = vel[:, 1]

    K = len(times)
    predictedx = np.zeros((K, n_submovements))
    predictedy = np.zeros((K, n_submovements))
    predicted = np.zeros((K, n_submovements))

    Jx = np.zeros((K, 4 * n_submovements))
    Jy = np.zeros((K, 4 * n_submovements))
    J = np.zeros((K, 4 * n_submovements))

    # t0s = params[:, 0]
    # durs = params[:, 1]
    # dxs = params[:, 2]
    # dys = params[:, 3]

    for k, (t0, dur, dx, dy) in enumerate(zip(*params.T)):
        thisrng = np.where((times >= t0) & (times < t0 + dur))[0]

        (
            predictedx[thisrng, k],
            predictedy[thisrng, k],
            predicted[thisrng, k],
            Jx[thisrng, k*4:(k+1)*4],
            Jy[thisrng, k*4:(k+1)*4],
            J[thisrng, k*4:(k+1)*4]
        ) = MJxy(t0, dur, dx, dy, times[thisrng], return_type=1)

    sumpredictedx = np.sum(predictedx, axis=1)
    sumpredictedy = np.sum(predictedy, axis=1)
    sumpredicted = np.sum(predicted, axis=1)

    sumtrajsq = np.sum(trajectoryx**2) + np.sum(trajectoryy**2) + np.sum(tangvel**2)
    error_x = sumpredictedx - trajectoryx
    error_y = sumpredictedy - trajectoryy
    error_t = sumpredicted - tangvel
    grad = (2 / sumtrajsq * (Jx.T @ error_x + Jy.T @ error_y + J.T @ error_t))

    # Calculate epsilon (total error)
    epsilon = calc_error(vel, tangvel, np.vstack([sumpredictedx, sumpredictedy]).T)

    return epsilon, grad, np.stack([sumpredictedx, sumpredictedy]).T


def calc_error(v, tv, v_pred):
    """
    Evaluate the error of a velocity reconstruction.

    Parameters
    ----------
    v : np.ndarray
        Observed velocity (K x 2 matrix).
    tv : np.ndarray
        Observed tangential speed (K array).
    v_pred : np.ndarray
        Velocity profile reconstructed from submovement parameters (K x 2 matrix).

    Returns
    -------
    cost : float
        Cost of the current reconstruction.
    """
    v_pred_tang = np.linalg.norm(v_pred, axis=1)
    e = np.linalg.norm(v - v_pred, ord='fro')**2
    e_tang = np.linalg.norm(tv - v_pred_tang)**2
    cost = e + e_tang

    return cost


def MJxy(t0, D, Ax, Ay, t, return_type=0):
    """
    Evaluate a minimum jerk curve with separate displacements for x and y.
    
    Parameters
    ----------
    t0 : float
        Movement start time.
    D : float
        Movement duration.
    Ax : float
        Displacement in the x direction.
    Ay : float
        Displacement in the y direction.
    t : np.ndarray
        Times at which to evaluate the function.
        t must be range in [t0, t0 + D].
    
    Returns
    -------
    Bx : np.ndarray
        x velocity.
    By : np.ndarray
        y velocity.
    B : np.ndarray
        Tangential velocity.
    Jx : np.ndarray (optional)
        Gradients of x velocity.
    Jy : np.ndarray (optional)
        Gradients of y velocity.
    J : np.ndarray (optional)
        Gradients of tangential velocity.
    Hx : np.ndarray (optional)
        Hessians of x velocity.
    Hy : np.ndarray (optional)
        Hessians of y velocity.
    H : np.ndarray (optional)
        Hessians of tangential velocity.
    """
    # Normalized time (0 <= nt <= 1)
    nt = (t - t0) / D
    nt_sq = nt**2
    nt_cubed = nt**3
    nt_fourth = nt**4

    # Shape function
    shape = (-60 * nt_cubed + 30 * nt_fourth + 30 * nt_sq)

    # Velocities
    Bx = Ax / D * shape
    By = Ay / D * shape

    A_tang = np.sqrt((Ax / D)**2 + (Ay / D)**2)
    B = A_tang * shape

    if return_type == 0:
        return Bx, By, B

    # Gradients and Hessians
    K = len(t)
    Jx = np.zeros((K, 4))
    Jy = np.zeros((K, 4))
    J = np.zeros((K, 4))

    shape_dt0 = -1 / D**2 * (120 * nt_cubed - 180 * nt_sq + 60 * nt)
    shape_dD = 1 / D**2 * (-150 * nt_fourth + 240 * nt_cubed - 90 * nt_sq)

    Jx[:,0] = Ax * shape_dt0
    Jx[:,1] = Ax * shape_dD
    Jx[:,2] = 1 / D * shape

    Jy[:,0] = Ay * shape_dt0
    Jy[:,1] = Ay * shape_dD
    Jy[:,3] = 1 / D * shape

    J[:,0] = A_tang * shape_dt0
    J[:,1] = A_tang * shape_dD
    J[:,2] = Ax / A_tang * shape
    J[:,3] = Ay / A_tang * shape

    if return_type == 1:
        return Bx, By, B, Jx, Jy, J

    Hx = np.zeros((4, K, 4))
    Hy = np.zeros((4, K, 4))
    H = np.zeros((4, K, 4))

    # Slice 1: First derivative w.r.t. t0
    shape_dt0dt0 = 1 / D**3 * (360 * nt_sq - 360 * nt + 60)
    Hx[0, :, 0] = Ax * shape_dt0dt0
    Hy[0, :, 0] = Ay * shape_dt0dt0
    H[0, :, 0] = A_tang * shape_dt0dt0

    shape_dDdt0 = -1 / D**3 * (-600 * nt_cubed + 720 * nt_sq - 180 * nt)
    Hx[0, :, 1] = Ax * shape_dDdt0
    Hy[0, :, 1] = Ay * shape_dDdt0
    H[0, :, 1] = A_tang * shape_dDdt0

    Hx[0, :, 2] = shape_dt0
    H[0, :, 2] = Ax / A_tang * shape_dt0

    Hy[0, :, 3] = shape_dt0
    H[0, :, 3] = Ay / A_tang * shape_dt0

    # Slice 2: First derivative w.r.t. D
    shape_dt0dD = shape_dDdt0
    Hx[1, :, 0] = Ax * shape_dt0dD
    Hy[1, :, 0] = Ay * shape_dt0dD
    H[1, :, 0] = A_tang * shape_dt0dD

    shape_dDdD = 1 / D**3 * (900 * nt_fourth - 1200 * nt_cubed + 360 * nt_sq)
    Hx[1, :, 1] = Ax * shape_dDdD
    Hy[1, :, 1] = Ay * shape_dDdD
    H[1, :, 1] = A_tang * shape_dDdD

    Hx[1, :, 2] = shape_dD
    H[1, :, 2] = Ax / A_tang * shape_dD

    Hy[1, :, 3] = shape_dD
    H[1, :, 3] = Ay / A_tang * shape_dD

    # Slice 3: First derivative w.r.t. Ax
    Hx[2, :, 0] = shape_dt0
    H[2, :, 0] = Ax / A_tang * shape_dt0

    Hx[2, :, 1] = shape_dD
    H[2, :, 1] = Ax / A_tang * shape_dD

    H[2, :, 2] = 1 / A_tang * shape

    H[2, :, 3] = -Ax * Ay / A_tang**3 * shape

    # Slice 4: First derivative w.r.t. Ay
    Hy[3, :, 0] = shape_dt0
    H[3, :, 0] = Ay / A_tang * shape_dt0

    Hy[3, :, 1] = shape_dD
    H[3, :, 1] = Ay / A_tang * shape_dD

    H[3, :, 2] = -Ax * Ay / A_tang**3 * shape

    H[3, :, 3] = 1 / A_tang * shape


    return Bx, By, B, Jx, Jy, J, Hx, Hy, H

