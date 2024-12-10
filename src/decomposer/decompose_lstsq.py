import numpy as np
from scipy.optimize import minimize
import warnings

from ..decomposer.reconstruct_subm import reconstruct_min_jerk
from ..decomposer.greedy_onset_sampling import greedy_onset_sampling
from ..decomposer.cost_functions import min_jerk_cost_fn, calculateerrorMJxy

def decompose_lstsq(
    times: np.ndarray,
    vel: np.ndarray,
    n_submovements: int,
    min_duration: float = 0.09,
    max_duration: float = 1.0,
    min_interval: float = 0.1,
    term_cond = -np.inf,
    fn_type: str = "min_jerk",
    prev_decomp: np.ndarray = np.array([]),
    add_new_subm: bool = True,
    sample_randomly: bool = False,
    n_iterations: int = 10,
    optimizer: str = 'scipy.optimize.minimize',
    verbose=False
):
    """
    Decompose a trajectory into submovements using min_jerk_full model.

    Parameters
    ----------
    times : np.ndarray
        1D array of time stamps for velocity samples (must be evenly spaced).
    vel : np.ndarray
        K x 2 array of x-y Cartesian velocities.
    n_submovements : int
        Number of submovements to fit.
    min_duration : float, optional
        Minimum duration of a submovement, in seconds.
    max_duration : float, optional
        Maximum duration of a submovement, in seconds.
    min_interval : float, optional
        Minimum time between submovements, in seconds.
    term_cond : float, optional
        Termination threshold for optimization.
    sample_randomly : bool, optional
        If True, sample parameters randomly.
    n_iterations : int, optional
        Number of optimization restarts.

    Returns
    -------
    best_cost : float
        Best cost achieved during optimization.
    bestresult : np.ndarray
        Optimal parameters for submovements.
    """

    assert fn_type in ["min_jerk", "min_jerk_full"]

    prev_decomp_size = len(prev_decomp)
    if n_submovements < prev_decomp_size:
        raise ValueError(f"Previous decomposition is larger than current: {n_submovements} < {prev_decomp_size}.")
    
    assert len(times) == len(vel), f"times ({len(times)}) and vel ({len(vel)}) do not match in length."

    time_interval = np.mean(np.diff(times))
    best_cost = np.inf

    N_PARAM_PER_SUBM = 2 if fn_type == "min_jerk" else 4

    ##################################################
    # Create parameter by parameter lower/upper bounds
    ##################################################

    t0_max = times[-1] - min_duration
    T_max = len(times) * time_interval

    vx_min = np.min(vel[:, 0])
    vx_max = np.max(vel[:, 0])
    vy_min = np.min(vel[:, 1])
    vy_max = np.max(vel[:, 1])

    lb = []
    ub = []
    
    if fn_type == "min_jerk":
        for k in range(n_submovements):
            lb.extend([min_interval * k, min_duration])
            ub.extend([t0_max, min(max_duration, T_max)])
    elif fn_type == "min_jerk_full":
        for k in range(n_submovements):
            lb.extend([min_interval * k, min_duration, vx_min, vy_min])
            ub.extend([t0_max, min(max_duration, T_max), vx_max, vy_max])

    lb = np.array(lb)
    ub = np.array(ub)

    if np.any(lb > ub):
        raise ValueError(f'Lower bounds exceed upper bound: lb: {lb}, ub: {ub}')

    # Linear inequality constraints for inter-submovement intervals
    if min_interval > 0 or n_submovements > 1:
        A = np.zeros((n_submovements - 1, n_submovements * N_PARAM_PER_SUBM))
        b = -min_interval * np.ones(n_submovements - 1)
        for k in range(n_submovements - 1):
            A[k][N_PARAM_PER_SUBM * k] = 1
            A[k][N_PARAM_PER_SUBM * (k+1)] = -1
    else:
        A = np.array([], dtype=float)
        b = np.array([], dtype=float)
    

    tv = np.linalg.norm(vel, axis=1)

    
    # n_iterations = 10
    recon = reconstruct_min_jerk(prev_decomp, len(times), time_interval)
    res_vel = vel - recon
    res_speed = np.linalg.norm(res_vel, axis=1)

    if not sample_randomly:
        t0_samples, dur_samples = greedy_onset_sampling(
            res_vel,
            res_speed,
            n_iterations,
            time_interval,
            greed_factor=prev_decomp_size + 1,
            plot_dist=False
        )
        t0_samples, indices = np.unique(t0_samples, return_index=True)
        dur_samples = dur_samples[indices]

        if len(t0_samples) < n_iterations:
            n_rand_samples = n_iterations - len(t0_samples)
            t0_samples = np.concatenate([
                t0_samples,
                np.random.uniform(0, t0_max, size=n_rand_samples)
            ])
            dur_samples = np.concatenate([
                dur_samples,
                np.random.uniform(min_duration, max_duration, size=n_rand_samples)
            ])

    # n_opt_iter = 0
    # n_func_evals = 0

    count = 0
    while count < n_iterations:
        ###############################################
        # Initialize estimate of submovement parameters
        ###############################################
        if n_submovements == 1 and count == 0:
            if fn_type == "min_jerk":
                init_param = np.array([0, np.random.uniform(min_duration, max_duration)])
            elif fn_type == "min_jerk_full":
                init_param = np.array([
                    0,
                    np.random.uniform(min_duration, max_duration),
                    np.random.uniform(vx_min, vx_max),
                    np.random.uniform(vy_min, vy_max)
                ])
            
        elif sample_randomly:
            n_new_subm = n_submovements - prev_decomp_size

            # Sample start time and duration of new submovements randomly
            new_subm_init_params = list()
            for m in range(n_new_subm):
                if fn_type == "min_jerk":
                    _t0 = np.random.uniform(0, t0_max)
                    _dur = np.random.uniform(min_duration, max_duration)
                    new_subm_init_params.append([_t0, _dur])

                elif fn_type == "min_jerk_full":
                    _p = [
                        np.random.uniform(0, t0_max),   # t0
                        np.random.uniform(min_duration, max_duration),  # dur
                        np.random.uniform(vx_min, vx_max),  # Ax
                        np.random.uniform(vy_min, vy_max)   # Ay
                    ]
                    new_subm_init_params.append(_p)
            
            new_subm_init_params = np.array(new_subm_init_params)
            
            if prev_decomp_size > 0:
                init_param = np.vstack((prev_decomp[:,0:N_PARAM_PER_SUBM], new_subm_init_params))
            else:
                init_param = new_subm_init_params
            
            init_param = init_param[init_param[:,0].argsort()].flatten()
        
        elif add_new_subm:
            n_new_subm = n_submovements - prev_decomp_size
            new_subm_init_params = list()

            for m in range(n_new_subm):
                _t0 = t0_samples[count]
                _dur = dur_samples[count]   # count < n_iterations is guaranteed

                if fn_type == "min_jerk":
                    new_subm_init_params.append([_t0, _dur])
                elif fn_type == "min_jerk_full":
                    new_subm_init_params.append([_t0, _dur, vx_max * _dur, vy_max * _dur])
            
            new_subm_init_params = np.array(new_subm_init_params)
            
            if prev_decomp_size > 0:
                init_param = np.vstack((prev_decomp[:,0:N_PARAM_PER_SUBM], new_subm_init_params))
            else:
                init_param = new_subm_init_params
            init_param = init_param[init_param[:,0].argsort()].flatten()

        else:
            init_param = prev_decomp[:,0:N_PARAM_PER_SUBM]
            init_param = init_param[init_param[:,0].argsort()].flatten()
        
        init_param = init_param.astype(np.float64)


        ###########################
        # Run optimization routines
        ###########################

        if fn_type == "min_jerk":
            func = min_jerk_cost_fn
        elif fn_type == "min_jerk_full":
            func = calculateerrorMJxy

        # Select optimizer
        if optimizer == "scipy.optimize.minimize":
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            
            def objective(params):
                cost, *_ = func(params, times, vel, tv)
                return cost
            
            optimization_options = {
                'maxiter': 5000,        # Maximum number of iterations
                'disp': verbose,           # Display optimization progress
                'ftol': 1e-6            # Function tolerance for stopping criteria
            }
            if len(b) == 0:
                kwargs = {
                    "method": "SLSQP",
                    "bounds": list(zip(lb, ub)),
                }
            else:
                kwargs = {
                    "method": "SLSQP",
                    "bounds": list(zip(lb, ub)),
                    "constraints": {'type': 'ineq', 'fun': lambda x: A @ x - b}
                }

            res = minimize(
                objective,
                init_param,
                options=optimization_options,
                **kwargs
            )
            result = res.x
            # niter = res.nit
            # neval = res.nfev

        ### Numpy version conflict with scipy
        # elif optimizer == "nlopt":
        #     def objective(params, grad):
        #         cost, g, *_ = func(params, times, vel, tv)
        #         if grad.size > 0:
        #             grad[:] = g
        #         return cost
            
        #     opt = nlopt.opt(nlopt.LD_SLSQP, len(init_param))
        #     opt.set_min_objective(objective)
        #     opt.set_lower_bounds(lb)
        #     opt.set_upper_bounds(ub)
        #     if len(b) > 0:
        #         def inequality_constraint(x, grad, row_idx, A, b):
        #             if grad.size > 0:
        #                 grad[:] = A[row_idx, :]
        #             return np.dot(A[row_idx, :], x) - b[row_idx]
        #         for i in range(n_submovements - 1):
        #             opt.add_inequality_constraint(lambda x, grad, i=i: inequality_constraint(x, grad, i, A, b))
        #     opt.set_ftol_rel(1e-6)
        #     opt.set_maxeval(5000)
        #     result = opt.optimize(init_param)

        else:
            raise NotImplementedError(f"Cannot specify optimizer function {optimizer}")
        
        cores = func(result, times, vel, tv)
        epsilon, fitresult = cores[0], cores[2]
    
        #########
        # Cleanup
        #########

        # Cleanup loop variables
        # n_opt_iter += niter  # Add the number of iterations
        # n_func_evals += neval     # Add the number of function evaluations

        # Check for invalid results
        # if not np.isreal(result[0]):
        #     raise ValueError("Found an imaginary value")

        # Update best result if current result is better
        if epsilon < best_cost:
            best_cost = epsilon
            bestresult = result
            bestfitresult = fitresult.copy()

        # Check termination condition
        if best_cost < term_cond:
            if verbose:
                print("\t\tOptimization target met, terminating early")
            break
        else:
            count += 1

    # Recalculate Ax, Ay if needed
    if fn_type == "min_jerk":
        *_, amplitude = min_jerk_cost_fn(bestresult, times, vel, tv)
        bestresult = np.hstack([bestresult.reshape(-1, N_PARAM_PER_SUBM), amplitude])
        bestresult = bestresult[bestresult[:,0].argsort()]

    elif fn_type == "min_jerk_full":
        bestresult = bestresult.reshape(-1, N_PARAM_PER_SUBM)
        bestresult = bestresult[bestresult[:,0].argsort()]

    return best_cost, bestresult, bestfitresult, n_iterations
