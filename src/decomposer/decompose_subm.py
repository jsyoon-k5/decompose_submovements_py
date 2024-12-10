"""
Python-base submovement decomposition function, with multiprocessing
Originally written in Matlab: https://github.com/sgowda/decompose_submovements

Matlab to Python migration done by June-Seop Yoon

Reference:
Gowda, S., Overduin, S. A., Chen, M., Chang, Y. H., Tomlin, C. J., & Carmena, J. M. (2015). 
Accelerating submovement decomposition with search-space reduction heuristics. 
IEEE Transactions on Biomedical Engineering, 62(10), 2508-2515.
"""

import numpy as np
import os, psutil
from joblib import delayed, Parallel
from tqdm import tqdm
from typing import Optional
import pandas as pd

from ..decomposer.segment_hand_vel import segment_hand_vel
from ..decomposer.residual_sse import residual_sse
from ..decomposer.decompose_lstsq import decompose_lstsq
from ..decomposer.reconstruct_subm import reconstruct_min_jerk
from ..utils.myutils import pickle_save

def decompose_submovements(
    hand_vel: np.ndarray,
    time_interval: float = 0.01,
    n_submovements: int = -1,
    still_threshold: float = 0.005,
    min_amplitude: float = 0.02,
    min_interval: float = 0.1,
    min_duration: float = 0.150,
    max_duration: float = 1.50,
    mse_perc_thresh: float = 0.02,
    min_segment_length: float = 0.2,
    use_cost_diff: bool = True,
    fn_type: str = "min_jerk",
    zero_greed: bool = False,
    verbose: bool = False,
    num_cpu: int = psutil.cpu_count(logical=False),
    save_path: Optional[str] = None
):
    """
    Decompose hand kinematics into submovements.

    Parameters
    ----------
    hand_vel : np.ndarray (unit: m/s)
        N x 2 matrix of hand velocity; column 0 and 1 refers to vel_x and vel_y, respectively.
    verbose : bool, optional, default=False
        Whether to print debugging statements.
    n_submovements : int, optional, default=-1
        Number of submovements to decompose. Default is to iterate starting from 1.
    still_threshold : float, optional, default=0.005 (unit: m/s)
        Threshold of endpoint speed in m/s to indicate the start/stop of movement segments.
    min_amplitude : float, optional, default=0.02 (unit: m/s)
        Minimum peak speed of a movement segment to qualify for decomposition.
    time_interval : float, optional, default=0.01 (unit: s)
        Sampling period, in seconds.
    min_interval : float, optional, default=0.1 (unit: s)
        Minimum time interval between submovements, in seconds.
    min_duration : float, optional, default=0.150 (unit: s)
        Minimum duration of a submovement, in seconds.
    mse_perc_thresh : float, optional, default=0.02
        Percentage MSE threshold at which to end decomposition.
    use_cost_diff : bool, optional, default=True
        Whether to terminate optimization if the MSE improvement is less than 0.1% 
        with the addition of a submovement.
    fn_type : str, optional, default="min_jerk"
        Prototype submovement function to fit the sum of.
    vel_indices : np.ndarray, optional, default=np.array([3, 4])
        Column indices of the `hand_kin` matrix corresponding to the velocity dimension.
    zero_greed : bool, optional, default=True
        Whether to sample initial parameters randomly.

    Returns
    -------
    movements : np.ndarray
        N x M matrix of submovement parameters, where M is the number of parameters per submovement.
    submovement_recon : np.ndarray
        K x 2 matrix of reconstructed hand velocity from the submovement parameters.
    segments : list
        Indices of the start and end of each segment split from the `hand_kin` matrix.
    costs : list
        Cost values associated with the decomposition.
    fits : list
        Fit results for the decomposition process.
    fit_costs : list
        Costs of the fit results.
    runtime : float
        Total time required to decompose the entire `hand_kin`.
    time_elapsed : np.ndarray
        Time required to decompose each segment.
    n_iterations : np.ndarray
        Number of optimization routine iterations required for each segment.
    n_func_evals : np.ndarray
        Number of cost function evaluations required for each segment.
    """

    assert hand_vel.shape[1] == 2
    iterate_n_submovements = (n_submovements == -1)

    #########################
    ### Movement segmentation
    #########################

    segments = segment_hand_vel(hand_vel, 
                                still_thresh=still_threshold, 
                                min_amplitude=min_amplitude, 
                                time_interval=time_interval)
    min_segment_len = int(np.floor(min_segment_length / time_interval))
    seg_lengths = segments[:, 1] - segments[:, 0]
    segments = segments[seg_lengths >= min_segment_len]
    n_segments = len(segments)

    if verbose:
        print(f'# of movement segments: {n_segments}')
    
    hand_speed = np.linalg.norm(hand_vel, axis=1)

    ###################################
    ### Begin submovement decomposition
    ###################################

    def decompose_segment(k):
        # Onset and offset
        s = segments[k][0]
        e = segments[k][1] + 1

        hand_vel_seg = hand_vel[s:e]
        hand_speed_seg = hand_speed[s:e]

        timestamp = np.arange(hand_vel_seg.shape[0]) * time_interval
        time_size = len(hand_vel_seg)

        costs_tr = []
        bestresult = []
        bestfitresult = []
        prev_decomp = np.array([])
        n_iterations_ = []
        # n_func_evals_ = []
        costs_so_far = [np.inf]

        # Determine the max number of submovements
        seg_len = len(timestamp)
        max_submovements = max(round(np.ceil((seg_len * time_interval - min_duration) / max(min_interval, 0.1))), 1)
        if not iterate_n_submovements:
            max_submovements = min(max_submovements, n_submovements)

        orig_error, _ = residual_sse(hand_vel_seg, hand_speed_seg)

        submov_iter = 0
        while (submov_iter < 1) or (submov_iter < max_submovements and not iterate_n_submovements) or \
                (costs_so_far[submov_iter] > mse_perc_thresh and submov_iter < max_submovements - 1):
            
            submov_iter += 1

            if zero_greed:
                prev_decomp = np.array([])
                sample_randomly = 1
            else:
                sample_randomly = 0

            _c, _r, _f, _nit = decompose_lstsq(timestamp, hand_vel_seg, submov_iter, 
                                                     min_duration=min_duration,
                                                     max_duration=max_duration,
                                                     min_interval=min_interval,
                                                     prev_decomp=prev_decomp,
                                                     term_cond=orig_error * mse_perc_thresh,
                                                     fn_type=fn_type,
                                                     sample_randomly=sample_randomly,
                                                     verbose=False)

            costs_tr.append(_c)
            bestresult.append(_r)       # Submovements' parameters in N x M
            bestfitresult.append(_f)    # Reconstrcuted velocity in K x 2
            n_iterations_.append(_nit)
            # n_func_evals_.append(_nfe)

            prev_decomp = _r.copy()     # Already reshaped to 2D in decompose_lstsq
            costs_so_far.append(_c / orig_error)

            if (submov_iter > 2 and iterate_n_submovements and 
                    (costs_so_far[-1] - costs_so_far[-2]) > -0.001 and use_cost_diff):
                break

        bestresult = prev_decomp
        best_recon_traj = reconstruct_min_jerk(prev_decomp, time_size, time_interval)
        bestresult[:,0] += s * time_interval    # Add onset time to parameter t0

        return bestresult, (s, e, best_recon_traj), costs_tr[-1], n_iterations_ #, n_func_evals_
    
    result = Parallel(n_jobs=num_cpu)(
        delayed(decompose_segment)(k) for k in (tqdm(range(n_segments)) if verbose else range(n_segments))
    )

    movements = []
    submovement_recon = np.zeros_like(hand_vel)
    n_iterations = []
    # n_func_evals = []
    costs = []

    for r in result:
        movements.append(r[0])
        submovement_recon[r[1][0]:r[1][1]] = r[1][2]
        costs.append(r[2])
        n_iterations.append(r[3])
        # n_func_evals.append(r[4])
    
    # Save decomposed results
    os.makedirs(save_path, exist_ok=True)

    mv_data = dict(
        timestamp = np.arange(len(hand_vel)) * time_interval,
        orig_vx = hand_vel[:,0],
        orig_vy = hand_vel[:,1],
        recn_vx = submovement_recon[:,0],
        recn_vy = submovement_recon[:,1]
    )
    mv_data = pd.DataFrame(mv_data)
    if save_path is not None:
        mv_data.to_csv(os.path.join(save_path, "movement.csv"), index=False)

    num_of_subm_per_seg = [m.shape[0] for m in movements]
    movements = np.concatenate(movements)

    param_data = dict(
        segment = np.repeat(np.arange(segments.shape[0]), num_of_subm_per_seg),
        t0 = movements[:,0],
        dur = movements[:,1],
        Ax = movements[:,2],
        Ay = movements[:,3],
    )
    param_data = pd.DataFrame(param_data)
    if save_path is not None:
        param_data.to_csv(os.path.join(save_path, "submovements.csv"), index=False)

    seg_data = dict(
        segment = np.arange(segments.shape[0]),
        onset = segments[:,0],
        offset = segments[:,1],
        len = segments[:,1] - segments[:,0]
    )
    seg_data = pd.DataFrame(seg_data)
    if save_path is not None:
        seg_data.to_csv(os.path.join(save_path, "segments.csv"), index=False)

    return mv_data, param_data, seg_data
