import numpy as np
import matplotlib.pyplot as plt

from ..decomposer.residual_sse import residual_sse

def greedy_onset_sampling(hand_vel, hand_speed, n_samples, time_interval, greed_factor=3, plot_dist=False):
    _, pt_by_pt_cost = residual_sse(hand_vel, hand_speed)

    pdist = pt_by_pt_cost / np.sum(pt_by_pt_cost)
    pdist = pdist ** greed_factor
    pdist = pdist / np.sum(pdist)

    local_mins = local_extrema(pdist, fn_type='min')
    local_min_indices = np.where(local_mins)[0]

    if plot_dist:
        plt.scatter(np.arange(len(pdist)), pdist, color='k', s=1)
        plt.scatter(local_min_indices, pdist[local_min_indices], color='r')
        plt.show()
    
    # cdist = np.cumsum(pdist)
    # samples = list()

    t0_sample_indices = np.random.randint(0, max(1, len(local_min_indices)-1), size=n_samples)
    t0_samples = local_min_indices[t0_sample_indices].astype(float)
    dur_samples = local_min_indices[t0_sample_indices + 1].astype(float) - t0_samples

    t0_samples *= time_interval
    dur_samples *= time_interval

    return t0_samples, dur_samples



def local_extrema(data, fn_type='max'):
    assert fn_type in ['min', 'max', 'both'], f'fn_type must be one of min, max, or both.'

    N = len(data)
    lm = np.zeros(N)

    if (data[0] <= data[1]) and (fn_type in ["min", "both"]):
        lm[0] = 1
    elif (data[0] >= data[1]) and (fn_type in ["max", "both"]):
        lm[0] = 1
    
    for k in range(1, N-1):
        if (data[k] >= data[k-1]) and (data[k] > data[k+1]) and (fn_type in ["max", "both"]):
            lm[k] = 1
        elif (data[k] <= data[k-1]) and (data[k] < data[k+1]) and (fn_type in ["max", "both"]):
            lm[k] = 1

    if (data[-1] <= data[-2]) and (fn_type in ["min", "both"]):
        lm[-1] = 1
    elif (data[-1] >= data[-2]) and (fn_type in ["max", "both"]):
        lm[-1] = 1

    if np.sum(lm) <= 1:
        lm[0] = 1
        lm[-1] = 1

    return lm