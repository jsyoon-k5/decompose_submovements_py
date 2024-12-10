import numpy as np

def residual_sse(v: np.ndarray, tv: np.ndarray):
    """
    Compute the residual sum of squared errors.

    Parameters
    ----------
    v : np.ndarray
        N x D array representing velocity components.
    tv : np.ndarray
        N x 1 array representing tangential velocity.

    Returns
    -------
    cost : float
        Total residual sum of squared errors.
    pt_by_pt_cost : np.ndarray
        Point-by-point residual costs.
    """
    # Compute squared errors for velocity components
    e = v**2
    
    # Compute squared errors for tangential velocity
    e_tang = tv.flatten()**2
    
    # Compute point-by-point cost
    pt_by_pt_cost = np.sum(e, axis=1) + e_tang
    
    # Compute total cost
    cost = np.sum(pt_by_pt_cost)
    
    return cost, pt_by_pt_cost
