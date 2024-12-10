import numpy as np

def segment_hand_vel(
    hand_vel: np.ndarray, 
    still_thresh: float = 0.005,
    min_amplitude: float = 0.02,
    time_interval: float = 0.01
):
    """
    Segment hand velocity data into continuous motion segments.

    Parameters
    ----------
    hand_vel : np.ndarray
        N x 2 array of hand velocity in x and y directions.
    still_thresh : float, optional
        Threshold to define "still" motion, default is 0.005.
    min_amplitude : float, optional
        Minimum peak velocity amplitude to qualify as a valid segment, default is 0.02.
    time_interval : float, optional
        Sampling time interval, default is 0.01 seconds.

    Returns
    -------
    segments : np.ndarray
        M x 2 array where each row defines the start and end indices of a segment.
    """
    # Identify "still" and "moving" frames
    still = np.all(np.abs(hand_vel) < still_thresh, axis=1)
    moving = ~still

    try:
        # Identify segments where the velocity is "moving"
        segments = segment(moving)
        n_segments = segments.shape[0]

        # Apply the minimum amplitude constraint to the submovements
        keep_inds = []
        for k in range(n_segments):
            start, end = segments[k]
            max_velocity = np.max(np.abs(hand_vel[start:end + 1, :]))
            
            if max_velocity > min_amplitude and (end - start) * time_interval >= 0.2:
                keep_inds.append(k)
        
        # Keep only the segments that meet the constraints
        segments = segments[keep_inds]

        # Merge segments that are not separated by more than 1 bin
        segments_flat = segments.flatten()
        while np.min(np.diff(segments_flat)) == 1:
            n_segments = len(segments_flat) // 2
            for k in range(n_segments - 1):
                if segments_flat[2 * k + 1] == segments_flat[2 * k + 2] - 1:
                    segments_flat = np.delete(segments_flat, [2 * k + 1, 2 * k + 2])
                    break
        
        segments = segments_flat.reshape(-1, 2)

    except Exception:
        # Return a default segment if an error occurs
        segments = np.array([[0, hand_vel.shape[0] - 1]])  # Adjust index to match Python's zero-based indexing

    return segments


def segment(vec: np.ndarray):
    """
    Identify continuous segments of `True` values in a boolean array.

    Parameters
    ----------
    vec : np.ndarray
        1D boolean array where `True` indicates the presence of a segment.

    Returns
    -------
    segments : np.ndarray
        M x 2 array where each row contains the start and end indices of a segment.
    """
    vec = vec.astype(bool).flatten()

    # Identify the starts and ends of segments
    segment_starts = np.where(vec[1:] & ~vec[:-1])[0] + 1
    segment_ends = np.where(~vec[1:] & vec[:-1])[0] + 1

    # Handle the case where the first segment starts at the beginning
    if segment_starts.size == 0 or (segment_starts[0] > segment_ends[0]):
        segment_starts = np.insert(segment_starts, 0, 0)

    # Handle the case where the last segment ends at the last index
    if segment_ends.size == 0 or (segment_ends[-1] < segment_starts[-1]):
        segment_ends = np.append(segment_ends, len(vec))

    # Combine starts and ends into a single array
    segments = np.vstack((segment_starts, segment_ends)).T

    return segments


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ..datamanager.load_data import load_curve

    time_interval, vel = load_curve("curve1.csv", downsample_rate=100)

    segments = segment_hand_vel(vel, time_interval=time_interval)
    time = np.arange(len(vel)) * time_interval
    
    plt.plot(time, np.linalg.norm(vel, axis=1), color='k', linewidth=0.5)
    for x in segments[:,0]:
        plt.axvline(time[x], color='red', linestyle='--', linewidth=0.5)
    for x in segments[:,1]:
        plt.axvline(time[x], color='b', linestyle='--', linewidth=0.5)

    plt.show()