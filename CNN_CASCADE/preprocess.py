
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

from config import *

def _as_1d_float_array(value):
    return np.asarray(value, dtype=np.float32).reshape(-1)

def _iter_attached_sweeps(data, mat_path):
    if "CAttached" not in data:
        required = {"fluo_time", "fluo_mean", "events_AP"}
        if required.issubset(data):
            yield data
            return

        available = sorted(k for k in data if not k.startswith("__"))
        raise KeyError(
            f"{mat_path} has neither top-level fluo fields nor CAttached. "
            f"Available keys: {available}"
        )

    for sweep in np.ravel(data["CAttached"]):
        yield sweep

def _get_field(sweep, field_name, mat_path, sweep_idx):
    if isinstance(sweep, dict):
        if field_name in sweep:
            return sweep[field_name]
    elif hasattr(sweep, field_name):
        return getattr(sweep, field_name)

    raise KeyError(
        f"{mat_path} sweep {sweep_idx} is missing required field "
        f"{field_name!r}"
    )

def estimate_frame_rate(fluo_time):
    dt = np.diff(fluo_time.astype(np.float64))
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        return FRAME_RATE

    return float(1.0 / np.median(dt))

def robust_normalize(x, median_x=None, mad_x=None):
    if median_x is None:
        median_x = np.median(x)

    if mad_x is None:
        mad_x = np.median(np.abs(x - median_x))

    x_norm = (x - median_x) / (1.4826 * mad_x + EPS)

    return x_norm.astype(np.float32)

def spike_times_to_counts(spike_times_sec, fluo_time):

    spike_counts = np.zeros(len(fluo_time), dtype=np.float32)

    spike_times_sec = spike_times_sec[np.isfinite(spike_times_sec)]
    spike_times_sec = spike_times_sec[
        (spike_times_sec >= fluo_time[0]) &
        (spike_times_sec <= fluo_time[-1])
    ]

    indices = np.searchsorted(fluo_time, spike_times_sec)

    valid = indices < len(fluo_time)
    indices = indices[valid]

    np.add.at(spike_counts, indices, 1.0)

    return spike_counts

def smooth_spike_counts(spike_counts, frame_rate=None):
    if frame_rate is None:
        frame_rate = FRAME_RATE

    sigma_frames = SIGMA_SEC * frame_rate

    smoothed = gaussian_filter1d(
        spike_counts,
        sigma=sigma_frames,
        mode="nearest"
    )

    return smoothed * frame_rate

def _load_raw_sweeps(mat_path):
    data = loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False
    )

    raw_sweeps = []

    for sweep_idx, sweep in enumerate(_iter_attached_sweeps(data, mat_path)):
        fluo_time = _as_1d_float_array(
            _get_field(sweep, "fluo_time", mat_path, sweep_idx)
        )
        fluo_mean = _as_1d_float_array(
            _get_field(sweep, "fluo_mean", mat_path, sweep_idx)
        )
        events_AP = _as_1d_float_array(
            _get_field(sweep, "events_AP", mat_path, sweep_idx)
        )

        if len(fluo_time) != len(fluo_mean):
            raise ValueError(
                f"{mat_path} sweep {sweep_idx} has mismatched fluo_time "
                f"({len(fluo_time)}) and fluo_mean ({len(fluo_mean)}) lengths"
            )

        if len(fluo_time) == 0:
            continue

        raw_sweeps.append((fluo_time, fluo_mean, events_AP))

    if not raw_sweeps:
        raise ValueError(f"{mat_path} did not contain any usable sweeps")

    return raw_sweeps

def load_neuron(mat_path):

    raw_sweeps = _load_raw_sweeps(mat_path)

    all_fluo = np.concatenate(
        [fluo_mean for _, fluo_mean, _ in raw_sweeps]
    )
    median_x = np.median(all_fluo)
    mad_x = np.median(np.abs(all_fluo - median_x))

    processed_sweeps = []

    for fluo_time, fluo_mean, events_AP in raw_sweeps:
        spike_times_sec = events_AP / 10000.0

        x_norm = robust_normalize(
            fluo_mean,
            median_x=median_x,
            mad_x=mad_x
        )

        spike_counts = spike_times_to_counts(
            spike_times_sec,
            fluo_time
        )

        frame_rate = estimate_frame_rate(fluo_time)

        y_spike_rate = smooth_spike_counts(
            spike_counts,
            frame_rate=frame_rate
        )

        processed_sweeps.append((x_norm, y_spike_rate))

    return processed_sweeps

def load_neuron_for_final_eval(mat_path):

    raw_sweeps = _load_raw_sweeps(mat_path)

    all_fluo = np.concatenate(
        [fluo_mean for _, fluo_mean, _ in raw_sweeps]
    )
    median_x = np.median(all_fluo)
    mad_x = np.median(np.abs(all_fluo - median_x))

    eval_sweeps = []

    for sweep_idx, (fluo_time, fluo_mean, events_AP) in enumerate(raw_sweeps):
        spike_times_sec = events_AP / 10000.0
        spike_times_sec = spike_times_sec[np.isfinite(spike_times_sec)]
        spike_times_sec = spike_times_sec[
            (spike_times_sec >= fluo_time[0]) &
            (spike_times_sec <= fluo_time[-1])
        ]
        spike_times_sec = np.sort(spike_times_sec)

        x_norm = robust_normalize(
            fluo_mean,
            median_x=median_x,
            mad_x=mad_x
        )

        eval_sweeps.append(
            {
                "fluo_time": fluo_time,
                "x_norm": x_norm,
                "spike_times_sec": spike_times_sec,
                "sweep_idx": sweep_idx,
            }
        )

    return eval_sweeps

def create_windows(x_norm, y_spike_rate):

    T = len(x_norm)

    if T <= WINDOW_SIZE:
        return (
            np.empty((0, WINDOW_SIZE, 1), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    num_windows = T - WINDOW_SIZE
    X = np.lib.stride_tricks.sliding_window_view(
        x_norm,
        WINDOW_SIZE
    )[:num_windows]
    y = y_spike_rate[HALF_WINDOW:HALF_WINDOW + num_windows]

    return X[:, :, None].astype(np.float32), y.astype(np.float32)
