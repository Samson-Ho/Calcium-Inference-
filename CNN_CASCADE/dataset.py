from dataclasses import dataclass

import numpy as np

from config import HALF_WINDOW, WINDOW_SIZE
from preprocess import load_neuron

import torch


@dataclass
class SweepRecord:
    x_norm: np.ndarray
    y_spike_rate: np.ndarray
    mat_path: str
    sweep_idx: int
    num_windows: int


class CalciumDataset:
    """Lazy window dataset.

    The split is still neuron-level: every .mat file is loaded only through the
    file list handed to this object.  Windows are materialized per batch instead
    of being copied into one huge [num_windows, 64, 1] array.
    """

    def __init__(self, file_list):
        self.sweeps = []
        self.file_origin = []

        for mat_path in file_list:
            processed_sweeps = load_neuron(mat_path)

            for sweep_idx, (x_norm, y_spike_rate) in enumerate(processed_sweeps):
                num_windows = len(x_norm) - WINDOW_SIZE

                if num_windows <= 0:
                    continue

                self.sweeps.append(
                    SweepRecord(
                        x_norm=np.ascontiguousarray(x_norm, dtype=np.float32),
                        y_spike_rate=np.ascontiguousarray(
                            y_spike_rate,
                            dtype=np.float32
                        ),
                        mat_path=str(mat_path),
                        sweep_idx=sweep_idx,
                        num_windows=num_windows,
                    )
                )
                self.file_origin.append(str(mat_path))

        if not self.sweeps:
            raise ValueError("No CNN windows were created from file_list")

        self.window_counts = np.array(
            [sweep.num_windows for sweep in self.sweeps],
            dtype=np.int64
        )
        self.window_ends = np.cumsum(self.window_counts)
        self.window_starts = self.window_ends - self.window_counts
        self.total_windows = int(self.window_ends[-1])
        self.window_offsets = np.arange(WINDOW_SIZE, dtype=np.int64)

    def __len__(self):
        return self.total_windows

    def num_batches(self, batch_size, drop_last=False):
        if drop_last:
            return self.total_windows // batch_size

        return (self.total_windows + batch_size - 1) // batch_size

    def get_batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)

        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")

        if len(indices) == 0:
            raise ValueError("Cannot build an empty batch")

        sweep_ids = np.searchsorted(
            self.window_ends,
            indices,
            side="right"
        )
        local_window_starts = indices - self.window_starts[sweep_ids]

        X = np.empty(
            (len(indices), 1, WINDOW_SIZE),
            dtype=np.float32
        )
        y = np.empty(len(indices), dtype=np.float32)

        order = np.argsort(sweep_ids, kind="stable")
        sorted_sweep_ids = sweep_ids[order]
        group_starts = np.r_[
            0,
            np.flatnonzero(np.diff(sorted_sweep_ids)) + 1,
            len(order)
        ]

        for start, stop in zip(group_starts[:-1], group_starts[1:]):
            rows = order[start:stop]
            sweep_id = int(sorted_sweep_ids[start])
            sweep = self.sweeps[sweep_id]
            starts = local_window_starts[rows]

            X[rows, 0, :] = sweep.x_norm[
                starts[:, None] + self.window_offsets[None, :]
            ]
            y[rows] = sweep.y_spike_rate[starts + HALF_WINDOW]

        return torch.from_numpy(X), torch.from_numpy(y)

    def __getitem__(self, idx):
        X, y = self.get_batch(np.array([idx], dtype=np.int64))
        return X[0], y[0]

    def iter_batches(
        self,
        batch_size,
        shuffle=False,
        seed=None,
        drop_last=False,
    ):
        if shuffle:
            rng = np.random.default_rng(seed)
            order = rng.permutation(self.total_windows)

            for start in range(0, self.total_windows, batch_size):
                stop = start + batch_size

                if stop > self.total_windows and drop_last:
                    break

                yield self.get_batch(order[start:min(stop, self.total_windows)])

            return

        stop_at = self.total_windows

        if drop_last:
            stop_at = (self.total_windows // batch_size) * batch_size

        for start in range(0, stop_at, batch_size):
            stop = min(start + batch_size, self.total_windows)
            indices = np.arange(start, stop, dtype=np.int64)
            yield self.get_batch(indices)
