# CNN_CASCADE

Pure data-driven 1D CNN for spike-rate inference from CASCADE GCaMP8 calcium
imaging ground-truth data.

## Core Rule: No Data Leakage

The split is strictly neuron-level:

- One `.mat` file is treated as one neuron.
- Windows from the same `.mat` file must stay in only one split.
- Time points from one neuron are never split across train/validation/test.
- Split ratio is train : validation : test = 80 : 10 : 10.
- DS30, DS31, and DS32 are split separately first, then concatenated.

This preserves the GCaMP8f/8m/8s composition in every split.

## Datasets

Data root:

```text
/Users/samsonho/KSL/data/Ground_truth
```

Used datasets:

```text
DS30-GCaMP8f-m-V1
DS31-GCaMP8m-m-V1
DS32-GCaMP8s-m-V1
```

Each `.mat` file contains one neuron. Internally, each file may contain
multiple `CAttached` sweeps; these sweeps remain under the same neuron split.

## Run

From this directory:

```bash
python train.py
```

The script prints runtime status at startup, including PyTorch version, selected
device, CUDA/MPS availability, AMP status, batch size, eval batch size, epochs,
prefetch, and pin-memory settings.

## Outputs

Each completed training run writes:

- `best_model.pt`: best checkpoint selected by validation MSE.
- `loss_{mmdd-hhmm}.png`: validation loss and final testing loss plot.
- `log_{mmdd-hhmm}.txt`: training log with device status, model shape, split
  counts, epoch history, best validation epoch, final test metrics, and
  isolated single-AP detection metrics.

The test set is evaluated after training with the best validation checkpoint.
It is not used for early stopping or checkpoint selection.

## Training Notes

- Progress bars are shown for training, validation, and final testing.
- Windows are generated lazily per batch, so the code does not materialize all
  `[num_windows, 64, 1]` samples in memory.
- CPU batch construction is prefetched in the background to reduce GPU idle
  time.
- `BATCH_SIZE`, `EVAL_BATCH_SIZE`, `NUM_EPOCHS`, and other hyperparameters are
  controlled in `config.py`.
- Smaller training batch sizes, such as 32, are valid but may reduce GPU
  utilization for this small 1D CNN.

## Python Files

### `config.py`

Central configuration file.

Defines:

- Data root and dataset names.
- Split ratios and random seed.
- Window size and Gaussian smoothing parameters.
- Training hyperparameters such as batch size, epoch count, learning rate, AMP,
  prefetching, and checkpoint path.
- Isolated AP detection settings: minimum spike isolation interval, local frame
  radius, and prediction threshold.

Edit this file when changing experiment settings.

### `split.py`

Builds the neuron-level train/validation/test split.

Main responsibility:

- Finds all `.mat` files in DS30, DS31, and DS32.
- Shuffles each dataset separately with fixed seed `42`.
- Splits each dataset independently into 80:10:10.
- Concatenates DS30/DS31/DS32 train files, validation files, and test files.
- Asserts that no `.mat` file appears in more than one split.

This file protects the no-data-leakage rule at the file/neuron level.

### `preprocess.py`

Loads and preprocesses one neuron file.

Main responsibility:

- Reads MATLAB `.mat` files.
- Extracts `fluo_time`, `fluo_mean`, and `events_AP` from each `CAttached`
  sweep.
- Converts spike times from `events_AP / 10000` into seconds.
- Robust-normalizes fluorescence with median and MAD.
- Bins spike times into fluorescence frames.
- Smooths spike counts with a Gaussian kernel to create spike-rate targets.
- Provides window creation utilities for 64-frame CNN inputs.

### `dataset.py`

Defines the lazy CNN dataset.

Main responsibility:

- Loads only the normalized traces and target spike-rate traces.
- Does not precompute and store all overlapping windows.
- Maps global sample indices to the correct sweep and frame.
- Builds each batch of `[batch, 1, 64]` windows on demand.
- Keeps every sweep under its original `.mat` file split.

This file is the main memory-efficiency improvement.

### `model.py`

Defines the CNN architecture.

Model summary:

- Input shape: `[batch, 1, 64]`.
- Conv1d channel flow: `1 -> 32 -> 64 -> 128`.
- Uses ReLU activations, max pooling, adaptive average pooling, and a small
  linear regressor.
- Output shape: `[batch]`, predicted spike rate at the center frame.

### `train.py`

Main training and evaluation script.

Main responsibility:

- Selects CUDA, MPS, or CPU device.
- Prints startup runtime status.
- Creates train/validation/test datasets.
- Trains the CNN with Adam optimizer.
- Uses validation MSE for early stopping and checkpoint selection.
- Evaluates the test set only after training.
- Reports final metrics for all test data and separately for DS30, DS31, DS32.
- Reports isolated single-AP detection on test neurons after training. Isolated
  APs are spikes whose previous and next spike are both more than 0.5 seconds
  away; detection is based on the prediction peak within +/- 3 frames.
- Saves `best_model.pt`, `loss_{mmdd-hhmm}.png`, and `log_{mmdd-hhmm}.txt`.

## Dependencies

Install requirements if needed:

```bash
pip install -r requirements.txt
```

Required packages:

- `numpy`
- `scipy`
- `torch`
- `scikit-learn`
- `tqdm`
- `matplotlib`
