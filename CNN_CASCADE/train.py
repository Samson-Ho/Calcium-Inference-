import math
import queue
import threading
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np

from config import *
from dataset import CalciumDataset
from preprocess import load_neuron_for_final_eval
from split import split_dataset_files

import torch
import torch.nn as nn

from model import SpikeCNN

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def configure_device(device):
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def print_runtime_status(device, use_amp):
    print("Runtime status:")
    print("  PyTorch:", torch.__version__)
    print("  Selected device:", device)
    print("  CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        current_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_idx)
        total_gb = props.total_memory / (1024 ** 3)

        print("  CUDA version:", torch.version.cuda)
        print("  CUDA device count:", torch.cuda.device_count())
        print("  CUDA current device:", current_idx)
        print("  CUDA device name:", props.name)
        print(f"  CUDA total memory: {total_gb:.2f} GB")
        print("  cuDNN enabled:", torch.backends.cudnn.enabled)
        print("  cuDNN benchmark:", torch.backends.cudnn.benchmark)
        print("  CUDA matmul TF32:", torch.backends.cuda.matmul.allow_tf32)
        print("  cuDNN TF32:", torch.backends.cudnn.allow_tf32)
    else:
        print("  CUDA version: unavailable")

    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    print("  MPS available:", mps_available)
    print("  AMP enabled:", use_amp)
    print("  Batch size:", BATCH_SIZE)
    print("  Eval batch size:", EVAL_BATCH_SIZE)
    print("  Epochs:", NUM_EPOCHS)
    print("  Prefetch batches:", PREFETCH_BATCHES)
    print("  Pin memory:", PIN_MEMORY)


def progress(iterator, total, desc, leave=False):
    if not SHOW_PROGRESS or tqdm is None:
        return iterator

    return tqdm(
        iterator,
        total=total,
        desc=desc,
        unit="batch",
        leave=leave,
        dynamic_ncols=True,
    )


def prefetch(iterator, max_prefetch):
    if max_prefetch <= 0:
        yield from iterator
        return

    batches = queue.Queue(maxsize=max_prefetch)
    sentinel = object()

    def worker():
        try:
            for item in iterator:
                batches.put(item)
        except BaseException as exc:
            batches.put(exc)
        finally:
            batches.put(sentinel)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = batches.get()

        if item is sentinel:
            break

        if isinstance(item, BaseException):
            raise item

        yield item


def autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda")

        return torch.cuda.amp.autocast()

    return nullcontext()


def make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)

    return torch.cuda.amp.GradScaler(enabled=enabled)


def move_batch_to_device(x, y, device):
    non_blocking = device.type == "cuda"

    if PIN_MEMORY and device.type == "cuda":
        x = x.pin_memory()
        y = y.pin_memory()

    return (
        x.to(device, non_blocking=non_blocking),
        y.to(device, non_blocking=non_blocking),
    )


def update_streaming_metrics(pred, target, state):
    pred = pred.detach().float().cpu().numpy().reshape(-1)
    target = target.detach().float().cpu().numpy().reshape(-1)

    diff = pred - target
    n = len(pred)

    state["n"] += n
    state["mse_sum"] += float(np.sum(diff * diff))
    state["mae_sum"] += float(np.sum(np.abs(diff)))
    state["sum_pred"] += float(np.sum(pred))
    state["sum_target"] += float(np.sum(target))
    state["sum_pred2"] += float(np.sum(pred * pred))
    state["sum_target2"] += float(np.sum(target * target))
    state["sum_pred_target"] += float(np.sum(pred * target))


def finalize_metrics(state):
    n = state["n"]

    if n == 0:
        return {
            "mse": math.nan,
            "mae": math.nan,
            "pearson": math.nan,
        }

    numerator = (
        n * state["sum_pred_target"]
        - state["sum_pred"] * state["sum_target"]
    )
    denom_pred = n * state["sum_pred2"] - state["sum_pred"] ** 2
    denom_target = n * state["sum_target2"] - state["sum_target"] ** 2

    if denom_pred > 0 and denom_target > 0:
        pearson = numerator / math.sqrt(denom_pred * denom_target)
    else:
        pearson = math.nan

    return {
        "mse": state["mse_sum"] / n,
        "mae": state["mae_sum"] / n,
        "pearson": pearson,
    }


def evaluate(model, dataset, device, desc="Eval"):
    model.eval()

    state = {
        "n": 0,
        "mse_sum": 0.0,
        "mae_sum": 0.0,
        "sum_pred": 0.0,
        "sum_target": 0.0,
        "sum_pred2": 0.0,
        "sum_target2": 0.0,
        "sum_pred_target": 0.0,
    }

    num_batches = dataset.num_batches(EVAL_BATCH_SIZE)
    batch_iter = dataset.iter_batches(
        EVAL_BATCH_SIZE,
        shuffle=False
    )
    batch_iter = prefetch(batch_iter, PREFETCH_BATCHES)

    with torch.inference_mode():
        for x, y in progress(batch_iter, num_batches, desc):
            x, y = move_batch_to_device(x, y, device)
            pred = model(x)
            update_streaming_metrics(pred, y, state)

    return finalize_metrics(state)


def print_metrics(label, metrics):
    print(
        f"{label}: "
        f"MSE={metrics['mse']:.6f} | "
        f"MAE={metrics['mae']:.6f} | "
        f"Pearson={metrics['pearson']:.4f}"
    )


def model_shape_summary(model):
    conv_channels = []
    linear_features = []

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            if not conv_channels:
                conv_channels.append(module.in_channels)

            conv_channels.append(module.out_channels)
        elif isinstance(module, nn.Linear):
            if not linear_features:
                linear_features.append(module.in_features)

            linear_features.append(module.out_features)

    shape = conv_channels + linear_features[1:]
    param_count = sum(p.numel() for p in model.parameters())

    return {
        "shape": " -> ".join(str(value) for value in shape),
        "conv_channels": " -> ".join(str(value) for value in conv_channels),
        "linear_features": " -> ".join(
            str(value) for value in linear_features
        ),
        "parameters": param_count,
    }


def gpu_status_label(device):
    if device.type == "cuda":
        return "yes (CUDA)"

    if device.type == "mps":
        return "yes (MPS)"

    return "no (CPU)"


def split_tree_lines(split_summary):
    total_train = sum(item["train"] for item in split_summary.values())
    total_val = sum(item["val"] for item in split_summary.values())
    total_test = sum(item["test"] for item in split_summary.values())
    total_neurons = sum(item["total"] for item in split_summary.values())

    lines = [
        f"Total neurons ({total_neurons})",
        f"├── Train: {total_train}",
        f"├── Val:   {total_val}",
        f"└── Test:  {total_test}",
        "",
    ]

    for dataset_name in DATASETS:
        item = split_summary[dataset_name]
        short_name = dataset_name.split("-")[0]

        lines.extend(
            [
                f"{short_name} ({item['total']})",
                f"├── Train: {item['train']}",
                f"├── Val:   {item['val']}",
                f"└── Test:  {item['test']}",
                "",
            ]
        )

    return lines


def predict_trace(model, x_norm, device, use_amp):
    model.eval()

    T = len(x_norm)
    pred_rate = np.full(T, np.nan, dtype=np.float32)
    num_windows = T - WINDOW_SIZE

    if num_windows <= 0:
        return pred_rate

    offsets = np.arange(WINDOW_SIZE, dtype=np.int64)

    with torch.inference_mode():
        for start in range(0, num_windows, EVAL_BATCH_SIZE):
            stop = min(start + EVAL_BATCH_SIZE, num_windows)
            starts = np.arange(start, stop, dtype=np.int64)
            X = x_norm[starts[:, None] + offsets[None, :]][:, None, :]
            X = np.ascontiguousarray(X, dtype=np.float32)
            x = torch.from_numpy(X)

            if PIN_MEMORY and device.type == "cuda":
                x = x.pin_memory()

            x = x.to(device, non_blocking=device.type == "cuda")

            with autocast_context(device, use_amp):
                pred = model(x)

            pred_rate[
                HALF_WINDOW + start:HALF_WINDOW + stop
            ] = pred.detach().float().cpu().numpy()

    return pred_rate


def nearest_frame_index(fluo_time, spike_time):
    idx = int(np.searchsorted(fluo_time, spike_time))

    if idx <= 0:
        return 0

    if idx >= len(fluo_time):
        return len(fluo_time) - 1

    prev_idx = idx - 1

    if abs(fluo_time[prev_idx] - spike_time) <= abs(fluo_time[idx] - spike_time):
        return prev_idx

    return idx


def find_isolated_spikes(spike_times_sec):
    isolated = []

    for idx, spike_time in enumerate(spike_times_sec):
        if idx == 0:
            prev_distance = math.inf
        else:
            prev_distance = spike_time - spike_times_sec[idx - 1]

        if idx == len(spike_times_sec) - 1:
            next_distance = math.inf
        else:
            next_distance = spike_times_sec[idx + 1] - spike_time

        if (
            prev_distance > ISOLATED_AP_MIN_INTERVAL_SEC
            and next_distance > ISOLATED_AP_MIN_INTERVAL_SEC
        ):
            isolated.append(float(spike_time))

    return isolated


def empty_isolated_ap_counts():
    return {
        "neurons": 0,
        "sweeps": 0,
        "spikes": 0,
        "isolated": 0,
        "tp": 0,
        "fn": 0,
        "peak_sum": 0.0,
        "peak_count": 0,
    }


def add_isolated_ap_counts(total, update):
    for key in ["neurons", "sweeps", "spikes", "isolated", "tp", "fn"]:
        total[key] += update[key]

    total["peak_sum"] += update["peak_sum"]
    total["peak_count"] += update["peak_count"]


def isolated_ap_recall(counts):
    denom = counts["tp"] + counts["fn"]

    if denom == 0:
        return math.nan

    return counts["tp"] / denom


def isolated_ap_mean_peak(counts):
    if counts["peak_count"] == 0:
        return math.nan

    return counts["peak_sum"] / counts["peak_count"]


def evaluate_isolated_single_spikes(model, test_files, device, use_amp):
    total = empty_isolated_ap_counts()
    by_dataset = {
        dataset_name: empty_isolated_ap_counts()
        for dataset_name in DATASETS
    }
    per_neuron = []

    file_iter = progress(
        iter(test_files),
        len(test_files),
        "Isolated AP final",
        leave=False,
    )

    for mat_path in file_iter:
        dataset_name = mat_path.parent.name
        file_counts = empty_isolated_ap_counts()
        file_counts["neurons"] = 1

        eval_sweeps = load_neuron_for_final_eval(mat_path)

        for sweep in eval_sweeps:
            fluo_time = sweep["fluo_time"]
            spike_times = sweep["spike_times_sec"]
            pred_rate = predict_trace(
                model,
                sweep["x_norm"],
                device,
                use_amp,
            )
            single_spikes = find_isolated_spikes(spike_times)

            file_counts["sweeps"] += 1
            file_counts["spikes"] += len(spike_times)
            file_counts["isolated"] += len(single_spikes)

            for spike_time in single_spikes:
                spike_frame = nearest_frame_index(fluo_time, spike_time)
                start = max(
                    0,
                    spike_frame - ISOLATED_AP_LOCAL_RADIUS_FRAMES
                )
                stop = min(
                    len(pred_rate),
                    spike_frame + ISOLATED_AP_LOCAL_RADIUS_FRAMES + 1
                )
                local_pred = pred_rate[start:stop]
                finite_pred = local_pred[np.isfinite(local_pred)]

                if len(finite_pred) == 0:
                    file_counts["fn"] += 1
                    continue

                pred_peak = float(np.max(finite_pred))
                file_counts["peak_sum"] += pred_peak
                file_counts["peak_count"] += 1

                if pred_peak > ISOLATED_AP_THRESHOLD:
                    file_counts["tp"] += 1
                else:
                    file_counts["fn"] += 1

        add_isolated_ap_counts(total, file_counts)

        if dataset_name in by_dataset:
            add_isolated_ap_counts(by_dataset[dataset_name], file_counts)

        per_neuron.append(
            {
                "dataset": dataset_name,
                "file": mat_path.name,
                **file_counts,
                "recall": isolated_ap_recall(file_counts),
                "mean_peak": isolated_ap_mean_peak(file_counts),
            }
        )

    return {
        "threshold": ISOLATED_AP_THRESHOLD,
        "min_interval_sec": ISOLATED_AP_MIN_INTERVAL_SEC,
        "local_radius_frames": ISOLATED_AP_LOCAL_RADIUS_FRAMES,
        "total": {
            **total,
            "recall": isolated_ap_recall(total),
            "mean_peak": isolated_ap_mean_peak(total),
        },
        "by_dataset": {
            dataset_name: {
                **counts,
                "recall": isolated_ap_recall(counts),
                "mean_peak": isolated_ap_mean_peak(counts),
            }
            for dataset_name, counts in by_dataset.items()
        },
        "per_neuron": per_neuron,
    }


def format_float(value):
    if value is None or math.isnan(value):
        return "nan"

    return f"{value:.8f}"


def isolated_ap_log_lines(metrics):
    total = metrics["total"]
    lines = [
        "",
        "Final isolated single-AP detection evaluation:",
        (
            "  Criterion: previous spike distance > "
            f"{metrics['min_interval_sec']} sec and next spike distance > "
            f"{metrics['min_interval_sec']} sec"
        ),
        (
            "  Local prediction window: spike_frame ± "
            f"{metrics['local_radius_frames']} frames"
        ),
        f"  Detection threshold: pred_peak > {metrics['threshold']}",
        (
            "  Total: "
            f"neurons={total['neurons']} "
            f"sweeps={total['sweeps']} "
            f"spikes={total['spikes']} "
            f"isolated={total['isolated']} "
            f"TP={total['tp']} "
            f"FN={total['fn']} "
            f"recall={format_float(total['recall'])} "
            f"mean_peak={format_float(total['mean_peak'])}"
        ),
        "",
        "  Per-dataset isolated AP detection:",
    ]

    for dataset_name in DATASETS:
        counts = metrics["by_dataset"][dataset_name]
        lines.append(
            "    "
            f"{dataset_name}: "
            f"neurons={counts['neurons']} "
            f"sweeps={counts['sweeps']} "
            f"spikes={counts['spikes']} "
            f"isolated={counts['isolated']} "
            f"TP={counts['tp']} "
            f"FN={counts['fn']} "
            f"recall={format_float(counts['recall'])} "
            f"mean_peak={format_float(counts['mean_peak'])}"
        )

    lines.extend(["", "  Per-neuron isolated AP detection:"])

    for item in metrics["per_neuron"]:
        lines.append(
            "    "
            f"{item['file']}: "
            f"dataset={item['dataset']} "
            f"sweeps={item['sweeps']} "
            f"spikes={item['spikes']} "
            f"isolated={item['isolated']} "
            f"TP={item['tp']} "
            f"FN={item['fn']} "
            f"recall={format_float(item['recall'])} "
            f"mean_peak={format_float(item['mean_peak'])}"
        )

    return lines


def save_loss_plot(history, test_loss, output_path):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    epochs = [item["epoch"] for item in history]
    val_losses = [item["val_mse"] for item in history]

    if not epochs:
        raise ValueError("Cannot save loss plot without epoch history")

    val_min_idx = int(np.argmin(val_losses))
    val_min_epoch = epochs[val_min_idx]
    val_min_loss = val_losses[val_min_idx]
    test_losses = [test_loss] * len(epochs)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(
        epochs,
        val_losses,
        color="orange",
        linestyle="-",
        linewidth=1.8,
        label="Validation loss",
    )
    ax.plot(
        epochs,
        test_losses,
        color="#003153",
        linestyle="-",
        linewidth=1.8,
        label="Testing loss",
    )

    ax.annotate(
        f"{val_min_loss:.6f}",
        xy=(val_min_epoch, val_min_loss),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        color="red",
        fontsize=8,
    )
    ax.annotate(
        f"{test_loss:.6f}",
        xy=(val_min_epoch, test_loss),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        color="red",
        fontsize=8,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Validation and Testing Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=450)
    plt.close(fig)


def write_training_log(
    output_path,
    started_at,
    finished_at,
    device,
    use_amp,
    model_info,
    history,
    best_epoch,
    best_val_loss,
    test_metrics,
    per_dataset_metrics,
    isolated_ap_metrics,
    split_summary,
):
    lines = [
        "CNN_CASCADE training log",
        "",
        f"開始訓練時間: {started_at.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"訓練結束時間: {finished_at.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"是否使用GPU(MPS/CUDA): {gpu_status_label(device)}",
        f"Selected device: {device}",
        f"AMP enabled: {use_amp}",
        f"Model shape: {model_info['shape']}",
        f"Conv1d channels: {model_info['conv_channels']}",
        f"Linear features: {model_info['linear_features']}",
        f"Trainable parameters: {model_info['parameters']}",
        f"Batch size: {BATCH_SIZE}",
        f"Eval batch size: {EVAL_BATCH_SIZE}",
        f"設定的epoch: {NUM_EPOCHS}",
        f"真的有跑完的epoch數量: {len(history)}",
        f"Best epoch by validation loss: {best_epoch}",
        f"Best validation loss (MSE): {best_val_loss:.8f}",
        (
            "Test loss at best validation checkpoint (MSE): "
            f"{test_metrics['mse']:.8f}"
        ),
        f"Test MAE at best validation checkpoint: {test_metrics['mae']:.8f}",
        (
            "Test Pearson at best validation checkpoint: "
            f"{test_metrics['pearson']:.8f}"
        ),
        "",
        "Leakage policy:",
        (
            "  Test set is evaluated once after training using the best "
            "validation checkpoint; it is not used for early stopping or "
            "checkpoint selection."
        ),
        "",
        "Split summary:",
        f"  {split_summary}",
        "",
        "Neuron counts by dataset:",
    ]

    lines.extend(split_tree_lines(split_summary))
    lines.extend(
        [
        "Epoch history:",
        ]
    )

    for item in history:
        lines.append(
            "  "
            f"epoch={item['epoch']} "
            f"train_loss={item['train_loss']:.8f} "
            f"val_loss={item['val_mse']:.8f} "
            f"val_mae={item['val_mae']:.8f} "
            f"val_pearson={item['val_pearson']:.8f} "
            f"samples_per_sec={item['samples_per_sec']:.2f} "
            f"elapsed_sec={item['elapsed_sec']:.2f}"
        )

    lines.extend(["", "Per-dataset test metrics:"])

    for dataset_name, metrics in per_dataset_metrics.items():
        lines.append(
            "  "
            f"{dataset_name}: "
            f"mse={metrics['mse']:.8f} "
            f"mae={metrics['mae']:.8f} "
            f"pearson={metrics['pearson']:.8f}"
        )

    lines.extend(isolated_ap_log_lines(isolated_ap_metrics))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    started_at = datetime.now().astimezone()
    run_stamp = started_at.strftime("%m%d-%H%M")
    loss_plot_path = Path(f"loss_{run_stamp}.png")
    log_path = Path(f"log_{run_stamp}.txt")

    device = get_device()
    configure_device(device)

    use_amp = USE_AMP and device.type == "cuda"

    print_runtime_status(device, use_amp)

    train_files, val_files, test_files, summary = split_dataset_files()

    print("Split summary:")
    print(summary)

    print("\nNO DATA LEAKAGE CHECK")
    print("Train neurons:", len(train_files))
    print("Val neurons:", len(val_files))
    print("Test neurons:", len(test_files))

    print("\nPreparing lazy datasets...")
    train_dataset = CalciumDataset(train_files)
    val_dataset = CalciumDataset(val_files)
    test_dataset = CalciumDataset(test_files)

    print("Train windows:", len(train_dataset))
    print("Val windows:", len(val_dataset))
    print("Test windows:", len(test_dataset))

    model = SpikeCNN().to(device)
    model_info = model_shape_summary(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    scaler = make_grad_scaler(use_amp)

    best_val = np.inf
    best_epoch = None
    patience = 10
    wait = 0
    history = []

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.perf_counter()
        model.train()

        running_loss = 0.0
        seen = 0
        num_batches = train_dataset.num_batches(
            BATCH_SIZE,
            drop_last=DROP_LAST_TRAIN_BATCH
        )
        batch_iter = train_dataset.iter_batches(
            BATCH_SIZE,
            shuffle=True,
            seed=SEED + epoch,
            drop_last=DROP_LAST_TRAIN_BATCH,
        )
        batch_iter = prefetch(batch_iter, PREFETCH_BATCHES)
        bar = progress(
            batch_iter,
            num_batches,
            f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            leave=True,
        )

        for x, y in bar:
            batch_size = len(y)
            x, y = move_batch_to_device(x, y, device)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, use_amp):
                pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * batch_size
            seen += batch_size

            if SHOW_PROGRESS and tqdm is not None:
                bar.set_postfix(
                    loss=f"{running_loss / max(seen, 1):.6f}"
                )

        train_loss = running_loss / max(seen, 1)

        val_metrics = evaluate(
            model,
            val_dataset,
            device,
            desc=f"Val {epoch + 1}/{NUM_EPOCHS}",
        )

        elapsed = time.perf_counter() - epoch_start
        samples_per_sec = seen / elapsed if elapsed > 0 else 0.0

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss={train_loss:.6f} | "
            f"Val MSE={val_metrics['mse']:.6f} | "
            f"Val Corr={val_metrics['pearson']:.4f} | "
            f"{samples_per_sec:.0f} samples/s | "
            f"{elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "val_pearson": val_metrics["pearson"],
                "samples_per_sec": samples_per_sec,
                "elapsed_sec": elapsed,
            }
        )

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print("Early stopping triggered.")
            break

    print("\nFINAL TEST EVALUATION")
    print("Loading best checkpoint...")

    model.load_state_dict(
        torch.load(MODEL_SAVE_PATH, map_location=device)
    )

    test_metrics = evaluate(
        model,
        test_dataset,
        device,
        desc="Test all",
    )
    print_metrics("Test all", test_metrics)

    print("\nPer-dataset test metrics:")
    per_dataset_metrics = {}

    for dataset_name in DATASETS:
        dataset_test_files = [
            path for path in test_files
            if path.parent.name == dataset_name
        ]

        if not dataset_test_files:
            continue

        dataset_test = CalciumDataset(dataset_test_files)
        dataset_metrics = evaluate(
            model,
            dataset_test,
            device,
            desc=f"Test {dataset_name}",
        )
        per_dataset_metrics[dataset_name] = dataset_metrics
        print_metrics(dataset_name, dataset_metrics)

    print("\nIsolated single-AP detection evaluation:")
    isolated_ap_metrics = evaluate_isolated_single_spikes(
        model,
        test_files,
        device,
        use_amp,
    )
    isolated_total = isolated_ap_metrics["total"]
    print(
        "Isolated AP total: "
        f"isolated={isolated_total['isolated']} | "
        f"TP={isolated_total['tp']} | "
        f"FN={isolated_total['fn']} | "
        f"Recall={format_float(isolated_total['recall'])}"
    )

    finished_at = datetime.now().astimezone()

    save_loss_plot(
        history,
        test_metrics["mse"],
        loss_plot_path,
    )
    write_training_log(
        log_path,
        started_at,
        finished_at,
        device,
        use_amp,
        model_info,
        history,
        best_epoch,
        best_val,
        test_metrics,
        per_dataset_metrics,
        isolated_ap_metrics,
        summary,
    )

    print("\nSaved artifacts:")
    print("Loss plot:", loss_plot_path)
    print("Training log:", log_path)


if __name__ == "__main__":
    main()
