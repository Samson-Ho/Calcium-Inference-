
from pathlib import Path
import random
from config import *

def split_dataset_files():
    random.seed(SEED)

    train_files = []
    val_files = []
    test_files = []

    split_summary = {}

    for dataset_name in DATASETS:

        dataset_path = DATA_ROOT / dataset_name

        files = sorted(dataset_path.glob("*.mat"))

        files = list(files)
        random.shuffle(files)

        n = len(files)

        n_train = round(TRAIN_RATIO * n)
        n_val = round(VAL_RATIO * n)
        n_test = n - n_train - n_val

        dataset_train = files[:n_train]
        dataset_val = files[n_train:n_train+n_val]
        dataset_test = files[n_train+n_val:]

        train_files.extend(dataset_train)
        val_files.extend(dataset_val)
        test_files.extend(dataset_test)

        split_summary[dataset_name] = {
            "total": n,
            "train": len(dataset_train),
            "val": len(dataset_val),
            "test": len(dataset_test),
        }

    assert len(set(train_files).intersection(set(val_files))) == 0
    assert len(set(train_files).intersection(set(test_files))) == 0
    assert len(set(val_files).intersection(set(test_files))) == 0

    return train_files, val_files, test_files, split_summary
