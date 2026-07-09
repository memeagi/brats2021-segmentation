# ================================================================
# Utility functions for reproducibility and dataset handling
# ================================================================

import os
import tarfile
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_dataset(tar_path, extract_path):
    """
    Extract the BraTS 2021 dataset from a TAR archive.

    Parameters
    ----------
    tar_path : str
        Path to the BraTS 2021 TAR file.

    extract_path : str
        Destination folder for extraction.
    """

    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)

        print("Extracting BraTS 2021 dataset...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_path)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")


def get_patient_folders(extract_path):
    """
    Find valid BraTS 2021 patient folders.

    Parameters
    ----------
    extract_path : str
        Extracted dataset directory.

    Returns
    -------
    list
        Sorted list of patient folder paths.
    """

    patient_folders = []

    for root, dirs, files in os.walk(extract_path):
        for folder in dirs:
            if folder.startswith("BraTS2021_"):
                patient_folders.append(os.path.join(root, folder))

    patient_folders = sorted(patient_folders)

    return patient_folders