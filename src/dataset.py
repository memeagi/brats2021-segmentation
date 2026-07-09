# ================================================================
# PyTorch Dataset for BraTS 2021 multi-modal MRI segmentation
# ================================================================

import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing import normalize_image
from config import CFG


class BraTSSegmentationDataset(Dataset):
    """
    Dataset class for loading BraTS 2021 2D tumor-containing slices.

    Input:
        Four MRI modalities: T1, T1ce, T2, FLAIR

    Output:
        image tensor: [4, 240, 240]
        mask tensor:  [1, 240, 240]
    """

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        patient_folder, z = self.records[idx]
        patient_id = os.path.basename(patient_folder)

        image_channels = []

        for modality in CFG.modalities:
            image_path = os.path.join(
                patient_folder,
                f"{patient_id}_{modality}.nii.gz"
            )

            image_volume = nib.load(image_path).get_fdata()
            image_slice = image_volume[:, :, z]
            image_slice = normalize_image(image_slice)

            image_channels.append(image_slice)

        image = np.stack(image_channels, axis=0)

        mask_path = os.path.join(patient_folder, f"{patient_id}_seg.nii.gz")
        mask_volume = nib.load(mask_path).get_fdata()

        mask_slice = (mask_volume[:, :, z] > 0).astype(np.float32)
        mask_slice = np.expand_dims(mask_slice, axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        mask_slice = torch.tensor(mask_slice, dtype=torch.float32)

        return image, mask_slice


def create_slice_records(patient_folders):
    """
    Create records of tumor-containing slices.

    Parameters
    ----------
    patient_folders : list
        List of BraTS patient folders.

    Returns
    -------
    list
        List of tuples: (patient_folder, slice_index)
    """

    slice_records = []

    for patient_folder in patient_folders:
        patient_id = os.path.basename(patient_folder)
        seg_path = os.path.join(patient_folder, f"{patient_id}_seg.nii.gz")

        seg_volume = nib.load(seg_path).get_fdata()

        for z in range(seg_volume.shape[2]):
            if np.sum(seg_volume[:, :, z] > 0) > 0:
                slice_records.append((patient_folder, z))

    return slice_records