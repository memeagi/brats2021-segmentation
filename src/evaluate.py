# ================================================================
# Evaluation script for BraTS 2021 brain tumor segmentation
# ================================================================

import random

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import CFG
from utils import seed_everything, extract_dataset, get_patient_folders
from dataset import BraTSSegmentationDataset, create_slice_records
from model import EnhancedUNet
from metrics import dice_coefficient, iou_score, precision_score, recall_score


def evaluate():
    """
    Evaluate the trained Enhanced U-Net model on the validation subset.
    """

    seed_everything(CFG.seed)

    print("Using device:", CFG.device)

    extract_dataset(CFG.tar_path, CFG.extract_path)

    patient_folders = get_patient_folders(CFG.extract_path)
    slice_records = create_slice_records(patient_folders)

    train_records, val_records = train_test_split(
        slice_records,
        test_size=0.2,
        random_state=CFG.seed,
        shuffle=True
    )

    random.seed(CFG.seed)

    fast_val_records = random.sample(
        val_records,
        min(CFG.fast_val_slices, len(val_records))
    )

    val_dataset = BraTSSegmentationDataset(fast_val_records)

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers
    )

    model = EnhancedUNet(in_channels=4, out_channels=1).to(CFG.device)
    model.load_state_dict(torch.load(CFG.model_save_path, map_location=CFG.device))
    model.eval()

    val_dice = 0.0
    val_iou = 0.0
    val_precision = 0.0
    val_recall = 0.0

    with torch.no_grad():
        for images_batch, masks_batch in val_loader:
            images_batch = images_batch.to(CFG.device)
            masks_batch = masks_batch.to(CFG.device)

            outputs = model(images_batch)

            val_dice += dice_coefficient(outputs, masks_batch).item()
            val_iou += iou_score(outputs, masks_batch).item()
            val_precision += precision_score(outputs, masks_batch).item()
            val_recall += recall_score(outputs, masks_batch).item()

    avg_dice = val_dice / len(val_loader)
    avg_iou = val_iou / len(val_loader)
    avg_precision = val_precision / len(val_loader)
    avg_recall = val_recall / len(val_loader)

    print("\nFinal Validation Results")
    print("------------------------")
    print(f"Dice:      {avg_dice:.4f}")
    print(f"IoU:       {avg_iou:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")


if __name__ == "__main__":
    evaluate()