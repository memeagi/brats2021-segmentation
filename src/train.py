# ================================================================
# Training script for BraTS 2021 brain tumor segmentation
# ================================================================

import random
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import CFG
from utils import seed_everything, extract_dataset, get_patient_folders
from dataset import BraTSSegmentationDataset, create_slice_records
from model import EnhancedUNet
from losses import BCEDiceLoss
from metrics import dice_coefficient, iou_score, precision_score, recall_score


def train():
    """
    Train Enhanced U-Net on a representative BraTS 2021 subset.
    """

    seed_everything(CFG.seed)

    print("Using device:", CFG.device)

    # ------------------------------------------------------------
    # Extract dataset and create patient list
    # ------------------------------------------------------------

    extract_dataset(CFG.tar_path, CFG.extract_path)

    patient_folders = get_patient_folders(CFG.extract_path)

    print("Number of patient folders:", len(patient_folders))

    # ------------------------------------------------------------
    # Create tumor-containing slice records
    # ------------------------------------------------------------

    slice_records = create_slice_records(patient_folders)

    print("Total tumor-containing slices:", len(slice_records))

    train_records, val_records = train_test_split(
        slice_records,
        test_size=0.2,
        random_state=CFG.seed,
        shuffle=True
    )

    random.seed(CFG.seed)

    fast_train_records = random.sample(
        train_records,
        min(CFG.fast_train_slices, len(train_records))
    )

    fast_val_records = random.sample(
        val_records,
        min(CFG.fast_val_slices, len(val_records))
    )

    print("Training slices:", len(fast_train_records))
    print("Validation slices:", len(fast_val_records))

    # ------------------------------------------------------------
    # Datasets and dataloaders
    # ------------------------------------------------------------

    train_dataset = BraTSSegmentationDataset(fast_train_records)
    val_dataset = BraTSSegmentationDataset(fast_val_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers
    )

    # ------------------------------------------------------------
    # Model, loss, optimizer
    # ------------------------------------------------------------

    model = EnhancedUNet(in_channels=4, out_channels=1).to(CFG.device)

    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)

    best_val_dice = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": []
    }

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------

    for epoch in range(CFG.epochs):
        print(f"\nEpoch {epoch + 1}/{CFG.epochs}")

        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc="Training", leave=False)

        for images_batch, masks_batch in train_bar:
            images_batch = images_batch.to(CFG.device)
            masks_batch = masks_batch.to(CFG.device)

            optimizer.zero_grad()

            outputs = model(images_batch)
            loss = criterion(outputs, masks_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_precision = 0.0
        val_recall = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation", leave=False)

            for images_batch, masks_batch in val_bar:
                images_batch = images_batch.to(CFG.device)
                masks_batch = masks_batch.to(CFG.device)

                outputs = model(images_batch)
                loss = criterion(outputs, masks_batch)

                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks_batch).item()
                val_iou += iou_score(outputs, masks_batch).item()
                val_precision += precision_score(outputs, masks_batch).item()
                val_recall += recall_score(outputs, masks_batch).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_precision = val_precision / len(val_loader)
        avg_val_recall = val_recall / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)
        history["val_iou"].append(avg_val_iou)
        history["val_precision"].append(avg_val_precision)
        history["val_recall"].append(avg_val_recall)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Val Dice:   {avg_val_dice:.4f}")
        print(f"Val IoU:    {avg_val_iou:.4f}")
        print(f"Precision:  {avg_val_precision:.4f}")
        print(f"Recall:     {avg_val_recall:.4f}")

        # --------------------------------------------------------
        # Save best model
        # --------------------------------------------------------

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_model_weights = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), CFG.model_save_path)

            print("Best model saved.")

    print("\nTraining complete.")
    print("Best validation Dice:", best_val_dice)

    return history


if __name__ == "__main__":
    train()