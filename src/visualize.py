# ================================================================
# Visualization utilities for BraTS 2021 segmentation
# ================================================================

import random

import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from config import CFG
from utils import seed_everything, extract_dataset, get_patient_folders
from dataset import BraTSSegmentationDataset, create_slice_records
from model import EnhancedUNet


def visualize_prediction():
    """
    Visualize MRI modalities, ground-truth mask, and predicted mask.
    """

    seed_everything(CFG.seed)

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

    model = EnhancedUNet(in_channels=4, out_channels=1).to(CFG.device)
    model.load_state_dict(torch.load(CFG.model_save_path, map_location=CFG.device))
    model.eval()

    sample_image, sample_mask = val_dataset[0]

    input_image = sample_image.unsqueeze(0).to(CFG.device)

    with torch.no_grad():
        logits = model(input_image)
        probability = torch.sigmoid(logits)
        predicted_mask = (probability > 0.5).float()

    sample_image_np = sample_image.cpu().numpy()
    sample_mask_np = sample_mask.squeeze().cpu().numpy()
    predicted_mask_np = predicted_mask.squeeze().cpu().numpy()

    plt.figure(figsize=(18, 5))

    titles = [
        "T1",
        "T1ce",
        "T2",
        "FLAIR",
        "Ground Truth",
        "Predicted Mask"
    ]

    images_to_show = [
        sample_image_np[0],
        sample_image_np[1],
        sample_image_np[2],
        sample_image_np[3],
        sample_mask_np,
        predicted_mask_np
    ]

    for i, (title, img) in enumerate(zip(titles, images_to_show)):
        plt.subplot(1, 6, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_prediction()