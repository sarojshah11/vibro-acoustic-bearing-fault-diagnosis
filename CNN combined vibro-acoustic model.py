# ================================================================
# COMBINED VIBRATION + ACOUSTIC CNN
# Bearing Fault Classification using Feature-Level Fusion
#
# Classes:
# H = Healthy
# I = Inner Race
# O = Outer Race
# B = Ball
# C = Cage
#
# Model:
# Vibration Spectrogram -> ResNet18 -> Feature Vector
# Acoustic Spectrogram  -> ResNet18 -> Feature Vector
#                              ↓
#                    Feature Concatenation
#                              ↓
#                       Fully Connected
#                              ↓
#                     5-Class Prediction
# ================================================================

import os
import re
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize


# ================================================================
# 1. SETTINGS
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("COMBINED VIBRATION + ACOUSTIC CNN")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ================================================================
# 2. DATASET PATHS
# ================================================================

VIBRATION_DIR = Path(
    r"C:\Users\saroj\OneDrive\Desktop\PYTHON\4_Spectrogram__accelerometer_datasets_21kHz_range"
)

ACOUSTIC_DIR = Path(
    r"C:\Users\saroj\OneDrive\Desktop\PYTHON\5_Spectrogram__acoustic_datasets_21kHz_range"
)


# ================================================================
# 3. OUTPUT DIRECTORY
# ================================================================

OUTPUT_DIR = Path("combined_vibration_acoustic_cnn_results")

FIGURE_DIR = OUTPUT_DIR / "Figures"
TABLE_DIR = OUTPUT_DIR / "Tables"
MODEL_DIR = OUTPUT_DIR / "Models"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("\nOutput directory:")
print(OUTPUT_DIR.resolve())


# ================================================================
# 4. CLASS INFORMATION
# ================================================================

CLASS_NAMES = ["H", "I", "O", "B", "C"]

CLASS_TO_INDEX = {
    "H": 0,
    "I": 1,
    "O": 2,
    "B": 3,
    "C": 4
}

NUM_CLASSES = len(CLASS_NAMES)


# ================================================================
# 5. CNN PARAMETERS
# ================================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 1e-4
BACKBONE_LR = 2e-5

WEIGHT_DECAY = 1e-4
DROPOUT = 0.30

LABEL_SMOOTHING = 0.05

PATIENCE = 8

NUM_WORKERS = 0


# ================================================================
# 6. FILE LABEL EXTRACTION
# ================================================================

# Expected naming pattern:
#
# H-1-0-001
# I-2-1-047
# O-7-2-323
#
# Also accepts "_" instead of "-":
#
# H_1_0_001
#
# The first three components are used to identify the recording/group.

LABEL_PATTERN = re.compile(
    r"([HIOBC])[-_](\d+)[-_]([012])",
    re.IGNORECASE
)


def extract_information(filepath):

    name = filepath.stem.upper()

    match = LABEL_PATTERN.search(name)

    if match is None:
        return None

    class_name = match.group(1)

    bearing_id = match.group(2)

    condition = match.group(3)

    if class_name not in CLASS_TO_INDEX:
        return None

    # Important:
    # All spectrograms from the same original recording
    # remain in the same train/validation/test split.

    group_id = f"{class_name}-{bearing_id}-{condition}"

    return {
        "class_name": class_name,
        "class_index": CLASS_TO_INDEX[class_name],
        "bearing_id": bearing_id,
        "condition": condition,
        "group_id": group_id
    }


# ================================================================
# 7. FIND IMAGES
# ================================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}


def collect_images(folder):

    files = []

    if not folder.exists():

        raise FileNotFoundError(
            f"\nFolder not found:\n{folder}\n"
        )

    for path in folder.rglob("*"):

        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:

            files.append(path)

    return sorted(files)


# ================================================================
# 8. BUILD VIBRATION DATAFRAME
# ================================================================

print("\nScanning vibration dataset...")

vibration_files = collect_images(VIBRATION_DIR)

print("Vibration images found:", len(vibration_files))

vibration_records = []

for path in vibration_files:

    info = extract_information(path)

    if info is None:
        continue

    vibration_records.append({
        "key": path.name.lower(),
        "vibration_path": str(path),
        **info
    })


vibration_df = pd.DataFrame(vibration_records)

print(
    "Vibration labelled:",
    len(vibration_df)
)


# ================================================================
# 9. BUILD ACOUSTIC DATAFRAME
# ================================================================

print("\nScanning acoustic dataset...")

acoustic_files = collect_images(ACOUSTIC_DIR)

print("Acoustic images found:", len(acoustic_files))

acoustic_records = []

for path in acoustic_files:

    info = extract_information(path)

    if info is None:
        continue

    acoustic_records.append({
        "key": path.name.lower(),
        "acoustic_path": str(path),
        **info
    })


acoustic_df = pd.DataFrame(acoustic_records)

print(
    "Acoustic labelled:",
    len(acoustic_df)
)


# ================================================================
# 10. MATCH VIBRATION AND ACOUSTIC DATA
# ================================================================

print("\nMatching vibration and acoustic images...")

combined_df = pd.merge(
    vibration_df,
    acoustic_df[
        [
            "key",
            "acoustic_path"
        ]
    ],
    on="key",
    how="inner"
)


# ------------------------------------------------
# Check that labels are consistent
# ------------------------------------------------

# Use vibration labels as the final labels.

print(
    "\nMatched vibration + acoustic pairs:",
    len(combined_df)
)


if len(combined_df) == 0:

    raise RuntimeError(
        "\nNo vibration/acoustic image pairs were found.\n"
        "Check that corresponding vibration and acoustic files "
        "have matching filenames."
    )


# ================================================================
# 11. CHECK CLASS DISTRIBUTION
# ================================================================

print("\nCombined class distribution:")

class_distribution = (
    combined_df
    .groupby("class_name")
    .size()
    .reindex(CLASS_NAMES, fill_value=0)
)

print(class_distribution)


# ================================================================
# 12. SAVE DATASET DISTRIBUTION TABLE
# ================================================================

table1 = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Class_Name": [
        "Healthy",
        "Inner Race",
        "Outer Race",
        "Ball",
        "Cage"
    ],
    "Combined_Pairs": [
        class_distribution[c]
        for c in CLASS_NAMES
    ]
})

table1["Percentage"] = (
    table1["Combined_Pairs"]
    / table1["Combined_Pairs"].sum()
    * 100
)

table1.to_csv(
    TABLE_DIR / "Table_1_Combined_Dataset_Distribution.csv",
    index=False
)


# ================================================================
# 13. GROUP-BASED DATA SPLIT
# ================================================================

# VERY IMPORTANT:
#
# Multiple spectrogram images may originate from the same recording.
#
# Therefore:
#
# Same group -> same train/validation/test split.
#
# This prevents data leakage.

groups = combined_df["group_id"].values


# ------------------------------------------------
# First split:
# 80% development
# 20% test
# ------------------------------------------------

gss_test = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

development_idx, test_idx = next(
    gss_test.split(
        combined_df,
        groups=groups
    )
)

development_df = combined_df.iloc[
    development_idx
].reset_index(drop=True)

test_df = combined_df.iloc[
    test_idx
].reset_index(drop=True)


# ------------------------------------------------
# Second split:
# 80% train
# 20% validation
# ------------------------------------------------

gss_val = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train_idx, val_idx = next(
    gss_val.split(
        development_df,
        groups=development_df["group_id"]
    )
)

train_df = development_df.iloc[
    train_idx
].reset_index(drop=True)

val_df = development_df.iloc[
    val_idx
].reset_index(drop=True)


# ================================================================
# 14. CHECK GROUP LEAKAGE
# ================================================================

train_groups = set(train_df["group_id"])
val_groups = set(val_df["group_id"])
test_groups = set(test_df["group_id"])

assert len(train_groups & val_groups) == 0
assert len(train_groups & test_groups) == 0
assert len(val_groups & test_groups) == 0

print("\nGroup leakage check: PASSED")

print("\nDataset split:")

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nGroups:")

print("Train groups:", len(train_groups))
print("Validation groups:", len(val_groups))
print("Test groups:", len(test_groups))


# ================================================================
# 15. SAVE SPLIT INFORMATION
# ================================================================

train_df.to_csv(
    TABLE_DIR / "train_split.csv",
    index=False
)

val_df.to_csv(
    TABLE_DIR / "validation_split.csv",
    index=False
)

test_df.to_csv(
    TABLE_DIR / "test_split.csv",
    index=False
)


# ================================================================
# 16. TABLE 2 — SPLIT DISTRIBUTION
# ================================================================

split_rows = []

for split_name, df in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    for class_name in CLASS_NAMES:

        class_df = df[
            df["class_name"] == class_name
        ]

        split_rows.append({

            "Split": split_name,

            "Class": class_name,

            "Groups": class_df["group_id"].nunique(),

            "Image_Pairs": len(class_df),

            "Percentage": (
                len(class_df)
                / len(df)
                * 100
                if len(df) > 0 else 0
            )
        })


table2 = pd.DataFrame(split_rows)

table2.to_csv(
    TABLE_DIR / "Table_2_Train_Validation_Test_Distribution.csv",
    index=False
)


# ================================================================
# 17. DATA AUGMENTATION
# ================================================================

train_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomAffine(
        degrees=5,
        translate=(0.02, 0.02),
        scale=(0.95, 1.05)
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================================================================
# 18. COMBINED DATASET
# ================================================================

class CombinedBearingDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        # ---------------------------------------------
        # Load vibration spectrogram
        # ---------------------------------------------

        vibration_image = Image.open(
            row["vibration_path"]
        ).convert("RGB")


        # ---------------------------------------------
        # Load acoustic spectrogram
        # ---------------------------------------------

        acoustic_image = Image.open(
            row["acoustic_path"]
        ).convert("RGB")


        # ---------------------------------------------
        # Apply transforms
        # ---------------------------------------------

        if self.transform is not None:

            vibration_image = self.transform(
                vibration_image
            )

            acoustic_image = self.transform(
                acoustic_image
            )


        # ---------------------------------------------
        # Label
        # ---------------------------------------------

        label = int(
            row["class_index"]
        )


        return (
            vibration_image,
            acoustic_image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ================================================================
# 19. CREATE DATASETS
# ================================================================

train_dataset = CombinedBearingDataset(
    train_df,
    train_transform
)

val_dataset = CombinedBearingDataset(
    val_df,
    test_transform
)

test_dataset = CombinedBearingDataset(
    test_df,
    test_transform
)


# ================================================================
# 20. CREATE DATALOADERS
# ================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ================================================================
# 21. COMBINED CNN MODEL
# ================================================================

class CombinedResNet18(nn.Module):

    def __init__(
        self,
        num_classes=5
    ):

        super().__init__()


        # ========================================================
        # VIBRATION CNN
        # ========================================================

        vibration_model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        vibration_features = vibration_model.fc.in_features

        vibration_model.fc = nn.Identity()

        self.vibration_cnn = vibration_model


        # ========================================================
        # ACOUSTIC CNN
        # ========================================================

        acoustic_model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        acoustic_features = acoustic_model.fc.in_features

        acoustic_model.fc = nn.Identity()

        self.acoustic_cnn = acoustic_model


        # ========================================================
        # FEATURE FUSION
        # ========================================================

        combined_features = (
            vibration_features
            +
            acoustic_features
        )


        # ========================================================
        # CLASSIFIER
        # ========================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                combined_features,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(
        self,
        vibration,
        acoustic
    ):

        # Vibration feature vector
        vibration_features = self.vibration_cnn(
            vibration
        )


        # Acoustic feature vector
        acoustic_features = self.acoustic_cnn(
            acoustic
        )


        # --------------------------------------------------------
        # Feature-level fusion
        # --------------------------------------------------------

        combined = torch.cat(
            [
                vibration_features,
                acoustic_features
            ],
            dim=1
        )


        # --------------------------------------------------------
        # Final classification
        # --------------------------------------------------------

        output = self.classifier(
            combined
        )


        return output


# ================================================================
# 22. CREATE MODEL
# ================================================================

model = CombinedResNet18(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)

print("\nModel:")
print(model)


# ================================================================
# 23. MODEL PARAMETER COUNT
# ================================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nTotal parameters:", total_parameters)

print(
    "Trainable parameters:",
    trainable_parameters
)


# ================================================================
# 24. TABLE 3 — HYPERPARAMETERS
# ================================================================

table3 = pd.DataFrame({

    "Parameter": [

        "Model",

        "Fusion method",

        "CNN backbone",

        "Pretrained weights",

        "Number of classes",

        "Input image size",

        "Batch size",

        "Maximum epochs",

        "Classifier learning rate",

        "Backbone learning rate",

        "Optimizer",

        "Weight decay",

        "Dropout",

        "Label smoothing",

        "Scheduler",

        "Early stopping patience",

        "Random seed",

        "Total parameters",

        "Trainable parameters"
    ],

    "Value": [

        "Combined ResNet-18",

        "Feature-level fusion",

        "ResNet-18 × 2",

        "ImageNet",

        NUM_CLASSES,

        f"{IMAGE_SIZE} × {IMAGE_SIZE}",

        BATCH_SIZE,

        EPOCHS,

        LEARNING_RATE,

        BACKBONE_LR,

        "AdamW",

        WEIGHT_DECAY,

        DROPOUT,

        LABEL_SMOOTHING,

        "CosineAnnealingLR",

        PATIENCE,

        SEED,

        total_parameters,

        trainable_parameters
    ]
})

table3.to_csv(
    TABLE_DIR / "Table_3_CNN_Hyperparameters.csv",
    index=False
)


# ================================================================
# 25. CLASS WEIGHTS
# ================================================================

class_counts = (
    train_df["class_index"]
    .value_counts()
    .sort_index()
)

# Mild sqrt balancing

weights = (
    class_counts.sum()
    /
    class_counts
)

weights = np.sqrt(weights)

weights = weights / weights.mean()

class_weights = torch.tensor(
    weights.values,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        class_name,
        ":",
        float(class_weights[i])
    )


# ================================================================
# 26. LOSS FUNCTION
# ================================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=LABEL_SMOOTHING
)


# ================================================================
# 27. OPTIMIZER
# ================================================================

optimizer = torch.optim.AdamW(

    [

        {
            "params":
            model.vibration_cnn.parameters(),

            "lr":
            BACKBONE_LR
        },

        {
            "params":
            model.acoustic_cnn.parameters(),

            "lr":
            BACKBONE_LR
        },

        {
            "params":
            model.classifier.parameters(),

            "lr":
            LEARNING_RATE
        }
    ],

    weight_decay=WEIGHT_DECAY
)


# ================================================================
# 28. LR SCHEDULER
# ================================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ================================================================
# 29. TRAINING FUNCTION
# ================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    running_loss = 0.0

    all_targets = []

    all_predictions = []


    for vibration, acoustic, targets in loader:

        vibration = vibration.to(
            DEVICE,
            non_blocking=True
        )

        acoustic = acoustic.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad()


        outputs = model(
            vibration,
            acoustic
        )


        loss = criterion(
            outputs,
            targets
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * targets.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_targets.extend(
            targets.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


    epoch_loss = (
        running_loss
        /
        len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )


    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ================================================================
# 30. VALIDATION FUNCTION
# ================================================================

def evaluate(
    model,
    loader,
    criterion,
    return_probabilities=False
):

    model.eval()

    running_loss = 0.0

    all_targets = []

    all_predictions = []

    all_probabilities = []


    with torch.no_grad():

        for vibration, acoustic, targets in loader:

            vibration = vibration.to(
                DEVICE,
                non_blocking=True
            )

            acoustic = acoustic.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )


            outputs = model(
                vibration,
                acoustic
            )


            loss = criterion(
                outputs,
                targets
            )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            predictions = torch.argmax(
                probabilities,
                dim=1
            )


            running_loss += (
                loss.item()
                * targets.size(0)
            )


            all_targets.extend(
                targets.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )


    epoch_loss = (
        running_loss
        /
        len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )


    if return_probabilities:

        return (
            epoch_loss,
            accuracy,
            precision,
            recall,
            f1,
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_probabilities)
        )


    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ================================================================
# 31. TRAINING LOOP
# ================================================================

history = []

best_val_f1 = -np.inf

best_epoch = 0

patience_counter = 0

best_model_path = (
    MODEL_DIR /
    "combined_vibration_acoustic_resnet18_best.pt"
)


print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(1, EPOCHS + 1):

    start_time = time.time()


    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    train_loss, train_acc, train_precision, train_recall, train_f1 = (
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )
    )


    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    (
        val_loss,
        val_acc,
        val_precision,
        val_recall,
        val_f1
    ) = evaluate(
        model,
        val_loader,
        criterion
    )


    scheduler.step()


    epoch_time = time.time() - start_time


    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "train_accuracy": train_acc,

        "train_macro_precision": train_precision,

        "train_macro_recall": train_recall,

        "train_macro_f1": train_f1,

        "val_loss": val_loss,

        "val_accuracy": val_acc,

        "val_macro_precision": val_precision,

        "val_macro_recall": val_recall,

        "val_macro_f1": val_f1,

        "learning_rate": optimizer.param_groups[-1]["lr"],

        "epoch_time_seconds": epoch_time
    })


    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train F1: {train_f1:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )


    # ------------------------------------------------------------
    # SAVE BEST MODEL
    # ------------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_epoch = epoch

        patience_counter = 0

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                model.state_dict(),

                "optimizer_state_dict":
                optimizer.state_dict(),

                "best_val_f1":
                best_val_f1,

                "class_names":
                CLASS_NAMES
            },

            best_model_path
        )

        print(
            ">>> Best model saved."
        )

    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/{PATIENCE})"
        )


    # ------------------------------------------------------------
    # EARLY STOPPING
    # ------------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ================================================================
# 32. SAVE TRAINING HISTORY
# ================================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    TABLE_DIR /
    "training_history.csv",
    index=False
)


# ================================================================
# 33. LOAD BEST MODEL
# ================================================================

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    "\nBest epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation Macro-F1:",
    checkpoint["best_val_f1"]
)


# ================================================================
# 34. TEST EVALUATION
# ================================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


start_time = time.time()


(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    y_true,
    y_pred,
    y_prob
) = evaluate(
    model,
    test_loader,
    criterion,
    return_probabilities=True
)


total_test_time = time.time() - start_time

latency_ms_per_image = (
    total_test_time
    /
    len(test_df)
    * 1000
)


print(
    f"\nTest Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Macro Precision: {test_precision:.4f}"
)

print(
    f"Macro Recall: {test_recall:.4f}"
)

print(
    f"Macro F1: {test_f1:.4f}"
)

print(
    f"Average inference time/image: "
    f"{latency_ms_per_image:.3f} ms"
)


# ================================================================
# 35. CLASSIFICATION REPORT
# ================================================================

report_dict = classification_report(

    y_true,

    y_pred,

    target_names=CLASS_NAMES,

    output_dict=True,

    zero_division=0
)


classification_df = pd.DataFrame(
    report_dict
).T


classification_df.to_csv(
    TABLE_DIR /
    "Table_4_Classification_Report.csv"
)


print("\nClassification Report:")
print(
    classification_df
)


# ================================================================
# 36. TABLE 5 — OVERALL PERFORMANCE
# ================================================================

table5 = pd.DataFrame({

    "Metric": [

        "Test Loss",

        "Accuracy",

        "Macro Precision",

        "Macro Recall",

        "Macro F1",

        "Test Images",

        "Test Groups",

        "Best Epoch",

        "Average Inference Time (ms/image)"
    ],

    "Combined_Vibration_Acoustic_CNN": [

        test_loss,

        test_accuracy,

        test_precision,

        test_recall,

        test_f1,

        len(test_df),

        len(test_groups),

        checkpoint["epoch"],

        latency_ms_per_image
    ]
})


table5.to_csv(
    TABLE_DIR /
    "Table_5_Overall_Test_Performance.csv",
    index=False
)


# ================================================================
# 37. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(NUM_CLASSES)
)


cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)


cm_df.to_csv(
    TABLE_DIR /
    "confusion_matrix.csv"
)


# ------------------------------------------------
# Normalized confusion matrix
# ------------------------------------------------

cm_normalized = (
    cm.astype(float)
    /
    cm.sum(
        axis=1,
        keepdims=True
    )
)


cm_normalized_df = pd.DataFrame(
    cm_normalized,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)


cm_normalized_df.to_csv(
    TABLE_DIR /
    "normalized_confusion_matrix.csv"
)


# ================================================================
# 38. FIGURE 3 — ACCURACY
# ================================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    history_df["epoch"],
    history_df["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Combined CNN - Training and Validation Accuracy"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "Figure_3_Combined_CNN_Accuracy.png",
    dpi=300
)

plt.close()


# ================================================================
# 39. FIGURE 4 — LOSS
# ================================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Combined CNN - Training and Validation Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "Figure_4_Combined_CNN_Loss.png",
    dpi=300
)

plt.close()


# ================================================================
# 40. FIGURE 5 — CONFUSION MATRIX
# ================================================================

plt.figure(
    figsize=(8, 7)
)

sns.heatmap(
    cm_normalized_df,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.title(
    "Combined CNN - Normalized Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "Figure_5_Combined_CNN_Confusion_Matrix.png",
    dpi=300
)

plt.close()


# ================================================================
# 41. FIGURE 6 — ROC CURVES
# ================================================================

y_true_binary = label_binarize(
    y_true,
    classes=range(NUM_CLASSES)
)


plt.figure(
    figsize=(9, 7)
)


for i, class_name in enumerate(CLASS_NAMES):

    fpr, tpr, _ = roc_curve(
        y_true_binary[:, i],
        y_prob[:, i]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} (AUC={roc_auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Combined CNN - Multiclass ROC Curves"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "Figure_6_Combined_CNN_ROC_Curves.png",
    dpi=300
)

plt.close()


# ================================================================
# 42. FIGURE 7 — PRECISION-RECALL CURVES
# ================================================================

plt.figure(
    figsize=(9, 7)
)


for i, class_name in enumerate(CLASS_NAMES):

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_true_binary[:, i],
            y_prob[:, i]
        )
    )

    ap = average_precision_score(
        y_true_binary[:, i],
        y_prob[:, i]
    )

    plt.plot(
        recall_curve,
        precision_curve,
        label=f"{class_name} (AP={ap:.3f})"
    )


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Combined CNN - Multiclass Precision-Recall Curves"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "Figure_7_Combined_CNN_Precision_Recall_Curves.png",
    dpi=300
)

plt.close()


# ================================================================
# 43. SAVE FINAL SUMMARY
# ================================================================

summary = {

    "Model":
    "Combined ResNet-18 Vibration + Acoustic",

    "Fusion":
    "Feature-level fusion",

    "Classes":
    ", ".join(CLASS_NAMES),

    "Train Images":
    len(train_df),

    "Validation Images":
    len(val_df),

    "Test Images":
    len(test_df),

    "Train Groups":
    len(train_groups),

    "Validation Groups":
    len(val_groups),

    "Test Groups":
    len(test_groups),

    "Best Epoch":
    checkpoint["epoch"],

    "Best Validation Macro F1":
    checkpoint["best_val_f1"],

    "Test Accuracy":
    test_accuracy,

    "Test Macro Precision":
    test_precision,

    "Test Macro Recall":
    test_recall,

    "Test Macro F1":
    test_f1
}


summary_df = pd.DataFrame(
    [summary]
)


summary_df.to_csv(
    OUTPUT_DIR /
    "combined_cnn_final_summary.csv",
    index=False
)


# ================================================================
# 44. FINISHED
# ================================================================

print("\n")
print("=" * 70)
print("TRAINING AND EVALUATION COMPLETED")
print("=" * 70)

print("\nResults saved in:")

print(
    OUTPUT_DIR.resolve()
)

print("\nFigures:")

for file in sorted(FIGURE_DIR.glob("*.png")):

    print(
        " ",
        file.name
    )


print("\nTables:")

for file in sorted(TABLE_DIR.glob("*.csv")):

    print(
        " ",
        file.name
    )


print("\nBest model:")

print(
    best_model_path
)

print("\nDone.")