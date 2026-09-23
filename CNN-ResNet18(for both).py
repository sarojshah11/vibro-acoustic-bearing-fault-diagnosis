# ============================================================
# IMPROVED CNN BASELINE
# Bearing Fault Classification from Spectrograms
#
# Classes:
# H, I, O, B, C
#
# Modalities:
# Vibration
# Acoustic
#
# CNN:
# Pretrained ResNet-18
#
# Automatically generates:
#
# TABLES
# Table 1: Dataset / class distribution
# Table 2: Train / validation / test group distribution
# Table 3: CNN hyperparameters
# Table 4: Classification report
# Table 5: Overall test performance
#
# FIGURES
# Figure 3: Training vs validation accuracy
# Figure 4: Training vs validation loss
# Figure 5: Confusion matrix
# Figure 6: ROC curves
# Figure 7: Precision-Recall curves
#
# ============================================================

import re
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


# ============================================================
# 1. CONFIGURATION
# ============================================================

VIBRATION_DIR = Path(
    r"C:\Users\saroj\OneDrive\Desktop\PYTHON\4_Spectrogram__accelerometer_datasets_21kHz_range"
)

ACOUSTIC_DIR = Path(
    r"C:\Users\saroj\OneDrive\Desktop\PYTHON\5_Spectrogram__acoustic_datasets_21kHz_range"
)

# Main result folder
OUTPUT_DIR = Path("bearing_cnn_results")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMAGE / TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 30

# Learning rate for new classifier
LEARNING_RATE = 1e-4

# Learning rate for pretrained ResNet backbone
BACKBONE_LR = 2e-5

WEIGHT_DECAY = 1e-4

DROPOUT = 0.30

LABEL_SMOOTHING = 0.05

PATIENCE = 8

NUM_WORKERS = 0

SEED = 42


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = [
    "H",
    "I",
    "O",
    "B",
    "C"
]

LABEL_TO_INDEX = {
    label: i
    for i, label in enumerate(CLASS_NAMES)
}

INDEX_TO_LABEL = {
    i: label
    for label, i in LABEL_TO_INDEX.items()
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("DEVICE:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("=" * 70)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 3. LABEL EXTRACTION
# ============================================================

LABEL_PATTERN = re.compile(
    r"([HIOBC])[-_](\d+)[-_]([012])",
    re.IGNORECASE
)


def extract_label_and_group(
    image_path,
    root_dir
):

    parts = [
        image_path.stem
    ]

    for parent in image_path.parents:

        parts.append(
            parent.name
        )

        if parent == root_dir:
            break

    search_text = " ".join(parts)

    match = LABEL_PATTERN.search(
        search_text
    )

    if match:

        label = (
            match.group(1)
            .upper()
        )

        bearing_number = (
            match.group(2)
        )

        condition_number = (
            match.group(3)
        )

        group_id = (
            f"{label}_"
            f"{bearing_number}_"
            f"{condition_number}"
        )

        return label, group_id


    # --------------------------------------------------------
    # Fallback for folders named H, I, O, B, C
    # --------------------------------------------------------

    for parent in image_path.parents:

        folder_label = (
            parent.name
            .strip()
            .upper()
        )

        if folder_label in CLASS_NAMES:

            relative_parent = (
                image_path
                .relative_to(root_dir)
                .parent
            )

            group_id = str(
                relative_parent
            )

            return (
                folder_label,
                group_id
            )

        if parent == root_dir:
            break


    return None, None


# ============================================================
# 4. BUILD METADATA
# ============================================================

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}


def build_metadata(
    root_dir,
    modality_name,
    modality_output
):

    if not root_dir.exists():

        raise FileNotFoundError(
            f"\nDataset folder not found:\n"
            f"{root_dir}"
        )


    image_paths = [
        p
        for p in root_dir.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower()
            in VALID_EXTENSIONS
        )
    ]


    print(
        f"\n{modality_name}: "
        f"Found {len(image_paths)} image files"
    )


    records = []

    skipped = []


    for image_path in image_paths:

        label, group_id = (
            extract_label_and_group(
                image_path,
                root_dir
            )
        )


        if label is None:

            skipped.append(
                str(image_path)
            )

            continue


        records.append({

            "image_path":
                str(image_path),

            "label":
                label,

            "target":
                LABEL_TO_INDEX[label],

            "group_id":
                group_id,

            "modality":
                modality_name

        })


    metadata = pd.DataFrame(
        records
    )


    if metadata.empty:

        raise ValueError(
            "No labels were recognized."
        )


    print(
        f"Labelled images: "
        f"{len(metadata)}"
    )

    print(
        f"Skipped images: "
        f"{len(skipped)}"
    )


    print("\nClass distribution:")

    class_counts = (
        metadata["label"]
        .value_counts()
        .reindex(
            CLASS_NAMES,
            fill_value=0
        )
    )

    print(class_counts)


    print("\nDetected class mapping:")

    for label in CLASS_NAMES:

        print(
            f"{label} -> "
            f"{LABEL_TO_INDEX[label]}"
        )


    print(
        "\nNumber of independent groups:"
    )

    group_counts = (
        metadata
        .groupby("label")
        ["group_id"]
        .nunique()
        .reindex(
            CLASS_NAMES,
            fill_value=0
        )
    )

    print(group_counts)


    # --------------------------------------------------------
    # Save complete metadata
    # --------------------------------------------------------

    metadata.to_csv(
        modality_output /
        "metadata.csv",
        index=False
    )


    return metadata


# ============================================================
# 5. TABLE 1
# DATASET / CLASS DISTRIBUTION
# ============================================================

def create_table_1(
    metadata,
    modality_output
):

    table = []

    total_images = len(metadata)

    for label in CLASS_NAMES:

        count = int(
            (
                metadata["label"]
                == label
            ).sum()
        )

        percentage = (
            count /
            total_images *
            100
        )

        groups = (
            metadata[
                metadata["label"]
                == label
            ]["group_id"]
            .nunique()
        )

        table.append({

            "Class": label,

            "Number_of_Images":
                count,

            "Percentage":
                round(
                    percentage,
                    2
                ),

            "Independent_Groups":
                groups

        })


    # Total row
    table.append({

        "Class": "Total",

        "Number_of_Images":
            total_images,

        "Percentage":
            100.00,

        "Independent_Groups":
            metadata[
                "group_id"
            ].nunique()

    })


    df = pd.DataFrame(table)


    df.to_csv(
        modality_output /
        "Tables" /
        "Table_1_Dataset_Class_Distribution.csv",
        index=False
    )


    return df


# ============================================================
# 6. GROUP-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_by_group(
    metadata,
    seed=42
):

    metadata = (
        metadata
        .reset_index(drop=True)
    )


    group_table = (
        metadata[
            [
                "group_id",
                "label"
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


    groups_per_class = (
        group_table
        .groupby("label")
        ["group_id"]
        .nunique()
        .reindex(
            CLASS_NAMES,
            fill_value=0
        )
    )


    print(
        "\nIndependent groups per class:"
    )

    print(
        groups_per_class
    )


    if not all(
        groups_per_class >= 3
    ):

        raise ValueError(
            "\nSome classes have fewer "
            "than 3 independent groups.\n"
        )


    rng = np.random.RandomState(
        seed
    )


    train_groups = []

    val_groups = []

    test_groups = []


    # --------------------------------------------------------
    # Split separately for each class
    # --------------------------------------------------------

    for label in CLASS_NAMES:

        class_groups = (
            group_table.loc[
                group_table["label"]
                == label,
                "group_id"
            ]
            .astype(str)
            .to_numpy()
        )


        rng.shuffle(
            class_groups
        )


        n_groups = len(
            class_groups
        )


        n_test = max(
            1,
            round(
                n_groups * 0.20
            )
        )


        n_val = max(
            1,
            round(
                n_groups * 0.16
            )
        )


        while (
            n_test +
            n_val >=
            n_groups
        ):

            if n_val > 1:

                n_val -= 1

            elif n_test > 1:

                n_test -= 1

            else:

                break


        test_groups.extend(
            class_groups[
                :n_test
            ]
        )


        val_groups.extend(
            class_groups[
                n_test:
                n_test + n_val
            ]
        )


        train_groups.extend(
            class_groups[
                n_test + n_val:
            ]
        )


    train_df = metadata[
        metadata["group_id"]
        .astype(str)
        .isin(train_groups)
    ].copy()


    val_df = metadata[
        metadata["group_id"]
        .astype(str)
        .isin(val_groups)
    ].copy()


    test_df = metadata[
        metadata["group_id"]
        .astype(str)
        .isin(test_groups)
    ].copy()


    train_df.reset_index(
        drop=True,
        inplace=True
    )

    val_df.reset_index(
        drop=True,
        inplace=True
    )

    test_df.reset_index(
        drop=True,
        inplace=True
    )


    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    train_set = set(
        train_df["group_id"]
    )

    val_set = set(
        val_df["group_id"]
    )

    test_set = set(
        test_df["group_id"]
    )


    assert train_set.isdisjoint(
        val_set
    )

    assert train_set.isdisjoint(
        test_set
    )

    assert val_set.isdisjoint(
        test_set
    )


    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# 7. TABLE 2
# TRAIN / VALIDATION / TEST GROUP DISTRIBUTION
# ============================================================

def create_table_2(
    train_df,
    val_df,
    test_df,
    modality_output
):

    rows = []


    for split_name, df in [

        ("Train", train_df),

        ("Validation", val_df),

        ("Test", test_df)

    ]:

        for label in CLASS_NAMES:

            class_df = df[
                df["label"]
                == label
            ]


            rows.append({

                "Split":
                    split_name,

                "Class":
                    label,

                "Images":
                    len(class_df),

                "Groups":
                    class_df[
                        "group_id"
                    ].nunique()

            })


        # Total for split

        rows.append({

            "Split":
                split_name,

            "Class":
                "Total",

            "Images":
                len(df),

            "Groups":
                df[
                    "group_id"
                ].nunique()

        })


    df_table = pd.DataFrame(
        rows
    )


    df_table.to_csv(
        modality_output /
        "Tables" /
        "Table_2_Train_Validation_Test_Group_Distribution.csv",
        index=False
    )


    return df_table


# ============================================================
# 8. SPECTROGRAM TRANSFORMS
# ============================================================

MEAN = (
    0.485,
    0.456,
    0.406
)

STD = (
    0.229,
    0.224,
    0.225
)


train_transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.RandomAffine(
        degrees=5,
        translate=(
            0.02,
            0.02
        ),
        scale=(
            0.97,
            1.03
        )
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        MEAN,
        STD
    )

])


eval_transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        MEAN,
        STD
    )

])


# ============================================================
# 9. PYTORCH DATASET
# ============================================================

class BearingImageDataset(
    Dataset
):

    def __init__(
        self,
        metadata,
        transform=None
    ):

        self.metadata = (
            metadata
            .reset_index(drop=True)
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.metadata
        )


    def __getitem__(
        self,
        index
    ):

        row = self.metadata.iloc[
            index
        ]


        try:

            with Image.open(
                row["image_path"]
            ) as image_file:

                image = (
                    image_file
                    .convert("RGB")
                )


        except Exception as e:

            raise RuntimeError(

                f"Could not read image:\n"
                f"{row['image_path']}\n"
                f"Error: {e}"

            )


        if self.transform:

            image = self.transform(
                image
            )


        label = int(
            row["target"]
        )


        return image, label


# ============================================================
# 10. RESNET-18 CNN
# ============================================================

class CNNBaseline(
    nn.Module
):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()


        weights = (
            models.ResNet18_Weights.DEFAULT
        )


        self.backbone = (
            models.resnet18(
                weights=weights
            )
        )


        num_features = (
            self.backbone.fc.in_features
        )


        self.backbone.fc = nn.Sequential(

            nn.Dropout(
                p=DROPOUT
            ),

            nn.Linear(
                num_features,
                num_classes
            )

        )


    def forward(
        self,
        x
    ):

        return self.backbone(x)


# ============================================================
# 11. CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    train_df
):

    counts = (
        train_df["target"]
        .value_counts()
        .reindex(
            range(
                len(CLASS_NAMES)
            ),
            fill_value=0
        )
        .values
        .astype(np.float32)
    )


    weights = np.sqrt(

        counts.sum()
        /
        (
            len(CLASS_NAMES)
            *
            counts
        )

    )


    weights = (
        weights /
        weights.mean()
    )


    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# 12. METRICS
# ============================================================

def calculate_metrics(
    targets,
    predictions
):

    precision, recall, f1, _ = (

        precision_recall_fscore_support(

            targets,

            predictions,

            labels=list(
                range(
                    len(CLASS_NAMES)
                )
            ),

            average="macro",

            zero_division=0

        )

    )


    return {

        "accuracy":
            accuracy_score(
                targets,
                predictions
            ),

        "macro_precision":
            precision,

        "macro_recall":
            recall,

        "macro_f1":
            f1

    }


# ============================================================
# 13. TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None
):

    training = (
        optimizer is not None
    )


    if training:

        model.train()

    else:

        model.eval()


    total_loss = 0.0

    all_targets = []

    all_predictions = []


    for images, targets in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )


        if training:

            optimizer.zero_grad(
                set_to_none=True
            )


        with torch.set_grad_enabled(
            training
        ):

            logits = model(
                images
            )


            loss = criterion(
                logits,
                targets
            )


            if training:

                loss.backward()


                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    max_norm=1.0

                )


                optimizer.step()


        total_loss += (

            loss.item()
            *
            images.size(0)

        )


        predictions = (
            logits
            .argmax(dim=1)
        )


        all_targets.extend(

            targets
            .detach()
            .cpu()
            .numpy()

        )


        all_predictions.extend(

            predictions
            .detach()
            .cpu()
            .numpy()

        )


    average_loss = (

        total_loss
        /
        len(loader.dataset)

    )


    metrics = calculate_metrics(

        all_targets,

        all_predictions

    )


    metrics["loss"] = (
        average_loss
    )


    return metrics


# ============================================================
# 14. TABLE 3
# CNN HYPERPARAMETERS
# ============================================================

def create_table_3(
    modality_output,
    total_params,
    trainable_params
):

    parameters = [

        ("Model", "Pretrained ResNet-18"),

        ("Architecture type", "CNN"),

        ("Input image size",
         f"{IMAGE_SIZE} × {IMAGE_SIZE}"),

        ("Input channels", "3 RGB"),

        ("Number of classes",
         len(CLASS_NAMES)),

        ("Classes",
         ", ".join(CLASS_NAMES)),

        ("Batch size",
         BATCH_SIZE),

        ("Maximum epochs",
         EPOCHS),

        ("Classifier learning rate",
         LEARNING_RATE),

        ("Backbone learning rate",
         BACKBONE_LR),

        ("Optimizer",
         "AdamW"),

        ("Weight decay",
         WEIGHT_DECAY),

        ("Dropout",
         DROPOUT),

        ("Label smoothing",
         LABEL_SMOOTHING),

        ("Scheduler",
         "CosineAnnealingLR"),

        ("Minimum learning rate",
         "1e-6"),

        ("Early stopping patience",
         PATIENCE),

        ("Random seed",
         SEED),

        ("Data split",
         "Group-based"),

        ("Data augmentation",
         "RandomAffine + ColorJitter"),

        ("Loss function",
         "Weighted CrossEntropyLoss"),

        ("Total parameters",
         total_params),

        ("Trainable parameters",
         trainable_params)

    ]


    table = pd.DataFrame(

        parameters,

        columns=[
            "Parameter",
            "Value"
        ]

    )


    table.to_csv(

        modality_output /
        "Tables" /
        "Table_3_CNN_Hyperparameters.csv",

        index=False

    )


    return table


# ============================================================
# 15. TRAINING
# ============================================================

def train_model(

    model,

    train_loader,

    val_loader,

    train_df,

    modality_name,

    modality_output

):

    model = model.to(
        DEVICE
    )


    # --------------------------------------------------------
    # Class-weighted loss
    # --------------------------------------------------------

    class_weights = (

        calculate_class_weights(
            train_df
        )
        .to(DEVICE)

    )


    print(
        "\nClass weights:"
    )


    for i, name in enumerate(
        CLASS_NAMES
    ):

        print(

            f"{name}: "
            f"{class_weights[i].item():.4f}"

        )


    criterion = (

        nn.CrossEntropyLoss(

            weight=class_weights,

            label_smoothing=
                LABEL_SMOOTHING

        )

    )


    # --------------------------------------------------------
    # Differential learning rates
    # --------------------------------------------------------

    backbone_params = []

    classifier_params = []


    for name, param in (

        model.named_parameters()

    ):

        if "backbone.fc" in name:

            classifier_params.append(
                param
            )

        else:

            backbone_params.append(
                param
            )


    optimizer = torch.optim.AdamW(

        [

            {
                "params":
                    backbone_params,

                "lr":
                    BACKBONE_LR

            },

            {
                "params":
                    classifier_params,

                "lr":
                    LEARNING_RATE

            }

        ],

        weight_decay=
            WEIGHT_DECAY

    )


    scheduler = (

        torch.optim.lr_scheduler.CosineAnnealingLR(

            optimizer,

            T_max=EPOCHS,

            eta_min=1e-6

        )

    )


    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    best_val_f1 = -1.0

    best_state = copy.deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0

    history = []


    print("\n" + "=" * 70)

    print(
        f"STARTING "
        f"{modality_name.upper()} CNN TRAINING"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):


        train_metrics = run_epoch(

            model,

            train_loader,

            criterion,

            optimizer

        )


        val_metrics = run_epoch(

            model,

            val_loader,

            criterion

        )


        scheduler.step()


        # ----------------------------------------------------
        # Learning rates
        # ----------------------------------------------------

        backbone_lr_current = (

            optimizer.param_groups[0]["lr"]

        )

        classifier_lr_current = (

            optimizer.param_groups[1]["lr"]

        )


        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append({

            "epoch":
                epoch,

            "train_loss":
                train_metrics["loss"],

            "train_accuracy":
                train_metrics["accuracy"],

            "train_macro_precision":
                train_metrics["macro_precision"],

            "train_macro_recall":
                train_metrics["macro_recall"],

            "train_macro_f1":
                train_metrics["macro_f1"],

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_macro_precision":
                val_metrics["macro_precision"],

            "val_macro_recall":
                val_metrics["macro_recall"],

            "val_macro_f1":
                val_metrics["macro_f1"],

            "backbone_learning_rate":
                backbone_lr_current,

            "classifier_learning_rate":
                classifier_lr_current

        })


        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(

            f"{modality_name} | "

            f"Epoch "
            f"{epoch:02d}/{EPOCHS} | "

            f"Train Loss "
            f"{train_metrics['loss']:.4f} | "

            f"Train Acc "
            f"{train_metrics['accuracy']:.4f} | "

            f"Val Loss "
            f"{val_metrics['loss']:.4f} | "

            f"Val Acc "
            f"{val_metrics['accuracy']:.4f} | "

            f"Val Macro-F1 "
            f"{val_metrics['macro_f1']:.4f} | "

            f"LR "
            f"{classifier_lr_current:.2e}"

        )


        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if (

            val_metrics["macro_f1"]
            >
            best_val_f1

        ):

            best_val_f1 = (

                val_metrics["macro_f1"]

            )


            best_state = copy.deepcopy(

                model.state_dict()

            )


            epochs_without_improvement = 0


            print(

                f"  ✓ BEST MODEL "
                f"(Macro-F1 = "
                f"{best_val_f1:.4f})"

            )


        else:

            epochs_without_improvement += 1


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (

            epochs_without_improvement
            >= PATIENCE

        ):

            print(
                "\nEarly stopping."
            )

            break


    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (

        modality_output
        /
        f"{modality_name.lower()}_cnn_best.pt"

    )


    torch.save(

        {

            "model_state_dict":
                model.state_dict(),

            "class_names":
                CLASS_NAMES,

            "class_to_idx":
                LABEL_TO_INDEX,

            "best_val_macro_f1":
                best_val_f1

        },

        model_path

    )


    print(
        f"\nBest model saved to:"
        f"\n{model_path}"
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(

        modality_output
        /
        "training_history.csv",

        index=False

    )


    return (
        model,
        history_df,
        best_val_f1
    )


# ============================================================
# 16. FIGURE 3
# TRAINING vs VALIDATION ACCURACY
# ============================================================

def plot_accuracy(
    history,
    modality_name,
    figures_dir
):

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(

        history["epoch"],

        history["train_accuracy"],

        label="Training Accuracy",

        linewidth=2

    )


    plt.plot(

        history["epoch"],

        history["val_accuracy"],

        label="Validation Accuracy",

        linewidth=2

    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        f"{modality_name} CNN: "
        f"Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(

        figures_dir /
        "Figure_3_Training_Validation_Accuracy.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


# ============================================================
# 17. FIGURE 4
# TRAINING vs VALIDATION LOSS
# ============================================================

def plot_loss(
    history,
    modality_name,
    figures_dir
):

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(

        history["epoch"],

        history["train_loss"],

        label="Training Loss",

        linewidth=2

    )


    plt.plot(

        history["epoch"],

        history["val_loss"],

        label="Validation Loss",

        linewidth=2

    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        f"{modality_name} CNN: "
        f"Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(

        figures_dir /
        "Figure_4_Training_Validation_Loss.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


# ============================================================
# 18. TEST EVALUATION
# ============================================================

@torch.no_grad()

def evaluate_model(

    model,

    test_loader,

    test_df,

    modality_name,

    modality_output,

    figures_dir

):

    model.eval()


    all_targets = []

    all_predictions = []

    all_probabilities = []


    start_time = (
        time.perf_counter()
    )


    for images, targets in (
        test_loader
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )


        logits = model(
            images
        )


        probabilities = torch.softmax(

            logits,

            dim=1

        )


        predictions = (

            probabilities
            .argmax(dim=1)

        )


        all_targets.extend(

            targets.numpy()

        )


        all_predictions.extend(

            predictions
            .cpu()
            .numpy()

        )


        all_probabilities.extend(

            probabilities
            .cpu()
            .numpy()

        )


    elapsed = (

        time.perf_counter()
        -
        start_time

    )


    all_targets = np.array(
        all_targets
    )


    all_predictions = np.array(
        all_predictions
    )


    all_probabilities = np.array(
        all_probabilities
    )


    latency_ms = (

        1000.0
        *
        elapsed
        /
        max(
            len(test_df),
            1
        )

    )


    # ========================================================
    # Overall metrics
    # ========================================================

    metrics = calculate_metrics(

        all_targets,

        all_predictions

    )


    print("\n")

    print("=" * 70)

    print(
        f"{modality_name.upper()} "
        f"CNN TEST RESULTS"
    )

    print("=" * 70)


    print(

        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"

    )


    print(

        f"Macro Precision: "
        f"{metrics['macro_precision']:.4f}"

    )


    print(

        f"Macro Recall: "
        f"{metrics['macro_recall']:.4f}"

    )


    print(

        f"Macro F1: "
        f"{metrics['macro_f1']:.4f}"

    )


    print(

        f"Latency: "
        f"{latency_ms:.3f} ms/image"

    )


    # ========================================================
    # TABLE 4
    # CLASSIFICATION REPORT
    # ========================================================

    report_dict = classification_report(

        all_targets,

        all_predictions,

        labels=list(
            range(
                len(CLASS_NAMES)
            )
        ),

        target_names=CLASS_NAMES,

        output_dict=True,

        zero_division=0

    )


    report_df = pd.DataFrame(
        report_dict
    ).transpose()


    report_df.to_csv(

        modality_output
        /
        "Tables"
        /
        "Table_4_Classification_Report.csv"

    )


    # Also save TXT version

    report_text = classification_report(

        all_targets,

        all_predictions,

        labels=list(
            range(
                len(CLASS_NAMES)
            )
        ),

        target_names=CLASS_NAMES,

        zero_division=0

    )


    with open(

        modality_output /
        "classification_report.txt",

        "w"

    ) as f:

        f.write(
            report_text
        )


    # ========================================================
    # FIGURE 5
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        all_targets,

        all_predictions,

        labels=list(
            range(
                len(CLASS_NAMES)
            )
        )

    )


    cm_normalized = (

        cm.astype(float)
        /
        cm.sum(
            axis=1,
            keepdims=True
        )

    )


    cm_normalized = np.nan_to_num(
        cm_normalized
    )


    cm_df = pd.DataFrame(

        cm,

        index=CLASS_NAMES,

        columns=CLASS_NAMES

    )


    cm_df.to_csv(

        modality_output
        /
        "confusion_matrix.csv"

    )


    cm_norm_df = pd.DataFrame(

        cm_normalized,

        index=CLASS_NAMES,

        columns=CLASS_NAMES

    )


    cm_norm_df.to_csv(

        modality_output
        /
        "normalized_confusion_matrix.csv"

    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 6)
    )


    plt.imshow(
        cm_normalized,
        interpolation="nearest"
    )


    plt.title(

        f"{modality_name} CNN "
        f"Normalized Confusion Matrix"

    )


    plt.colorbar()


    tick_marks = np.arange(
        len(CLASS_NAMES)
    )


    plt.xticks(
        tick_marks,
        CLASS_NAMES
    )


    plt.yticks(
        tick_marks,
        CLASS_NAMES
    )


    plt.xlabel(
        "Predicted Class"
    )


    plt.ylabel(
        "True Class"
    )


    for i in range(
        len(CLASS_NAMES)
    ):

        for j in range(
            len(CLASS_NAMES)
        ):

            plt.text(

                j,

                i,

                f"{cm_normalized[i, j]:.2f}",

                ha="center",

                va="center"

            )


    plt.tight_layout()


    plt.savefig(

        figures_dir /
        "Figure_5_Confusion_Matrix.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    # ========================================================
    # FIGURE 6
    # ROC CURVES
    # ========================================================

    y_true_binary = label_binarize(

        all_targets,

        classes=list(
            range(
                len(CLASS_NAMES)
            )
        )

    )


    plt.figure(
        figsize=(8, 6)
    )


    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        fpr, tpr, _ = roc_curve(

            y_true_binary[:, i],

            all_probabilities[:, i]

        )


        roc_auc = auc(
            fpr,
            tpr
        )


        plt.plot(

            fpr,

            tpr,

            linewidth=2,

            label=f"{class_name} "
                  f"(AUC = {roc_auc:.3f})"

        )


    # Macro-average ROC

    all_fpr = np.unique(

        np.concatenate([

            roc_curve(

                y_true_binary[:, i],

                all_probabilities[:, i]

            )[0]

            for i in range(
                len(CLASS_NAMES)
            )

        ])

    )


    mean_tpr = np.zeros_like(
        all_fpr
    )


    for i in range(
        len(CLASS_NAMES)
    ):

        fpr, tpr, _ = roc_curve(

            y_true_binary[:, i],

            all_probabilities[:, i]

        )


        mean_tpr += np.interp(

            all_fpr,

            fpr,

            tpr

        )


    mean_tpr /= len(
        CLASS_NAMES
    )


    macro_auc = auc(

        all_fpr,

        mean_tpr

    )


    plt.plot(

        all_fpr,

        mean_tpr,

        linestyle="--",

        linewidth=2,

        label=f"Macro-average "
              f"(AUC = {macro_auc:.3f})"

    )


    plt.plot(

        [0, 1],

        [0, 1],

        linestyle=":",

        linewidth=1

    )


    plt.xlabel(
        "False Positive Rate"
    )


    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(

        f"{modality_name} CNN "
        f"ROC Curves"

    )


    plt.legend(
        loc="lower right"
    )


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(

        figures_dir /
        "Figure_6_ROC_Curves.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    # ========================================================
    # FIGURE 7
    # PRECISION-RECALL CURVES
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )


    average_precision_values = []


    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        precision, recall, _ = (

            precision_recall_curve(

                y_true_binary[:, i],

                all_probabilities[:, i]

            )

        )


        average_precision = (

            average_precision_score(

                y_true_binary[:, i],

                all_probabilities[:, i]

            )

        )


        average_precision_values.append(

            average_precision

        )


        plt.plot(

            recall,

            precision,

            linewidth=2,

            label=f"{class_name} "
                  f"(AP = "
                  f"{average_precision:.3f})"

        )


    # --------------------------------------------------------
    # Macro average precision
    # --------------------------------------------------------

    macro_ap = np.mean(

        average_precision_values

    )


    plt.xlabel(
        "Recall"
    )


    plt.ylabel(
        "Precision"
    )


    plt.title(

        f"{modality_name} CNN "
        f"Precision-Recall Curves"

    )


    plt.legend(
        loc="best"
    )


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(

        figures_dir /
        "Figure_7_Precision_Recall_Curves.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    # ========================================================
    # TABLE 5
    # OVERALL TEST PERFORMANCE
    # ========================================================

    table_5 = pd.DataFrame({

        "Metric": [

            "Accuracy",

            "Macro Precision",

            "Macro Recall",

            "Macro F1",

            "Macro ROC-AUC",

            "Macro Average Precision",

            "Test Images",

            "Test Groups",

            "Inference Latency (ms/image)"

        ],


        "Value": [

            metrics["accuracy"],

            metrics["macro_precision"],

            metrics["macro_recall"],

            metrics["macro_f1"],

            macro_auc,

            macro_ap,

            len(test_df),

            test_df[
                "group_id"
            ].nunique(),

            latency_ms

        ]

    })


    table_5.to_csv(

        modality_output
        /
        "Tables"
        /
        "Table_5_Overall_Test_Performance.csv",

        index=False

    )


    # ========================================================
    # Save prediction results
    # ========================================================

    prediction_df = pd.DataFrame({

        "True_Class":

            [
                INDEX_TO_LABEL[x]
                for x in all_targets
            ],

        "Predicted_Class":

            [
                INDEX_TO_LABEL[x]
                for x in all_predictions
            ]

    })


    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        prediction_df[
            f"Probability_{class_name}"
        ] = all_probabilities[:, i]


    prediction_df.to_csv(

        modality_output
        /
        "test_predictions.csv",

        index=False

    )


    return {

        "modality":
            modality_name,

        "accuracy":
            metrics["accuracy"],

        "macro_precision":
            metrics["macro_precision"],

        "macro_recall":
            metrics["macro_recall"],

        "macro_f1":
            metrics["macro_f1"],

        "macro_roc_auc":
            macro_auc,

        "macro_average_precision":
            macro_ap,

        "latency_ms_per_image":
            latency_ms,

        "test_images":
            len(test_df),

        "test_groups":
            test_df[
                "group_id"
            ].nunique()

    }


# ============================================================
# 19. MODEL SUMMARY
# ============================================================

def save_model_summary(
    model,
    modality_output
):

    total_params = sum(

        p.numel()

        for p in model.parameters()

    )


    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


    with open(

        modality_output /
        "model_summary.txt",

        "w"

    ) as f:

        f.write(
            str(model)
        )

        f.write(
            "\n\n"
        )

        f.write(
            f"Total parameters: "
            f"{total_params:,}\n"
        )

        f.write(
            f"Trainable parameters: "
            f"{trainable_params:,}\n"
        )


    return (
        total_params,
        trainable_params
    )


# ============================================================
# 20. TRAIN ONE MODALITY
# ============================================================

def train_one_modality(
    root_dir,
    modality_name
):

    print("\n\n")

    print("#" * 70)

    print(
        f"PROCESSING: "
        f"{modality_name.upper()}"
    )

    print("#" * 70)


    # --------------------------------------------------------
    # Create modality-specific folders
    # --------------------------------------------------------

    modality_output = (

        OUTPUT_DIR /
        modality_name

    )


    tables_dir = (

        modality_output /
        "Tables"

    )


    figures_dir = (

        modality_output /
        "Figures"

    )


    modality_output.mkdir(
        parents=True,
        exist_ok=True
    )


    tables_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = build_metadata(

        root_dir,

        modality_name,

        modality_output

    )


    # --------------------------------------------------------
    # TABLE 1
    # --------------------------------------------------------

    table_1 = create_table_1(

        metadata,

        modality_output

    )


    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_df,
        val_df,
        test_df
    ) = split_by_group(

        metadata,

        seed=SEED

    )


    # --------------------------------------------------------
    # TABLE 2
    # --------------------------------------------------------

    table_2 = create_table_2(

        train_df,

        val_df,

        test_df,

        modality_output

    )


    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = (

        BearingImageDataset(

            train_df,

            transform=train_transform

        )

    )


    val_dataset = (

        BearingImageDataset(

            val_df,

            transform=eval_transform

        )

    )


    test_dataset = (

        BearingImageDataset(

            test_df,

            transform=eval_transform

        )

    )


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        )

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        )

    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        )

    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CNNBaseline(

        num_classes=len(
            CLASS_NAMES
        )

    )


    total_params = sum(

        p.numel()

        for p in model.parameters()

    )


    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


    print(

        f"\nTotal parameters: "
        f"{total_params:,}"

    )


    print(

        f"Trainable parameters: "
        f"{trainable_params:,}"

    )


    # --------------------------------------------------------
    # TABLE 3
    # --------------------------------------------------------

    create_table_3(

        modality_output,

        total_params,

        trainable_params

    )


    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    save_model_summary(

        model,

        modality_output

    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    (

        model,

        history,

        best_val_f1

    ) = train_model(

        model,

        train_loader,

        val_loader,

        train_df,

        modality_name,

        modality_output

    )


    # --------------------------------------------------------
    # FIGURE 3
    # --------------------------------------------------------

    plot_accuracy(

        history,

        modality_name,

        figures_dir

    )


    # --------------------------------------------------------
    # FIGURE 4
    # --------------------------------------------------------

    plot_loss(

        history,

        modality_name,

        figures_dir

    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    result = evaluate_model(

        model,

        test_loader,

        test_df,

        modality_name,

        modality_output,

        figures_dir

    )


    return result


# ============================================================
# 21. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nStarting CNN experiments..."
    )


    # ========================================================
    # VIBRATION
    # ========================================================

    vibration_result = train_one_modality(

        VIBRATION_DIR,

        "Vibration"

    )


    # ========================================================
    # ACOUSTIC
    # ========================================================

    acoustic_result = train_one_modality(

        ACOUSTIC_DIR,

        "Acoustic"

    )


    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    comparison = pd.DataFrame([

        vibration_result,

        acoustic_result

    ])


    comparison.to_csv(

        OUTPUT_DIR /
        "vibration_vs_acoustic_cnn_results.csv",

        index=False

    )


    print("\n")

    print("=" * 70)

    print(
        "FINAL CNN COMPARISON"
    )

    print("=" * 70)


    print(

        comparison.to_string(
            index=False
        )

    )


    print(

        f"\nAll results saved to:"
        f"\n{OUTPUT_DIR.resolve()}"

    )