# ============================================================
# RAW DATA QUALITY ANALYSIS
# University of Ottawa Rolling-Element Bearing Dataset
#
# Column 1 = Vibration / Accelerometer
# Column 2 = Acoustic / Microphone
#
# Sampling frequency = 42 kHz
# Recording duration = 10 seconds
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. PATH SETTINGS
# ============================================================

DATA_PATH = Path(
    r"C:\Users\saroj\OneDrive\Desktop\PYTHON\archive (1)\University of Ottawa Rolling-element Dataset – Vibration and Acoustic Faults under Constant Load and Speed conditions (UORED-VAFCLS)\1_CSV_Raw_Data_Files (.csv)"
)

OUTPUT_PATH = Path(
    r"C:\Users\saroj\OneDrive\Desktop\PYTHON\Raw_Data_Quality_Analysis"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATA SETTINGS
# ============================================================

FS = 42000
DURATION = 10
EXPECTED_SAMPLES = FS * DURATION


# ============================================================
# 3. CLASS INFORMATION
# ============================================================

class_names = {
    "H": "Healthy",
    "I": "Inner Race Fault",
    "O": "Outer Race Fault",
    "B": "Ball Fault",
    "C": "Cage Fault"
}


# Fixed colors for all graphs
colors = {
    "H": "blue",
    "I": "orange",
    "O": "green",
    "B": "red",
    "C": "purple"
}


# Class order
class_order = ["H", "I", "O", "B", "C"]


# ============================================================
# 4. FIND ALL CSV FILES
# ============================================================

csv_files = list(DATA_PATH.rglob("*.csv"))

print("=" * 70)
print("RAW DATA QUALITY ANALYSIS")
print("=" * 70)

print("\nData folder:")
print(DATA_PATH)

print(f"\nTotal CSV files found: {len(csv_files)}")


if len(csv_files) == 0:

    print("\nERROR: No CSV files found.")
    print("Please check DATA_PATH.")

    raise SystemExit


# ============================================================
# 5. FUNCTION TO IDENTIFY CLASS
# ============================================================

def get_class_from_path(file_path):

    path_string = str(file_path).lower()

    if "healthy" in path_string:
        return "H"

    elif "inner_race" in path_string:
        return "I"

    elif "outer_race" in path_string:
        return "O"

    elif "ball_fault" in path_string:
        return "B"

    elif "cage_fault" in path_string:
        return "C"

    else:
        return "Unknown"


# ============================================================
# 6. STORAGE
# ============================================================

records = []

representative_waveforms = {}


# ============================================================
# 7. READ ALL CSV FILES
# ============================================================

print("\nReading CSV files...\n")


for i, csv_file in enumerate(csv_files, start=1):

    try:

        # ----------------------------------------------------
        # Read ONLY first two columns
        #
        # Column 1 = Vibration
        # Column 2 = Acoustic
        # ----------------------------------------------------

        df = pd.read_csv(
            csv_file,
            usecols=[0, 1]
        )

        # Remove missing values
        df = df.dropna()

        if len(df) == 0:
            continue


        # ----------------------------------------------------
        # Extract signals
        # ----------------------------------------------------

        vibration = df.iloc[:, 0].to_numpy(dtype=float)

        acoustic = df.iloc[:, 1].to_numpy(dtype=float)


        # ----------------------------------------------------
        # Identify class
        # ----------------------------------------------------

        class_code = get_class_from_path(csv_file)


        if class_code == "Unknown":

            print(
                f"WARNING: Unknown class: {csv_file.name}"
            )

            continue


        # ----------------------------------------------------
        # VIBRATION STATISTICS
        # ----------------------------------------------------

        vibration_mean = np.mean(vibration)

        vibration_sd = np.std(vibration)

        vibration_rms = np.sqrt(
            np.mean(vibration ** 2)
        )

        vibration_min = np.min(vibration)

        vibration_max = np.max(vibration)

        vibration_peak = np.max(
            np.abs(vibration)
        )


        # ----------------------------------------------------
        # ACOUSTIC STATISTICS
        # ----------------------------------------------------

        acoustic_mean = np.mean(acoustic)

        acoustic_sd = np.std(acoustic)

        acoustic_rms = np.sqrt(
            np.mean(acoustic ** 2)
        )

        acoustic_min = np.min(acoustic)

        acoustic_max = np.max(acoustic)

        acoustic_peak = np.max(
            np.abs(acoustic)
        )


        # ----------------------------------------------------
        # SAVE STATISTICS
        # ----------------------------------------------------

        records.append({

            "File": csv_file.name,

            "Class": class_code,

            "Class_Name": class_names[class_code],

            "Vibration_Samples": len(vibration),

            "Acoustic_Samples": len(acoustic),

            "Vibration_Mean": vibration_mean,

            "Vibration_SD": vibration_sd,

            "Vibration_RMS": vibration_rms,

            "Vibration_Min": vibration_min,

            "Vibration_Max": vibration_max,

            "Vibration_Peak": vibration_peak,

            "Acoustic_Mean": acoustic_mean,

            "Acoustic_SD": acoustic_sd,

            "Acoustic_RMS": acoustic_rms,

            "Acoustic_Min": acoustic_min,

            "Acoustic_Max": acoustic_max,

            "Acoustic_Peak": acoustic_peak

        })


        # ----------------------------------------------------
        # SAVE ONE REPRESENTATIVE RECORDING PER CLASS
        # ----------------------------------------------------

        if class_code not in representative_waveforms:

            time = np.arange(
                len(vibration)
            ) / FS

            representative_waveforms[class_code] = {

                "time": time,

                "vibration": vibration,

                "acoustic": acoustic,

                "file": csv_file.name

            }


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if i % 10 == 0 or i == len(csv_files):

            print(
                f"Processed {i}/{len(csv_files)} files"
            )


    except Exception as e:

        print("\nERROR reading:")

        print(csv_file)

        print(e)


# ============================================================
# 8. CREATE DATAFRAME
# ============================================================

stats_df = pd.DataFrame(records)


print("\n" + "=" * 70)

print("PROCESSING COMPLETE")

print("=" * 70)


print(
    f"\nTotal recordings successfully processed: "
    f"{len(stats_df)}"
)


# ============================================================
# 9. SAVE ALL RECORDING STATISTICS
# ============================================================

statistics_file = (
    OUTPUT_PATH /
    "Raw_Data_Statistics_All_Recordings.csv"
)


stats_df.to_csv(
    statistics_file,
    index=False
)


print("\nSaved:")

print(statistics_file)


# ============================================================
# 10. CLASS-WISE SUMMARY
# ============================================================

summary_df = (

    stats_df

    .groupby(
        ["Class", "Class_Name"]
    )

    .agg({

        "Vibration_Mean": ["mean", "std"],

        "Vibration_SD": ["mean", "std"],

        "Vibration_RMS": ["mean", "std"],

        "Acoustic_Mean": ["mean", "std"],

        "Acoustic_SD": ["mean", "std"],

        "Acoustic_RMS": ["mean", "std"]

    })

)


summary_file = (
    OUTPUT_PATH /
    "Class_Wise_Summary.csv"
)


summary_df.to_csv(
    summary_file
)


print("Saved:")

print(summary_file)


# ============================================================
# GRAPH 1
# ALL VIBRATION WAVEFORMS
# ============================================================

plt.figure(figsize=(16, 8))


for cls in class_order:

    if cls in representative_waveforms:

        data = representative_waveforms[cls]

        plt.plot(

            data["time"],

            data["vibration"],

            color=colors[cls],

            label=class_names[cls],

            alpha=0.7,

            linewidth=0.8

        )


plt.xlabel(
    "Time (s)",
    fontsize=12
)

plt.ylabel(
    "Acceleration (m/s²)",
    fontsize=12
)


plt.title(
    "Representative Vibration Waveforms of All Bearing Classes",
    fontsize=14
)


plt.legend(
    title="Bearing Condition",
    fontsize=10
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    OUTPUT_PATH /
    "01_All_Vibration_Waveforms.png",

    dpi=300

)


plt.close()


# ============================================================
# GRAPH 2
# ALL ACOUSTIC WAVEFORMS
# ============================================================

plt.figure(figsize=(16, 8))


for cls in class_order:

    if cls in representative_waveforms:

        data = representative_waveforms[cls]

        plt.plot(

            data["time"],

            data["acoustic"],

            color=colors[cls],

            label=class_names[cls],

            alpha=0.7,

            linewidth=0.8

        )


plt.xlabel(
    "Time (s)",
    fontsize=12
)

plt.ylabel(
    "Acoustic Signal (V)",
    fontsize=12
)


plt.title(
    "Representative Acoustic Waveforms of All Bearing Classes",
    fontsize=14
)


plt.legend(
    title="Bearing Condition",
    fontsize=10
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    OUTPUT_PATH /
    "02_All_Acoustic_Waveforms.png",

    dpi=300

)


plt.close()


# ============================================================
# GRAPH 3
# VIBRATION RMS BOXPLOT
# ============================================================

plt.figure(figsize=(10, 7))


data_to_plot = []

labels = []


for cls in class_order:

    values = stats_df[
        stats_df["Class"] == cls
    ]["Vibration_RMS"].values

    data_to_plot.append(values)

    labels.append(
        class_names[cls]
    )


# IMPORTANT:
# New Matplotlib uses tick_labels
# instead of labels

box = plt.boxplot(

    data_to_plot,

    tick_labels=labels,

    patch_artist=False

)


plt.xlabel(
    "Bearing Condition"
)

plt.ylabel(
    "Vibration RMS (m/s²)"
)


plt.title(
    "Vibration RMS Distribution"
)


plt.xticks(
    rotation=20
)


plt.grid(
    True,
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    OUTPUT_PATH /
    "03_Vibration_RMS_Boxplot.png",

    dpi=300

)


plt.close()


# ============================================================
# GRAPH 4
# ACOUSTIC RMS BOXPLOT
# ============================================================

plt.figure(figsize=(10, 7))


data_to_plot = []

labels = []


for cls in class_order:

    values = stats_df[
        stats_df["Class"] == cls
    ]["Acoustic_RMS"].values

    data_to_plot.append(values)

    labels.append(
        class_names[cls]
    )


plt.boxplot(

    data_to_plot,

    tick_labels=labels

)


plt.xlabel(
    "Bearing Condition"
)

plt.ylabel(
    "Acoustic RMS (V)"
)


plt.title(
    "Acoustic RMS Distribution"
)


plt.xticks(
    rotation=20
)


plt.grid(
    True,
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    OUTPUT_PATH /
    "04_Acoustic_RMS_Boxplot.png",

    dpi=300

)


plt.close()


# ============================================================
# GRAPH 5
# VIBRATION MEAN
# ============================================================

plt.figure(figsize=(10, 7))


for class_index, cls in enumerate(class_order, start=1):

    values = stats_df[
        stats_df["Class"] == cls
    ]["Vibration_Mean"].values


    # Small horizontal jitter
    x = np.random.normal(

        loc=class_index,

        scale=0.04,

        size=len(values)

    )


    plt.scatter(

        x,

        values,

        color=colors[cls],

        label=class_names[cls],

        alpha=0.6

    )


plt.xlabel(
    "Bearing Condition"
)

plt.ylabel(
    "Vibration Mean (m/s²)"
)


plt.title(
    "Vibration Mean of All Recordings"
)


plt.xticks(

    [1, 2, 3, 4, 5],

    [
        "Healthy",
        "Inner Race",
        "Outer Race",
        "Ball",
        "Cage"
    ],

    rotation=20

)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    OUTPUT_PATH /
    "05_Vibration_Mean.png",

    dpi=300

)


plt.close()


# ============================================================
# GRAPH 6
# ACOUSTIC MEAN
# ============================================================

plt.figure(figsize=(10, 7))


for class_index, cls in enumerate(class_order, start=1):

    values = stats_df[
        stats_df["Class"] == cls
    ]["Acoustic_Mean"].values


    x = np.random.normal(

        loc=class_index,

        scale=0.04,

        size=len(values)

    )


    plt.scatter(

        x,

        values,

        color=colors[cls],

        label=class_names[cls],

        alpha=0.6

    )


plt.xlabel(
    "Bearing Condition"
)

plt.ylabel(
    "Acoustic Mean (V)"
)


plt.title(
    "Acoustic Mean of All Recordings"
)


plt.xticks(

    [1, 2, 3, 4, 5],

    [
        "Healthy",
        "Inner Race",
        "Outer Race",
        "Ball",
        "Cage"
    ],

    rotation=20

)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    OUTPUT_PATH /
    "06_Acoustic_Mean.png",

    dpi=300

)


plt.close()


# ============================================================
# GRAPH 7
# VIBRATION STANDARD DEVIATION
# ============================================================

plt.figure(figsize=(10, 7))


for class_index, cls in enumerate(class_order, start=1):

    values = stats_df[
        stats_df["Class"] == cls
    ]["Vibration_SD"].values


    x = np.random.normal(

        loc=class_index,

        scale=0.04,

        size=len(values)

    )


    plt.scatter(

        x,

        values,

        color=colors[cls],

        label=class_names[cls],

        alpha=0.6

    )


plt.xlabel(
    "Bearing Condition"
)

plt.ylabel(
    "Vibration Standard Deviation (m/s²)"
)


plt.title(
    "Vibration Standard Deviation of All Recordings"
)


plt.xticks(

    [1, 2, 3, 4, 5],

    [
        "Healthy",
        "Inner Race",
        "Outer Race",
        "Ball",
        "Cage"
    ],

    rotation=20

)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    OUTPUT_PATH /
    "07_Vibration_SD.png",

    dpi=300

)


plt.close()


# ============================================================
# GRAPH 8
# ACOUSTIC STANDARD DEVIATION
# ============================================================

plt.figure(figsize=(10, 7))


for class_index, cls in enumerate(class_order, start=1):

    values = stats_df[
        stats_df["Class"] == cls
    ]["Acoustic_SD"].values


    x = np.random.normal(

        loc=class_index,

        scale=0.04,

        size=len(values)

    )


    plt.scatter(

        x,

        values,

        color=colors[cls],

        label=class_names[cls],

        alpha=0.6

    )


plt.xlabel(
    "Bearing Condition"
)

plt.ylabel(
    "Acoustic Standard Deviation (V)"
)


plt.title(
    "Acoustic Standard Deviation of All Recordings"
)


plt.xticks(

    [1, 2, 3, 4, 5],

    [
        "Healthy",
        "Inner Race",
        "Outer Race",
        "Ball",
        "Cage"
    ],

    rotation=20

)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    OUTPUT_PATH /
    "08_Acoustic_SD.png",

    dpi=300

)


plt.close()


# ============================================================
# GRAPH 9
# VIBRATION MEAN VS STANDARD DEVIATION
# ============================================================

plt.figure(figsize=(10, 7))


for cls in class_order:

    subset = stats_df[
        stats_df["Class"] == cls
    ]


    plt.scatter(

        subset["Vibration_Mean"],

        subset["Vibration_SD"],

        color=colors[cls],

        label=class_names[cls],

        alpha=0.6

    )


plt.xlabel(
    "Vibration Mean (m/s²)"
)

plt.ylabel(
    "Vibration Standard Deviation (m/s²)"
)


plt.title(
    "Vibration Mean vs Standard Deviation"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    OUTPUT_PATH /
    "09_Vibration_Mean_vs_SD.png",

    dpi=300

)


plt.close()


# ============================================================
# GRAPH 10
# ACOUSTIC MEAN VS STANDARD DEVIATION
# ============================================================

plt.figure(figsize=(10, 7))


for cls in class_order:

    subset = stats_df[
        stats_df["Class"] == cls
    ]


    plt.scatter(

        subset["Acoustic_Mean"],

        subset["Acoustic_SD"],

        color=colors[cls],

        label=class_names[cls],

        alpha=0.6

    )


plt.xlabel(
    "Acoustic Mean (V)"
)

plt.ylabel(
    "Acoustic Standard Deviation (V)"
)


plt.title(
    "Acoustic Mean vs Standard Deviation"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    OUTPUT_PATH /
    "10_Acoustic_Mean_vs_SD.png",

    dpi=300

)


plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("ALL ANALYSIS COMPLETED SUCCESSFULLY")

print("=" * 70)


print("\nTotal graphs created: 10")


print("\nOutput folder:")

print(OUTPUT_PATH)


print("\nCOLOR CODING:")

print("Blue   = Healthy")

print("Orange = Inner Race Fault")

print("Green  = Outer Race Fault")

print("Red    = Ball Fault")

print("Purple = Cage Fault")


print("\nCSV FILES CREATED:")

print("1. Raw_Data_Statistics_All_Recordings.csv")

print("2. Class_Wise_Summary.csv")


print("\nGRAPH FILES CREATED:")

print("1. 01_All_Vibration_Waveforms.png")

print("2. 02_All_Acoustic_Waveforms.png")

print("3. 03_Vibration_RMS_Boxplot.png")

print("4. 04_Acoustic_RMS_Boxplot.png")

print("5. 05_Vibration_Mean.png")

print("6. 06_Acoustic_Mean.png")

print("7. 07_Vibration_SD.png")

print("8. 08_Acoustic_SD.png")

print("9. 09_Vibration_Mean_vs_SD.png")

print("10. 10_Acoustic_Mean_vs_SD.png")

print("\nAll graphs saved at 300 DPI.")

print("=" * 70)