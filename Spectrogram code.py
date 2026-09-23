import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io

# Define parameters for spectrogram computation
fs = 42000
nperseg = 505
noverlap = 460
segment_size = 512

# ============================================================
# CHANGE ONLY THESE PATHS
# ============================================================

input_dir = r"C:\Users\saroj\OneDrive\Desktop\PYTHON\archive (1)\University of Ottawa Rolling-element Dataset – Vibration and Acoustic Faults under Constant Load and Speed conditions (UORED-VAFCLS)\1_CSV_Raw_Data_Files (.csv)\1_Healthy"

output_dir_main = r"C:\Users\saroj\OneDrive\Desktop\PYTHON\Ottawa_Spectrograms\1_Healthy"

# ============================================================

for filename in os.listdir(input_dir):
    if filename.endswith('.mat'):

        folder_name = os.path.splitext(filename)[0]

        output_dir = os.path.join(
            output_dir_main,
            folder_name
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        mat_contents = scipy.io.loadmat(
            os.path.join(input_dir, filename)
        )

        data = mat_contents[list(mat_contents.keys())[3]][:, 1]

        for i in range(0, segment_size * 400, segment_size):

            segment = data[i:i + segment_size]

            f, t, Sxx = signal.stft(
                segment,
                nperseg=1024
            )

            fig = plt.figure(figsize=(8, 6))

            plt.imshow(
                np.fliplr(abs(Sxx).T).T,
                cmap='viridis',
                aspect='auto',
                extent=[
                    t.min(),
                    t.max(),
                    f.min(),
                    f.max()
                ]
            )

            plt.ylabel('Frequency [kHz]')
            plt.xlabel('Number of Samples')

            plt.axis('off')

            output_filename = os.path.join(
                output_dir,
                folder_name + '_{}.png'.format(int(i / 512))
            )

            plt.savefig(
                output_filename,
                bbox_inches='tight',
                pad_inches=0
            )

            plt.close(fig)

print('Complete!')