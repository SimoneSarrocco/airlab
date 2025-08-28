# Check slo and faf in omega data dir
import cv2
import os
from pathlib import Path
from glob import glob
import pydicom
import matplotlib.pyplot as plt
from lakefsimaged import LoadLakeFSImaged
from lakefsloader import LakeFSLoader
from datalist import DataList
import yaml
import re
import numpy as np
from collections import defaultdict

src_path = Path(__file__).parent
project_path = src_path.parent

# Paths of the config files
config_path = '/home/simone.sarrocco/OMEGA_study/image_registration/airlab/configs/config.yaml'
lakefs_config_path = '/home/simone.sarrocco/OMEGA_study/image_registration/airlab/configs/lakefs_cfg.yaml'

# Load the config files
with open('/home/simone.sarrocco/OMEGA_study/image_registration/airlab/configs/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
with open('/home/simone.sarrocco/OMEGA_study/image_registration/airlab/configs/lakefs_cfg.yaml') as f:
    lakefs_config = yaml.load(f, Loader=yaml.FullLoader)

# Create a datalist from the filepath
input_path = Path(config["input_path"])

datalist_slo = DataList.from_lakefs(data_config=config['data'], lakefs_config=lakefs_config, filepath=str(input_path), include_root=True, image_modality='Spectralis_slo')
datalist_faf = DataList.from_lakefs(data_config=config['data'], lakefs_config=lakefs_config, filepath=str(input_path), include_root=True, image_modality='Spectralis_faf')
data_list_slo = datalist_slo.data
data_list_faf = datalist_faf.data
print("Data list SLO len: ", len(data_list_slo['full_dataset']))
print("Data list FAF len: ", len(data_list_faf["full_dataset"]))

pattern_faf = r"(OMEGA\d{2})/(OD|OS)/(V\d{2})/Spectralis_faf/.*\.dcm"
pattern_slo = r"(OMEGA\d{2})/(OD|OS)/(V\d{2})/Spectralis_slo/.*\.dcm"

combined_paths = {}

for entry in data_list_faf['full_dataset']:
    file_path_faf = entry["image"]
    # if 'OMEGA_04_271.nii.gz' in file_path:
    #    continue
        # volume_dict[(patient_id, eye)][visit_id] = file_path
    m_oct = re.search(pattern_faf, file_path_faf)
    if m_oct:
        patient_id, eye, visit_id = m_oct.groups()
    # Ensure nested dictionaries exist
        if patient_id not in combined_paths:
            combined_paths[patient_id] = {}
        if eye not in combined_paths[patient_id]:
            combined_paths[patient_id][eye] = {}
        if visit_id not in combined_paths[patient_id][eye]:
            combined_paths[patient_id][eye][visit_id] = {}
        # oct_paths[(patient_id, eye)][visit_id] = file_path_oct
        combined_paths[patient_id][eye][visit_id]["faf"] = file_path_faf

for entry in data_list_slo['full_dataset']:
    file_path_slo = entry["image"]
    m_slo = re.search(pattern_slo, file_path_slo)
    if m_slo:
        patient_id, eye, visit_id = m_slo.groups()
        if patient_id not in combined_paths:
            combined_paths[patient_id] = {}
        if eye not in combined_paths[patient_id]:
            combined_paths[patient_id][eye] = {}
        if visit_id not in combined_paths[patient_id][eye]:
            combined_paths[patient_id][eye][visit_id] = {}
        # slo_paths[(patient_id, eye)][visit_id] = file_path_slo
        combined_paths[patient_id][eye][visit_id]["slo"] = file_path_slo


# Global counters
total_faf = 0
total_slo = 0

# Per patient/eye/visit counters
patient_counts = defaultdict(lambda: defaultdict(dict))

for patient_id, eyes in combined_paths.items():
    for eye, visits in eyes.items():
        for visit_id, data in visits.items():
            faf_exists = "faf" in data
            slo_exists = "slo" in data

            if faf_exists:
                total_faf += 1
            if slo_exists:
                total_slo += 1

            # Save status for this visit
            patient_counts[patient_id][eye][visit_id] = {
                "faf": faf_exists,
                "slo": slo_exists
            }

print("=== Global counts ===")
print(f"Total FAF images: {total_faf}")
print(f"Total SLO images: {total_slo}")

print("\n=== Per patient/eye/visit availability ===")
for patient_id, eyes in patient_counts.items():
    print(f"\n{patient_id}:")
    for eye, visits in eyes.items():
        print(f"  {eye}:")
        for visit_id, status in sorted(visits.items()):
            print(f"    {visit_id}: FAF={'Yes' if status['faf'] else 'No'}, "
                  f"SLO={'Yes' if status['slo'] else 'No'}")

img_size_plot = 96
img_plot_size = 4
rows = sum(len(eyes) for eyes in combined_paths.values())  # one row per eye
cols = max(len(v) for p in combined_paths.values() for e, v in p.items()) * 2  # max visits × 2
fig = plt.figure(figsize=(img_plot_size*cols, img_plot_size*rows))
row = 1

for s, (patient_id, eyes) in enumerate(combined_paths.items()):
    print(f"Processing subject {s+1}/{len(combined_paths.items())}: {patient_id}")
    for eye, visits in eyes.items():
        col = 1
        visit_ids = sorted(visits.keys())
        for visit_id in visit_ids:
            data = visits[visit_id]
            # if "-" in visit_id:
            #    continue
            # if "faf" not in data or "slo" not in data:
            #    print(f"Missing data for {patient_id} {eye} {visit_id}, skipping (missing FAF or SLO)")
            #    continue
            if "slo" in data:
                slo_image = pydicom.dcmread(data["slo"]).pixel_array
                slo_img_plot = cv2.resize(slo_image, (img_size_plot, img_size_plot), interpolation=cv2.INTER_NEAREST)
            else: 
                slo_img_plot = np.ones((img_size_plot, img_size_plot), dtype=np.uint8) * 255  # white square
            
            idx_slo = (row-1) * cols + col
            plt.subplot(rows, cols, idx_slo)
            plt.imshow(slo_img_plot, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            if row == 1:
                plt.title(f"SLO {patient_id} {eye}", fontsize=12)
            col += 1

            # Handle FAF
            if "faf" in data:
                faf_image = pydicom.dcmread(data["faf"]).pixel_array
                faf_img_plot = cv2.resize(faf_image, (img_size_plot, img_size_plot), interpolation=cv2.INTER_NEAREST)
            else:
                faf_img_plot = np.ones((img_size_plot, img_size_plot), dtype=np.uint8) * 255  # white square
            idx_faf = (row-1) * cols + col
            plt.subplot(rows, cols, idx_faf)
            plt.imshow(faf_img_plot, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            if row == 1:
                plt.title(f"FAF {patient_id} {eye}", fontsize=12)
            col += 1
        fig.text(0.01, (rows-row+0.5)/rows, f"{patient_id} {eye}", va='center', fontsize=14)
        row += 1
plt.tight_layout(rect=[0.05, 0, 1, 1]) 
plt.show()
                
"""img_size_plot = 48
modalities = ["Spectralis_slo", "Spectralis_faf"]
rows, cols = 70, 8
img_plot_size = 5
subject_dirs = sorted(glob(os.path.join(OMEGA_DATA_DIR, "*")))
fig = plt.figure(figsize=(img_plot_size*cols, img_plot_size*rows))
row = 1
for s, subject_dir in enumerate(subject_dirs):
    subject_id = os.path.basename(subject_dir)
    # if s > 5:
    #    break
    print(f"Processing subject {s+1}/{len(subject_dirs)}: {subject_id}")
    laterality_dirs = sorted(glob(os.path.join(subject_dir, "*")))
    for l, laterality_dir in enumerate(laterality_dirs):
        laterality = os.path.basename(laterality_dir)
        eye_id = f"{subject_id}_{laterality}"
        print(f"Processing: {eye_id}")
        visit_dirs = sorted(glob(os.path.join(laterality_dir, "*")))
        col = 1
        for v, visit_dir in enumerate(visit_dirs):
            visit_id = os.path.basename(visit_dir)
            if "-" in visit_id:
                continue
            modality_dirs = sorted(glob(os.path.join(visit_dir, "*")))
            for modality in modalities:
                modality_dir = os.path.join(visit_dir, modality)
                dcm_path = glob(os.path.join(modality_dir, "*.dcm"))
                if not dcm_path:
                    print(f"No DICOM files found for {eye_id} {visit_id} {modality}")
                    continue
                dcm_path = dcm_path[0]
                img = pydicom.dcmread(dcm_path).pixel_array
                img_plot = cv2.resize(img, (img_size_plot, img_size_plot), interpolation=cv2.INTER_NEAREST)
                idx = (row-1) * cols + col
                plt.subplot(rows, cols, idx)
                plt.imshow(img_plot, cmap='gray')
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                if col == 1:
                    plt.title(f"{eye_id}", fontsize=20)
                col += 1
        row += 1
plt.show()"""