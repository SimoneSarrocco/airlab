import itk 
import sys
from itkwidgets import view, compare, checkerboard
import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import time
import math
import csv
import glob
import gc
import subprocess
from lakefsimaged import LoadLakeFSImaged
from lakefsloader import LakeFSLoader
from monai.metrics.regression import PSNRMetric, SSIMMetric, MSEMetric
from monai import transforms
from nibabel.processing import resample_to_output
import traceback
from pathlib import Path
import re
from collections import defaultdict
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import airlab as al
import datetime
import shutil
import io
from datalist import DataList
import tempfile
import logging
import http.client as http_client
import botocore
from botocore.config import Config
import cv2
import pandas as pd


# Set path
src_path = Path(__file__).parent
project_path = src_path.parent

# Paths of the config files
config_path = 'configs/config.yaml'
lakefs_config_path = 'configs/lakefs_cfg.yaml'

# Load the config files
with open(src_path / config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
with open(src_path / lakefs_config_path) as f:
    lakefs_config = yaml.load(f, Loader=yaml.FullLoader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# transformations = ['rigid', 'affine', 'bspline']
target_shape = (512, 496, 192)

# Create the experiment folder, setup tensorboard, copy the used config
current_datetime = datetime.datetime.now()
timestamp = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
experiment_name = config['run_name']
# writer = SummaryWriter(experiment_path)
experiment_path = project_path / 'experiments' / (experiment_name + f"_{timestamp}")
os.makedirs(experiment_path, exist_ok=True)
output_csv = os.path.join(experiment_path, 'oct_to_oct_and_slo_to_enface_oct_registration_metrics.csv')

# Copy the config files to the corresponding experiment folder
shutil.copyfile(src_path / config_path, experiment_path / "config.yaml")
shutil.copyfile(src_path / lakefs_config_path, experiment_path / "lakefs_cfg.yaml")


def resample_image(image, target_spacing, interpolator=sitk.sitkLinear):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    return resample.Execute(image)

def normalize_image(image):
    """Normalize the image to the range [0, 1]."""
    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / (image_max - image_min)

def compute_metrics(fixed_th, warped_th, moving_th, device, dim):
    """Compute metrics between fixed and warped images"""
    # fixed_th = fixed_th.permute(2, 1, 0)
    # fixed_th = fixed_th.unsqueeze(0).unsqueeze(0)
    # fixed_th = fixed_th.to(device=device)
    if dim == 2:
        SSIM = SSIMMetric(spatial_dims=2, reduction='mean', data_range=1.0)
        PSNR = PSNRMetric(reduction='mean', max_val=1.0)
        MSE = MSEMetric(reduction='mean')       
        fixed_th = fixed_th.permute(1, 0)
        fixed_th = fixed_th.unsqueeze(0).unsqueeze(0).to(device=device)

        warped_th = warped_th.permute(1, 0)
        warped_th = warped_th.unsqueeze(0).unsqueeze(0).to(device)

        moving_th = moving_th.permute(1, 0)
        moving_th = moving_th.unsqueeze(0).unsqueeze(0).to(device)

        assert fixed_th.shape == warped_th.shape == moving_th.shape and len(fixed_th.shape) == 4, "Fixed and warped images must have the same shape of kind (B, C, H, W)"
    
    elif dim == 3:
        SSIM = SSIMMetric(spatial_dims=3, reduction='mean', data_range=1.0)
        PSNR = PSNRMetric(reduction='mean', max_val=1.0)
        MSE = MSEMetric(reduction='mean')

        fixed_th = fixed_th.permute(2, 1, 0)
        fixed_th = fixed_th.unsqueeze(0).unsqueeze(0)
        fixed_th = fixed_th.to(device=device)

        warped_th = warped_th.permute(2, 1, 0)
        warped_th = warped_th.unsqueeze(0).unsqueeze(0)
        warped_th = warped_th.to(device=device)

        moving_th = moving_th.permute(2, 1, 0)
        moving_th = moving_th.unsqueeze(0).unsqueeze(0)
        moving_th = moving_th.to(device=device)
    
        assert fixed_th.shape == warped_th.shape == moving_th.shape and len(fixed_th.shape) == 5, "Fixed and warped images must have the same shape of kind (B, C, Z, H, W)"
    
    # Similarity metrics between fixed and warped image
    psnr_fixed_warped = PSNR(warped_th, fixed_th)
    ssim_fixed_warped = SSIM(warped_th, fixed_th)
    mse_fixed_warped = MSE(warped_th, fixed_th)

    # Similarity metrics between fixed and moving image (to use as baseline values)
    psnr_fixed_moving = PSNR(moving_th, fixed_th)
    ssim_fixed_moving = SSIM(moving_th, fixed_th)
    mse_fixed_moving = MSE(moving_th, fixed_th)
    
    return psnr_fixed_warped, ssim_fixed_warped, mse_fixed_warped, psnr_fixed_moving, ssim_fixed_moving, mse_fixed_moving

def resize_volume(image, target_shape):
    original_size = image.GetSize()
    scale_factors = [ts / osz for ts, osz in zip(target_shape, original_size)]
    transform = sitk.ScaleTransform(3, scale_factors)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetSize(target_shape)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)

def save_to_lakefs_cli(sitk_image: sitk.Image, patient_id: str, eye: str, visit_id: str, repo: str, branch: str):
    # Step 1: Save the NIfTI image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_file:
        sitk.WriteImage(sitk_image, tmp_file.name)
        tmp_file_path = tmp_file.name
        print(f'Written to temp file at: {tmp_file_path}')

    # Step 2: Construct LakeFS S3 URI
    object_key = f"{branch}/data/{patient_id}/{eye}/{visit_id}/Spectralis_oct/{patient_id}_{eye}_{visit_id}_registered.nii.gz"
    s3_uri = f"s3://{repo}/{object_key}"

    # Step 3: Call `aws s3 cp` using subprocess
    print(f'Uploading to LakeFS using aws s3 cp...')
    start = time.time()
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", tmp_file_path, s3_uri, "--endpoint-url", lakefs_config['s3_endpoint']],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Upload failed:\n{e.stderr}")
    finally:
        end = time.time()
        print(f"Upload finished in {end - start:.2f} seconds")

        # Optional: delete the temp file
        os.remove(tmp_file_path)


def save_to_lakefs_cli_singlepart(
    sitk_image: sitk.Image,
    patient_id: str,
    eye: str,
    visit_id: str,
    output_filename: str,
    repo: str,
    branch: str,
    folder: str,
    file_extension: str,
):
    """
    Save a registered image as DICOM (.dcm) in case of SLO or NIFTI (.nii.gz) in case of OCT, and upload it to LakeFS
    using aws s3api put-object.
    """

    # Step 0: Cast to UInt16 since DICOM doesn’t support float32 directly
    if sitk_image.GetPixelID() in [sitk.sitkFloat32, sitk.sitkFloat64]:
        print(f"Casting float image to UInt16 for {file_extension} export...")
        sitk_image = sitk.Cast(sitk.RescaleIntensity(sitk_image, 0, 65535), sitk.sitkUInt16)

    # Step 1: Save image to temp DICOM file
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
        sitk.WriteImage(sitk_image, tmp_file.name)
        tmp_file_path = tmp_file.name
        print(f'Written {file_extension} file to temp file at: {tmp_file_path}')

    # Step 2: Build S3 key and URI
    if file_extension == '.dcm':
        object_key = f"{branch}/{folder}/{patient_id}/{eye}/{visit_id}/Spectralis_slo/{output_filename}"
    elif file_extension == '.nii.gz':
        object_key = f"{branch}/{folder}/{patient_id}/{eye}/{visit_id}/Spectralis_oct/{output_filename}"

    s3_uri = f"s3://{repo}/{object_key}"

    # Step 3: Upload using aws s3api put-object
    print(f"Uploading to LakeFS using s3api put-object (single-part)...")
    try:
        result = subprocess.run(
            [
                "aws", "s3api", "put-object",
                "--bucket", repo,
                "--key", object_key,
                "--body", tmp_file_path,
                "--endpoint-url", lakefs_config['s3_endpoint']
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Upload failed:\n{e.stderr}")
    finally:
        os.remove(tmp_file_path)
        print("Temporary file removed.")


def extract_oct_id(oct_path: str) -> str:
    """
    Given a path like ".../OMEGA_01_176.nii.gz",
    return "OMEGA_01_176".
    """
    p = Path(oct_path)
    # Remove double extension (.nii.gz)
    return Path(p.stem).stem



def get_oct_fov_from_metadata(oct_path: str, metadata_root: str) -> dict:
    """
    Given a single OCT volume, load its CSV metadata and compute its FOV bounding box.
    
    Args:
        oct_path (str): path to the OCT .nii.gz file
        metadata_root (str): path to "250529_OMEGA_oct_slo_metadata/SLO"

    Returns:
        dict with bounding box for this OCT volume
    """
    oct_id = extract_oct_id(oct_path)  # e.g. "OMEGA_01_176"
    visit_folder = os.path.join(metadata_root, oct_id)
    bscans_file = os.path.join(visit_folder, f"{oct_id}_bscans.csv")
    scaninfo_file = os.path.join(visit_folder, f"{oct_id}_scaninfo.csv")

    if not os.path.exists(bscans_file):
        raise FileNotFoundError(f"Metadata not found for {oct_id} at {bscans_file}")
    
    if not os.path.exists(scaninfo_file):
        raise FileNotFoundError(f"Metadata not found for {oct_id} at {scaninfo_file}")

    # Load the b-scans metadata
    df = pd.read_csv(bscans_file)
    df_scaninfo = pd.read_csv(scaninfo_file)

    # Compute bounding box for this OCT volume
    start_x = df["StartX"].min()
    end_x   = df["EndX"].max()
    start_y = df["StartY"].min()
    end_y   = df["EndY"].max()

    scale_x = df_scaninfo["ScaleX"]
    scale_y = df_scaninfo["Distance"]
    scale_x_slo = df_scaninfo["ScaleXSlo"]
    scale_y_slo = df_scaninfo["ScaleYSlo"]

    fov = {
        "oct_id": oct_id,
        "StartX": start_x,
        "EndX": end_x,
        "StartY": start_y,
        "EndY": end_y,
        "Width": end_x - start_x,
        "Height": end_y - start_y,
        "ScaleX": scale_x,
        "ScaleY": scale_y,
        "ScaleXSlo": scale_x_slo,
        "ScaleYSlo": scale_y_slo,
    }

    return fov



def resample_and_resize(fixed_image_sitk, moving_image_sitk):
    # print('Fixed image shape before resampling: ', fixed_image_sitk.GetSize())
    # print(f'Fixed image pixel range before resampling: {fixed_image_np.min()}, {fixed_image_np.max()}')

    # print('Moving image shape before resampling: ', moving_image_sitk.GetSize())
    # print(f'Moving image pixel range before resampling: {moving_image_np.min()}, {moving_image_np.max()}')

    print('Fixed image voxel spacing before resampling: ', fixed_image_sitk.GetSpacing())
    print('Moving image voxel spacing before resampling: ', moving_image_sitk.GetSpacing())

    # Resample fixed and moving volumes to the target spacing and resize them to the target shape [512, 496, 192]
    fixed_resampled = resample_image(fixed_image_sitk, target_spacing)
    moving_resampled = resample_image(moving_image_sitk, target_spacing)

    # print('Fixed image voxel spacing after resampling: ', fixed_resampled.GetSpacing())
    # print('Moving image voxel spacing after resampling: ', moving_resampled.GetSpacing())

    # print('Fixed image shape after resampling: ', fixed_resampled.GetSize())

    # print('Moving image shape before resampling: ', moving_resampled.GetSize())

    fixed_resampled_and_resized = resize_volume(fixed_resampled, target_shape)
    moving_resampled_and_resized = resize_volume(moving_resampled, target_shape)

    # print('Fixed shape after resizing: ', fixed_resampled_and_resized.GetSize())
    # print('Moving shape after resizing: ', moving_resampled_and_resized.GetSize())

    return fixed_resampled_and_resized, moving_resampled_and_resized


def register_slo_to_oct_enface(fixed_image_path, moving_image_path, baseline_fov, patient, eye, followup_visit, method):
    start = time.time()

    StartX = baseline_fov['StartX']
    EndX = baseline_fov['EndX']
    StartY = baseline_fov['StartY']
    EndY = baseline_fov['EndY']

    ScaleX = baseline_fov['ScaleX'][0]
    ScaleY = baseline_fov['ScaleY'][0]
    ScaleXSlo = baseline_fov['ScaleXSlo'][0]
    ScaleYSlo = baseline_fov['ScaleYSlo'][0]

    print('StartX: ', StartX)
    print('EndX: ', EndX)
    print('StartY: ', StartY)
    print('EndY: ', EndY)
    print('ScaleX: ', ScaleX)
    print('ScaleY: ', ScaleY)
    print('ScaleXSlo: ', ScaleXSlo)
    print('ScaleYSlo: ', ScaleYSlo)

    # Try to compute the en-face projection of the whole 3D OCT volume
    baseline_oct_volume_sitk = sitk.ReadImage(fixed_image_path)
    
    baseline_oct_volume_np = sitk.GetArrayFromImage(baseline_oct_volume_sitk)

    # Compute En-Face projection of the OCT volume
    enface_baseline_oct_np = np.mean(baseline_oct_volume_np[:, 170:320, :], axis=1)
    enface_baseline_oct_sitk = sitk.GetImageFromArray(enface_baseline_oct_np.astype(np.float32))

    print('OCT volume sitk metadata')
    print(' Size (cols, rows):', baseline_oct_volume_sitk.GetSize())
    print(' Spacing (x, y, z):', baseline_oct_volume_sitk.GetSpacing())
    print(' Origin:', baseline_oct_volume_sitk.GetOrigin())
    print(' Direction:', baseline_oct_volume_sitk.GetDirection())

    print('Enface sitk metadata:')
    print(' Size (cols, rows): ', enface_baseline_oct_sitk.GetSize())
    print(' Spacing (x, z):', enface_baseline_oct_sitk.GetSpacing())
    print(' Origin:', enface_baseline_oct_sitk.GetOrigin())
    print(' Direction:', enface_baseline_oct_sitk.GetDirection())

    # Vertically flip en-face projection to match it with the SLO orientation
    enface_baseline_oct_sitk = sitk.Flip(enface_baseline_oct_sitk, [False, True])

    # Set spacing and origin according to the physical link with the SLO (metadata from the CSV files) 
    enface_baseline_oct_sitk.SetSpacing([(EndX-StartX)/enface_baseline_oct_sitk.GetSize()[0], (EndY-StartY)/enface_baseline_oct_sitk.GetSize()[1]])
    enface_baseline_oct_sitk.SetOrigin([StartX, StartY])
    enface_baseline_oct_sitk.SetDirection([1.0, 0.0, 0.0, 1.0])  # 2x2 identity

    # Rescale final enface baseline projection between 0 and 1
    enface_baseline_oct_sitk = sitk.RescaleIntensity(enface_baseline_oct_sitk, outputMinimum=0, outputMaximum=1)

    # Load moving image --> SLO at the same visit as the OCT volume
    moving_slo_sitk = sitk.ReadImage(moving_image_path)
    moving_slo_sitk.SetSpacing([ScaleXSlo, ScaleYSlo])
    print('Slo sitk spacing: ', moving_slo_sitk.GetSpacing())
    # print(f'Slo original spacing in the CSV: {ScaleXSlo}, {ScaleYSlo}')

    # Rescale between 0 and 1
    moving_slo_sitk = sitk.RescaleIntensity(moving_slo_sitk, outputMinimum=0, outputMaximum=1)

    # Resample SLO to the en-face OCT projection physical space and size
    moving_slo_resampled = sitk.Resample(moving_slo_sitk, enface_baseline_oct_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, moving_slo_sitk.GetPixelID())

    print(f'Slo resampled shape: {moving_slo_resampled.GetSize()}')
    print(f'Slo resampled spacing: {moving_slo_resampled.GetSpacing()}')

    # REGISTRATION SLO --> OCT en-face projection

    # set the used data type
    dtype = torch.float32
    # set the device for the computaion to CPU
    device = torch.device("cuda")

    fixed_image_al = al.create_tensor_image_from_itk_image(enface_baseline_oct_sitk, dtype=dtype, device=device)
    moving_image_al = al.create_tensor_image_from_itk_image(moving_slo_resampled, dtype=dtype, device=device)
    print(f"Fixed image airlab shape: {fixed_image_al.image.shape}, Moving image airlab shape: {moving_image_al.image.shape}")

    print("Fixed image range after normalization: ", fixed_image_al.image.min(), fixed_image_al.image.max())
    print("Moving image range after normalization: ", moving_image_al.image.min(), moving_image_al.image.max())

    # convert intensities so that the object intensities are 1 and the background 0. This is important in order to calculate the center of mass of the object
    fixed_image_al.image = 1 - fixed_image_al.image
    moving_image_al.image = 1 - moving_image_al.image

    registration = al.PairwiseRegistration(verbose=True)

    # define the transformation
    transformation = al.transformation.pairwise.RigidTransformation(moving_image_al, opt_cm=True)
    transformation.init_translation(fixed_image_al)

    registration.set_transformation(transformation)

    # choose the Mutual Information as image loss
    image_loss = al.loss.pairwise.MI(fixed_image_al, moving_image_al)

    registration.set_image_loss([image_loss])

    # define the optimizer
    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.01, amsgrad=True)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(500)

    # start the registration
    registration.start()

    # set the intensities back to the original for the visualisation
    fixed_image_al.image = 1 - fixed_image_al.image
    moving_image_al.image = 1 - moving_image_al.image

    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image_al, displacement)

    # cv2.imwrite('/home/simone.sarrocco/OMEGA_study/scripts/visualisation/warped_image.png', warped_image.image.squeeze().cpu().numpy()*255)
    end = time.time()

    print("=================================================================")

    print("Registration done in:", end - start, "s")

    print('Warped image shape: ', warped_image.image.shape)
    end = time.time()
    print(f'Time for registration: {end-start}')
    
    # compute metrics
    start = time.time()
    psnr, ssim, mse, baseline_psnr, baseline_ssim, baseline_mse = compute_metrics(fixed_image_al.image.squeeze(), warped_image.image.squeeze(), moving_image_al.image.squeeze(), device, len(fixed_image_al.image.squeeze().shape))
    end = time.time()
    print(f'Time to compute metrics: {end-start}')
    # save warped image

    out_dir = os.path.join(experiment_path, patient, eye, followup_visit, method, 'slo_registration')
    os.makedirs(out_dir, exist_ok=True)

    warped_itk = warped_image.itk()
    print('Warped itk shape: ', warped_itk.GetSize())
    transpose = sitk.PermuteAxesImageFilter()
    transpose.SetOrder([1, 0])
    warped_itk_transposed = transpose.Execute(warped_itk)
    # warped_itk_flipped = sitk.Flip(warped_itk_transposed, [False, True])
    print('Warped itk transposed shape: ', warped_itk_transposed.GetSize())

    start = time.time()

    # Images to plot

    fixed_np = fixed_image_al.image.squeeze().cpu().numpy()
    moving_np = moving_image_al.image.squeeze().cpu().numpy()
    warped_np = warped_image.image.squeeze().cpu().numpy()

    print('Fixed_np shape: ', fixed_np.shape)
    print('Moving_np shape: ', moving_np.shape)
    print('Warped_np shape: ', warped_np.shape)

    # Flip back the fixed en-face OCT projection
    # fixed_np = np.flip(fixed_np, 0)

    # Extract rigid params in (theta, tx, tz) with pixel translations
    # theta_rad, tx_pix, tz_pix = extract_rigid_params_from_airlab(transformation)

    # Plot Fixed, Moving and Warped images for a visual evaluation
    # fixed image
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im1 = axes[0].imshow(fixed_np, cmap='gray')
    axes[0].axis("off")
    axes[0].set_title("Fixed En-Face OCT Image")
    # moving image
    im2 = axes[1].imshow(moving_np, cmap='gray')
    axes[1].axis("off")
    axes[1].set_title('Moving SLO Image')
    # warped image
    im3 = axes[2].imshow(warped_np, cmap='gray')
    axes[2].axis("off")
    axes[2].set_title('Warped Moving SLO Image')

    outfile = os.path.join(out_dir, f'fixed_moving_warped_slo_to_enface_oct.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Create a checkerboard
    checker = sitk.CheckerBoard(sitk.GetImageFromArray(fixed_np), sitk.GetImageFromArray(warped_np), [8,8])  # [x_tiles, y_tiles]

    # Convert to numpy for visualization
    checker_np = sitk.GetArrayFromImage(checker)  # shape: [y, x]

    plt.imshow(checker_np, cmap='gray')
    plt.title("Checkerboard Overlay")
    outfile = os.path.join(out_dir, f'slo_to_oct_enface_checkerboard_{method}_registration.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()

    # del fixed_image_al, moving_image_al, warped_image, fixed_resampled_sitk, moving_resampled_sitk
    torch.cuda.empty_cache()
    gc.collect()

    return psnr, baseline_psnr, ssim, baseline_ssim, mse, baseline_mse, warped_itk_transposed # , theta_rad, tx_pix, tz_pix


def register_oct_to_oct_baseline_enface(fixed_image_path, moving_image_path, baseline_fov, patient, eye, followup_visit, method):
    start = time.time()

    StartX = baseline_fov['StartX']
    EndX = baseline_fov['EndX']
    StartY = baseline_fov['StartY']
    EndY = baseline_fov['EndY']

    # Try to compute the en-face projection of the whole BASELINE 3D OCT volume
    baseline_oct_volume_sitk = sitk.ReadImage(fixed_image_path)
    baseline_oct_volume_np = sitk.GetArrayFromImage(baseline_oct_volume_sitk)

    print('Baseline OCT Volume shape before computing en-face projection: ', baseline_oct_volume_np.shape)

    # Compute En-Face projection of the OCT volume
    enface_baseline_oct_np = np.mean(baseline_oct_volume_np[:, 170:320, :], axis=1)
    enface_baseline_oct_sitk = sitk.GetImageFromArray(enface_baseline_oct_np.astype(np.float32))

    print('BASELINE OCT volume sitk metadata')
    print(' Size (cols, rows):', baseline_oct_volume_sitk.GetSize())
    print(' Spacing (x, y, z):', baseline_oct_volume_sitk.GetSpacing())
    print(' Origin:', baseline_oct_volume_sitk.GetOrigin())
    print(' Direction:', baseline_oct_volume_sitk.GetDirection())

    print('BASELINE Enface sitk metadata:')
    print(' Size (cols, rows): ', enface_baseline_oct_sitk.GetSize())
    print(' Spacing (x, z):', enface_baseline_oct_sitk.GetSpacing())
    print(' Origin:', enface_baseline_oct_sitk.GetOrigin())
    print(' Direction:', enface_baseline_oct_sitk.GetDirection())

    # Set spacing and origin according to the physical link with the SLO (metadata from the CSV files) 
    enface_baseline_oct_sitk.SetSpacing([(EndX-StartX)/enface_baseline_oct_sitk.GetSize()[0], (EndY-StartY)/enface_baseline_oct_sitk.GetSize()[1]])
    enface_baseline_oct_sitk.SetOrigin([StartX, StartY])
    enface_baseline_oct_sitk.SetDirection([1.0, 0.0, 0.0, 1.0])  # 2x2 identity

    # Vertically flip en-face projection to match it with the SLO orientation
    enface_baseline_oct_sitk = sitk.Flip(enface_baseline_oct_sitk, [False, True])

    # Rescale to [0,1]
    enface_baseline_oct_sitk = sitk.RescaleIntensity(enface_baseline_oct_sitk, outputMinimum=0, outputMaximum=1)

    # Try to compute the en-face projection of the whole FOLLOW-UP 3D OCT volume
    moving_oct_volume_sitk = sitk.ReadImage(moving_image_path)
    moving_oct_volume_np = sitk.GetArrayFromImage(moving_oct_volume_sitk)

    # Compute En-Face projection of the MOVING OCT volume
    enface_moving_oct_np = np.mean(moving_oct_volume_np[:, 170:320, :], axis=1)
    enface_moving_oct_sitk = sitk.GetImageFromArray(enface_moving_oct_np.astype(np.float32))

    print('MOVING OCT volume sitk metadata')
    print(' Size (cols, rows):', moving_oct_volume_sitk.GetSize())
    print(' Spacing (x, y, z):', moving_oct_volume_sitk.GetSpacing())
    print(' Origin:', moving_oct_volume_sitk.GetOrigin())
    print(' Direction:', moving_oct_volume_sitk.GetDirection())

    print('MOVING Enface sitk metadata:')
    print(' Size (cols, rows): ', enface_moving_oct_sitk.GetSize())
    print(' Spacing (x, z):', enface_moving_oct_sitk.GetSpacing())
    print(' Origin:', enface_moving_oct_sitk.GetOrigin())
    print(' Direction:', enface_moving_oct_sitk.GetDirection())

    # Set spacing and origin of the MOVING ENFACE OCT according to the physical link with the SLO (metadata from the CSV files) 
    enface_moving_oct_sitk.SetSpacing([(EndX-StartX)/enface_moving_oct_sitk.GetSize()[0], (EndY-StartY)/enface_moving_oct_sitk.GetSize()[1]])
    enface_moving_oct_sitk.SetOrigin([StartX, StartY])
    enface_moving_oct_sitk.SetDirection([1.0, 0.0, 0.0, 1.0])  # 2x2 identity

    # Vertically flip en-face projection
    enface_moving_oct_sitk = sitk.Flip(enface_moving_oct_sitk, [False, True])

    # Rescale to [0,1]
    enface_moving_oct_sitk = sitk.RescaleIntensity(enface_moving_oct_sitk, outputMinimum=0, outputMaximum=1)

    # Resample En-Face follow-up to the en-face OCT baseline physical space and size
    enface_moving_oct_resampled = sitk.Resample(enface_moving_oct_sitk, enface_baseline_oct_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, enface_moving_oct_sitk.GetPixelID())

    print(f'En-face moving OCT resampled shape: {enface_moving_oct_resampled.GetSize()}')
    print(f'En-face moving OCT resampled spacing: {enface_moving_oct_resampled.GetSpacing()}')

    # set the used data type
    dtype = torch.float32
    # set the device for the computaion to CPU
    device = torch.device("cuda")

    # REGISTRATION en-face follow-up --> OCT en-face projection
    fixed_image_al = al.create_tensor_image_from_itk_image(enface_baseline_oct_sitk, dtype=dtype, device=device)
    moving_image_al = al.create_tensor_image_from_itk_image(enface_moving_oct_resampled, dtype=dtype, device=device)
    print(f"Fixed image airlab shape: {fixed_image_al.image.shape}, Moving image airlab shape: {moving_image_al.image.shape}")

    # fixed_image_al, moving_image_al = al.utils.normalize_images(fixed_image_al, moving_image_al)
    # second_moving_image_al = (second_moving_image_al - second_moving_image_al.image.min()) - (second_moving_image_al.image.max() - second_moving_image_al.image.min())
    print("Fixed image range after normalization: ", fixed_image_al.image.min(), fixed_image_al.image.max())
    print("Moving image range after normalization: ", moving_image_al.image.min(), moving_image_al.image.max())

    # convert intensities so that the object intensities are 1 and the background 0. This is important in order to calculate the center of mass of the object
    # fixed_image_al.image = 1 - fixed_image_al.image
    # moving_image_al.image = 1 - moving_image_al.image

    registration = al.PairwiseRegistration(verbose=True)

    # define the transformation
    transformation = al.transformation.pairwise.RigidTransformation(moving_image_al, opt_cm=True)
    transformation.init_translation(fixed_image_al)

    registration.set_transformation(transformation)

    # choose the MSE as image loss
    image_loss = al.loss.pairwise.MSE(fixed_image_al, moving_image_al)

    registration.set_image_loss([image_loss])

    # define the optimizer
    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.01, amsgrad=True)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(500)

    # start the registration
    registration.start()

    # set the intensities back to the original for the visualisation
    # fixed_image_al.image = 1 - fixed_image_al.image
    # moving_image_al.image = 1 - moving_image_al.image

    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image_al, displacement)

    # cv2.imwrite('/home/simone.sarrocco/OMEGA_study/scripts/visualisation/warped_image.png', warped_image.image.squeeze().cpu().numpy()*255)
    end = time.time()

    print("=================================================================")

    print("Registration done in:", end - start, "s")

    print('Warped image shape: ', warped_image.image.shape)
    end = time.time()
    print(f'Time for registration: {end-start}')
    
    # compute metrics
    start = time.time()
    psnr, ssim, mse, baseline_psnr, baseline_ssim, baseline_mse = compute_metrics(fixed_image_al.image.squeeze(), warped_image.image.squeeze(), moving_image_al.image.squeeze(), device, len(fixed_image_al.image.squeeze().shape))
    end = time.time()
    print(f'Time to compute metrics: {end-start}')
    # save warped image

    out_dir = os.path.join(experiment_path, patient, eye, followup_visit, method, 'oct_registration')
    os.makedirs(out_dir, exist_ok=True)

    warped_itk = warped_image.itk()
    print('Warped itk shape: ', warped_itk.GetSize())
    transpose = sitk.PermuteAxesImageFilter()
    transpose.SetOrder([1, 0])
    warped_itk_transposed = transpose.Execute(warped_itk)
    print('Warped itk transposed shape: ', warped_itk_transposed.GetSize())

    start = time.time()

    # Images to plot
    fixed_np = fixed_image_al.image.squeeze().cpu().numpy()
    moving_np = moving_image_al.image.squeeze().cpu().numpy()
    warped_np = warped_image.image.squeeze().cpu().numpy()

    print('Fixed_np shape: ', fixed_np.shape)
    print('Moving_np shape: ', moving_np.shape)
    print('Warped_np shape: ', warped_np.shape)

    # Flip back all images
    # fixed_np = np.flip(fixed_np, 0)
    # moving_np = np.flip(moving_np, 0)
    # warped_np = np.flip(warped_np, 0)

    # Plot Fixed, Moving and Warped images for a visual evaluation
    # fixed image
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im1 = axes[0].imshow(fixed_np, cmap='gray')
    axes[0].axis("off")
    axes[0].set_title("Fixed En-Face Image")
    # moving image
    im2 = axes[1].imshow(moving_np, cmap='gray')
    axes[1].axis("off")
    axes[1].set_title('Moving En-Face Image')
    # warped image
    im3 = axes[2].imshow(warped_np, cmap='gray')
    axes[2].axis("off")
    axes[2].set_title('Warped Moving En-Face Image')

    outfile = os.path.join(out_dir, f'fixed_moving_warped_oct_to_oct_enface.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Create a checkerboard
    checker = sitk.CheckerBoard(sitk.GetImageFromArray(fixed_np), sitk.GetImageFromArray(warped_np), [8,8])  # [x_tiles, y_tiles]

    # Convert to numpy for visualization
    checker_np = sitk.GetArrayFromImage(checker)  # shape: [y, x]

    plt.imshow(checker_np, cmap='gray')
    plt.title("Checkerboard Overlay")
    outfile = os.path.join(out_dir, f'oct_to_oct_enface_checkerboard_{method}_registration.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()

    # del fixed_image_al, moving_image_al, warped_image, fixed_resampled_sitk, moving_resampled_sitk
    torch.cuda.empty_cache()
    gc.collect()

    return psnr, baseline_psnr, ssim, baseline_ssim, mse, baseline_mse, warped_itk_transposed # , theta_rad, tx_pix, tz_pix


def register_oct_to_oct_baseline_volume(fixed_image_path, moving_image_path, baseline_fov, patient, eye, followup_visit, method):
    start = time.time()

    # Read BASELINE 3D OCT volume
    baseline_oct_volume_sitk = sitk.ReadImage(fixed_image_path)

    # Rescale to [0,1]
    baseline_oct_volume_sitk = sitk.RescaleIntensity(baseline_oct_volume_sitk, outputMinimum=0, outputMaximum=1)

    # Read FOLLOW-UP 3D OCT volume
    moving_oct_volume_sitk = sitk.ReadImage(moving_image_path)

    # Rescale to [0,1]
    moving_oct_volume_sitk = sitk.RescaleIntensity(moving_oct_volume_sitk, outputMinimum=0, outputMaximum=1)

    # Resample Follow-up OCT volume to the physical size and spacing of the baseline OCT volume
    moving_oct_resampled = sitk.Resample(moving_oct_volume_sitk, baseline_oct_volume_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, moving_oct_volume_sitk.GetPixelID())

    print(f'Moving OCT resampled shape: {moving_oct_resampled.GetSize()}')
    print(f'Moving OCT resampled spacing: {moving_oct_resampled.GetSpacing()}')

    # set the used data type
    dtype = torch.float32
    # set the device for the computaion to CPU
    device = torch.device("cuda")

    # REGISTRATION en-face follow-up --> OCT en-face projection
    fixed_image_al = al.create_tensor_image_from_itk_image(baseline_oct_volume_sitk, dtype=dtype, device=device)
    moving_image_al = al.create_tensor_image_from_itk_image(moving_oct_resampled, dtype=dtype, device=device)
    print(f"Fixed image airlab shape: {fixed_image_al.image.shape}, Moving image airlab shape: {moving_image_al.image.shape}")

    # fixed_image_al, moving_image_al = al.utils.normalize_images(fixed_image_al, moving_image_al)
    # second_moving_image_al = (second_moving_image_al - second_moving_image_al.image.min()) - (second_moving_image_al.image.max() - second_moving_image_al.image.min())
    print("Fixed image range after normalization: ", fixed_image_al.image.min(), fixed_image_al.image.max())
    print("Moving image range after normalization: ", moving_image_al.image.min(), moving_image_al.image.max())

    # convert intensities so that the object intensities are 1 and the background 0. This is important in order to calculate the center of mass of the object
    # fixed_image_al.image = 1 - fixed_image_al.image
    # moving_image_al.image = 1 - moving_image_al.image

    registration = al.PairwiseRegistration(verbose=True)

    # define the transformation
    transformation = al.transformation.pairwise.RigidTransformation(moving_image_al, opt_cm=True)
    transformation.init_translation(fixed_image_al)

    registration.set_transformation(transformation)

    # choose the MSE as image loss
    image_loss = al.loss.pairwise.MSE(fixed_image_al, moving_image_al)

    registration.set_image_loss([image_loss])

    # define the optimizer
    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.1, amsgrad=True)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(500)

    # start the registration
    registration.start()

    # set the intensities back to the original for the visualisation
    # fixed_image_al.image = 1 - fixed_image_al.image
    # moving_image_al.image = 1 - moving_image_al.image

    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image_al, displacement)

    # cv2.imwrite('/home/simone.sarrocco/OMEGA_study/scripts/visualisation/warped_image.png', warped_image.image.squeeze().cpu().numpy()*255)
    end = time.time()

    print("=================================================================")

    print("Registration done in:", end - start, "s")

    print('Warped image shape: ', warped_image.image.shape)
    
    # compute metrics
    start = time.time()
    psnr, ssim, mse, baseline_psnr, baseline_ssim, baseline_mse = compute_metrics(fixed_image_al.image.squeeze(), warped_image.image.squeeze(), moving_image_al.image.squeeze(), device, len(fixed_image_al.image.squeeze().shape))
    end = time.time()
    print(f'Time to compute metrics: {end-start}')
    # save warped image

    out_dir = os.path.join(experiment_path, patient, eye, followup_visit, method, '3doct_registration')
    os.makedirs(out_dir, exist_ok=True)

    warped_itk = warped_image.itk()
    print('Warped itk shape: ', warped_itk.GetSize())
    transpose = sitk.PermuteAxesImageFilter()
    transpose.SetOrder([2, 1, 0])
    warped_itk_transposed = transpose.Execute(warped_itk)
    print('Warped itk transposed shape: ', warped_itk_transposed.GetSize())

    start = time.time()

    # Images to plot
    fixed_np = fixed_image_al.image.squeeze().cpu().numpy()
    moving_np = moving_image_al.image.squeeze().cpu().numpy()
    warped_np = warped_image.image.squeeze().cpu().numpy()

    print('Fixed_np shape: ', fixed_np.shape)
    print('Moving_np shape: ', moving_np.shape)
    print('Warped_np shape: ', warped_np.shape)

    middle_slice = fixed_np.shape[0] // 2
    # Plot Fixed, Moving and Warped images for a visual evaluation
    # fixed image
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im1 = axes[0].imshow(fixed_np[middle_slice, :, :], cmap='gray')
    axes[0].axis("off")
    axes[0].set_title("Fixed Middle Slice Image")
    # moving image
    im2 = axes[1].imshow(moving_np[middle_slice, :, :], cmap='gray')
    axes[1].axis("off")
    axes[1].set_title('Moving Middle Slice Image')
    # warped image
    im3 = axes[2].imshow(warped_np[middle_slice, :, :], cmap='gray')
    axes[2].axis("off")
    axes[2].set_title('Warped Moving Middle Image')

    outfile = os.path.join(out_dir, f'fixed_moving_warped_oct_to_oct_3d.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Create a checkerboard
    checker = sitk.CheckerBoard(sitk.GetImageFromArray(fixed_np), sitk.GetImageFromArray(warped_np), [8,8,1])  # [x_tiles, y_tiles, z_tiles]

    # Convert to numpy for visualization
    checker_np = sitk.GetArrayFromImage(checker)  # shape: [z, y, x]

    # Show a middle slice
    slice_idx = checker_np.shape[0] // 2
    plt.imshow(checker_np[slice_idx], cmap='gray')
    plt.title("Checkerboard Overlay")
    outfile = os.path.join(out_dir, f'oct_to_oct_volume_slice_checkerboard_{method}_registration.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()

    # del fixed_image_al, moving_image_al, warped_image, fixed_resampled_sitk, moving_resampled_sitk
    torch.cuda.empty_cache()
    gc.collect()

    return psnr, baseline_psnr, ssim, baseline_ssim, mse, baseline_mse, warped_itk_transposed # , theta_rad, tx_pix, tz_pix


def main():
    # Create a datalist from the filepath
    input_path = Path(config["input_path"])
    datalist_oct = DataList.from_lakefs(data_config=config["data"], lakefs_config=lakefs_config, filepath=str(input_path), include_root=True, image_modality='Spectralis_oct')
    datalist_slo = DataList.from_lakefs(data_config=config['data'], lakefs_config=lakefs_config, filepath=str(input_path), include_root=True, image_modality='Spectralis_slo')
    # datalist_faf = DataList.from_lakefs(data_config=config["data"], lakefs_config=lakefs_config, filepath=str(input_path), include_root=True, image_modality='Spectralis_faf')
    data_list_oct = datalist_oct.data
    data_list_slo = datalist_slo.data
    # data_list_faf = datalist_faf.data

    print("Data list OCT len: ", len(data_list_oct['full_dataset']))
    print("Data list SLO len: ", len(data_list_slo['full_dataset']))

    # load_image_from_lakefs = LoadLakeFSImaged(lakefs_loader=lakefs_loader, image_only=False, keys=['image'], reader="ITKReader")

    mse_all_3d, psnr_all_3d, ssim_all_3d = [], [], []
    baseline_mse_all_3d, baseline_psnr_all_3d, baseline_ssim_all_3d = [], [], []

    mse_all_2d, psnr_all_2d, ssim_all_2d = [], [], []
    baseline_mse_all_2d, baseline_psnr_all_2d, baseline_ssim_all_2d = [], [], []
    
    pattern_oct = r"(OMEGA\d{2})/(OD|OS)/(V\d{2})/Spectralis_oct/.*\.nii\.gz"
    pattern_slo = r"(OMEGA\d{2})/(OD|OS)/(V\d{2})/Spectralis_slo/.*\.tiff"
    # pattern_faf = r"(OMEGA\d{2})/(OD|OS)/(V\d{2})/Spectralis_faf/.*\.bmp"

    combined_paths = {}

    for entry in data_list_oct['full_dataset']:
        file_path_oct = entry["image"]
        # if 'OMEGA_04_271.nii.gz' in file_path:
        #    continue
            # volume_dict[(patient_id, eye)][visit_id] = file_path
        m_oct = re.search(pattern_oct, file_path_oct)
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
            combined_paths[patient_id][eye][visit_id]["oct"] = file_path_oct

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
    
    # with open(output_csv, 'a', newline='') as csvfile:
        # writer = csv.writer(csvfile)
        # writer.writerow(['patient','eye','visit','method','MSE_3D', 'BASELINE_MSE_3D', 'PSNR_3D','BASELINE_PSNR_3D', 'SSIM_3D', 'BASELINE_SSIM_3D', 'MSE_2D', 'BASELINE_MSE_2D', 'PSNR_2D', 'BASELINE_PSNR_2D', 'SSIM_2D', 'BASELINE_SSIM_2D'])
    for patient_id, eyes in combined_paths.items():
        for eye, visits in eyes.items():
            baseline = visits.get("V01")
            if not baseline or "oct" not in baseline or "slo" not in baseline:
                print(f"No baseline (V01) for {patient_id} {eye}, skipping.")
                continue

            baseline_oct_path = baseline["oct"]
            baseline_slo_path = baseline["slo"]

            baseline_oct_id = extract_oct_id(baseline_oct_path)
            baseline_fov = get_oct_fov_from_metadata(oct_path=baseline_oct_id, metadata_root='/home/simone.sarrocco/OMEGA_study/data/250529_OMEGA_oct_slo_metadata/SLO')

            for visit_id, data in visits.items():
                if visit_id == "V01":
                    # REGISTER BASELINE SLO TO THE BASELINE EN-FACE OCT PROJECTION
                    try:
                        psnr_2d, ssim_2d, mse_2d, baseline_psnr_2d, baseline_ssim_2d, baseline_mse_2d, registered_baseline_slo_itk = register_slo_to_oct_enface(
                            baseline_oct_path, baseline_slo_path, baseline_fov, patient_id, eye, visit_id, "rigid"
                        )
                        save_to_lakefs_cli_singlepart(
                                sitk_image=registered_baseline_slo_itk,
                                patient_id=patient_id,
                                eye=eye,
                                visit_id=visit_id,
                                output_filename=os.path.basename(baseline_slo_path),
                                repo=lakefs_config["data_repository"],
                                branch=lakefs_config["branch"],
                                folder=config["input_path"],
                                file_extension='.dcm',
                            )
                        psnr_all_2d.append(psnr_2d)
                        ssim_all_2d.append(ssim_2d)
                        mse_all_2d.append(mse_2d)
                        baseline_psnr_all_2d.append(baseline_psnr_2d)
                        baseline_ssim_all_2d.append(baseline_ssim_2d)
                        baseline_mse_all_2d.append(baseline_mse_2d)
                    except Exception as e:
                        print(f"Error registering {patient_id} {eye} {visit_id} baseline SLO to baseline en-face OCT projection: {e}")
                        traceback.print_exc()
                        continue
                    continue

                if "oct" not in data or "slo" not in data:
                    print(f"Missing data for {patient_id} {eye} {visit_id}, skipping")
                    continue
                
                moving_oct_path = data["oct"]
                moving_slo_path = data["slo"]

                # REGISTER EACH FOLLOW-UP OCT VOLUME TO THE BASELINE OCT VOLUME
                print(f"Registering {patient_id} {eye} {visit_id} follow-up OCT volume to baseline OCT volume (3D rigid registration)...")
                if (baseline_oct_path and moving_oct_path) and patient_id not in ['OMEGA02', 'OMEGA04', 'OMEGA13', 'OMEGA15', 'OMEGA20', 'OMEGA23', 'OMEGA24', 'OMEGA28', 'OMEGA29']:
                    if patient_id == 'OMEGA06' and eye == 'OD':
                        continue
                    if patient_id == 'OMEGA12' and eye == 'OS':
                        continue
                    try:
                        # Read images
                        psnr_3d, ssim_3d, mse_3d, baseline_psnr_3d, baseline_ssim_3d, baseline_mse_3d, registered_oct_itk = register_oct_to_oct_baseline_volume(
                        baseline_oct_path, moving_oct_path, baseline_fov, patient_id, eye, visit_id, "rigid"
                        )
                        save_to_lakefs_cli_singlepart(
                            sitk_image=registered_oct_itk,
                            patient_id=patient_id,
                            eye=eye,
                            visit_id=visit_id,
                            output_filename=os.path.basename(moving_oct_path),
                            repo=lakefs_config["data_repository"],
                            branch=lakefs_config["branch"],
                            folder=config["input_path"],
                            file_extension='.nii.gz',
                        )
                        psnr_all_3d.append(psnr_3d)
                        ssim_all_3d.append(ssim_3d)
                        mse_all_3d.append(mse_3d)
                        baseline_psnr_all_3d.append(baseline_psnr_3d)
                        baseline_ssim_all_3d.append(baseline_ssim_3d)
                        baseline_mse_all_3d.append(baseline_mse_3d)
                    except Exception as e:
                        print(f"[WARN] Failed to register follow-up OCT to baseline OCT for {patient_id} {eye} {visit_id}: {e}")
                else:
                    print(f"[INFO] OCT for {patient_id} {eye} {visit_id} were already registered by the Heidelberg device. The 3D OCT registration is skipped.")
                
                # REGISTER EACH FOLLOW-UP SLO TO BASELINE EN-FACE OCT PROJECTION
                print(f"Registering {patient_id} {eye} {visit_id} SLO to baseline en-face OCT projection...")
                if moving_slo_path:
                    try:
                        psnr_2d, ssim_2d, mse_2d, baseline_psnr_2d, baseline_ssim_2d, baseline_mse_2d, registered_slo_itk = register_slo_to_oct_enface(
                            baseline_oct_path, moving_slo_path, baseline_fov, patient_id, eye, visit_id, "rigid"
                        )
                        save_to_lakefs_cli_singlepart(
                                sitk_image=registered_slo_itk,
                                patient_id=patient_id,
                                eye=eye,
                                visit_id=visit_id,
                                output_filename=os.path.basename(moving_slo_path),
                                repo=lakefs_config["data_repository"],
                                branch=lakefs_config["branch"],
                                folder=config["input_path"],
                                file_extension='.dcm',
                            )
                        psnr_all_2d.append(psnr_2d)
                        ssim_all_2d.append(ssim_2d)
                        mse_all_2d.append(mse_2d)
                        baseline_psnr_all_2d.append(baseline_psnr_2d)
                        baseline_ssim_all_2d.append(baseline_ssim_2d)
                        baseline_mse_all_2d.append(baseline_mse_2d)
                        # writer.writerow([patient_id, eye, visit_id, 'rigid', mse_3d, baseline_mse_3d, psnr_3d, baseline_psnr_3d, ssim_3d, baseline_ssim_3d, mse_2d, baseline_mse_2d, psnr_2d, baseline_psnr_2d, ssim_2d, baseline_ssim_2d])
                    except Exception as e:
                        print(f"[WARN] Failed to register follow-up SLO to baseline OCT en-face for {patient_id} {eye} {visit_id}: {e}")
                        traceback.print_exc()
                else: 
                    print(f"[INFO] SLO not found for {patient_id} {eye} {visit_id}; skipping SLO-OCT en-face registration.")

    avg_mse_3d, std_mse_3d = np.mean(mse_all_3d), np.std(mse_all_3d)
    avg_psnr_3d, std_psnr_3d = np.mean(psnr_all_3d), np.std(psnr_all_3d)
    avg_ssim_3d, std_ssim_3d = np.mean(ssim_all_3d), np.std(ssim_all_3d)
    baseline_avg_mse_3d, baseline_std_mse_3d = np.mean(baseline_mse_all_3d), np.std(baseline_mse_all_3d)
    baseline_avg_psnr_3d, baseline_std_psnr_3d = np.mean(baseline_psnr_all_3d), np.std(baseline_psnr_all_3d)
    baseline_avg_ssim_3d, baseline_std_ssim_3d = np.mean(baseline_ssim_all_3d), np.std(baseline_ssim_all_3d)

    avg_mse_2d, std_mse_2d = np.mean(mse_all_2d), np.std(mse_all_2d)
    avg_psnr_2d, std_psnr_2d = np.mean(psnr_all_2d), np.std(psnr_all_2d)
    avg_ssim_2d, std_ssim_2d = np.mean(ssim_all_2d), np.std(ssim_all_2d)
    baseline_avg_mse_2d, baseline_std_mse_2d = np.mean(baseline_mse_all_2d), np.std(baseline_mse_all_2d)
    baseline_avg_psnr_2d, baseline_std_psnr_2d = np.mean(baseline_psnr_all_2d), np.std(baseline_psnr_all_2d)
    baseline_avg_ssim_2d, baseline_std_ssim_2d = np.mean(baseline_ssim_all_2d), np.std(baseline_ssim_all_2d)

    print(f"METRICS 3D OCT REGISTRATION: Average MSE: {avg_mse_3d:.4f} ± {std_mse_3d:.4f}, PSNR: {avg_psnr_3d:.4f} ± {std_psnr_3d:.4f}, SSIM: {avg_ssim_3d:.4f} ± {std_ssim_3d:.4f}")
    print(f"METRICS 3D OCT REGISTRATION: Average BASELINE MSE: {baseline_avg_mse_3d:.4f} ± {baseline_std_mse_3d:.4f}, PSNR: {baseline_avg_psnr_3d:.4f} ± {baseline_std_psnr_3d:.4f}, SSIM: {baseline_avg_ssim_3d:.4f} ± {baseline_std_ssim_3d:.4f}")

    print(f"METRICS 2D SLO-Enface OCT REGISTRATION: Average MSE: {avg_mse_2d:.4f} ± {std_mse_2d:.4f}, PSNR: {avg_psnr_2d:.4f} ± {std_psnr_2d:.4f}, SSIM: {avg_ssim_2d:.4f} ± {std_ssim_2d:.4f}")
    print(f"METRICS 2D SLO-Enface OCT REGISTRATION: Average BASELINE MSE: {baseline_avg_mse_2d:.4f} ± {baseline_std_mse_2d:.4f}, PSNR: {baseline_avg_psnr_2d:.4f} ± {baseline_std_psnr_2d:.4f}, SSIM: {baseline_avg_ssim_2d:.4f} ± {baseline_std_ssim_2d:.4f}")

if __name__ == "__main__":
    main()
