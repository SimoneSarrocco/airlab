import itk 
import sys
from itkwidgets import view, compare, checkerboard
import numpy as np
import os
import SimpleITK as sitk
import ants
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import time
import math
import csv
import glob
from monai.metrics.regression import PSNRMetric, SSIMMetric, MSEMetric
from monai import transforms
from nibabel.processing import resample_to_output
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import airlab as al

SSIM = SSIMMetric(spatial_dims=3, reduction='mean', data_range=1.0)
PSNR = PSNRMetric(reduction='mean', max_val=1.0)
MSE = MSEMetric(reduction='mean')
DATA_ROOT = '/home/simone.sarrocco/OMEGA_study/data/OMEGA_cleaned'
OUTPUT_ROOT = '/home/simone.sarrocco/OMEGA_study/data/registration_outputs'
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
output_csv = os.path.join(OUTPUT_ROOT, 'metrics.csv')
# transformations = ['rigid', 'affine', 'bspline']
target_spacing = (0.011422, 0.003872, 0.030460)  # X, Y, Z spacing in mm --> median values on the entire dataset
target_shape = (512, 496, 192)

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

def compute_metrics(fixed_np, warped_np, moving_np):
    """Compute metrics between fixed and warped images."""
    fixed_th = torch.from_numpy(fixed_np).squeeze().unsqueeze(0).unsqueeze(0)
    warped_th = torch.from_numpy(warped_np).squeeze().unsqueeze(0).unsqueeze(0)
    moving_th = torch.from_numpy(moving_np).squeeze().unsqueeze(0).unsqueeze(0)
    assert fixed_th.shape == warped_th.shape == moving_th.shape and len(fixed_th.shape) == 5, "Fixed and warped images must have the same shape of kind (B, C, D, H, W)"
    
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

def resample_and_resize(fixed_image_path, moving_image_path):
    # transform_spacing = transforms.Spacing(pixdim=(0.0115234375, 0.0038716698, 0.030569948186528), mode='bilinear')

    # Respacing with MONAI transforms
    # transforms_load = transforms.LoadImage(reader='NibabelReader', image_only=True, ensure_channel_first=True)
    # transforms_resample = transforms.Spacing(pixdim=target_spacing, mode='bilinear')
    # transforms_resize = transforms.Resize(target_shape, dtype=torch.float64)

    # fixed_nib = nib.load(fixed_image_path)
    # fixed_np = fixed_nib.get_fdata(dtype=np.float64)  # Converting nibabel metadata to image Array

    # Load fixed and moving volumes
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat64)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat64)

    # Transform to ndarray
    fixed_image_np = sitk.GetArrayFromImage(fixed_image)
    moving_image_np = sitk.GetArrayFromImage(moving_image)

    print('Fixed image shape before resampling: ', fixed_image.GetSize())
    print(f'Fixed image pixel range before resampling: {fixed_image_np.min()}, {fixed_image_np.max()}')

    print('Moving image shape before resampling: ', moving_image.GetSize())
    print(f'Moving image pixel range before resampling: {moving_image_np.min()}, {moving_image_np.max()}')

    print('Fixed image voxel spacing before resampling: ', fixed_image.GetSpacing())
    print('Moving image voxel spacing before resampling: ', moving_image.GetSpacing())

    # Resample fixed and moving volumes to the target spacing and resize them to the target shape [512, 496, 192]
    fixed_resampled = resample_image(fixed_image, target_spacing)
    moving_resampled = resample_image(moving_image, target_spacing)

    # Transform to ndarray
    fixed_resampled_np = sitk.GetArrayFromImage(fixed_resampled)
    moving_resampled_np = sitk.GetArrayFromImage(moving_resampled)

    print('Fixed image voxel spacing after resampling: ', fixed_resampled.GetSpacing())
    print('Moving image voxel spacing after resampling: ', moving_resampled.GetSpacing())

    print('Fixed image shape after resampling: ', fixed_resampled.GetSize())
    print(f'Fixed image pixel range before resampling: {fixed_resampled_np.min()}, {fixed_resampled_np.max()}')

    print('Moving image shape before resampling: ', moving_resampled.GetSize())
    print(f'Moving image pixel range before resampling: {moving_resampled_np.min()}, {moving_resampled_np.max()}')

    fixed_resampled_and_resized = resize_volume(fixed_resampled, target_shape)
    moving_resampled_and_resized = resize_volume(moving_resampled, target_shape)

    print('Fixed shape after resizing: ', fixed_resampled_and_resized.GetSize())
    print('Moving shape after resizing: ', moving_resampled_and_resized.GetSize())

    return fixed_resampled_and_resized, moving_resampled_and_resized


def register_and_evaluate(fixed_image_path, moving_image_path, patient, eye, followup_visit, method):
    start = time.time()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fixed_resampled_sitk, moving_resampled_sitk = resample_and_resize(fixed_image_path, moving_image_path)

    # Registration with AirLab
    fixed_image_al = al.Image(fixed_resampled_sitk, dtype=sitk.sitkFloat64, device='cuda')
    moving_image_al = al.Image(moving_resampled_sitk, dtype=sitk.sitkFloat64, device='cuda')
    print(f"Fixed image airlab shape: {fixed_image_al.image.shape}, Moving image airlab shape: {moving_image_al.image.shape}")

    fixed_image_al, moving_image_al = al.utils.normalize_images(fixed_image_al, moving_image_al)
    # second_moving_image_al = (second_moving_image_al - second_moving_image_al.image.min()) - (second_moving_image_al.image.max() - second_moving_image_al.image.min())
    print("Fixed image range after normalization: ", fixed_image_al.image.min(), fixed_image_al.image.max())
    print("Moving image range after normalization: ", moving_image_al.image.min(), moving_image_al.image.max())

    registration = al.PairwiseRegistration(verbose=True)

    # define the transformation
    transformation = al.transformation.pairwise.RigidTransformation(moving_image_al, opt_cm=False)
    transformation.init_translation(fixed_image_al)

    registration.set_transformation(transformation)
    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.MSE(fixed_image_al, moving_image_al)

    registration.set_image_loss([image_loss])

    # define the optimizer
    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.1)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(500)

    # start the registration
    registration.start()

    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image_al, displacement)
    print('Warped image shape: ', warped_image.image.shape)
    # compute metrics
    psnr, ssim, mse, baseline_psnr, baseline_ssim, baseline_mse = compute_metrics(fixed_image_al.image.squeeze().cpu().numpy(), warped_image.image.squeeze().cpu().numpy(), moving_image_al.image.squeeze().cpu().numpy())

    # save warped image
    out_dir = os.path.join(OUTPUT_ROOT, patient, eye, followup_visit, method, 'common_voxel_spacing')
    os.makedirs(out_dir, exist_ok=True)

    warped_itk = warped_image.itk()
    warped_np = sitk.GetArrayFromImage(warped_itk)
    # warped_np = np.transpose(warped_np, (2, 1, 0))
    # warped_itk = sitk.GetImageFromArray(warped_np)
    # print('Warped_itk size: ', warped_itk.GetSize())
    # sitk.WriteImage(warped_itk, os.path.join(out_dir, f'warped_{method}.nii.gz'))

    # nib.save(nib.Nifti1Image(warped_np, fixed_image_al.affine), os.path.join(out_dir, f'warped_{method}_second.nii.gz'))
    if patient == 'OMEGA01':
        # nib.save(nib.Nifti1Image(warped_image.numpy(), affine=fixed_image_al.image.affine), 
        #         os.path.join(out_dir, f'{patient}_{eye}_{followup_visit}.nii.gz'))
        warped_itk = warped_image.itk()
        print('Warped itk shape: ', warped_itk.GetSize())
        warped_np = sitk.GetArrayFromImage(warped_itk)
        warped_np = np.transpose(warped_np, (2, 1, 0))
        warped_transposed = sitk.GetImageFromArray(warped_np)
        sitk.WriteImage(warped_transposed, os.path.join(out_dir, f'warped_{patient}_{eye}_{followup_visit}_{method}.nii.gz'))

    end = time.time()

    # save checkerboard
    # Example slice index (axial middle slice)
    slice_idx = fixed_image_al.image.shape[0] // 2

    # Prepare figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    # Titles for subplots
    titles = ['Fixed', 'Moving', f'Warped ({method})']

    # Images to plot
    fixed_np = fixed_image_al.image.squeeze().cpu().numpy()
    moving_np = moving_image_al.image.squeeze().cpu().numpy()
    warped_np = warped_image.image.squeeze().cpu().numpy()

    print('Fixed_np shape: ', fixed_np.shape)
    print('Moving_np shape: ', moving_np.shape)
    print('Warped_np shape: ', warped_np.shape)

    # Plot XY plane --> frontal image
    images = [fixed_np[slice_idx, :, :],
            moving_np[slice_idx, :, :],
            warped_np[slice_idx, :, :]]

    cmaps = ['gray', 'gray', 'gray']

    # Plot
    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        if title == 'Diff':
            # Add a small colorbar only for the diff
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

    # Save
    outfile = os.path.join(out_dir, f'comparing_frontal_image_{method}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot YZ plane --> lateral image
    slice_idx = fixed_image_al.image.shape[2] // 2
    # Prepare figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    images = [fixed_np[:, :, slice_idx],
            moving_np[:, :, slice_idx],
            warped_np[:, :, slice_idx]]

    # Plot
    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        if title == 'Diff':
            # Add a small colorbar only for the diff
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

    # Save
    outfile = os.path.join(out_dir, f'comparing_lateral_image_{method}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot XZ plane --> from above image
    slice_idx = fixed_image_al.image.shape[1] // 2
    # Prepare figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    images = [fixed_np[:, slice_idx, :],
            moving_np[:, slice_idx, :],
            warped_np[:, slice_idx, :]]

    # Plot
    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        if title == 'Diff':
            # Add a small colorbar only for the diff
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

    # Save
    outfile = os.path.join(out_dir, f'comparing_above_image_{method}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

    # Compute and plot difference maps
    diff_np = np.abs(fixed_np - warped_np)
    diff_baseline_np = np.abs(fixed_np - moving_np)

    fig, axes = plt.subplots(1, 2, figsize=(16,4), constrained_layout=True)
    images = [diff_baseline_np, diff_np]
    titles = ['Fixed-Moving', 'Fixed-Warped']
    cmaps = ['gray', 'gray']
    # Plot
    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        if title == 'Fixed-Warped':
            # Add a small colorbar only for the diff
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

    return psnr, ssim, mse, baseline_psnr, baseline_ssim, baseline_mse

def main():
    corrupted_cases = [
        ('OMEGA04', 'L', 'V02'),
    ]
    #methods = []
    # psnr_means = []
    # ssim_means = []
    # mse_means = []
    # all_metrics = {method: [] for method in transformations}
    mse_all, psnr_all, ssim_all = [], [], []
    baseline_mse_all, baseline_psnr_all, baseline_ssim_all = [], [], []
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['patient','eye','visit','method','MSE', 'BASELINE_MSE', 'PSNR','BASELINE_PSNR', 'SSIM', 'BASELINE_SSIM'])
        for patient in sorted(os.listdir(DATA_ROOT)):
            if not patient.startswith('OMEGA'):
                continue
            print(f"Processing patient: {patient}")
            for eye in ['L','R']:
                eye_dir = os.path.join(DATA_ROOT, patient, eye)
                if not os.path.exists(eye_dir): continue

                visits = sorted(os.listdir(eye_dir))
                if len(visits) < 2: continue

                baseline_path = os.path.join(str(eye_dir), 'V01', 'Spectralis_oct')
                if not os.path.isdir(baseline_path):
                    print(f'Baseline folder missing for {patient} {eye}')

                # robustly pick only the .nii.gz file (exclude .npz)
                baseline_volumes = glob.glob(os.path.join(baseline_path, '*.nii.gz'))
                        
                if len(baseline_volumes) == 0:
                    print(f'No baseline volume found for {patient} {eye}')
                    continue
                elif len(baseline_volumes) > 1:
                    print(f'Multiple baseline volumes found for {patient} {eye}, please check:')
                    print(baseline_volumes)
                    continue

                baseline_volume_path = baseline_volumes[0]

                for followup_visit in ['V02', 'V03', 'V04']:
                    if (patient, eye, followup_visit) in corrupted_cases:
                        print(f'Skipping corrupted volume: {patient} {eye} {followup_visit}')
                        continue
                    visit_path = os.path.join(str(eye_dir), str(followup_visit), 'Spectralis_oct')
                    if not os.path.isdir(visit_path):
                        continue
                            
                    moving_volumes = glob.glob(os.path.join(visit_path, '*.nii.gz'))
                    # moving_volumes = [v for v in moving_volumes if not v.endswith('.npz')]

                    if len(moving_volumes) == 0:
                        print(f'No moving volume found for {patient} {eye} {followup_visit}')
                        continue
                    elif len(moving_volumes) > 1:
                        print(f'Multiple moving volumes found for {patient} {eye} {followup_visit}, please check:')
                        print(moving_volumes)
                        continue

                    moving_volume_path = moving_volumes[0]
                    print(f'Registering {moving_volume_path} to baseline')
                    try:
                        psnr_fixed_warped, ssim_fixed_warped, mse_fixed_warped, psnr_fixed_moving, ssim_fixed_moving, mse_fixed_moving = register_and_evaluate(
                            baseline_volume_path, moving_volume_path, patient, eye, followup_visit, "rigid"
                            )
                        mse_all.append(mse_fixed_warped.item())
                        psnr_all.append(psnr_fixed_warped.item())
                        ssim_all.append(ssim_fixed_warped.item())
                        baseline_mse_all.append(mse_fixed_moving.item())
                        baseline_psnr_all.append(psnr_fixed_moving.item())
                        baseline_ssim_all.append(ssim_fixed_moving.item())
                        writer.writerow([patient, eye, followup_visit, "rigid", f"{mse_fixed_warped.item():.4f}", f"{mse_fixed_moving.item():.4f}", f"{psnr_fixed_warped.item():.4f}", f"{psnr_fixed_moving.item():.4f}", f"{ssim_fixed_warped.item():.4f}", f"{ssim_fixed_moving.item():.4f}"])
                        csvfile.flush()
                    except Exception as e:
                        print(f"Failed on {patient} {eye} {followup_visit}: {e}")
                        traceback.print_exc()

    avg_mse, std_mse = np.mean(mse_all), np.std(mse_all)
    avg_psnr, std_psnr = np.mean(psnr_all), np.std(psnr_all)
    avg_ssim, std_ssim = np.mean(ssim_all), np.std(ssim_all)
    baseline_avg_mse, baseline_std_mse = np.mean(baseline_mse_all), np.std(baseline_mse_all)
    baseline_avg_psnr, baseline_std_psnr = np.mean(baseline_psnr_all), np.std(baseline_psnr_all)
    baseline_avg_ssim, baseline_std_ssim = np.mean(baseline_ssim_all), np.std(baseline_ssim_all)

    print(f"Average MSE: {avg_mse:.4f} ± {std_mse:.4f},, PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}, SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"Average BASELINE MSE: {baseline_avg_mse:.4f} ± {baseline_std_mse:.4f}, PSNR: {baseline_avg_psnr:.4f} ± {baseline_std_psnr:.4f}, SSIM: {baseline_avg_ssim:.4f} ± {baseline_std_ssim:.4f}")

if __name__ == "__main__":
    main()
