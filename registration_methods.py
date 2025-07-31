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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import airlab as al

"""# Load images with itk floats (itk.F). Necessary for elastix
fixed_image = itk.imread('/home/simone.sarrocco/OMEGA_study/data/OMEGA_cleaned/OMEGA06/R/V01/Spectralis_oct/OMEGA_06_295.nii.gz', itk.F)
moving_image = itk.imread('/home/simone.sarrocco/OMEGA_study/data/OMEGA_cleaned/OMEGA06/R/V02/Spectralis_oct/OMEGA_06_264.nii.gz', itk.F)

output_dir = '/home/simone.sarrocco/OMEGA_study/data/exampleoutput/'
os.makedirs(output_dir, exist_ok=True)

parameter_object = itk.ParameterObject.New()
default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid', 3)
parameter_object.AddParameterMap(default_rigid_parameter_map)

param0035_path = '/home/simone.sarrocco/OMEGA_study/data/exampleoutput/Par0035.SPREAD.MI.bs.1.ASGD.txt'
parameter_object.AddParameterFile(param0035_path)

# Call registration function
result_image, result_transform_parameters = itk.elastix_registration_method(
    fixed_image, moving_image,
    parameter_object=parameter_object,
    log_to_console=True,
    output_directory=output_dir)

# Save result
output_result_path = os.path.join(output_dir, 'result_image.nii.gz')
itk.imwrite(result_image, output_result_path)

compare(fixed_image, result_image)
checkerboard(fixed_image, result_image)"""

SSIM = SSIMMetric(spatial_dims=3, reduction='mean', data_range=1.0)
PSNR = PSNRMetric(reduction='mean', max_val=1.0)
MSE = MSEMetric(reduction='mean')
output_csv = os.path.join('outputs', 'metrics.csv')
DATA_ROOT = '/home/simone.sarrocco/OMEGA_study/data/OMEGA_cleaned'
OUTPUT_ROOT = '/home/simone.sarrocco/OMEGA_study/data/registration_outputs'
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
transformations = ['rigid', 'affine', 'bspline']


def normalize_image(image):
    """Normalize the image to the range [0, 1]."""
    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / (image_max - image_min)

def compute_metrics(fixed_np, warped_np):
    """Compute metrics between fixed and warped images."""
    mse = MSE(warped_np, fixed_np)
    psnr = PSNR(warped_np, fixed_np)
    ssim = SSIM(warped_np, fixed_np)
    return psnr, ssim, mse

def register_and_evaluate(fixed_image_path, moving_image_path, patient, eye, followup_visit, method):
    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    fixed_nib = nib.load(fixed_image_path)
    fixed_image = fixed_nib.get_fdata().astype(np.float32)
    fixed_image = torch.from_numpy(fixed_image).to(device)
    fixed_image= normalize_image(fixed_image)
    print("Fixed image shape: ", fixed_image.shape)
    print("Fixed image range: ", fixed_image.min(), fixed_image.max())

    moving_nib = nib.load(moving_image_path)
    moving_image = moving_nib.get_fdata().astype(np.float32)
    moving_image = torch.from_numpy(moving_image).to(device)
    moving_image = normalize_image(moving_image)
    print("Moving image shape: ", moving_image.shape)
    print("Moving image range: ", moving_image.min(), moving_image.max())

    fixed_image = fixed_image.permute(2, 1, 0)
    moving_image = moving_image.permute(2, 1, 0)

    fixed_image_al = al.Image(fixed_image, [193, 496, 512], [1,1,1], [0,0,0])
    moving_image_al = al.Image(moving_image, [193, 496, 512], [1,1,1], [0,0,0])

    registration = al.PairwiseRegistration()

    # Choose transformation
    if method == 'rigid':
        transformation = al.transformation.pairwise.RigidTransformation(moving_image_al, opt_cm=False)
    elif method == 'affine':
        transformation = al.transformation.pairwise.AffineTransformation(moving_image_al, opt_cm=False)
    elif method == 'bspline':
        transformation = al.transformation.pairwise.BsplineTransformation(moving_image_al)
    else:
        raise ValueError("Unknown transformation method: {}".format(method))
    
    transformation.init_translation(fixed_image_al)

    registration.set_transformation(transformation)

    image_loss = al.loss.pairwise.NCC(fixed_image_al, moving_image_al)

    registration.set_image_loss([image_loss])

    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.1)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(500)

    registration.start()

    # fixed_image_al.image = 1 - fixed_image_al.image
    # moving_image_al.image = 1 - moving_image_al.image

    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image_al, displacement)

    # compute metrics
    mse_val, psnr_val, ssim_val = compute_metrics(fixed_image_al.numpy(), warped_image.numpy())

    # save warped image
    warped_nib = nib.Nifti1Image(warped_image.numpy().transpose(2,1,0), affine=fixed_nib.affine)
    out_dir = os.path.join(OUTPUT_ROOT, patient, eye, followup_visit)
    os.makedirs(out_dir, exist_ok=True)
    nib.save(warped_nib, os.path.join(out_dir, f'warped_{method}.nii.gz'))

    end = time.time()

    # save checkerboard
    slice_idx = fixed_image_al.numpy().shape[0] // 2
    plt.figure(figsize=(12,4))
    plt.subplot(131); plt.imshow(fixed_image_al.numpy()[slice_idx,:,:], cmap='gray'); plt.title('Fixed')
    plt.subplot(132); plt.imshow(warped_image.numpy()[slice_idx,:,:], cmap='gray'); plt.title(f'Warped ({method})')
    plt.subplot(133); plt.imshow(np.abs(fixed_image_al.numpy()[slice_idx,:,:] - warped_image.numpy()[slice_idx,:,:]), cmap='hot'); plt.title('Diff')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'checkerboard_{method}.png'))
    plt.close()

    return psnr_val, ssim_val, mse_val, image_loss

def main():
    output_csv = os.path.join('outputs', 'metrics.csv')
    corrupted_cases = [
        ('OMEGA04', 'L', 'V02'),
    ]
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['patient','eye','visit','method','MSE','PSNR','NCC','SSIM'])

        for patient in sorted(os.listdir(DATA_ROOT)):
            for eye in ['L','R']:
                eye_dir = os.path.join(DATA_ROOT, patient, eye)
                if not os.path.exists(eye_dir): continue

                visits = sorted(os.listdir(eye_dir))
                if len(visits) < 2: continue

                baseline_path = os.path.join(eye_dir, 'V01', 'Spectralis_oct')
                if not os.path.isdir(baseline_path):
                    print(f'Baseline folder missing for {patient} {eye}')

                # robustly pick only the .nii.gz file (exclude .npz)
                baseline_volumes = glob.glob(os.path.join(baseline_path, '*.nii')) + \
                           glob.glob(os.path.join(baseline_path, '*.nii.gz'))
                
                if len(baseline_volumes) == 0:
                    print(f'No baseline volume found for {patient} {eye}')
                    continue
                elif len(baseline_volumes) > 1:
                    print(f'⚠ Multiple baseline volumes found for {patient} {eye}, please check:')
                    print(baseline_volumes)
                    continue

                baseline_volume_path = baseline_volumes[0]

                for followup_visit in ['V02', 'V03', 'V04']:
                    if (patient, eye, followup_visit) in corrupted_cases:
                        print(f'Skipping corrupted volume: {patient} {eye} {followup_visit}')
                        continue
                    visit_path = os.path.join(eye_dir, followup_visit, 'Spectralis_oct')
                    if not os.path.isdir(visit_path):
                        continue
                    
                    moving_volumes = glob.glob(os.path.join(visit_path, '*.nii')) + \
                    glob.glob(os.path.join(visit_path, '*.nii.gz'))
                    moving_volumes = [v for v in moving_volumes if not v.endswith('.npz')]

                    if len(moving_volumes) == 0:
                        print(f'No moving volume found for {patient} {eye} {followup_visit}')
                        continue
                    elif len(moving_volumes) > 1:
                        print(f'⚠ Multiple moving volumes found for {patient} {eye} {followup_visit}, please check:')
                        print(moving_volumes)
                        continue

                    moving_volume_path = moving_volumes[0]
                    print(f'Registering {moving_volume_path} to baseline')

                    for method in transformations:
                        print(f"Registering {patient} {eye} {followup_visit} with {method}")
                        try:
                            psnr, ssim, mse, ncc = register_and_evaluate(baseline_volume_path, moving_volume_path, patient, eye, followup_visit, method)
                            writer.writerow([patient, eye, followup_visit, method, mse, psnr, ncc, ssim])
                            csvfile.flush()
                        except Exception as e:
                            print(f"Failed on {patient} {eye} {followup_visit} with {method}: {e}")

if __name__ == "__main__":
    main()
