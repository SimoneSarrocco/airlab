import matplotlib.pyplot as plt
import nibabel as nib
import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import torch
from monai import transforms
import SimpleITK as sitk
from skimage.filters import threshold_mean
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import airlab as al
import torchvision.transforms as T
import tifffile as tiff


def resample_image(itk_image, out_spacing, out_size=None, is_label=False):
    """
    Resample itk_image to target voxel spacing.
    
    :param itk_image: SimpleITK Image object
    :param out_spacing: list of 3 floats, target spacing in mm (X, Y, Z)
    :param is_label: bool, if True use nearest neighbor interpolation (for masks)
    :return: resampled SimpleITK Image
    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    if out_size is None:
        out_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, out_spacing)
        ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(itk_image)


def main():
    start = time.time()

    output_dir = '/home/simone.sarrocco/OMEGA_study/data/random_images_to_visualize'

    """
    flattened = volume_monai.flatten()
    plt.figure()
    plt.hist(flattened, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title('Voxel Intensity Distribution of 3D OCT Volume')
    plt.xlabel('Intensity')
    plt.ylabel('Number of Voxels')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    root_dir = '/home/simone.sarrocco/OMEGA_study/data/registration_outputs'
    all_histograms = []
    all_labels = []

    # Settings for histogram
    n_bins = 100
    range_min, range_max = 0, 255 

    for patient_id in sorted(os.listdir(root_dir)):
        patient_path = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue
        for eye in ['L', 'R']:
            eye_path = os.path.join(patient_path, eye)
            if not os.path.isdir(eye_path):
                continue
            for visit_id in sorted(os.listdir(eye_path)):
                visit_path = os.path.join(eye_path, visit_id)
                spectralis_oct_path = os.path.join(visit_path, 'Spectralis_oct')
                if not os.path.isdir(spectralis_oct_path):
                    continue
                for file in os.listdir(spectralis_oct_path):
                    if file.endswith('.nii.gz'):
                        nii_path = os.path.join(spectralis_oct_path, file)
                        try:
                            volume = nib.load(nii_path).get_fdata()
                            flattened = volume.flatten()
                            # Optionally clip to remove outliers
                            flattened = np.clip(flattened, np.percentile(flattened, 1), np.percentile(flattened, 99))
                            hist, bin_edges = np.histogram(flattened, bins=n_bins, range=(range_min, range_max), density=True)
                            all_histograms.append((hist, bin_edges))
                            label = f"{patient_id}_{eye}_{visit_id}"
                            all_labels.append(label)
                        except Exception as e:
                            print(f"Failed to load {nii_path}: {e}")

    print(f"Collected {len(all_histograms)} histograms with {len(all_labels)} labels")
    # Plot all histograms
    plt.figure(figsize=(12, 7))
    for hist, bin_edges in all_histograms:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, hist, alpha=0.5)

    plt.title('Voxel Intensity Distributions of All OCT Volumes')
    plt.xlabel('Intensity')
    plt.ylabel('Normalized Frequency')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    """
    transform_load = transforms.LoadImage(image_only=True)
    volume_monai = transform_load('/home/simone.sarrocco/OMEGA_study/data/OMEGA_cleaned/OMEGA01/OD/V01/Spectralis_oct/OMEGA_01_308.nii.gz')

    transforms_channel_first = transforms.EnsureChannelFirst()
    volume_monai = transforms_channel_first(volume_monai)
    print(f"Volume shape after channel first: {volume_monai.shape}")
    print(f"Volume dtype after channel first: {volume_monai.dtype}")
    print(f"Volume range of pixels after channel first: {torch.min(volume_monai)}, {torch.max(volume_monai)}")

    volume_monai = volume_monai.permute(0, 3, 2, 1)
    print(f"Volume shape after permute: {volume_monai.shape}")
    print(f"Volume dtype after permute: {volume_monai.dtype}")
    print(f"Volume range of pixels after permute: {torch.min(volume_monai)}, {torch.max(volume_monai)}")

    # transforms_resize = transforms.Resize(spatial_size=(193, 128, 128), mode='trilinear')
    # volume_monai = transforms_resize(volume_monai)
    # print(f"Volume shape after resize: {volume_monai.shape}")
    # print(f"Volume dtype after resize: {volume_monai.dtype}")
    # print(f"Volume range of pixels after resize: {torch.min(volume_monai)}, {torch.max(volume_monai)}")

    # transforms_crop = transforms.SpatialCrop(roi_start=[48, 98, 127], roi_end = [144, 346, 383])
    # volume_monai = transforms_crop(volume_monai)
    # print(f"Volume shape after crop: {volume_monai.shape}")
    # print(f"Volume dtype after crop: {volume_monai.dtype}")
    # print(f"Volume range of pixels after crop: {torch.min(volume_monai)}, {torch.max(volume_monai)}")

    transforms_scale = transforms.ScaleIntensity(minv=0, maxv=1)
    volume_monai = transforms_scale(volume_monai)
    print(f"Volume shape after scale: {volume_monai.shape}")
    print(f"Volume dtype after scale: {volume_monai.dtype}")
    print(f"Volume range of pixels after scale: {torch.min(volume_monai)}, {torch.max(volume_monai)}")

    ScaleX = 0.01152956020087
    ScaleY = 0.0307455528527498
    ScaleXSlo = 0.01152956020087
    ScaleYSlo = 0.01152956020087

    StartX = 1.47578370571136
    EndX = 7.37891852855682

    EndY = 7.37891852855682
    StartY = 1.47578370571136

    # Try to compute the en-face projection of the whole 3D OCT volume
    volume_sitk = sitk.ReadImage('/home/simone.sarrocco/OMEGA_study/data/OMEGA_cleaned/OMEGA01/OD/V01/Spectralis_oct/OMEGA_01_308.nii.gz')
    
    volume_np = sitk.GetArrayFromImage(volume_sitk)
    enface_np = np.mean(volume_np[:, 170:320, :], axis=1)
    enface_sitk = sitk.GetImageFromArray(enface_np.astype(np.float32))

    print('OCT volume sitk metadata')
    print(' Size (cols, rows):', volume_sitk.GetSize())
    print(' Spacing (x, y, z):', volume_sitk.GetSpacing())
    print(' Origin:', volume_sitk.GetOrigin())
    print(' Direction:', volume_sitk.GetDirection())

    print('Enface sitk metadata:')
    print(' Size (cols, rows): ', enface_sitk.GetSize())
    print(' Spacing (x, z):', enface_sitk.GetSpacing())
    print(' Origin:', enface_sitk.GetOrigin())
    print(' Direction:', enface_sitk.GetDirection())

    enface_sitk.SetSpacing([(EndX-StartX)/512, (EndY-StartY)/193])
    enface_sitk.SetOrigin([StartX, StartY])
    enface_sitk.SetDirection([1.0, 0.0, 0.0, 1.0])  # 2x2 identity

    enface_sitk_flipped = sitk.Flip(enface_sitk, [False, True])

    plt.imshow(sitk.GetArrayFromImage(enface_sitk_flipped), cmap='gray')
    plt.title("En-face projection of 3D OCT volume - Mean")
    plt.axis("off")
    plt.show()
    plt.close()

    slo_image_tiff = tiff.imread('/home/simone.sarrocco/OMEGA_study/data/OMEGA_cleaned/OMEGA01/OD/V04/Spectralis_slo/OMEGA_01_176_SLO.tiff')
    print(f'Slo original shape: {slo_image_tiff.shape}')

    plt.imshow(slo_image_tiff[126:641, 128:640], cmap='gray')
    plt.title('SLO image - OMEGA01 - V01 - Manually Resized')
    # plt.axis('off')
    plt.show()
    plt.close()
    print(f'Shape slo manually resized: {slo_image_tiff[128:640, 126:641].shape}')

    slo_sitk = sitk.ReadImage('/home/simone.sarrocco/OMEGA_study/data/OMEGA_cleaned/OMEGA01/OD/V04/Spectralis_slo/OMEGA_01_176_SLO.tiff')
    slo_sitk.SetSpacing([ScaleXSlo, ScaleYSlo])
    print('Slo sitk spacing: ', slo_sitk.GetSpacing())
    # print(f'Slo original spacing in the CSV: {ScaleXSlo}, {ScaleYSlo}')
    slo_sitk = sitk.RescaleIntensity(slo_sitk, outputMinimum=0, outputMaximum=1)

    reference = sitk.Image([512, 193], slo_sitk.GetPixelID())
    reference.SetSpacing([(EndX-StartX)/512, (EndY-StartY)/193])
    reference.SetOrigin([StartX, StartY])
    reference.SetDirection([1.0, 0.0, 0.0, 1.0])  # 2x2 identity

    slo_resampled = sitk.Resample(slo_sitk, reference, sitk.Transform(), sitk.sitkLinear, 0.0, slo_sitk.GetPixelID())

    print(f'Slo resampled shape: {slo_resampled.GetSize()}')
    print(f'Slo resampled spacing: {slo_resampled.GetSpacing()}')

    plt.imshow(sitk.GetArrayFromImage(slo_resampled), cmap='gray')
    plt.title('SLO image - OMEGA01 - V01 - Resampled')
    # plt.axis('off')
    plt.show()
    plt.close()

    """# 3) Register SLO to OCT en-face (rigid)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                                        minStep=1e-4,
                                                        numberOfIterations=200)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetInitialTransform(sitk.TranslationTransform(2))
    transform = registration.Execute(enface_sitk_flipped, slo_resampled)

    slo_registered = sitk.Resample(slo_resampled, enface_sitk_flipped, transform,
                                sitk.sitkLinear, 0.0)
    
    print(f'Slo registered shape: {slo_registered.GetSize()}')
    print(f'Slo registered spacing: {slo_registered.GetSpacing()}')

    plt.imshow(sitk.GetArrayFromImage(slo_registered), cmap='gray')
    plt.title('SLO image - OMEGA01 - V01 - Registered')
    # plt.axis('off')
    plt.show()
    plt.close()

    print("Done: Generated both high-res registered SLO and downsampled SLO.")"""

    # set the used data type
    dtype = torch.float32
    # set the device for the computaion to CPU
    device = torch.device("cuda")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    # device = th.device("cuda:0")


    # load the image data and normalize to [0, 1]
    # fixed_image = al.read_image_as_tensor('/home/simone.sarrocco/OMEGA_study/scripts/visualisation/OMEGA01_Retina_20210316_000_fundus_0(2).dcm', dtype=dtype, device=device)
    # moving_image = al.read_image_as_tensor('/home/simone.sarrocco/OMEGA_study/scripts/visualisation/enface.png', dtype=dtype, device=device)
    fixed_image = al.create_tensor_image_from_itk_image(enface_sitk_flipped, dtype=dtype, device=device)
    moving_image = al.create_tensor_image_from_itk_image(slo_resampled, dtype=dtype, device=device)

    print('Fixed image shape with airlab: ', fixed_image.image.shape)
    print('Moving image shape with airlab: ', moving_image.image.shape)

    # fixed_image_th = fixed_image.image.squeeze().unsqueeze(0)
    # fixed_image_th = transforms_resize(fixed_image_th)

    # fixed_image_np = fixed_image_th.squeeze().cpu().numpy()
    # fixed_image = al.image_from_numpy(fixed_image_np, pixel_spacing=fixed_image_spacing, image_origin=fixed_image_origin, dtype=dtype, device=device)

    # fixed_image, moving_image = al.utils.normalize_images(fixed_image, moving_image)

    # convert intensities so that the object intensities are 1 and the background 0. This is important in order to
    # calculate the center of mass of the object
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # create pairwise registration object
    registration = al.PairwiseRegistration()

    # choose the affine transformation model
    transformation = al.transformation.pairwise.RigidTransformation(moving_image, opt_cm=True)
    # initialize the translation with the center of mass of the fixed image
    transformation.init_translation(fixed_image)

    registration.set_transformation(transformation)

    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.MI(fixed_image, moving_image)

    registration.set_image_loss([image_loss])

    # choose the Adam optimizer to minimize the objective
    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.01, amsgrad=True)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(100)

    # start the registration
    registration.start()

    # set the intensities back to the original for the visualisation
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image, displacement)

    cv2.imwrite('/home/simone.sarrocco/OMEGA_study/scripts/visualisation/warped_image.png', warped_image.image.squeeze().cpu().numpy()*255)
    end = time.time()

    print("=================================================================")

    print("Registration done in:", end - start, "s")
    print("Result parameters:")
    transformation.print()

    # plot the results
    plt.subplot(131)
    plt.imshow(fixed_image.image.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Fixed Image')
    plt.axis("off")

    plt.subplot(132)
    plt.imshow(moving_image.image.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Moving Image')
    plt.axis("off")

    plt.subplot(133)
    plt.imshow(warped_image.image.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Warped Moving Image')
    plt.axis("off")

    plt.show()
    plt.tight_layout()
    plt.savefig('/home/simone.sarrocco/OMEGA_study/scripts/visualisation/plot.png')
    plt.close()


if __name__ == '__main__':
    main()