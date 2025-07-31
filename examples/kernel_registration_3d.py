
# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys, time
import torch as th
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlab as al


def main():
    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    # device = th.device("cuda:0")

    # directory to store results
    tmp_directory = "/tmp/"

    # load the image data and normalize intensities to [0, 1]
    fixed_image = nib.load('/home/simone.sarrocco/OMEGA_study/data/OMEGA_cleaned/OMEGA06/R/V01/Spectralis_oct/OMEGA_06_295.nii.gz')
    fixed_image_numpy = fixed_image.get_fdata().astype(np.float32)
    fixed_image = torch.from_numpy(fixed_image_numpy).to(device)
    # fixed_image_torch = (fixed_image_torch - fixed_image_torch.min()) / (fixed_image_torch.max() - fixed_image_torch.min())
    print("Fixed image shape: ", fixed_image.shape)
    print("Fixed image range: ", fixed_image.min(), fixed_image.max())

    moving_image = nib.load('/home/simone.sarrocco/OMEGA_study/data/OMEGA_cleaned/OMEGA06/R/V02/Spectralis_oct/OMEGA_06_264.nii.gz')
    moving_image_numpy = moving_image.get_fdata().astype(np.float32)
    moving_image = torch.from_numpy(moving_image_numpy).to(device)
    # moving_image_torch = (moving_image_torch - moving_image_torch.min()) / (moving_image_torch.max() - moving_image_torch.min())
    print("Moving image shape: ", moving_image.shape)
    print("Moving image range: ", moving_image.min(), moving_image.max())

    fixed_image = fixed_image.permute(2, 1, 0)
    moving_image = moving_image.permute(2, 1, 0)

    fixed_image = al.Image(fixed_image, [193, 496, 512], [1,1,1], [0,0,0])
    moving_image = al.Image(moving_image, [193, 496, 512], [1,1,1], [0,0,0])

    fixed_points = None
    moving_points = None

    using_landmarks = True

    if fixed_points is None or moving_points is None:
        using_landmarks = False

    if using_landmarks:
        initial_tre = al.Points.TRE(fixed_points, moving_points)
        print("initial TRE: "+str(initial_tre))

    print("preprocessing images")
    (fixed_image, fixed_body_mask) = al.remove_bed_filter(fixed_image)
    (moving_image, moving_body_mask) = al.remove_bed_filter(moving_image)

    # normalize image intensities using common minimum and common maximum
    fixed_image, moving_image = al.utils.normalize_images(fixed_image, moving_image)

    # only perform center of mass alignment if inter subject registration is performed
    cm_alignment = False

    # Remove bed and auto-crop images
    f_image, m_image, _, _, cm_displacement = al.get_joint_domain_images(fixed_image, moving_image,
                                                                                   cm_alignment=cm_alignment,
                                                                                   compute_masks=False)

    # align also moving points
    if not cm_displacement is None and using_landmarks:
        moving_points_aligned = np.zeros_like(moving_points)
        for i in range(moving_points_aligned.shape[0]):
            moving_points_aligned[i, :] = moving_points[i, :] + cm_displacement
        print("aligned TRE: " + str(al.Points.TRE(fixed_points, moving_points_aligned)))
    else:
        moving_points_aligned = moving_points

    # create image pyramid size/8 size/4, size/2, size/1
    fixed_image_pyramid = al.create_image_pyramid(f_image, [[8, 8, 8], [4, 4, 4], [2, 2, 2]])
    moving_image_pyramid = al.create_image_pyramid(m_image, [[8, 8, 8], [4, 4, 4], [2, 2, 2]])

    constant_flow = None
    regularisation_weight = [1e-2, 1e-1, 1e-0, 1e+2]
    number_of_iterations = [300, 200, 100, 50]
    sigma = [[9, 9, 9], [9, 9, 9], [9, 9, 9], [9, 9, 9]]
    step_size = [1e-2, 4e-3, 2e-3, 2e-3]

    print("perform registration")
    for level, (mov_im_level, fix_im_level) in enumerate(zip(moving_image_pyramid,
                                                                                           fixed_image_pyramid,)):

        print("---- Level "+str(level)+" ----")
        registration = al.PairwiseRegistration()

        # define the transformation
        transformation = al.transformation.pairwise.BsplineTransformation(mov_im_level.size,
                                                                          sigma=sigma[level],
                                                                          order=3,
                                                                          dtype=dtype,
                                                                          device=device)

        if level > 0:
            constant_flow = al.transformation.utils.upsample_displacement(constant_flow,
                                                                          mov_im_level.size,
                                                                          interpolation="linear")

            transformation.set_constant_flow(constant_flow)

        registration.set_transformation(transformation)

        # choose the Mean Squared Error as image loss
        image_loss = al.loss.pairwise.MSE(fix_im_level, mov_im_level)

        registration.set_image_loss([image_loss])

        # define the regulariser for the displacement
        regulariser = al.regulariser.displacement.DiffusionRegulariser(mov_im_level.spacing)
        regulariser.SetWeight(regularisation_weight[level])

        registration.set_regulariser_displacement([regulariser])

        # define the optimizer
        optimizer = th.optim.Adam(transformation.parameters(), lr=step_size[level], amsgrad=True)

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(number_of_iterations[level])

        registration.start()

        # store current flow field
        constant_flow = transformation.get_flow()

        current_displacement = transformation.get_displacement()
        # generate SimpleITK displacement field and calculate TRE
        tmp_displacement = al.transformation.utils.upsample_displacement(current_displacement.clone().to(device='cpu'),
                                                                         m_image.size, interpolation="linear")
        tmp_displacement = al.transformation.utils.unit_displacement_to_dispalcement(tmp_displacement)  # unit measures to image domain measures
        tmp_displacement = al.create_displacement_image_from_image(tmp_displacement, m_image)
        tmp_displacement.write('/tmp/bspline_displacement_image_level_'+str(level)+'.vtk')

        # in order to not invert the displacement field, the fixed points are transformed to match the moving points
        if using_landmarks:
            print("TRE on that level: "+str(al.Points.TRE(moving_points_aligned, al.Points.transform(fixed_points, tmp_displacement))))

    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(m_image, displacement)
    displacement = al.transformation.utils.unit_displacement_to_dispalcement(displacement) # unit measures to image domain measures
    displacement = al.create_displacement_image_from_image(displacement, m_image)

    end = time.time()

    # in order to not invert the displacement field, the fixed points are transformed to match the moving points
    if using_landmarks:
        print("Initial TRE: "+str(initial_tre))
        fixed_points_transformed = al.Points.transform(fixed_points, displacement)
        print("Final TRE: " + str(al.Points.TRE(moving_points_aligned, fixed_points_transformed)))

    # write result images
    print("writing results")
    warped_image.write('/tmp/bspline_warped_image.vtk')
    m_image.write('/tmp/bspline_moving_image.vtk')
    m_mask.write('/tmp/bspline_moving_mask.vtk')
    f_image.write('/tmp/bspline_fixed_image.vtk')
    f_mask.write('/tmp/bspline_fixed_mask.vtk')
    displacement.write('/tmp/bspline_displacement_image.vtk')

    if using_landmarks:
        al.Points.write('/tmp/bspline_fixed_points_transformed.vtk', fixed_points_transformed)
        al.Points.write('/tmp/bspline_moving_points_aligned.vtk', moving_points_aligned)

    print("=================================================================")
    print("Registration done in: ", end - start, " seconds")


if __name__ == '__main__':
    main()
