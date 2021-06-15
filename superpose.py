# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:38:41 2021

@author: DELINTE Nicolas

"""

import os
import numpy as np
import nibabel as nib
from dipy.viz import regtools
from dipy.io.image import load_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)

static_data, static_affine = load_nifti('Data/P1_FA.nii.gz')
static_grid2world = static_affine

moving_data, moving_affine = load_nifti('Data/Brain_T1.nii')
moving_grid2world = moving_affine

# Affine registration -----------------------------------------------------

identity = np.eye(4)
affine_map = AffineMap(identity,
                       static_data.shape, static_grid2world,
                       moving_data.shape, moving_grid2world)
resampled = affine_map.transform(moving_data)
regtools.overlay_slices(static_data, resampled, None, 0,
                        "Static", "Moving", "resampled_0.png")
regtools.overlay_slices(static_data, resampled, None, 1,
                        "Static", "Moving", "resampled_1.png")
regtools.overlay_slices(static_data, resampled, None, 2,
                        "Static", "Moving", "resampled_2.png")

out=nib.Nifti1Image(resampled, static_affine)
out.to_filename('Data/MASK_to_diffSpace.nii.gz')

mask = nib.load('Data/segmentationKmeans.nii')
resampled = affine_map.transform(mask.get_fdata())
out=nib.Nifti1Image(resampled, static_affine)
out.to_filename('Data/MASK_to_diffSpace2.nii.gz')
#then save