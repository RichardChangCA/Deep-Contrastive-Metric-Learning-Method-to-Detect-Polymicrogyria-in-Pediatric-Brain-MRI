import os

base_dir = '/home/lingfeng/Downloads/PMG1'

# ax_dir = os.path.join(base_dir, 'PMG1_axT1')
ax_dir = os.path.join(base_dir, 'PMG1corT1')

filename_list = []

for filename in os.listdir(ax_dir):
    filename_list.append(filename)

# print(filename_list)

filename_list.sort()

filename_list_new = sorted(filename_list,key=len)

# for filename in filename_list_new:
#     print(filename)

from PIL import Image
from numpy import asarray
import numpy as np

# size = (128, 128)
size = (170, 170)

image_array = []
slice_num = 0
for filename in filename_list_new:
    image = Image.open(os.path.join(ax_dir, filename)).convert('L') # convert to gray scale
    image = image.resize(size) 
    data = asarray(image)
    image_array.append(data)
    slice_num += 1

print(slice_num)

# for _ in range(128-slice_num):
#     image_array.append(np.zeros(size))

image_array = np.array(image_array)

print(image_array.shape)

import nibabel as nib

ni_img = nib.Nifti1Image(image_array, affine=None)
# nib.save(ni_img, "PMG1_axT1_new.nii")
nib.save(ni_img, "PMG1_corT1_new.nii")

# view .nii format: https://socr.umich.edu/HTML5/BrainViewer/