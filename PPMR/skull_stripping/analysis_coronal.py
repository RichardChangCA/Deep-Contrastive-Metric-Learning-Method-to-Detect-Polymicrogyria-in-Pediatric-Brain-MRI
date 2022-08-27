import nibabel as nib

# data = nib.load("PMG1_corT1_new.nii").get_fdata()
# print(data.shape)

# from matplotlib import pyplot as plt
# plt.imshow(data[100,:,:])
# plt.savefig('PMG1_corT1_new_demo.png')

data = nib.load("coronal/brain.nii").get_fdata()
print(data.shape)

from matplotlib import pyplot as plt
plt.imshow(data[100,:,:])
plt.savefig('coronal_new_demo.png')