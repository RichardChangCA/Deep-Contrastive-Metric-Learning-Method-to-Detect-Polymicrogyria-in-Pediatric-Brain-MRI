import os
from PIL import Image
from numpy import asarray
import numpy as np
import nibabel as nib
from natsort import natsorted
from matplotlib import pyplot as plt
from tqdm import tqdm

# https://github.com/rockstreamguy/deepbrain

# pipeline:
# 1. convert .jpeg images into .nii
# 2. skull stripping
# 3. label them based on the .jpeg image name

# finished_list = ["12", "8", "29", "17", "30", "18", "22", "14", "9", "6", "3", "4"]

def skull_stripping_func(base_dir, save_base_dir, whether_normal):

    size = (256, 256)

    for patient_id in tqdm(os.listdir(base_dir)):
        patient_id_folder = os.path.join(base_dir, patient_id)

        # if(patient_id in finished_list):
        #     continue
        
        print("patient_id: ", patient_id)
        
        save_patient_id_folder = os.path.join(save_base_dir, patient_id)
        if not os.path.exists(save_patient_id_folder):
            os.makedirs(save_patient_id_folder)

        if(whether_normal == True):
            for control_id in os.listdir(patient_id_folder):

                control_path_saved = os.path.join(patient_id_folder, control_id)
                save_control_path_saved = os.path.join(save_patient_id_folder, control_id)
                
                if not os.path.exists(save_control_path_saved):
                    os.makedirs(save_control_path_saved)
                control_path_list = os.listdir(control_path_saved)
                control_path_list = natsorted(control_path_list)

                # control
                # 
                # 
                image_array = []
                image_name_array = []

                for slice_ in control_path_list:
                    image_path = os.path.join(control_path_saved, slice_)
                    label_num = slice_.split('.')[0].split('_')[-1]
                    label_num = int(label_num)
                    if(label_num!=3):
                        
                        image = Image.open(image_path).convert('L') # convert to gray scale
                        image = image.resize(size) 
                        data = asarray(image)
                        image_array.append(data)
                        image_name_array.append(slice_)

                image_array = np.array(image_array)
                image_name_array = np.array(image_name_array)

                # print(image_array.shape)
                # print(image_name_array.shape)

                ni_img = nib.Nifti1Image(image_array, affine=None)
                nib.save(ni_img, "control.nii")

                os.system("deepbrain-extractor -i control.nii -o .")
                
                nii_data = nib.load("brain.nii").get_fdata()
                # print(nii_data.shape)

                for slice_ in range(nii_data.shape[0]):
                    # plt.imshow(nii_data[slice_,:,:])
                    # plt.savefig(os.path.join(save_control_path_saved, image_name_array[slice_]))
                    Image.fromarray(nii_data[slice_,:,:]).convert('RGB').save(os.path.join(save_control_path_saved, image_name_array[slice_]))

        else: # coronal

            normal_abnormal = os.listdir(patient_id_folder)
            if(normal_abnormal[0].__contains__("cor")):
                coronal_path_saved = os.path.join(patient_id_folder, normal_abnormal[0])

                save_coronal_path_saved = os.path.join(save_patient_id_folder, normal_abnormal[0])

            else:
                coronal_path_saved = os.path.join(patient_id_folder, normal_abnormal[1])

                save_coronal_path_saved = os.path.join(save_patient_id_folder, normal_abnormal[1])

            if not os.path.exists(save_coronal_path_saved):
                os.makedirs(save_coronal_path_saved)

            coronal_path_list = os.listdir(coronal_path_saved)
            coronal_path_list = natsorted(coronal_path_list)
            
            # coronal
            # 
            # 
            image_array = []
            image_name_array = []
            for slice_ in coronal_path_list:
                image_path = os.path.join(coronal_path_saved, slice_)
                label_num = slice_.split('.')[0].split('_')[-1]
                label_num = int(label_num)
                if(label_num!=3):
                    
                    image = Image.open(image_path).convert('L') # convert to gray scale
                    image = image.resize(size) 
                    data = asarray(image)
                    image_array.append(data)
                    image_name_array.append(slice_)

            image_array = np.array(image_array)
            image_name_array = np.array(image_name_array)

            # print(image_array.shape)
            # print(image_name_array.shape)

            ni_img = nib.Nifti1Image(image_array, affine=None)
            nib.save(ni_img, "coronal.nii")

            os.system("deepbrain-extractor -i coronal.nii -o .")
            
            nii_data = nib.load("brain.nii").get_fdata()
            # print(nii_data.shape)

            for slice_ in range(nii_data.shape[0]):
                # plt.imshow(nii_data[slice_,:,:])
                # plt.savefig(os.path.join(save_coronal_path_saved, image_name_array[slice_]))
                Image.fromarray(nii_data[slice_,:,:]).convert('RGB').save(os.path.join(save_coronal_path_saved, image_name_array[slice_]))

# base_dir = '/home/lingfeng/Downloads/PMGControlsEditedDec2021'

# save_base_dir = '/home/lingfeng/Downloads/PMG_skull_stripping/normal'

# if not os.path.exists(save_base_dir):
#     os.makedirs(save_base_dir)

# skull_stripping_func(base_dir, save_base_dir, whether_normal=True)

base_dir = '/home/lingfeng/Desktop/Nested_CV/thesis_plots_drawing/coronal_images'

save_base_dir = '/home/lingfeng/Desktop/Nested_CV/thesis_plots_drawing/skull_stripping_images'

if not os.path.exists(save_base_dir):
    os.makedirs(save_base_dir)

skull_stripping_func(base_dir, save_base_dir, whether_normal=False)