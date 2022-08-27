import numpy as np
from PIL import Image
from numpy import asarray

# img_path = '/home/lingfeng/Downloads/PMG_skull_stripping/abnormal/12/12cor_1/12cor_1_83_2.jpg'
# img_path = '/home/lingfeng/Downloads/PMG_skull_stripping/normal/12/12control1/12control1_cor_0_051.jpg'

img_path = '/home/lingfeng/Desktop/Nested_CV/thesis_plots_drawing/skull_stripping_images/6/6cor_1/6cor_1_106_1.jpg'

image = Image.open(img_path)

# convert image to numpy array
data = asarray(image)

print("type(data):", type(data))

# summarize shape
print("data.shape:", data.shape)

print("max:", np.amax(data))
print("min:", np.amin(data))

# type(data): <class 'numpy.ndarray'>
# data.shape: (256, 256, 3)
# max: 255
# min: 0