# Import libraries
import os
import cv2
import numpy as np

from keras.models import load_model
from keras import backend as K
from keras.preprocessing import image

PATH = os.getcwd()
# Define data path
data_path = os.path.join(PATH, 'test')
data_dir_list = os.listdir(data_path)

img_rows = 224
img_cols = 224

img_data_list = []  # total img
total_img_filename = []
class_name = ['bruises', 'cellulitis', 'cuts', 'lowerlimbs', 'mouth', 'scrape']

for img in data_dir_list:

    input_img = cv2.imread(os.path.join(data_path, img))
    print(os.path.join(data_path, img))
    input_img_resize = cv2.resize(input_img, (img_rows, img_cols))

    img_data_list.append(input_img_resize)
    total_img_filename.append(img)


x = image.img_to_array(input_img_resize)
x = np.expand_dims(x, axis=0)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')



if K.image_data_format() == 'channels_first':
    img_data = img_data.reshape(img_data.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    img_data = img_data.reshape(img_data.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)



model = load_model(os.path.join(PATH, 'Best-weights-my_model-019-0.2421-0.9100.h5'))

predict = model.predict(img_data)

classes = np.argmax(predict, axis=1)

for i in range(len(classes)):
    print("file:" + total_img_filename[i] + " is " + class_name[classes[i]])
