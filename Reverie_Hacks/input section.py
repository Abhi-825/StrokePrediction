import tensorflow as tf
import cv2
import numpy as np
import os

model = tf.keras.models.load_model('C:\\Users\mkura\Reverie_Hacks\my_cnn_model.keras')


file = 'C:\\Users\mkura\Reverie_Hacks\FrontEndVideo\Signs of a Stroke - New.jpeg'

image = cv2.imread(file)
image_array = np.array(image)
dirname = os.path.dirname(file)
label = np.array(dirname, dtype=str)

#Assume images is a list of NumPy arrays with varying sizes
target_size = (64, 64)  # Target size (height, width)

# Resize all images to the target size


resized_image = cv2.resize(image, target_size)

resized_image = np.array(resized_image, dtype = str)   

resized_image = resized_image.astype('float32') / 255.0
    
# Add batch dimension (1, target_size[0], target_size[1], 3)
resized_image = np.expand_dims(resized_image, axis=0) 

print(resized_image.shape)

prediction = model.predict(resized_image)

print(prediction)



