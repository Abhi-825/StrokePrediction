# Import Libraries
import tensorflow as tf
import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

folder_path = "C:\\Temp\\main\\"

files_list = glob.glob(os.path.join(folder_path,'**','*.jpg'), recursive = True)  # Creating a list for the files in the data set

# Labels and Images List
images = []
labels12 = []
labels = []

# Reading the Files, Converting them to Numpy Arrays, appending the Numpy Array and Label into the lists.
# Index's Match

for file in files_list:
 image = cv2.imread(file)
 image_array = np.array(image)
 dirname = os.path.dirname(file)
 labels12.append(os.path.basename(dirname))
 images.append(image_array)
 labels = np.array(labels12, dtype=str)


#Assume images is a list of NumPy arrays with varying sizes
target_size = (64, 64)  # Target size (height, width)

# Resize all images to the target size
resized_images = []
for image in images:
    resized_image = cv2.resize(image, target_size)
    resized_images.append(resized_image)
    
resized_images = np.array(resized_images, dtype = str)    

X_train, X_test, y_train, y_test = train_test_split(resized_images, labels, test_size=0.2, random_state=42) 

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform labels to integer values
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

num_classes = len(np.unique(y_train))  # Number of unique labels

# Convert labels to one-hot encoding
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

X_train = X_train.astype('float32') / 255.0  # Normalize to [0, 1] if using float32
X_test = X_test.astype('float32') / 255.0

num_classes = 2  # e.g., "stroke image" and "not stroke image"

num_classes = 2  # Update this based on your dataset

# Convert labels to one-hot encoding
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train_one_hot, epochs=7, batch_size=64)

# Get predicted probabilities
y_pred_prob = model.predict(X_test)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_prob, axis=1)

# Convert one-hot encoded y_test to class labels if it's still one-hot encoded
y_test_labels = np.argmax(y_test_one_hot, axis=1)
    
# Calculate and print accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
print('Test accuracy:', accuracy)

model.save('C:\\Temp\\my_cnn_model.keras')  # Saving as .keras