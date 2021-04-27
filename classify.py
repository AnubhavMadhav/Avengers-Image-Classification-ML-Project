from zipfile import ZipFile
filename= "marvel.zip"
with ZipFile(filename,'r')as zip:
    zip.extractall()



#Importing
import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = os.path.join(os.path.dirname(filename), 'marvel')



# Accessing the images and setting 0.7 of images for training and the rest for testing
classes=['Black Widow','Captain America','Hulk','Iron Man','Thor']
for m in classes:
    img_path = os.path.join(base_dir, m)
    images = glob.glob(img_path + '/*.jpg')
    num_train = int(round(len(images)*0.7))
    train, val = images[:num_train], images[num_train:]
# Creating separate directories for training data
for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', m)):
        os.makedirs(os.path.join(base_dir, 'train', m))
        shutil.move(t, os.path.join(base_dir, 'train', m))
# Creating separate directories for validating data
for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', m)):
        os.makedirs(os.path.join(base_dir, 'val', m))
        shutil.move(v, os.path.join(base_dir, 'val', m))


train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')



# Setting batch size and a constant image shape
batch_size = 130
IMG_SHAPE = 150
# Rescaling the images so all the values lie between 0 and 1 and applying horizontal flip and training the data
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data_gen = image_gen.flow_from_directory(
batch_size=batch_size,
directory=train_dir,
shuffle=True,
target_size=(IMG_SHAPE,IMG_SHAPE)
)
# Rescaling the images so all the values lie between 0 and 1 and rotating and training the data
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
directory=train_dir,
shuffle=True,
target_size=(IMG_SHAPE, IMG_SHAPE))
#Rescaling and zooming the data
image_gen_train = ImageDataGenerator(
rescale=1./255,
rotation_range=45,
width_shift_range=.15,
height_shift_range=.15,
horizontal_flip=True,
zoom_range=0.5
)
train_data_gen = image_gen_train.flow_from_directory(
batch_size=batch_size,
directory=train_dir,
shuffle=True,
target_size=(IMG_SHAPE,IMG_SHAPE),
class_mode='sparse'
)





image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
directory=val_dir,
target_size=(IMG_SHAPE, IMG_SHAPE),
class_mode='sparse')
model = Sequential()
model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SHAPE,IMG_SHAPE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Adding dropout to turn down some neurons
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))





model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
epochs = 120
history = model.fit_generator(
train_data_gen,
steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
epochs=epochs,
validation_data=val_data_gen,
validation_steps=int(np.ceil(val_data_gen.n / float(batch_size)))
)