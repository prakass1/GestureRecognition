#A quick CNN experiment using keras
import numpy as np
import  pandas as pd
from keras import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dropout
import keras
from keras.models import load_model
from keras.callbacks import *
import cv2

dataset_path = "G:\\PycharmProjects\\HandRecognitionApplication\\dataset\\"

classes_dict = {0:"background", 1:"L", 2:"thumbsup", 3:"up"}
classes_inv_dict = {"background":0, "L":1, "thumbsup":2, "up":3}

#Read image to numpy and do something with float32 probably
classes = [0, 1, 2, 3]
import os
from pathlib import Path
def convert(image):
        img = cv2.imread(image)
        img_resize = cv2.resize(img, (32, 32))
        return np.float32(img_resize[:,:,0])


def create_data(data_path):
        data_list = []
        labels = []
        # Create train data and test data
        images_folder = os.listdir(data_path)
        for folder in images_folder:
                images = os.listdir(data_path + "\\" + folder)
                for image in images:
                        data_list.append(convert(data_path + "\\" + folder + "//" + image))
                        labels.append(classes_inv_dict.get(folder))
        return np.asarray(data_list, dtype="float32"), np.asarray(labels, dtype="float32")


# Train Dataset
X_train, y_train = create_data("G:\\PycharmProjects\\HandRecognitionApplication\\dataset\\training_set")

# Test Dataset
X_test, y_test = create_data("G:\\PycharmProjects\\HandRecognitionApplication\\dataset\\test_set")


# TODO: Dump and load numpy

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


# reshape dataset to have a single channel
w, h, c = X_train.shape[1], X_train.shape[2], X_train.shape[3]



# Initialize a classifier
classifier = Sequential()

# Add layers for a quick experiment without optimization
classifier.add(Conv2D(32, (3, 3), input_shape=(w, h, c), activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Conv2D(64, (3, 3), activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Conv2D(64, (3, 3), activation="relu"))
# classifier.add(Conv2D(64, (3, 3), activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2)))

# Add another convolution layer later
# use dropout later
classifier.add(Dropout(0.30))

# Flatten
classifier.add(Flatten())

# Dense layer
# Step 4 - Full connection. Units for softmax are 6 because there are 6 classes
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units=4, activation = 'softmax'))

# Compiling the CNN
adam = keras.optimizers.adam()
classifier.compile(optimizer = adam, loss = "categorical_crossentropy", metrics = ['accuracy'])

# Add the callback
filepath="models/epochs_{epoch:03d}_val_acc_{val_acc:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Create a datagen and setup the classifier for results
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=45.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

#test_datagen = ImageDataGenerator(rescale=1./255)
#Mean of the data
datagen.fit(X_train)

# Categorical conversion
from keras.utils import to_categorical
classes_cat_train = to_categorical(y_train)
classes_cat_test = to_categorical(y_test)

train_generator = datagen.flow(X_train, classes_cat_train, batch_size= 32, shuffle= True)
test_generator = datagen.flow(X_test, classes_cat_test, batch_size= 32, shuffle= True)
# train_generator = train_datagen.flow_from_directory(
#         'dataset/training_set',
#         #color_mode="grayscale",
#         target_size=(32, 32),
#         batch_size=32,
#         class_mode='categorical',
#         shuffle=True)

# validation_generator = test_datagen.flow_from_directory(
#         'dataset/test_set',
#         target_size=(32, 32),
#         batch_size=32,
#         class_mode='categorical',
#         #color_mode="grayscale",
#         shuffle=True)

# fit model with generator
classifier.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=11)
# evaluate model
_, acc = classifier.evaluate_generator(test_generator, steps=len(test_generator), verbose=0)
print('Test Accuracy: %.3f' % (acc * 100))


#classifier.predict(X_test[len(X_test)].reshape(1,32,32,3))

# classifier.fit_generator(
#         train_generator,
#         steps_per_epoch=4163,
#         epochs=20,
#         validation_data=validation_generator,
#         validation_steps=1821,
#         #use_multiprocessing=True,
#         callbacks = callbacks_list)

#Save classifier and weights
import datetime
#YYYYMMDDHHMMSS
date_time = str(datetime.datetime.utcnow()).replace("-", "").replace(" ", "").replace(":","").split(".")[0]
classifier.save("cl_models/" + date_time + "model.h5")
classifier.save_weights("cl_models/" + date_time + "model_weights.h5")

#lets predict an image
model = load_model("cl_models/20191103230955model.h5")

classes = {"l":0,"thumbsup":1,"up":2}
import cv2
single_img = convert("backed/testg/two/two_gesture227.png")

model.predict(single_img.reshape(1, 32, 32, 3))

single_img_resize = cv2.resize(single_img, (32, 32))

#single_img_resize = single_img_resize - single_img_resize.mean()
s_i_r_copy = single_img_resize.copy()
img_pred = s_i_r_copy.reshape(1, 32, 32, 3)

import numpy as np
from keras.preprocessing import image

img_width, img_height = 32, 32
img = image.load_img('backed/testg/four/four_gesture171.png', target_size = (img_width, img_height))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)

model.predict(img)


pred = model.predict(single_img)
pred
for i, val in enumerate(pred):
        if val > 0.9:
                pred_class_name = [key for key, val in classes.items() if val == i]
                print("Prediction is - {} with confidence of {}".format(pred_class_name, val*100))
