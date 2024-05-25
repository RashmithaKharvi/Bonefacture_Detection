import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image

alljoint_train_path= '/kaggle/input/joint-fracture-dataset/joint dataset/joint data/joint identification data/train'
alljoint_test_path='/kaggle/input/joint-fracture-dataset/joint dataset/joint data/joint identification data/val'

elbow_train_path= '/kaggle/input/joint-fracture-dataset/joint dataset/joint data/elbow fracture/train'
elbow_test_path='/kaggle/input/joint-fracture-dataset/joint dataset/joint data/elbow fracture/val'

finger_train_path= '/kaggle/input/joint-fracture-dataset/joint dataset/joint data/fingers fracture/train'
finger_test_path='/kaggle/input/joint-fracture-dataset/joint dataset/joint data/fingers fracture/val'

forearm_train_path= '/kaggle/input/joint-fracture-dataset/joint dataset/joint data/forearm fracture/train'
forearm_test_path='/kaggle/input/joint-fracture-dataset/joint dataset/joint data/forearm fracture/val'

hand_train_path= '/kaggle/input/joint-fracture-dataset/joint dataset/joint data/hand fracture/train'
hand_test_path='/kaggle/input/joint-fracture-dataset/joint dataset/joint data/hand fracture/val'

humerus_train_path= '/kaggle/input/joint-fracture-dataset/joint dataset/joint data/humerus fracture/train'
humerus_test_path='/kaggle/input/joint-fracture-dataset/joint dataset/joint data/humerus fracture/val'

shoulder_train_path= '/kaggle/input/joint-fracture-dataset/joint dataset/joint data/shoulder fracture/train'
shoulder_test_path='/kaggle/input/joint-fracture-dataset/joint dataset/joint data/shoulder fracture/train'

wrist_train_path= '/kaggle/input/joint-fracture-dataset/joint dataset/joint data/wrist fracture/train'
wrist_test_path='/kaggle/input/joint-fracture-dataset/joint dataset/joint data/wrist fracture/val'

train_datagen = image.ImageDataGenerator(
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)
val_datagen= image.ImageDataGenerator(    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1)

a_train_generator = train_datagen.flow_from_directory(
    alljoint_train_path,
    target_size = (224,224),
    batch_size = 4,
    class_mode = 'categorical')
a_validation_generator = val_datagen.flow_from_directory(
    alljoint_test_path,
    target_size = (224,224),
    batch_size = 4,
    shuffle=True,
    class_mode = 'categorical')

e_train_generator = train_datagen.flow_from_directory(
    elbow_train_path,
    target_size = (224,224),
    batch_size = 4,
    class_mode = 'binary')
e_validation_generator = val_datagen.flow_from_directory(
    elbow_test_path,
    target_size = (224,224),
    batch_size = 4,
    shuffle=True,
    class_mode = 'binary')

fi_train_generator = train_datagen.flow_from_directory(
    finger_train_path,
    target_size = (224,224),
    batch_size = 4,
    class_mode = 'binary')
fi_validation_generator = val_datagen.flow_from_directory(
    finger_test_path,
    target_size = (224,224),
    batch_size = 4,
    shuffle=True,
    class_mode = 'binary')

fo_train_generator = train_datagen.flow_from_directory(
    forearm_train_path,
    target_size = (224,224),
    batch_size = 4,
    class_mode = 'binary')
fo_validation_generator = val_datagen.flow_from_directory(
    forearm_test_path,
    target_size = (224,224),
    batch_size = 4,
    shuffle=True,
    class_mode = 'binary')

ha_train_generator = train_datagen.flow_from_directory(
    hand_train_path,
    target_size = (224,224),
    batch_size = 4,
    class_mode = 'binary')
ha_validation_generator = val_datagen.flow_from_directory(
    hand_test_path,
    target_size = (224,224),
    batch_size = 4,
    shuffle=True,
    class_mode = 'binary')

hu_train_generator = train_datagen.flow_from_directory(
    humerus_train_path,
    target_size = (224,224),
    batch_size = 4,
    class_mode = 'binary')
hu_validation_generator = val_datagen.flow_from_directory(
    humerus_test_path,
    target_size = (224,224),
    batch_size = 4,
    shuffle=True,
    class_mode = 'binary')

s_train_generator = train_datagen.flow_from_directory(
    shoulder_train_path,
    target_size = (224,224),
    batch_size = 4,
    class_mode = 'binary')
s_validation_generator = val_datagen.flow_from_directory(
    shoulder_test_path,
    target_size = (224,224),
    batch_size = 4,
    shuffle=True,
    class_mode = 'binary')

w_train_generator = train_datagen.flow_from_directory(
    wrist_train_path,
    target_size = (224,224),
    batch_size = 4,
    class_mode = 'binary')
w_validation_generator = val_datagen.flow_from_directory(
    wrist_test_path,
    target_size = (224,224),
    batch_size = 4,
    shuffle=True,
    class_mode = 'binary')

base_model = tf.keras.applications.EfficientNetB3(weights='imagenet', input_shape=(224,224,3), include_top=False)
for layer in base_model.layers:
    layer.trainable=False
all_model = Sequential()
all_model.add(base_model)
all_model.add(GaussianNoise(0.25))
all_model.add(Conv2D(32,kernel_size=(1,1),activation='relu'))
all_model.add(GlobalAveragePooling2D())
all_model.add(Dense(512,activation='relu'))
all_model.add(GaussianNoise(0.25))
all_model.add(Dropout(0.25))
all_model.add(Dense(7, activation='softmax'))

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
lrp=ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=2)
filepath='best_model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
call=[checkpoint,lrp]
history = all_model.fit(
    a_train_generator,
    epochs=1,
    validation_data=a_validation_generator,
    steps_per_epoch= 80,
    callbacks=call
    )
all_model.save('all joint identification model.h5')