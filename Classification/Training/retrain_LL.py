import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
from Model_Builder import ModelBuilder

imgdir = 'C:/Users/shloks/Documents/robproj/MLGamePieceDetector/Data/ImgData'
batch_size = 64
img_height = 180
img_width = 180
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  imgdir,
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='binary')
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  imgdir,
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='binary')

# train_ds = train_ds.repeat(500)
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
print(train_ds.class_names)

mb = ModelBuilder(training = True, ckpt = './checkpoints/LL_NEW')
model = mb.get_model()

model.fit(
  train_ds,
  batch_size=batch_size,
  validation_data=val_ds,
  epochs=1,
  shuffle = True
)

mb.save_LL()