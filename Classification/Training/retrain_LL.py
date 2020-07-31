import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt


imgdir = 'C:/Users/shloks/Documents/robproj/MLGamePieceDetector/Data/ImgData'
batch_size = 32
img_height = 180
img_width = 180
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  imgdir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='binary')
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  imgdir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='binary')

train_ds = train_ds.repeat(500)
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#print(train_ds.class_names)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

inception = tf.keras.applications.Xception(
    weights='imagenet',
    input_shape=(180, 180, 3),
    include_top=False)
inception.trainable = False

top_layer =  tf.keras.Sequential(
    [
    	tf.keras.layers.GlobalAveragePooling2D(),
    	tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ]
)

inputs = tf.keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(x)
x = inception(x, training=False)
print(x.shape)
inception.summary()
x = top_layer(x)
model = tf.keras.Model(inputs, x)
model.summary()

model.compile(
  optimizer= tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
   metrics=[tf.keras.metrics.BinaryAccuracy()])
model.fit(
  train_ds,
  batch_size=batch_size,
  validation_data=val_ds,
  epochs=1,
  steps_per_epoch = 100,
  shuffle = True
)

top_layer.save_weights('./checkpoints/LL')