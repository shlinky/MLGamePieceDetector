import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plts


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
x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)
x = inception(x, training=False)
print(x.shape)
inception.summary()
x = top_layer(x)
model = tf.keras.Model(inputs, x)
model.summary()

# model.compile(
#   optimizer= tf.keras.optimizers.Adam(
#     learning_rate=0.0001
#   ),
#   loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#   metrics=['accuracy'])

top_layer.load_weights('./checkpoints/LL')

path = 'C:/Users/shloks/Documents/1data39.png'
i = tf.keras.preprocessing.image.load_img(
    path, grayscale=False, color_mode='rgb', target_size=[180, 180],
    interpolation='nearest'
)
print(i.size)
i.show()
x = np.expand_dims(tf.keras.preprocessing.image.img_to_array(i), axis = 0)
print(x.dtype)
print(x.shape)

print(model.predict(x))