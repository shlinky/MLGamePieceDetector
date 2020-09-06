import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plts
from Model_Builder import ModelBuilder


mb = ModelBuilder(ckpt = './checkpoints/LL_NEWSA')
model = mb.get_model()

path = 'C:/Users/shloks/Documents/g7.jpg'
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