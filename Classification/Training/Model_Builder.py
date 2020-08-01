import numpy as np
import os
import tensorflow as tf

class ModelBuilder:
	def __init__(self, training=False, ckpt = None):
		self.training = training
		self.ckpt = ckpt
		self.build_model()

	def build_model(self):
		self.inception = tf.keras.applications.Xception(
		    weights='imagenet',
		    input_shape=(180, 180, 3),
		    include_top=False)
		self.inception.trainable = False

		self.top_layer =  tf.keras.Sequential(
		    [
		    	tf.keras.layers.GlobalAveragePooling2D(),
		    	tf.keras.layers.Dropout(0.5),
		        tf.keras.layers.Dense(1)
		    ]
		)

		self.data_augmentation = tf.keras.Sequential(
		    [
		        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
		        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
		    ]
		)

		inputs = tf.keras.Input(shape=(180, 180, 3))
		if self.training:
			x = self.data_augmentation(inputs)
			x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(x)
		else:
			x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)
		x = self.inception(x, training=False)
		x = self.top_layer(x)
		self.model = tf.keras.Model(inputs, x)
		self.model.summary()

		self.model.compile(
		  optimizer= tf.keras.optimizers.Adam(),
		  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
		   metrics=[tf.keras.metrics.BinaryAccuracy()])

		if (self.ckpt) and (not self.training):
			self.top_layer.load_weights(self.ckpt)

	def get_model(self):
		return(self.model)

	def init_LL(self, ckpt = None):
		if self.ckpt:
			self.top_layer.load_weights(self.ckpt)
		elif ckpt:
			self.top_layer.load_weights(ckpt)

	def save_LL(self, ckpt = None):
		if self.ckpt:
			self.top_layer.save_weights(self.ckpt)
		elif ckpt:
			self.top_layer.save_weights(ckpt)