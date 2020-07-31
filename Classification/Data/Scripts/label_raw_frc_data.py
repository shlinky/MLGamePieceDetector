import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import sys
from shutil import copyfile

#2555
class LabellerUi:
	def __init__(self):
		raw_data_path = ['C:/Users/shloks/Downloads/Raw Data/Filming Day 1 Images/img/', 'C:/Users/shloks/Downloads/Raw Data/Filming Day 1 Video/img/', 'C:/Users/shloks/Downloads/Raw Data/Filming Day 2 Video/img/']
		self.img_num = 2854
		self.prediction = None
		self.datadirectory = ['C:/Users/shloks/Documents/robproj/MLGamePieceDetector/Classification/Data/ImgData/0/', 'C:/Users/shloks/Documents/robproj/MLGamePieceDetector/Classification/Data/ImgData/1/']
		self.img_files = []
		self.model = self.build_model()
		for i in raw_data_path:
			for f in os.listdir(i):
				self.img_files.append([i, f])

		self.datacount = [0 for i in range(2)]
		for i in self.datadirectory:
			for filename in os.listdir(i):
				if filename[0] == '1':
					self.datacount[1] += 1
				else:
					self.datacount[0] += 1
		fig = plt.figure(figsize=(10, 10))
		fig.canvas.mpl_connect('key_press_event', self.press)

	def draw_next(self):
		while (self.img_num < len(self.img_files)):
			self.img_num += 1
			json_file = self.img_files[self.img_num][0][:-4] + 'ann/' + self.img_files[self.img_num][1] + '.json'
			print(json_file)
			annotations = json.load(open(json_file))

			if (len(annotations['tags']) == 0) or (annotations['tags'][0]['name'] == 'Invalid'):
				break
			break

		print(self.img_num)
		img_path = self.img_files[self.img_num][0] + self.img_files[self.img_num][1]
		i = tf.keras.preprocessing.image.load_img(
    		img_path, grayscale=False, color_mode='rgb', target_size=[180, 180],
    		interpolation='nearest')
		i = np.expand_dims(tf.keras.preprocessing.image.img_to_array(i), axis = 0)
		print(self.model.predict(i))
		if list(self.model.predict(i))[0] > 0:
			self.prediction = 1
		else:
			self.prediction = 0

		plt.clf()
		if self.prediction:
			plt.title('Game Piece Present; Correct?')
		else:
			plt.title('Game Piece Not Present; Correct?')
		image = Image.open(img_path)
		plt.imshow(image)
		plt.show()

	def press(self, event):
	    print('press', event.key)
	    sys.stdout.flush()
	    label = None
	    if event.key == 'n':
	        if self.prediction:
	        	label = 0
	        else:
	        	label = 1
	    elif event.key == 'y':
	    	label = self.prediction
	    else:
	    	self.draw_next()
	    	return True
	    self.move_to_dataset(self.img_files[self.img_num][0] + self.img_files[self.img_num][1], label)
	    self.draw_next()

	def build_model(self):
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
		x = top_layer(x)
		model = tf.keras.Model(inputs, x)
		model.summary()

		model.compile(
		  optimizer= tf.keras.optimizers.Adam(),
		  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
		   metrics=[tf.keras.metrics.BinaryAccuracy()])
		top_layer.load_weights('C:/Users/shloks/Documents/robproj/MLGamePieceDetector/Classification/Training/checkpoints/LL')
		return(model)

	def move_to_dataset(self, fname, label):
		self.datacount[label] += 1
		name = str(label) + 'data' + str(self.datacount[label]) + '.png'
		print(self.datadirectory[label] + name)
		copyfile(fname, self.datadirectory[label] + name)


sys.setrecursionlimit(100000)
ui = LabellerUi()
ui.draw_next()