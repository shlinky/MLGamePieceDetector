import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

class Exp(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(Exp, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.m = self.add_weight("m",
                                  shape=[1], initializer='ones')
    self.e = self.add_weight("e",
                                  shape=[1], initializer='ones')
    self.a = self.add_weight("a",
                                  shape=[1], initializer='zeros')

  def call(self, inputs):
    return tf.math.add(tf.math.multiply(tf.math.pow(self.e, inputs), self.m), self.a)

def weighted_loss(pred, act):
	return tf.keras.losses.MeanAbsoluteError()(act, pred)


inputs = tf.keras.Input(shape=())
exp = Exp(1)
outputs = exp(inputs)
model = tf.keras.Model(inputs, outputs)
print(model.summary())

X = []
Y = []
for i in range(40):
	X.append(i / 5)
	Y.append((0.9 ** (i / 5)))
X = tf.convert_to_tensor(X)
Y = tf.convert_to_tensor(Y)
print(X)
print(tf.executing_eagerly())
wl = tf.function(weighted_loss)

optimizer = tf.keras.optimizers.SGD()
epochs = 6000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        logits = model(X, training=True)  # Logits for this minibatch
        print(logits.numpy(), Y.numpy(), exp.variables)
        loss_value = wl(logits, Y)
        print(float(loss_value))

    grads = tape.gradient(loss_value, model.trainable_weights)
    print(grads)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

print(exp.variables)

# print(tf.math.pow([2], [[2], [4], [5]]))