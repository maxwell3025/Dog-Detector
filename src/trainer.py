import tensorflow as tf
import numpy as np

x_test = np.load("data/x_test.npy")
y_test = np.load("data/y_test.npy")
x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")

y_train = y_train.dot(np.array([0,1])) 

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(64, 64)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
