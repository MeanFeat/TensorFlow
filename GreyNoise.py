import tensorflow as tf
from tensorflow import keras
import numpy as np
import tools
import matplotlib.pyplot as plt
import matplotlib.animation as anim

trainCount = 30000

X = np.random.uniform( low=0, high=1, size=(trainCount, 10000))
y = np.ones((trainCount,10000)) * 0.5

testX = np.random.uniform( low=0, high=1, size=(trainCount, 10000))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10000, input_shape=(10000,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10000, activation = keras.activations.sigmoid)
])

model.compile(optimizer= keras.optimizers.Adam(), loss= keras.losses.MSE)

tb_callback, log_dir = tools.GetTensorboardCallback('Grey-')
model.fit(X,y, epochs=5, batch_size=10000, verbose=1, callbacks=([tb_callback]))

model.evaluate(testX,  y, verbose=2)

tools.LaunchTensorboard(log_dir)