from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras
import time
import numpy as np

data_size = 1000000
# 80% of the data is for training.
train_pct = 0.8

train_size = int(data_size * train_pct)

# Create some input data between -1 and 1 and randomize it.
x = np.linspace(-1, 1, data_size)
np.random.shuffle(x)

# Generate the output data.
# y = 0.5x + 2 + noise
y = 0.5 * x + 2 + np.random.normal(0, 0.05, (data_size, ))

# Split into test and train pairs.
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

log_dir = 'logs\\' + os.path.basename(__file__) + datetime.now().strftime("%m%d%Y%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= log_dir)

model = keras.models.Sequential([
    keras.layers.Dense(16, input_dim=1),
    keras.layers.Dense(1),
])

model.compile(
    loss='mse', # keras.losses.mean_squared_error
    optimizer=keras.optimizers.SGD(lr=0.015),
)

print("Training ... With default parameters, this takes less than 10 seconds.")
training_history = model.fit(
    x_train, # input
    y_train, # output
    batch_size=train_size,
    verbose=1, # Suppress chatty output; use Tensorboard instead
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
)

print("Average test loss: ", np.average(training_history.history['loss']))

os.system("start http://localhost:6006/")
os.system(("python -m tensorboard.main --logdir " + log_dir))


#%SystemRoot%\system32\WindowsPowerShell\v1.0\powershell.exe -command start http://localhost:6006/; cd c:\research; python -m tensorboard.main --logdir "%1"






# scatter plot of the circles dataset with points colored by class
from sklearn.datasets import make_circles
from numpy import where
from matplotlib import pyplot
# generate circles
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# select indices of points with each class label
for i in range(2):
	samples_ix = where(y == i)
	pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
pyplot.legend()
pyplot.show()

import numpy as np
test = np.random.uniform(low=-0.001, high=0.001, size=(50,50))
test