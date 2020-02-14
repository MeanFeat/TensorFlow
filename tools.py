import os
from datetime import datetime
import tensorflow as tf
from pandas import read_csv

def LoadCVS(file):
    return read_csv(file)

def GetTensorboardCallback(name):
    log_dir = 'logs\\' + name + datetime.now().strftime("%m%d%Y%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir= log_dir), log_dir

def LaunchTensorboard(directory):
    os.system("start http://localhost:6006/")
    os.system(("python -m tensorboard.main --logdir " + directory))