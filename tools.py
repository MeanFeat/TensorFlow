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

import sys
import numpy as np

def WriteWeightsToFile(m, file):
    np.set_printoptions(threshold=sys.maxsize)
    print("Writing to file: "+ file)
    str1 = ""
    f = open(file, "w")
    layerIndex = 0
    isBias = False
    progress = "."
    for v in m.trainable_variables:
        print( progress, end='\r' )
        var = v.numpy()
        if isBias:
            str1 += "   bias " + str(v.shape) + " {\n"
            str1 += "       " + np.array2string(var, formatter={'float_kind':lambda x: "%.10f" % x} , separator=',', max_line_width=sys.maxsize) + "\n"
            str1 += "   }\n"
            layerIndex += 1
            str1 += "}\n"
        else:
            str1 += "layer " + str(layerIndex) + " {\n"
            str1 += "   weight "+ str(v.shape) + " {\n"
            for e in var:
                str1 += "       " + np.array2string(e, formatter={'float_kind':lambda x: "%.10f" % x}, separator=',', max_line_width=sys.maxsize) + "\n"
            str1 += "   }\n"
        isBias = not isBias
        progress += "."
    f.write(str1)
    f.close()
    print("\nFinished\n")