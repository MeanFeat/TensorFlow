import os
from datetime import datetime
import tensorflow as tf
from pandas import read_csv
import sys
import numpy as np

def LoadCVS(file):
    return read_csv(file)

def GetTensorboardCallback(name):
    log_dir = 'logs\\' + name + datetime.now().strftime("%m%d%Y%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir= log_dir), log_dir

def LaunchTensorboard(directory):
    os.system("start http://localhost:6006/")
    os.system(("python -m tensorboard.main --logdir " + directory))

def WriteJson(m, file):
    np.set_printoptions(threshold=sys.maxsize)
    print("Writing to file: "+ file)
    str1 = "{\n"
    f = open(file, "w")
    layerIndex = 0
    isBias = False
    progress = "."
    for v in m.trainable_variables:
        print( progress, end='\r' )
        var = v.numpy()
        if isBias:
            str1 += "\t\t\"biasShape\" : [" + str(v.shape[0]) + ",1],\n"
            str1 += "\t\t\"biasVals\" :\n\t\t\t[\n\t\t\t\t"
            str1 += np.array2string(var, formatter={'float_kind':lambda x: "%.10f" % x} , separator=',', max_line_width=sys.maxsize) + "\n"
            str1 += "\t\t\t]\n"
            str1 += "\t},\n"
            layerIndex += 1
        else:
            str1 += "\t\"layer " + str(layerIndex) + "\" : {\n"
            str1 += "\t\t\"index\" :\"" + str(layerIndex) + "\",\n"
            str1 += "\t\t\"activation\" : \"" + m.layers[layerIndex].activation._tf_api_names[2] + "\",\n"
            str1 += "\t\t\"weightShape\" : "+ str(v.shape).replace('(', '[').replace(')',']') + ",\n"
            str1 += "\t\t\"weightVals\" :\n\t\t\t[\n"
            for e in range(len(var)):
                str1 += "\t\t\t\t" + np.array2string(var[e], formatter={'float_kind':lambda x: "%.10f" % x}, separator=',', max_line_width=sys.maxsize)
                if e == (len(var)-1):
                    str1 += "\n"
                else:
                 str1 += ",\n"
            str1 += "\t\t\t],\n"
        isBias = not isBias
        progress += "."
    str1 += "}"
    f.write(str1)
    f.close()
    print("\nFinished\n")
