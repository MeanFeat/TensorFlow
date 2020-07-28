import os
from datetime import datetime
import tensorflow as tf
from pandas import read_csv
import sys
import numpy as np
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def LoadCVS(file):
    return read_csv(file)

def GetDateTimeString():
    return datetime.now().strftime("%m%d%Y%H%M%S")

def GetTensorboardCallback(name):
    log_dir = 'logs\\' + name + GetDateTimeString()
    return tf.keras.callbacks.TensorBoard(log_dir= log_dir), log_dir

def LaunchTensorboard(directory):
    os.system("start http://localhost:6006/")
    os.system(("python -m tensorboard.main --logdir " + directory))

def WriteJson(m, filePrefix, results = -1.0, timeInFileName = False):
    resultStr = ""
    timeStr = ""
    if results >= 0:
        resultStr = "-" +str(round(results, 3))
    if timeInFileName:
        timeStr = "-" + GetDateTimeString()
    file = filePrefix + timeStr + resultStr + ".json"
    tvars = m.trainable_variables
    activations = []
    biases = []
    weights = []
    isWeight = True # Weights and biases alternate in the trainable_variables
    for v in range(len(tvars)):
        if isWeight:
            weights.append(tvars[v].numpy())
        else:
            biases.append(tvars[v].numpy())
        isWeight = not isWeight

    with open(file, "w") as write_file:
        write_file.write('[')
        for w in range(len(weights)):
            act = m.layers[w].activation._tf_api_names[0].split('.')
            layer = {
                "layer" : w,
                "activation" : act[len(act)-1],
                "weightShape" : [len(weights[w]), len(weights[w][0]-1)],
                "weightVals" : weights[w],
                "biasShape" : [len(biases[w]), 1],
                "biasVals" : biases[w]
            }
            json.dump(layer, write_file, indent = 4,  cls=NumpyArrayEncoder)
            if(w < len(weights)-1):
                write_file.write(',')
        write_file.write(']')
    print("Exported to file: ", file)