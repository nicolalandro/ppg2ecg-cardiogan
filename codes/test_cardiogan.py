import os
import socket
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import sklearn.preprocessing as skp
import tflib


import module 
import preprocessing

tf.keras.backend.set_floatx('float64')
tf.autograph.set_verbosity(0)

@tf.function
def sample_P2E(P, model):
    fake_ecg = model(P, training=False)
    return fake_ecg




########### params ###########
ecg_sampling_freq = 128
ppg_sampling_freq = 128
window_size = 4
ecg_segment_size = ecg_sampling_freq*window_size
ppg_segment_size = ppg_sampling_freq*window_size
model_dir = '../weights'

""" model """
Gen_PPG2ECG = module.generator_attention()
""" resotre """
tflib.Checkpoint(dict(Gen_PPG2ECG=Gen_PPG2ECG), model_dir).restore()
# Gen_PPG2ECG.load_model(os.path.join(model_dir, 'ckpt-1.index'))
# loaded_model = keras.models.load_model(model_dir)
print("model loaded successfully")

""" ONNX convert """
# import tf2onnx
# import onnx
# onnx_model, _ = tf2onnx.convert.from_keras(Gen_PPG2ECG)
# onnx.save_model(onnx_model, '../ppg2ecg.onnx')


""" please process the data as mentioned below before extracting ECG output """
# load the data: x_ppg = np.loadtxt()
# make sure loaded data is a numpy array: x_ppg = np.array(x_ppg)
# resample to 128 Hz using: cv2.resize(x_ppg, (1,ppg_segment_size), interpolation = cv2.INTER_LINEAR)
# filter the data using: preprocessing.filter_ppg(x_ppg, 128)
# make an array to N x 512 [this is the input shape of x_ppg], where Nx512=len(x_ppg)
# normalize the data b/w -1 to 1: x_ppg = skp.minmax_scale(x_ppg, (-1, 1), axis=1)
#######
#x_ecg = sample_P2E(x_ppg, Gen_PPG2ECG)
#######

x_ppg = np.loadtxt('ppg_example.txt')
x_ppg = np.array(x_ppg)
x_ppg = cv2.resize(x_ppg, (1, ppg_segment_size), interpolation = cv2.INTER_LINEAR)
x_ppg = np.moveaxis(x_ppg, -1, 0)
# x_ppg = preprocessing.filter_ppg(x_ppg, 128)
x_ppg = skp.minmax_scale(x_ppg, (-1, 1), axis=1)

x_ecg = sample_P2E(x_ppg, Gen_PPG2ECG)
# print(x_ecg[:,0].shape, x_ppg.shape)
one_ecg = x_ecg[0]

from matplotlib import pyplot as plt

fig, (ax1) = plt.subplots(1, 1,
                          sharex = False,
                          sharey = False,
                          figsize = (12,1))
fs = 128
t = np.arange(0,len(one_ecg)/fs,1.0/fs)
# fig.suptitle('PPG') 
ax1.plot(t, x_ppg[0], color = 'black')
ax1.plot(t, one_ecg, color = 'blue')
plt.savefig('test.png')
