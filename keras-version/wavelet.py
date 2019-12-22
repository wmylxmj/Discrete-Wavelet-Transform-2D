# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:56:25 2019

@author: wmy
"""

import pywt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.layers import Add, Conv2D, Input, Lambda, Activation
from keras.models import Model
from keras.layers import Conv3D, ZeroPadding3D, BatchNormalization
from keras.layers import LeakyReLU, concatenate, Reshape, Softmax
from IPython.display import SVG
from keras.utils import plot_model
from keras import layers

def dwt2d(x, wavelet='haar'):
    # shape x: (b, h, w, c)
    nc = int(x.shape.dims[3])
    # 小波波形
    w = pywt.Wavelet(wavelet)
    # 水平低频 垂直低频
    ll = np.outer(w.dec_lo, w.dec_lo)
    # 水平低频 垂直高频
    lh = np.outer(w.dec_hi, w.dec_lo)
    # 水平高频 垂直低频 
    hl = np.outer(w.dec_lo, w.dec_hi)
    # 水平高频 垂直高频
    hh = np.outer(w.dec_hi, w.dec_hi)
    # 卷积核
    core = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
    core[:, :, 0, 0] = ll[::-1, ::-1]
    core[:, :, 0, 1] = lh[::-1, ::-1]
    core[:, :, 0, 2] = hl[::-1, ::-1]
    core[:, :, 0, 3] = hh[::-1, ::-1]
    core = core.astype(np.float32)
    kernel = np.array([core], dtype=np.float32)
    kernel = K.constant(kernel, dtype='float32')
    p = 2 * (len(w.dec_lo) // 2 - 1)
    # padding odd length
    x = Lambda(tf.pad, arguments={'paddings':[[0, 0], [p, p+1], [p, p+1], [0, 0]]})(x)
    xh = K.shape(x)[1] - K.shape(x)[1] % 2
    xw = K.shape(x)[2] - K.shape(x)[2] % 2
    x = Lambda(lambda x: x[:, 0:xh, 0:xw, :])(x)
    # convert to 3d data
    x3d = Lambda(tf.expand_dims, arguments={'axis':1})(x)
    # 切开通道
    x3d = Lambda(tf.split, arguments={'num_or_size_splits':int(x3d.shape.dims[4]), 'axis':4})(x3d)
    # 贴到维度一
    x3d = Lambda(tf.concat, arguments={'axis':1})([a for a in x3d])
    # 三维卷积
    y3d = Lambda(tf.nn.conv3d, arguments={'filter':kernel, 'padding':'VALID', \
                                          'strides':[1, 1, 2, 2, 1]})(x3d)
    # 切开维度一
    y = Lambda(tf.split, arguments={'num_or_size_splits':int(y3d.shape.dims[1]), 'axis':1})(y3d)
    # 贴到通道维
    y = Lambda(tf.concat, arguments={'axis':4})([a for a in y])
    shape = (K.shape(y)[0], K.shape(y)[2], K.shape(y)[3], 4*nc)
    y = Lambda(tf.reshape, arguments={'shape':shape})(y)
    # 拼贴通道
    channels = Lambda(tf.split, arguments={'num_or_size_splits':nc, 'axis':3})(y)
    outputs = []
    for channel in channels:
        (cA, cH, cV, cD) = Lambda(tf.split, arguments={'num_or_size_splits':4, 'axis':3})(channel)
        AH = Lambda(tf.concat, arguments={'axis':2})([cA, cH])
        VD = Lambda(tf.concat, arguments={'axis':2})([cV, cD])
        outputs.append(Lambda(tf.concat, arguments={'axis':1})([AH, VD]))
        pass
    outputs = Lambda(tf.concat, arguments={'axis':-1})(outputs)
    return outputs

def idwt2d(x, wavelet='haar'):
    # shape x: (b, h, w, c)
    nc = int(x.shape.dims[3])
    # 小波波形
    w = pywt.Wavelet(wavelet)
    # 水平低频 垂直低频
    ll = np.outer(w.dec_lo, w.dec_lo)
    # 水平低频 垂直高频
    lh = np.outer(w.dec_hi, w.dec_lo)
    # 水平高频 垂直低频 
    hl = np.outer(w.dec_lo, w.dec_hi)
    # 水平高频 垂直高频
    hh = np.outer(w.dec_hi, w.dec_hi)
    # 卷积核
    core = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
    core[:, :, 0, 0] = ll[::-1, ::-1]
    core[:, :, 0, 1] = lh[::-1, ::-1]
    core[:, :, 0, 2] = hl[::-1, ::-1]
    core[:, :, 0, 3] = hh[::-1, ::-1]
    core = core.astype(np.float32)
    kernel = np.array([core], dtype=np.float32)
    kernel = K.constant(kernel, dtype='float32')
    s = 2 * (len(w.dec_lo) // 2 - 1)
    # 反变换
    hcA = tf.floordiv(tf.shape(x)[1], 2)
    wcA = tf.floordiv(tf.shape(x)[2], 2)
    y = []
    for c in range(nc):
        channel = Lambda(lambda x: x[:, :, :, c])(x)
        channel = Lambda(tf.expand_dims, arguments={'axis':-1})(channel)
        cA = Lambda(lambda x: x[:, 0:hcA, 0:wcA, :])(channel)
        cH = Lambda(lambda x: x[:, 0:hcA, wcA:, :])(channel)
        cV = Lambda(lambda x: x[:, hcA:, 0:wcA, :])(channel)
        cD = Lambda(lambda x: x[:, hcA:, wcA:, :])(channel)
        temp = Lambda(tf.concat, arguments={'axis':-1})([cA, cH, cV, cD])
        y.append(temp)
        pass
    # nc * 4
    y = Lambda(tf.concat, arguments={'axis':-1})(y)
    y3d = Lambda(tf.expand_dims, arguments={'axis':1})(y)
    y3d = Lambda(tf.split, arguments={'num_or_size_splits':nc, 'axis':4})(y3d)
    y3d = Lambda(tf.concat, arguments={'axis':1})([a for a in y3d])
    output_shape = [tf.shape(y)[0], tf.shape(y3d)[1], \
                    2*(tf.shape(y)[1]-1)+tf.shape(ll)[0], \
                    2*(tf.shape(y)[2]-1)+tf.shape(ll)[1], 1]
    x3d = Lambda(tf.nn.conv3d_transpose, arguments={'filter':kernel, 'output_shape':output_shape, \
                                                    'padding':'VALID', 'strides':[1, 1, 2, 2, 1]})(y3d)
    outputs = Lambda(tf.split, arguments={'num_or_size_splits':nc, 'axis':1})(x3d)
    outputs = Lambda(tf.concat, arguments={'axis':4})([x for x in outputs])
    shape = (tf.shape(outputs)[0], tf.shape(outputs)[2], tf.shape(outputs)[3], nc)
    outputs = Lambda(tf.reshape, arguments={'shape':shape})(outputs)
    outputs = Lambda(lambda x: x[:, s:2*(tf.shape(y)[1]-1)+np.shape(ll)[0]-s, \
                                 s:2*(tf.shape(y)[2]-1)+np.shape(ll)[1]-s, :])(outputs)
    return outputs

def wavedec2d(x, level=1, wavelet='haar'):
    if level == 0:
        return x
    y = dwt2d(x, wavelet=wavelet)
    hcA = tf.floordiv(tf.shape(y)[1], 2)
    wcA = tf.floordiv(tf.shape(y)[2], 2)
    cA = Lambda(lambda x: x[:, 0:hcA, 0:wcA, :])(y)
    cA = wavedec2d(cA, level=level-1, wavelet=wavelet)
    cA = Lambda(lambda x: x[:, 0:hcA, 0:wcA, :])(cA)
    hcA = tf.shape(cA)[1]
    wcA = tf.shape(cA)[2]
    cH = Lambda(lambda x: x[:, 0:hcA, wcA:, :])(y)
    cV = Lambda(lambda x: x[:, hcA:, 0:wcA, :])(y)
    cD = Lambda(lambda x: x[:, hcA:, wcA:, :])(y)
    AH = Lambda(tf.concat, arguments={'axis':2})([cA, cH])
    VD = Lambda(tf.concat, arguments={'axis':2})([cV, cD])
    outputs = Lambda(tf.concat, arguments={'axis':1})([AH, VD])
    return outputs
