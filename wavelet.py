# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:50:03 2019

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
import tensorflow as tf

def dwt2d(x, wave='haar'):
    # shape x: (b, h, w, c)
    nc = int(x.shape.dims[3])
    # 小波波形
    w = pywt.Wavelet(wave)
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
    kernel = tf.convert_to_tensor(kernel)
    p = 2 * (len(w.dec_lo) // 2 - 1)
    with tf.variable_scope('dwt2d'):
        # padding odd length
        if tf.shape(x)[1] % 2 == 1 and tf.shape(x)[2] % 2 == 1:
            x = tf.pad(x, tf.constant([[0, 0], [p, p+1], [p, p+1], [0, 0]]), mode='reflect')
            pass
        elif tf.shape(x)[1] % 2 == 1:
            x = tf.pad(x, tf.constant([[0, 0], [p, p+1], [p, p], [0, 0]]), mode='reflect')
            pass
        elif tf.shape(x)[2] % 2 == 1:
            x = tf.pad(x, tf.constant([[0, 0], [p, p], [p, p+1], [0, 0]]), mode='reflect')
            pass
        else:      
            x = tf.pad(x, tf.constant([[0, 0], [p, p], [p, p], [0, 0]]), mode='reflect')
            pass
        x3d = tf.expand_dims(x, 1)
        # 切开通道
        x3d = tf.split(x3d, int(x3d.shape.dims[4]), 4)
        # 贴到维度一
        x3d = tf.concat([a for a in x3d], 1)
        # 三维卷积
        y3d = tf.nn.conv3d(x3d, kernel, padding='VALID', strides=[1, 1, 2, 2, 1])
        # 切开维度一
        y = tf.split(y3d, int(y3d.shape.dims[1]), 1)
        # 贴到通道维
        y = tf.concat([a for a in y], 4)
        y = tf.reshape(y, (tf.shape(y)[0], tf.shape(y)[2], tf.shape(y)[3], tf.shape(y)[4]))
        # 拼贴通道
        channels = tf.split(y, nc, 3)
        outputs = []
        for channel in channels:
            (cA, cH, cV, cD) = tf.split(channel, 4, 3)
            AH = tf.concat([cA, cH], axis=2)
            VD = tf.concat([cV, cD], axis=2)
            outputs.append(tf.concat([AH, VD], axis=1))
            pass
        outputs = tf.concat(outputs, axis=-1)
        pass
    return outputs

tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [None, None, None, 3], name='inputs')
image = plt.imread('test.jpg')
x = np.array([image, image[:, ::-1, :]])
outputs = dwt2d(inputs)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(outputs, feed_dict={inputs:x})
    plt.imshow(np.array(result[0], dtype=np.uint8))
    plt.show()
    pass