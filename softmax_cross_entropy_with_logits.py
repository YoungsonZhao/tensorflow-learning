#!/usr/bin/env python

# ======================================================================
#       Copyright (C) 2017 Institute of Cyber-Systems & Control.
#
#               File Name: softmax_cross_entropy_with_logits.py
#                  Author: Dr. Yongsheng Zhao
#                   Email: zhaoyongsheng@zju.edu.cn
#           Creating Data: 11 07, 2017
#            Discription:
#
# ======================================================================

import tensorflow as tf

# Output of Neural Network

logits = tf.constant([[1.0, 2.0, 3.0],
                      [2.0, 3.0, 2.0],
                      [3.0, 4.0, 4.0]])

label = tf.constant([[0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0]])

logits_softmax = tf.nn.softmax(logits)

logits_exp = tf.exp(logits)
logits_exp_mean = tf.reduce_sum(logits_exp, 1)
logits_exp_mean_reshap = tf.reshape(logits_exp_mean, (-1, 1))
logits_softmax_ = tf.div(logits_exp, logits_exp_mean_reshap)

with tf.Session() as sess:
    print("softmax = :")
    print(sess.run(logits_softmax))
    print("softmax verified = :")
    print(sess.run(logits_softmax_))
