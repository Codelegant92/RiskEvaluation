#!/usr/bin/python
# -*- coding: utf-8 -*-
# Created by RobinCHEN on 12/8/2016

from commonFunction import *
import tensorflow as tf
from featureSupplement import *

def MLP(trainFeature, trainLabel, testFeature):
    N1 = trainFeature.shape[0]
    N2 = testFeature.shape[0]
    D = trainFeature.shape[1]
    x = tf.placeholder(tf.float32, [None, D])
    W = tf.Variable(tf.zeros([D, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 2])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    label1 = np.zeros([N1, 2])
    for item in range(N1):
        label1[item][trainLabel[item]] = 1
    sess = tf.Session()
    sess.run(init)
    idx = [i for i in range(N1)]
    for i in range(100):
        randomSamples = random.sample(idx, 5)
        batch_xs = trainFeature[randomSamples, :]
        batch_ys = label1[randomSamples]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 10 == 0:
            print(i, sess.run(W), sess.run(b))

    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    predicted_label = tf.arg_max(y, 1)
    return(sess.run(predicted_label, feed_dict={x: testFeature}))



if __name__ == "__main__":
    trainFeature, trainLabel, testFeature, testPlatform = readFeature(5, 0.5, 10, 0.6, 15, 0.6, 5, 0.6, 20)
    #testLabel = np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]])
    #testLabel = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
    testLabel = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
    #label = list(trainLabel)
    #label.extend(testLabel)
    #feature = np.concatenate([trainFeature, testFeature])[:, :]
    #label = np.array(label)

    print(MLP(trainFeature, trainLabel, testFeature))
    #featureFolder, labelFolder = crossValidation(feature, label, 3)

    #accu13, accu23 = crossValidationFunc(featureFolder, labelFolder, MLP)