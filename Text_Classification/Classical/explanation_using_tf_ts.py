'''
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

from collections import Counter
import itertools
import json
import os
import sys
from sklearn.tests.test_multiclass import n_classes
curr_path = os.path.dirname(__file__)
path = os.path.abspath(os.path.join(curr_path, '..'))
# path should be : '/Users/sarker/WorkSpaces/EclipseNeon/XAI/Text_Classification/'
print('path: ', path)
sys.path.append(path)

import pickle
import re
import time

import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.stem.porter import *
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2

from Classical import classifier_2_class as classifier_2_class
from Classical import data_helpers as dth
from Classical import multilayer_perceptron_custom
from Classical import utils as util
from multilayer_perceptron_custom import MLPClassifier_Custom
import tensorflow as tf

# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("../../data/mnist/", one_hot=True)

# get data
def get_data():
    
    '''
    '''
    x_text, y, y_label = dth.load_data_and_labels(util.train_data_folder, util.saving_data_file)

    words = classifier_2_class.make_word_embeddings(x_text, y, y_label)
    
    documents, classes = classifier_2_class.create_document_and_classes_from_data(x_text, y, y_label)
    
    '''documents is tuple
    documents = (text,label)
    classes is list of classes.
    classes = [0,0,0....1,1,1,]'''
    
    # remove duplicates
    classes = list(set(classes))
    
    # print(len(documents), " documents")
    # print(len(classes), " classes", classes)
    # print(len(words), " unique stemmed words")
    
    # training data
    training, output = classifier_2_class.convert_training_documents_to_vector(documents, classes, words)
    X_training = np.array(training)
    y_training = np.array(output)
    
    X_test, y_test = classifier_2_class.load_test_documents(util.test_file_dir,words)
    '''
    '''
    return X_training, X_test, y_training, y_test
        
train_X, test_X, train_y, test_y = get_data()

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
logs_path = util.saving_classifier_logs

# Network Parameters
n_hidden_1 = 4760 # 1st layer number of features
n_hidden_2 = 4760 # 2nd layer number of features
n_hidden_3 = 4760 # 1st layer number of features
n_hidden_4 = 4760 # 2nd layer number of features
n_input = 4760 # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 4760], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 2], name='LabelData')


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Create a summary to visualize the first layer ReLU activation
    tf.summary.histogram("sigmoid1", layer_1)
    
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Create another summary to visualize the second layer ReLU activation
    tf.summary.histogram("sigmoid2", layer_2)
    
        # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    # Create another summary to visualize the second layer ReLU activation
    tf.summary.histogram("sigmoid3", layer_3)
    
        # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['w4']), biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)
    # Create another summary to visualize the second layer ReLU activation
    tf.summary.histogram("sigmoid4", layer_4)
    
    # Output layer
    out_layer = tf.add(tf.matmul(layer_4, weights['w5']), biases['b5'])
    return out_layer

# Store layers weight & bias
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]), name='W3'),
    'w4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4]), name='W4'),
    'w5': tf.Variable(tf.random_normal([n_hidden_4, n_classes]), name='W5')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'b3': tf.Variable(tf.random_normal([n_hidden_3]), name='b3'),
    'b4': tf.Variable(tf.random_normal([n_hidden_4]), name='b4'),
    'b5': tf.Variable(tf.random_normal([n_classes]), name='b5')
}

# Encapsulating all ops into scopes, making Tensorboard's Graph
# Visualization more convenient
with tf.name_scope('Model'):
    # Build model
    pred = multilayer_perceptron(x, weights, biases)

with tf.name_scope('Loss'):
    # Softmax Cross entropy (cost function)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Op to calculate every variable gradient
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    # Op to update all variables according to their gradient
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
# Summarize all gradients
for grad, var in grads:
    print('var.name: ',var.name)
    tf.summary.histogram(var.name + '/gradient', grad)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path,
                                            graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = 1
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = train_X, train_y
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([apply_grads, loss, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", acc.eval({x: test_X, y: test_y}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")