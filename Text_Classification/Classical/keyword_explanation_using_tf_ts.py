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
    x_keyword, y_keyword, y_label_keyword = dth.load_keywords_and_labels(util.train_data_folder, util.saving_data_file,use_cache= False)
#     print('len(x_keyword), type(x_keyword): ',len(x_keyword),type(x_keyword))
    
    words = classifier_2_class.make_word_embeddings(x_text, y, y_label,use_cache = False)
    words_keywords = classifier_2_class.make_word_embeddings(x_keyword, y_keyword, y_label_keyword,use_cache=False)
    print('len(words), type(words): ',len(words),type(words))
    print('len(words_keywords), type(words_keywords): ',len(words_keywords),type(words_keywords))
    
    documents, classes = classifier_2_class.create_document_and_classes_from_data(x_text, y, y_label)
    documents_keyword, classes_keywords = classifier_2_class.create_document_and_classes_from_data(x_keyword, y_keyword, y_label_keyword)
#     print('type(documents), len(documents): ',type(documents), len(documents))
#     print('type(documents_keyword), len(documents_keyword): ',type(documents_keyword),len(documents_keyword))
#     print('words_keywords: ',words_keywords)
    #print('classes_keywords[0]: ',classes_keywords[0]) 
     
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
    training, output = classifier_2_class.convert_training_documents_to_vector(documents, classes, words,use_cache = False)
    training_keyword, output_keyword = classifier_2_class.convert_training_documents_to_vector(documents_keyword, classes_keywords, words_keywords,use_cache = False)
    print('len(training[0]), type(training[0]): ',len(training[0]),type(training[0]))
    print('len(training_keyword[0]), type(training_keyword[0]): ',len(training_keyword[0]),type(training_keyword[0]))
    
    X_training = np.array(training)
    y_training = np.array(output)
    
    X_training_keyword = np.array(training_keyword)
    y_training_keyword = np.array(output_keyword)
    
    X_test, y_test = classifier_2_class.load_test_documents(util.test_file_dir,words)
    '''
    '''
    return X_training, X_test, y_training, y_test , X_training_keyword, y_training_keyword
        
train_X, test_X, train_y, test_y, X_training_keyword, y_training_keyword = get_data()

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
logs_path = util.saving_classifier_logs

print('train_X.shape[1]',train_X.shape[1])
print('X_training_keyword.shape[1]',X_training_keyword.shape[1])

# Network Parameters
n_hidden_1 = 4760 # 1st layer number of features
n_hidden_2 = 4760 # 2nd layer number of features
n_hidden_3 = 4760 # 1st layer number of features
n_hidden_4 = 4760 # 2nd layer number of features
n_input = 4760 # MNIST data input (img shape: 28*28)
n_hidden_keyword_1 = 121 # 1st layer number of features
n_hidden_keyword_2 = 121 # 2nd layer number of features
n_hidden_keyword_3 = 121 # 1st layer number of features
n_hidden_keyword_4 = 121 # 2nd layer number of features
n_input_keyword = 121 # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 4760], name='Input_Data')
# keyword data
x_keyword = tf.placeholder(tf.float32, [None, 121], name='Input_keyword_Data')
print(x)
print(x_keyword)
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 2], name='LabelData')


# Create model
def multilayer_perceptron(x, weights, biases, x_keyword, weights_keyword,biases_key):
    # Hidden layer with RELU activation
    print('x.shape, weights[w1].shape: ', x.shape, weights['w1'].shape)
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Create a summary to visualize the first layer ReLU activation
    tf.summary.histogram("layer_1", layer_1)
    # Hidden layer with RELU activation
    print('x_keyword.shape, weights_keyword[w1_key].shape: ', x_keyword.shape, weights_keyword['w1_key'].shape)
    layer_1_keyword = tf.add(tf.matmul(x_keyword, weights_keyword['w1_key']), biases_key['b1_key'])
    layer_1_keyword = tf.nn.sigmoid(layer_1_keyword)
    # Create a summary to visualize the first layer ReLU activation
    tf.summary.histogram("layer_1_keyword", layer_1_keyword)
    
    
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Create another summary to visualize the second layer ReLU activation
    tf.summary.histogram("sigmoid2", layer_2)
    # Hidden layer with RELU activation
    layer_2_keyword = tf.add(tf.matmul(layer_1_keyword, weights_keyword['w2_key']), biases_key['b2_key'])
    layer_2_keyword = tf.nn.sigmoid(layer_2_keyword)
    # Create another summary to visualize the second layer ReLU activation
    tf.summary.histogram("layer_2_keyword", layer_2_keyword)
    
        # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    # Create another summary to visualize the second layer ReLU activation
    tf.summary.histogram("sigmoid3", layer_3)
        # Hidden layer with RELU activation
    layer_3_keyword = tf.add(tf.matmul(layer_2_keyword, weights_keyword['w3_key']), biases_key['b3_key'])
    layer_3_keyword = tf.nn.sigmoid(layer_3_keyword)
    # Create another summary to visualize the second layer ReLU activation
    tf.summary.histogram("layer_3_keyword", layer_3_keyword)
    
    
        # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['w4']), biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)
    # Create another summary to visualize the second layer ReLU activation
    tf.summary.histogram("sigmoid4", layer_4)
    # Hidden layer with RELU activation
    layer_4_keyword = tf.add(tf.matmul(layer_3_keyword, weights_keyword['w4_key']), biases_key['b4_key'])
    layer_4_keyword = tf.nn.sigmoid(layer_4_keyword)
    # Create another summary to visualize the second layer ReLU activation
    tf.summary.histogram("layer_4_keyword", layer_4_keyword)
    
    # Output layer
    out_layer = tf.add(tf.matmul(layer_4, weights['w5']), biases['b5'])
    out_layer_keyword = tf.add(tf.matmul(layer_4_keyword, weights_keyword['w5_key']), biases_key['b5_key'])
    return [out_layer, out_layer_keyword]

# Store layers weight & bias

weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]), name='W3'),
    'w4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4]), name='W4'),
    'w5': tf.Variable(tf.random_normal([n_hidden_4, n_classes]), name='W5')
}
weights_keyword = {
    'w1_key': tf.Variable(tf.random_normal([n_input_keyword, n_hidden_keyword_1]), name='W1_keyword'),
    'w2_key': tf.Variable(tf.random_normal([n_hidden_keyword_1, n_hidden_keyword_2]), name='W2_keyword'),
    'w3_key': tf.Variable(tf.random_normal([n_hidden_keyword_2, n_hidden_keyword_3]), name='W3_keyword'),
    'w4_key': tf.Variable(tf.random_normal([n_hidden_keyword_3, n_hidden_keyword_4]), name='W4_keyword'),
    'w5_key': tf.Variable(tf.random_normal([n_hidden_keyword_4, n_classes]), name='W5_keyword')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'b3': tf.Variable(tf.random_normal([n_hidden_3]), name='b3'),
    'b4': tf.Variable(tf.random_normal([n_hidden_4]), name='b4'),
    'b5': tf.Variable(tf.random_normal([n_classes]), name='b5')
}
biases_key = {
    'b1_key': tf.Variable(tf.random_normal([n_hidden_keyword_1]), name='b1_key'),
    'b2_key': tf.Variable(tf.random_normal([n_hidden_keyword_2]), name='b2_key'),
    'b3_key': tf.Variable(tf.random_normal([n_hidden_keyword_3]), name='b3_key'),
    'b4_key': tf.Variable(tf.random_normal([n_hidden_keyword_4]), name='b4_key'),
    'b5_key': tf.Variable(tf.random_normal([n_classes]), name='b5_key')
}

# Encapsulating all ops into scopes, making Tensorboard's Graph
# Visualization more convenient
with tf.name_scope('Model'):
    # Build model
    [pred, pred_keyword] = multilayer_perceptron(x, weights, biases, x_keyword, weights_keyword, biases_key)
    print(type(pred), type(pred_keyword))
    
with tf.name_scope('Loss'):
    # Softmax Cross entropy (cost function)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_keyword, labels=y))
    loss_keyword = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_keyword, labels=y))

with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Op to calculate every variable gradient
    grads = tf.gradients(loss, tf.trainable_variables())
    grads_keyword = tf.gradients(loss_keyword, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    grads_keyword = list(zip(grads_keyword, tf.trainable_variables()))
    for g in grads:
        print('grads: ', g)
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
    print('var.name: ', grad)
    pass
    '''problem is occuring here'''
    # print('var.name: ', var.name)
    # tf.summary.histogram(var.name + '/gradient', grad)
   
   
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
            batch_xs_key, batch_ys_key = X_training_keyword, train_y
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([apply_grads, loss, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys, x_keyword: batch_xs_key})
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