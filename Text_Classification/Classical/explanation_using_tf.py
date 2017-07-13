'''
Created on Jul 12, 2017

@author: sarker
'''
# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.

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


with tf.Graph().as_default():
    with tf.Session() as sess:

        RANDOM_SEED = 42
        tf.set_random_seed(RANDOM_SEED)
        
        
        def init_weights(shape):
            """ Weight initialization """
            weights = tf.random_normal(shape, stddev=0.1)
            return tf.Variable(weights)
        
        def forwardprop(X, w_1, w_2, w_3, w_4, w_5):
            """
            Forward-propagation.
            IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
            """
            ''' hidden layer 1'''
            h1 = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
            
            ''' hidden layer 2'''
            h2 = tf.nn.sigmoid(tf.matmul(h1, w_2))
            
            ''' hidden layer 3'''
            h3 = tf.nn.sigmoid(tf.matmul(h2, w_3))
            
            ''' hidden layer 4'''
            h4 = tf.nn.sigmoid(tf.matmul(h3, w_4))
            
            ''' output layer '''
            yhat = tf.matmul(h4, w_5)  # The \varphi function
            
            return yhat
        
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
        
        def main():
            train_X, test_X, train_y, test_y = get_data()
            # print(train_y)
            
            # Layer's sizes
            x_size = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
            h_size = 256  # Number of hidden nodes
            y_size = train_y.shape[1]  # Number of outcomes (3 iris flowers)
            #print('train_X.shape[1]: ', train_X.shape[1])
            
            # Symbols
            X = tf.placeholder("float", shape=[None, x_size])
            y = tf.placeholder("float", shape=[None, y_size])
        
            # Weight initializations
            w_1 = init_weights((x_size, h_size))
            w_2 = init_weights((h_size, h_size))
            w_3 = init_weights((h_size, h_size))
            w_4 = init_weights((h_size, h_size))
            w_5 = init_weights((h_size, y_size))
        
            # Forward propagation
            yhat = forwardprop(X, w_1, w_2, w_3, w_4, w_5)
            predict = tf.argmax(yhat, axis=1)
            
            # Backward propagation
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
            updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
            
            # Run initializer
            init = tf.global_variables_initializer()
            sess.run(init)
            
            # logging for tensorboard
            summary_writer = tf.summary.FileWriter(util.saving_classifier_logs, graph=tf.get_default_graph())
            
            # train
            for epoch in range(500):
                # Train with each example
                for i in range(len(train_X)):
                    sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
        
                train_accuracy = np.mean(np.argmax(train_y, axis=1) == 
                                         sess.run(predict, feed_dict={X: train_X, y: train_y}))
                test_accuracy = np.mean(np.argmax(test_y, axis=1) == 
                                          sess.run(predict, feed_dict={X: test_X, y: test_y}))
         
                print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                       % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
#                 print("Epoch = %d, train accuracy = %.2f%%"
#                       % (epoch + 1, 100. * train_accuracy))
        
            sess.close()
        
        if __name__ == '__main__':
            main()
