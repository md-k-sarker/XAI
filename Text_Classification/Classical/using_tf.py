'''
Created on Jul 9, 2017

@author: sarker
'''
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




with tf.Graph().as_default():
    with tf.Session() as sess:
        
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
        
        '''
        Changing y_training
        '''
        y_training = np.ones(4670)
        
        
        print('X_training.shape: ', X_training.shape)
        print('y_training.shape: ', y_training.shape)
        X_test, y_test = classifier_2_class.load_test_documents(util.test_file_dir, words)
        print('X_Test.shape: ', X_test.shape)
        print('y_test.shape: ', y_test.shape)
        
        no_of_input_neurons = ((len(X_training[0])))
        no_of_hidden_neurons = no_of_input_neurons
        hidden_layer_sizes = []
        for i in range(5):
            hidden_layer_sizes.append(no_of_hidden_neurons)
            
        # clf = classifier_2_class.train_NN(X_training, y_training, no_of_hidden_layer=5, max_iter=5)

        
#         # train data
        X_train_ = tf.constant(X_training)
        y_train_ = tf.constant(y_training)
        
        # test data
        X_test_ = tf.constant(X_test)
        y_test_ = tf.constant(y_test)
        
        def get_train_inputs():
            x = tf.constant(X_training,dtype=np.float32)
            y = tf.constant(y_training,dtype=np.float32)
            print('type(x): ', type(x))
            print('type(y): ', type(y))
            print('y: ', y)
            print('sess.run(y): ', sess.run(y))
            return x, y 
        
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=no_of_input_neurons)]
        clf = tf.contrib.learn.DNNClassifier(hidden_units=hidden_layer_sizes,
                                              model_dir=util.saving_classifier_logs,
                                              feature_columns=feature_columns,
                                              n_classes = 2)
        
        # fit model
        sess.run(clf.fit(input_fn=get_train_inputs(), steps=5))
