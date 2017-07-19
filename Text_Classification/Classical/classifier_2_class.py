'''
2 class Classifier and explainer.
Classifier using sk-learn model
'''
'''
Created on Jun 16, 2017

@author: sarker
'''
from collections import Counter
import itertools
import json
import os
import pickle
import re
import time

import nltk
from nltk.stem.porter import *
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2

from Classical import data_helpers as dth
from Classical import multilayer_perceptron_custom
from Classical import utils as util
import matplotlib.pyplot as plt
from multilayer_perceptron_custom import MLPClassifier_Custom
import numpy as np
import tensorflow as tf


# from sklearn.neural_network import MLPClassifier
# from bokeh.plotting import figure, output_file, show
def make_word_embeddings(x_text, y, y_label, use_cache=False):
    '''word embedding.
    vector of all words'''
    
    # If file is saved then just load the file
    if (os.path.isfile(util.saving_global_words_embedding_file) and use_cache):
        words = []
        with open(os.path.abspath(util.saving_global_words_embedding_file), 'rb') as f:
            words = pickle.load(f)

    else:
        words = []
        
        # loop through each documents in our training data
        for each_text, each_y, each_label in zip(x_text, y, y_label):
            # print('x_text: ', x_text)
            # tokenize each word in the document
            w = nltk.word_tokenize(each_text)
            # add to our words list
            words.extend(w)

        # stem and lower each word and remove duplicates
        # words = [stemmer.stem(w.lower())
        #     for w in words if w not in ignore_words]

        # save word embeddings to cache
        words = list(set(words))
        with open(os.path.abspath(util.saving_global_words_embedding_file), 'wb') as f:
            pickle.dump(words, f)

    return words

def create_document_and_classes_from_data(x_text, y, y_label):
    # loop through each sample in x_text and create document as pair of (text,label)
    documents = []
    classes = []
    for text, label in zip(x_text, y_label):

        # add to documents in our corpus
        documents.append((text, label))
        # add to our classes list
        # if label not in classes:
        classes.append(label)

    return documents, classes

def convert_training_documents_to_vector(documents, classes, words, use_cache=False):
    '''
    Convert documents to vector.
    Parameters:
    documents = (text,label)
    classes = ['class_1', 'class_2']

    Return:
    training: list_of[list_of_words_from_doc_1,list_of_words_from_doc_2....]
    [[]]
    Output: list_of[output_vector_for_doc_1,output_vector_for_doc_2...]
    '''
    print("convert_training_documents_to_vector started...")
    # create an empty array for our output
    training = []
    output = []
    output_empty = [0] * len(classes)
    # If file is saved then just load the file
    if os.path.isfile(util.saving_bag_of_words_data_file) and use_cache:
        print(util.saving_bag_of_words_data_file, ' found in disk. loading it.')
        with open(os.path.abspath(util.saving_bag_of_words_data_file), 'rb') as f:
            training = pickle.load(f)

        for doc in documents:
            # convert output to vector
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            output.append(output_row)

    else:
        # training set, bag of words for each document
        print(util.saving_bag_of_words_data_file,
              ' not found in cache. doing operations....')
        for doc in documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
#             pattern_words = [stemmer.stem(word.lower())
#                              for word in pattern_words]
            # create our bag of words array
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            training.append(bag)

            # convert output to vector
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            output.append(output_row)

        # save to cache
        with open(os.path.abspath(util.saving_bag_of_words_data_file), 'wb') as f:
            pickle.dump(training, f)
    print("convert_training_documents_to_vector finished.")
    return training, output

def train_NN(X_training, y_training, no_of_hidden_layer, max_iter, use_cache=False, for_keyword=False):
    start_time = time.time()
    print('train_NN started...')
    if  use_cache:
        if for_keyword:
            if os.path.isfile(util.saving_classifier_model_keyword_file): 
                print('existing keyword mlp model found in cache. loading it.')
                clf = joblib.load(util.saving_classifier_model_keyword_file)
                print('clf: ', clf)
        else:
            if os.path.isfile(util.saving_classifier_model_file): 
                print('existing mlp model found in cache. loading it.')
                clf = joblib.load(util.saving_classifier_model_file)
                print('clf: ', clf)
    else:
        print('mlp initilizing started')
        no_of_hidden_neurons = ((len(X_training[0])))
        hidden_layer_sizes = ()
        for i in range(no_of_hidden_layer):
            hidden_layer_sizes += (no_of_hidden_neurons,)

        mlp = MLPClassifier_Custom(hidden_layer_sizes=hidden_layer_sizes,
                            solver='adam', activation='relu',
                            learning_rate='adaptive', learning_rate_init=0.001,
                            max_iter=max_iter,
                            verbose=True, tol=0.000000001)
        print('clf: ', mlp)

        print('mlp fitting started...')
        mlp.fit(X_training, y_training)
        print('mlp fitting finished')

        '''save to disk'''
        if for_keyword:
            joblib.dump(mlp, util.saving_classifier_model_keyword_file)
        else:
            joblib.dump(mlp, util.saving_classifier_model_file)            
        clf = mlp
    end_time = time.time()
    print('trained in: ', end_time - start_time, ' seconds')
    print('train_NN finished')
    return clf

def load_test_documents(test_file_dir, words):
    X_test = []
    y_test = []
    class_names = {}
    folder_counter = 0
    for folder_name in os.listdir(test_file_dir):
        if not folder_name.startswith('.'):
            '''Do operations for each folder/class'''
            print('folder_name: ', folder_name)
            class_names[folder_counter] = folder_name
            folder_counter += 1
            file_counter = 0
            for file_name in os.listdir(os.path.join(test_file_dir, folder_name)):
                if not file_name.startswith('.') and not file_name.endswith('keyword'):
                    file_counter += 1
                    document = open(os.path.join(test_file_dir, folder_name, file_name),
                                    mode='r', encoding='utf-8', errors='ignore').read()

                    bag_0_1_vector, bag_words_vector, bag_words = dth.bow(
                        document.lower(), words, show_details=False)
                    y = [0, 0]
                    if folder_name == 'comp.windows.x':
                        y = [1, 0]
                    elif folder_name == 'rec.sport.hockey':
                        y = [0, 1]

                    X_test.append(bag_0_1_vector)
                    y_test.append(y)
    return np.array(X_test), np.array(y_test)

# append training and test for feature selection

def select_feature(X_training, X_test, y_training, y_test):
    train = len(X_training)
    X_training = np.vstack((X_training, X_test))
    y_training_ = np.vstack((y_training, y_test))
    print('after appending X_training: ', X_training.shape)
    print('after appending y_training: ', y_training_.shape)

    '''feature selection'''
    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # X_training_selected_features = sel.fit_transform(X_training)
    X_training_selected_features = SelectKBest(
        chi2, k=3474).fit_transform(X_training, y_training_)
    '''split into training and test'''
    X_training = X_training_selected_features[:train]
    X_test = X_training_selected_features[train:]

    y_training = y_training_[:train]
    y_test = y_training_[train:]
    print('after feature selection X_training: ', X_training.shape)
    print('after feature selection X_test: ', X_test.shape)
    print('after feature selection y_training: ', y_training.shape)
    print('after feature selection y_test: ', y_test.shape)

    return X_training, X_test, y_training, y_test



# load data
