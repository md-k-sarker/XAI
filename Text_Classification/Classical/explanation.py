'''
Created on Jul 9, 2017

@author: sarker
'''
from collections import Counter
import itertools
import json
import os
import sys
curr_path = os.path.dirname(__file__)
path = os.path.abspath(os.path.join(curr_path, '..'))
# path should be : '/Users/sarker/WorkSpaces/EclipseNeon/XAI/Text_Classification/'
print('path: ', path)
sys.path.append(path)

import pickle
import re
import time
from random import randint
from collections import Counter

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


# from sklearn.neural_network import MLPClassifier
# from bokeh.plotting import figure, output_file, show
print("Program started")

color = ['green', 'purple', 'black', 'cyan', 'green', 'blue']

min_activation = 8
min_activation_for_test = 1

_DEBUG_ = True

def preprocess_concepts(concepts):
    '''Convert panda dataframe concepts to list of concepts
    
    Parameters:
    ----------
    concepts: panda dataframe
        Example: [['TeamSport' 'Racing' 'OlympicGames' 'Golf']
             ['Sport' 0 0 0]
             ['Game' 'Smoking' 'SocialParty' 'Vacationing']
             ['RecreationOrExercise' 'Repairing' 'Maintaining' 'SocialInteraction']]
    Returns:
    -------
    concepts: concepts: [.....[n-1_th_hidden_layer_concepts],[n_th_hidden_layer_concepts],[output_layer_concepts]]         
    '''
    concepts = concepts.values
    concepts_as_list = []
    
    for i in range(concepts.shape[0]-1,-1,-1):
        concepts_as_list.append(concepts[i, :])
        
    return concepts_as_list



# get data

def get_data():
    '''
    DataSet
    Computer is class 0 = [1 0]
    Baseball is class 1 = [0 1]
    Actually this is not always true. It depends on how order of file appears to python. It may not be in sorted order. 
    '''
    x_text, y, y_label = dth.load_data_and_labels(util.train_data_folder, util.saving_data_file)
        
    x_keyword, y_keyword, y_label_keyword = dth.load_keywords_and_labels(util.train_data_folder, util.saving_data_file, use_cache=False)
#     print('len(x_keyword), type(x_keyword): ',len(x_keyword),type(x_keyword))
    
    words = classifier_2_class.make_word_embeddings(x_text, y, y_label, use_cache=False)
    words_keywords = classifier_2_class.make_word_embeddings(x_keyword, y_keyword, y_label_keyword, use_cache=False)
    print('len(words), type(words): ', len(words), type(words))
    print('len(words_keywords), type(words_keywords): ', len(words_keywords), type(words_keywords))
    
    documents, classes = classifier_2_class.create_document_and_classes_from_data(x_text, y, y_label)
    documents_keyword, classes_keywords = classifier_2_class.create_document_and_classes_from_data(x_keyword, y_keyword, y_label_keyword)
#     print('type(documents), len(documents): ',type(documents), len(documents))
#     print('type(documents_keyword), len(documents_keyword): ',type(documents_keyword),len(documents_keyword))
#     print('words_keywords: ',words_keywords)
    # print('classes_keywords[0]: ',classes_keywords[0]) 
     
    '''documents is tuple
    documents = (text,label)
    classes is list of classes.
    classes = [0,0,0....1,1,1,]'''
    
    # remove duplicates
    classes = list(set(classes))
    classes_keywords = list(set(classes_keywords))
    
    # print(len(documents), " documents")
    # print(len(classes), " classes", classes)
    # print(len(words), " unique stemmed words")
    
    # training data
    training, output = classifier_2_class.convert_training_documents_to_vector(documents, classes, words, use_cache=False)
    training_keyword, output_keyword = classifier_2_class.convert_training_documents_to_vector(documents_keyword, classes_keywords, words_keywords, use_cache=False)
    print('len(training[0]), type(training[0]): ', len(training[0]), type(training[0]))
    print('len(training_keyword[0]), type(training_keyword[0]): ', len(training_keyword[0]), type(training_keyword[0]))
    
    X_training = np.array(training)
    y_training = np.array(output)
    
    X_training_keyword = np.array(training_keyword)
    y_training_keyword = np.array(output_keyword)
    
    X_test, y_test = classifier_2_class.load_test_documents(util.test_file_dir, words)
    '''
    '''
    
    ''' concepts_baseball, concepts_computer are panda dataframe''' 
    concepts_baseball = dth.load_concepts(util.concepts_file_baseball, None, use_cache=False)
    concepts_computer = dth.load_concepts(util.concepts_file_computer, None, use_cache=False)
    
    if _DEBUG_:
        import csv
        with open('baseball_concepts_raw_0.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                    quoting=csv.QUOTE_MINIMAL)
            for c in concepts_baseball:
                csvwriter.writerow(c)
                
        with open('computer_concepts_raw_0.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                     quoting=csv.QUOTE_MINIMAL)
            for c in concepts_computer:
                csvwriter.writerow(c)
            
    concepts_baseball = preprocess_concepts(concepts_baseball)
    concepts_computer = preprocess_concepts(concepts_computer)
    if _DEBUG_:
        import csv
        with open('baseball_concepts_raw_1.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                    quoting=csv.QUOTE_MINIMAL)
            for c in concepts_baseball:
                csvwriter.writerow(c)
                
        with open('computer_concepts_raw_1.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                     quoting=csv.QUOTE_MINIMAL)
            for c in concepts_computer:
                csvwriter.writerow(c)
    
    print('#######################')
    print('X_training[5][:20]: ', X_training[5][:20])
    print('y_training[5]: ', y_training[5])
    print('y_label[5]: ', y_label[5])
    print('X_training[20][:20]: ', X_training[15][:20])
    print('y_training[20]: ', y_training[15])
    print('y_label[20]: ', y_label[15])
    print('#######################')
    
    y_label_mapping = {y_label[0]:y_training[0]}
    
    return X_training, X_test, y_training, y_test , X_training_keyword, \
        y_training_keyword, words, words_keywords, concepts_baseball, concepts_computer, y_label_mapping

   
def match_ontology_concepts_with_no_of_neurons(neurons, concepts, random_prob=True):
    '''
    no. of neuron and no. of concepts should be same.
    it will try to wrap up backwards.
    i.e. from output layer to hidden layer
    if hierarchy of hidden layer > hierarchy of concepts then it will 
    only attach concepts to hidden layer closest to the output layer.
    for the remaining layers it will copy the last (in reverse) layer concepts and assign those.
    
    Strategy:
    1. With fixed probability
    2. Copy the concepts with random probability the match with no of neurons.

    
    Parameters:
    neurons_size: [input_layer_size,hidden_layer_1_size....output_layer_size]
    concepts: [.....[n-1_th_hidden_layer_concepts],[n_th_hidden_layer_concepts],[output_layer_concepts]]
    
    Returns:
    modified_concepts: [....[same_size_of_n-1_th_hidden_layer_nueorns],[same_size_of_n_th_hidden_layer_nueorns]]
    '''
    
    diff = len(neurons) - len(concepts)
    modified_concepts = []
    last_layer_concepts = []  # in backward
    
    
    for index, concepts_of_layer_k in reversed(list(enumerate(concepts))):
        last_layer_concepts = concepts_of_layer_k
        no_of_concepts = len(concepts_of_layer_k)
        no_of_neurons = neurons[index]
        
        modified_concepts_list = []
        
        if random_prob:
            for i in range(0, neurons[index], 1):
                _concept = concepts_of_layer_k[randint(0, no_of_concepts - 1)]
                modified_concepts_list.append(_concept)
        
        else:
            no_of_times = ((no_of_neurons // no_of_concepts) + 1)
            while no_of_times:
                modified_concepts_list.extend(concepts_of_layer_k)
                no_of_times -= 1
                
            modified_concepts_list = modified_concepts_list[0:no_of_neurons]
            
        modified_concepts.append(modified_concepts_list)
        
        
    ''' assign last layers concepts to remaining layers'''
    for index, neurons_of_layer_k in reversed(list(enumerate(neurons))):
        if index < len(concepts):
            continue
        else:
            no_of_concepts = len(last_layer_concepts)
            no_of_neurons = neurons[index]
        
            modified_concepts_list = []
            
            if random_prob:
                for i in range(0, neurons[index], 1):
                    _concept = last_layer_concepts[randint(0, no_of_concepts - 1)]
                    modified_concepts_list.append(_concept)
            
            else:
                no_of_times = ((no_of_neurons // no_of_concepts) + 1)
                while no_of_times:
                    modified_concepts_list.extend(last_layer_concepts)
                    no_of_times -= 1
                    
                modified_concepts_list = modified_concepts_list[0:no_of_neurons]
                
            modified_concepts.append(modified_concepts_list)
        
        
    # reverse the layers so it's compatible with  
    # [....[same_size_of_n-1_th_hidden_layer_nueorns],[same_size_of_n_th_hidden_layer_nueorns]]    
    modified_concepts = list(reversed(modified_concepts))
    
    return modified_concepts


def feed_ontology_contepts(concepts, feed_option='Top_to_Bottom'):
    '''
    feed concepts to the neural network
    There are different ways to feed ontology concepts
    * Top_to_Bottom : Output layer will have Top (more general) concepts while 
     hidden to input layer will have more specific concepts (sub class of general concepts)
    
    * Bottom_to_Top : Output layer will have Bottom (more specific) concepts while 
     hidden to input layer will have more general concepts (super class of specific concepts)
    Parameters
    ----------
    concepts from the ontology/background knowledge
    concepts: [...[c1,c2,..],[c1,c2..],[c1,c2]] 
    
    Returns
    ---------
    
    '''
    pass


def _activation_pattern_over_all_instance(activations):
    '''
    Find the activation pattern over all instance
    Parameters:
    ----------
    activations: activations of the neurons for all instances.
    --activations[instances][layers]
    
    Returns:
    ---------
    neuron_activations_dict{1_0: n_times,..}
    neuron_activation_list[layer,index,activated_or_not,no_of_times_activated]
    neurons_activations_np = len(activations[0]) * len(activations[0][0])
                           = layer * neurons
    '''
    
    neuron_activations_dict = {}
    neuron_activations_list = []
    neurons_activations_np = np.zeros((len(activations[0]), len(activations[0][0])))
    
    for instance_index, instance_j_activations in enumerate(activations):
        no_of_times_activated = 0
        for layer_index, layer_i_activations in enumerate(instance_j_activations):
            # instance_j_activations contains the activation of neurons for a \
            # particular instance for a layer
            activation_mean_value = np.mean(layer_i_activations, axis=0)
#             activated_neurons = [1 if value >= activation_mean_value else 0 for value in instance_j_activations ]
            
            for neuron_index, neuron_value in enumerate(layer_i_activations):
                activated = 0
                if neuron_value > activation_mean_value:
                    activated = 1
                    # index_of_this_neuron 
                    # no_of_times_activated = neuron_activations.index([layer_index + 1,neuron_index,activated,no_of_times_activated]) 1
                
                    _key = str(layer_index + 1) + '_' + str(neuron_index)
                    if _key in neuron_activations_dict:
                        neuron_activations_dict[_key] = neuron_activations_dict[_key] + 1  
                    else:
                        neuron_activations_dict[_key] = 1
                else:
                    _key = str(layer_index + 1) + '_' + str(neuron_index)
                    if _key not in neuron_activations_dict:
                        neuron_activations_dict[_key] = 0
                neuron_activation_list = [layer_index + 1, neuron_index, activated, neuron_activations_dict[_key]]        
                neurons_activations_np[layer_index, neuron_index] = neuron_activations_dict[_key]
                
                neuron_activations_list.append(neuron_activation_list) 
            
    return neuron_activations_dict , neuron_activations_list , neurons_activations_np


def _activation_pattern_for_a_single_instance(activations):
    
    if _DEBUG_:
#         print('\n\n\n')
#         print('type(activations): ', type(activations))
#         print('len(activations): ', len(activations))
#         print('type(activations[0]): ', type(activations[0]))
#         print('activations[0].shape: ', activations[0].shape)
#         print('activations[0]: ', activations[0])
        print('\n\n\n')
    
    '''
    
    Parameters:
    ----------
    activations: activations of the neurons for single instances.
    --activations[layers][activations_as_np_array]
    
    Returns:
    ---------
    neuron_activations_dict{1_0: n_times,..}
    neuron_activation_list[layer,index,activated_or_not,no_of_times_activated]
    neurons_activations_np = len(activations[0]) * len(activations[0][0])
                           = layer * neurons
    '''

    neuron_activations_dict = {}
    neuron_activations_list = []
    neurons_activations_np = np.zeros((len(activations), activations[0].shape[1]))
    

    for layer_index, layer_i_activations in enumerate(activations):
        # layer_i_activations contains the activation of neurons for a \
        # particular layer
        
#             activated_neurons = [1 if value >= activation_mean_value else 0 for value in instance_j_activations ]
        # make it a list
        layer_i_activations = layer_i_activations.tolist()
        # remove the outer empty list
        layer_i_activations = layer_i_activations[0]
        activation_mean_value = np.mean(layer_i_activations)
        
#         if _DEBUG_:
#             print('activation_mean_value: ', activation_mean_value)
#             print('layer_i_activations: ', layer_i_activations)
        
        for neuron_index, neuron_value in enumerate(layer_i_activations):
            activated = 0
            if neuron_value > activation_mean_value:
                activated = 1
                # index_of_this_neuron 
                # no_of_times_activated = neuron_activations.index([layer_index + 1,neuron_index,activated,no_of_times_activated]) 1
            
                _key = str(layer_index + 1) + '_' + str(neuron_index)
                if _key in neuron_activations_dict:
                    neuron_activations_dict[_key] = neuron_activations_dict[_key] + 1  
                else:
                    neuron_activations_dict[_key] = 1
            else:
                _key = str(layer_index + 1) + '_' + str(neuron_index)
                if _key not in neuron_activations_dict:
                    neuron_activations_dict[_key] = 0
            neuron_activation_list = [layer_index + 1, neuron_index, activated, neuron_activations_dict[_key]]        
            neurons_activations_np[layer_index, neuron_index] = neuron_activations_dict[_key]
            
            neuron_activations_list.append(neuron_activation_list) 
            
    return neuron_activations_dict , neuron_activations_list , neurons_activations_np
    

def plot_figure(patterns_all, patterns_baseball, concepts_baseball, patterns_computer, concepts_computer, patterns_test_instance):
    '''plot figures
    Parameters:
    ----------
    patterns: 2 dimentional numpy array
        layer * neurons_in_layer
    
    Returns:
    -------  
    '''
    
    s_scale = 1
    x_scale = 1
    max_neuron_for_display = 50
    x_ticks = [1,2,3,4,5]
    x_label = 'Hidden layers'
    y_label = 'Neurons'
    # use only the max_neuron_for_display neurons semantic
    print('concepts_baseball.shape: ', concepts_baseball.shape)
    concepts_baseball = concepts_baseball[0:5,0:max_neuron_for_display]
    concepts_computer = concepts_computer[0:5,0:max_neuron_for_display]
    
    if _DEBUG_:
        with open('baseball_concepts_for_plot.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                    quoting=csv.QUOTE_MINIMAL)
            for c in concepts_baseball:
                csvwriter.writerow(c)
                
        with open('computer_concepts_for_plot.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                     quoting=csv.QUOTE_MINIMAL)
            for c in concepts_computer:
                csvwriter.writerow(c)
    


    ax_all = plt.figure(1).add_subplot(111)
    ax_all_with_semantics = plt.figure(2).add_subplot(111)
    ax_test_instance = plt.figure(3).add_subplot(111)
    
    '''figure without semantics'''
    '''for all'''
    x = np.zeros(patterns_all.shape[0] * max_neuron_for_display)
    y = np.zeros(patterns_all.shape[0] * max_neuron_for_display)
    s_all = np.zeros(patterns_all.shape[0] * max_neuron_for_display)    
    
    for i, layer_l in enumerate(patterns_all):
        for j, neuron_n in enumerate(layer_l):
            if j > max_neuron_for_display -1 :
                break
            x[i * max_neuron_for_display + j] = (i + 1) / x_scale
            y[i * max_neuron_for_display + j] = j + 1
            s_all[i * max_neuron_for_display + j] = neuron_n * s_scale
    ax_all.scatter(x, y, s=s_all, color=color[0])
    ax_all.set_xticks(x_ticks)
    
    '''figure with semantics'''
    '''for baseball'''
    x = np.zeros(patterns_baseball.shape[0] * max_neuron_for_display)
    y = np.zeros(patterns_baseball.shape[0] * max_neuron_for_display)
    s_baseball = np.zeros(patterns_baseball.shape[0] * max_neuron_for_display)    
    
    for i, layer_l in enumerate(patterns_baseball):
        for j, neuron_n in enumerate(layer_l):
            if j > max_neuron_for_display -1:
                break
            x[i * max_neuron_for_display + j] = (i + 1) / x_scale
            y[i * max_neuron_for_display + j] = j + 1
            s_baseball[i * max_neuron_for_display + j] = neuron_n * s_scale
    ax_all_with_semantics.scatter(x, y, s=s_baseball, color=color[0])
    
    '''annotate/attach the concepts/semantics'''
    for _x, _y, _s, _c in zip(x, y, s_baseball, concepts_baseball.flatten()):
        if _s > min_activation:
            ax_all_with_semantics.annotate(_c, xy=(_x, _y), textcoords='data', color=color[0])
    
    '''for computer'''
    x = np.zeros(patterns_computer.shape[0] * max_neuron_for_display)
    y = np.zeros(patterns_computer.shape[0] * max_neuron_for_display)
    s_computer = np.zeros(patterns_computer.shape[0] * max_neuron_for_display)
    
    for i, layer_l in enumerate(patterns_computer):
        for j, neuron_n in enumerate(layer_l):
            if j > max_neuron_for_display -1:
                break
            x[i * max_neuron_for_display + j] = (i + 1) / x_scale
            y[i * max_neuron_for_display + j] = j + 1
            s_computer[i * max_neuron_for_display + j] = neuron_n * s_scale
    ax_all_with_semantics.scatter(x, y, s=s_computer, color=color[1])
    
    # set xticks
    ax_all_with_semantics.set_xticks(x_ticks)
    
    '''annotate/attach the concepts/semantics'''
    for _x, _y, _s, _c in zip(x, y, s_computer, concepts_computer.flatten()):
        if _s > min_activation:
            ax_all_with_semantics.annotate(_c, xy=(_x , _y + .3), textcoords='data', color=color[1])
    
    ax_all_with_semantics.legend(['class baseball', 'class computer'], loc='upper left')
            
            
    '''figure for test instance'''
    x = np.zeros(patterns_test_instance.shape[0] * max_neuron_for_display)
    y = np.zeros(patterns_test_instance.shape[0] * max_neuron_for_display)
    s_test_instance = np.zeros(patterns_test_instance.shape[0] * max_neuron_for_display)    
    
    for i, layer_l in enumerate(patterns_test_instance):
        for j, neuron_n in enumerate(layer_l):
            if j > max_neuron_for_display -1 :
                break
            x[i * max_neuron_for_display + j] = (i + 1) / x_scale
            y[i * max_neuron_for_display + j] = j + 1
            s_test_instance[i * max_neuron_for_display + j] = neuron_n * s_scale
    ax_test_instance.scatter(x, y, s=s_test_instance, color=color[2])
    ax_test_instance.set_xticks(x_ticks)
    
    '''annotate/attach the concepts/semantics'''
    for _x, _y, _s_b, _s_c, _s_t, _c_b, _c_c in zip(x, y, s_baseball, s_computer, s_test_instance, concepts_baseball.flatten(), concepts_computer.flatten()):
        if _s_b > min_activation and _s_t >= min_activation_for_test:
            ax_test_instance.annotate(_c_b, xy=(_x, _y), textcoords='data', color=color[0])
        if _s_c > min_activation and _s_t >= min_activation_for_test:
            ax_test_instance.annotate(_c_c, xy=(_x , _y + .3), textcoords='data', color=color[1])
    
    
    '''legends, title, axis_labels etc'''
    ax_all.set_title('Activated neurons in DNN')
    ax_all.set_xlabel(x_label)
    ax_all.set_ylabel(y_label)
    
    ax_all_with_semantics.set_title('Semantic trace of DNN')
    ax_all_with_semantics.legend(['class baseball', 'class computer'], loc='upper left')
    ax_all_with_semantics.set_xlabel(x_label)
    ax_all_with_semantics.set_ylabel(y_label)
    
    #ax_test_instance.legend(['class baseball', 'class computer'], loc='upper left')
    ax_test_instance.set_title('Semantic trace for single instance')
    ax_test_instance.set_xlabel(x_label)
    ax_test_instance.set_ylabel(y_label)
    
    
    

def display_figure():
    '''Display all the plotted figures'''
    plt.show()


def analyze_activations(clf, activations, X_train, y_train, concepts_baseball, concepts_computer):
    '''
    analyze the activations of the DNN.
    
    1. find the cluster of neurons
        1.1. Cluster over multiple layer
        1.2. Cluster over a single layer
        
    2. find differentiating neurons for different class
        2.1. for different classes how the activation pattern is changing
    
    Parameters
    ----------
    activations: activations of the neurons over all iterations for all instances.
    --activations[iterations][layers][instances]
    
    concepts: concepts as ndarray. concepts dimension must match with hidden layer dimention
    
    Returns
    --------
    pattern_all_np_array, pattern_baseball_np_array, pattern_computer_np_array
    '''
    
    # taking activation of last iteration
    # activations for all instance after training
    activations = activations[-1]
    
    X_baseball = []
    X_computer = []
    for y, X in zip(y_train, X_train):
        if y[0] == 0:
            X_baseball.append(X)
        else:
            X_computer.append(X)
    
    '''activation_all_instance will be
    activation_all_instance[instances][layers]
    '''
    activation_all_instance = []
    for i in range(len(X_train)):
        activations_single_instance = []
        for layer_i in  activations[1:6]:
            activations_single_instance.append(layer_i[i])
        activation_all_instance.append(activations_single_instance)
    
#     if _DEBUG_:
#         print('X_train.shape: ', X_train.shape)
#         print('len(activation_all_instance): ', len(activation_all_instance))
    
    # clf.predict_proba()
    activations_for_baseball = []
    activations_for_computer = []
    for y, activs_ in zip(y_train, activation_all_instance):
        if(y[0] == 0):
            activations_for_baseball.append(activs_)
        else:
            activations_for_computer.append(activs_)
    
#     print('activations_for_baseball[-1]: ', len(activations_for_baseball[-1]))
#     print('activations_for_baseball[-1][-2]: ', len(activations_for_baseball[-1][-2]))
#     print('activations_for_baseball[-1][-2][-1]: ', len(activations_for_baseball[-1][-2][-1]))
#     print('activations_for_baseball[-1][-1][-1]: ', (activations_for_baseball[-1][-1]))
    
    pattern_all_dict, pattern_all_list , pattern_all_np_array = _activation_pattern_over_all_instance(activation_all_instance)
    pattern_baseball_dict, pattern_baseball_list , pattern_baseball_np_array = _activation_pattern_over_all_instance(activations_for_baseball)
    pattern_computer_dict, pattern_computer_list, pattern_computer_np_array = _activation_pattern_over_all_instance(activations_for_computer)
    
    # do not plot here
    # plot_figure(pattern_all_np_array, pattern_baseball_np_array, np.array(concepts_baseball), pattern_computer_np_array, np.array(concepts_computer))
    
    return pattern_all_np_array, pattern_baseball_np_array, pattern_computer_np_array
    

def get_hidden_neurons_sizes(X_train, no_of_hidden_layer):
    no_of_hidden_neurons = ((len(X_train[0])))
    hidden_layer_sizes = ()
    for i in range(no_of_hidden_layer):
        hidden_layer_sizes += (no_of_hidden_neurons,)
    return hidden_layer_sizes
    

'''
activations_over_all_itr[iterations][layers][instances]
'''

def train_network(X_train, y_train, hidden_layer_sizes, max_iter=1, use_cache=False, for_keyword=None, should_save=False):
    '''Train the DNN
    Parameters:
    ----------
    X_train, y_train, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, use_cache=False
    
    Returns:
    --------
    clf, activations_over_all_itr
    '''
    clf, activations_over_all_itr = classifier_2_class.train_NN(X_train, y_train, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, use_cache=use_cache, for_keyword=for_keyword, should_save=should_save)
    
    return clf, activations_over_all_itr
    
    
# Get train and test data
# Get Concepts        


def explain_instance(classifier, instance, pattern_all, pattern_baseball, pattern_computer, concepts_baseball, concepts_computer, y_label_mapping):
    '''
    Predict an instance and explain its decision
    Parameters:
    ----------
    
    Returns:
    --------
    neurons_activations_np
    '''
    predict_proba, activations, activated_neurons, activated_neurons_raw_sum = classifier.predict_proba(instance)
    
    # take only hidden layer activations
    activations = activations[1:6]
    neuron_activations_dict , neuron_activations_list , neurons_activations_np = _activation_pattern_for_a_single_instance(activations)
    
#     if _DEBUG_:
#         print('instance: ', instance[:20])  
#         print('type(activated_neurons): ', type(activated_neurons), '\tlen(activated_neurons): ', len(activated_neurons))
#         print('type(activated_neurons[0]): ', type(activated_neurons[0]), '\tlen(activated_neurons[0]): ', len(activated_neurons[0]))
#         print('activated_neurons[1]: ', activated_neurons[1])

    '''match with baseball'''
    activated_concepts_for_baseball = []
    for a_neurons, patterns, concepts in zip(activated_neurons[:5], pattern_baseball, concepts_baseball):
        a_neurons = list(a_neurons)
        a_neurons.sort()
        activated_concepts_this_l = []

        # a_neurons is list of activated neurons
        for index in a_neurons:
            if patterns[index] > min_activation: 
                activated_concepts_this_l.append(concepts[index])
        
        activated_concepts_for_baseball.append(activated_concepts_this_l)   
     
     
    '''match with computer'''
    activated_concepts_for_computer = []
    for a_neurons, patterns, concepts in zip(activated_neurons[:5], pattern_computer, concepts_computer):
        a_neurons = list(a_neurons)
        a_neurons.sort()
        activated_concepts_this_l = []

        # a_neurons is list of activated neurons
        for index in a_neurons:
            if patterns[index] > min_activation: 
                activated_concepts_this_l.append(concepts[index])
        
        activated_concepts_for_computer.append(activated_concepts_this_l)
    
    '''convert to flat list'''
    activated_concepts_for_computer_flat_list = [item for sublist in activated_concepts_for_computer for item in sublist]
    activated_concepts_for_baseball_flat_list = [item for sublist in activated_concepts_for_baseball for item in sublist]
    
#     if _DEBUG_:
#         print('activated_concepts_for_computer: ', activated_concepts_for_computer)
#         print('activated_concepts_for_baseball: ', activated_concepts_for_baseball) 
# 
#         print('predict_proba: ', predict_proba)
#         print('y_label_mapping: ', y_label_mapping)
#         print('len(activated_concepts_for_computer): ', len(set(activated_concepts_for_computer_flat_list)))
#         print('len(activated_concepts_for_baseball): ', len(set(activated_concepts_for_baseball_flat_list)))     
#         print('activated_concepts_for_computer: ', set(activated_concepts_for_computer_flat_list))
#         print('activated_concepts_for_baseball: ', set(activated_concepts_for_baseball_flat_list))  
#         print('Counter(activated_concepts_for_computer): ', Counter(activated_concepts_for_computer_flat_list))
#         print('Counter(activated_concepts_for_baseball): ', Counter(activated_concepts_for_baseball_flat_list))   
#         print('\n\n')
#         print('')
        
    return neurons_activations_np

# Get Data


X_train, X_test, y_train, y_test, X_training_keyword, \
    y_training_keyword, words, words_keywords, concepts_baseball, concepts_computer, y_label_mapping = get_data()

if _DEBUG_:
    import csv
    with open('baseball_concepts_raw.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quoting=csv.QUOTE_MINIMAL)
        for c in concepts_baseball:
            csvwriter.writerow(c)
            
    with open('computer_concepts_raw.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                 quoting=csv.QUOTE_MINIMAL)
        for c in concepts_computer:
            csvwriter.writerow(c)

# Parameters for DNN
no_of_hidden_layer = 5
hidden_layer_sizes = get_hidden_neurons_sizes(X_train, no_of_hidden_layer=no_of_hidden_layer)
max_iter = 1

# match concept and neurons by number and layer  
concepts_baseball = match_ontology_concepts_with_no_of_neurons(list(hidden_layer_sizes), concepts_baseball)
concepts_computer = match_ontology_concepts_with_no_of_neurons(list(hidden_layer_sizes), concepts_computer)

if _DEBUG_:
    import csv
    with open('baseball_concepts_mod.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quoting=csv.QUOTE_MINIMAL)
        for c in concepts_baseball:
            csvwriter.writerow(c)
            
    with open('computer_concepts_mod.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                 quoting=csv.QUOTE_MINIMAL)
        for c in concepts_computer:
            csvwriter.writerow(c)
            
#     print('len(concepts_baseball): ',len(concepts_baseball), '\n type(concepts_baseball): ',type(concepts_baseball))
#     for c in concepts_baseball:
#         print(c[0:10])
#     print('-------')
#     print('len(concepts_computer): ',len(concepts_computer))

# train the network
clf, activations_over_all_itr = train_network(X_train, y_train, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, use_cache=False, should_save=False)
# analyze the activations
pattern_all_np, pattern_baseball_np, pattern_computer_np = analyze_activations(clf, activations_over_all_itr, X_train, y_train, concepts_baseball, concepts_computer)
# explain a single instance
neurons_activations_test_instance_np = explain_instance(clf, X_test[0], pattern_all_np.tolist(), pattern_baseball_np.tolist(), pattern_computer_np.tolist(), concepts_baseball, concepts_computer, y_label_mapping)
# plot figures
plot_figure(pattern_all_np, pattern_baseball_np, np.array(concepts_baseball), pattern_computer_np, np.array(concepts_computer), neurons_activations_test_instance_np)
# Display figures
display_figure()





# # print('activations_over_all_itr[-1]: ', len(activations_over_all_itr[-1]))
# # print('activations_over_all_itr[-1][-2]: ', len(activations_over_all_itr[-1][-2]))
# # print('activations_over_all_itr[-1][-2][-1]: ', len(activations_over_all_itr[-1][-2][-1]))
# # print('activations_over_all_itr[-1][-1][-1]: ', (activations_over_all_itr[-1][-1]))
# 
# # activations_over_all_itr_as_np = np.array(activations_over_all_itr[-1][-2])
# # print('activations_over_all_itr_as_np.shape: ', activations_over_all_itr_as_np.shape)
# # plt.imshow(activations_over_all_itr_as_np.T, cmap='hot')
# # plt.show()
# 
# 
# clf_keyword , activations_over_all_itr_keyword = train_network(X_training_keyword, y_training_keyword, \
#                                            hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, use_cache=False, for_keyword=True)
# 
# # for index, activations in enumerate(activations_over_all_itr_keyword):
# #     for index_, activation in enumerate(activations):
# #         # print('activations[%d]' % index_, activation[index_])
# #         # pass
#     
# # visualize informations
# 
# 
# '''keyword train phase:
#  input_keyword = X_training_keyword
#  l1_keyword = from ontoloy
#  l2_keyword = from ontology
#  l3_keyword = from ontology
#  l4_keyword = from ontology
#  
#  explanation phase
#  input_keyword = from ontology only related keywords others are 0
#  l1_keyword = from ontoloy
#  l2_keyword = from ontology
#  l3_keyword = from ontology
#  l4_keyword = from ontology
#  mapping keywords with cluster of activation'''
# 
# '''for explanation''' 
# l1_key = X_training_keyword
# l2_key = ['sport', 'entertainment'] 
# l3_key = ['intentional_process', 'motion', 'power_generation'] 
# l4_key = [ 'process', 'object', 'physical_system'] 
# l5_key = ['OWL_Thing'] 
# 
# 
# l1_r_key = 'OWL_Thing'
# l2_r_key = [ 'process', 'object', 'physical_system']
# l3_r_key = ['intentional_process', 'motion', 'power_generation'] 
# l4_r_key = ['sport', 'entertainment']  
# l5_r_key = [''] 
#  
#  
# print('predicting started...')
# '''see difference in activated neurons for each class'''
# '''class_names: dict'''
# X_test_comp_windows = X_test[:2]
# X_rec_sport_hockey = X_test[2:4]
#  
#  
# predict_proba, activated_neurons_comp_windows, activated_neurons_raw_sum_comp_windows = clf.predict_proba(
#     X_test_comp_windows)
# predict_proba, activated_neurons_sport_hockey, activated_neurons_raw_sum_sport_hockey = clf.predict_proba(
#     X_rec_sport_hockey)
#  
#  
# predict_proba_all, activated_neurons_all, activated_neurons_raw_sum_all = clf.predict_proba(
#     X_test)
# print('len(X_test): ', len(X_test))
#  
#  
#  
# '''Plot for each class'''
# # 3474
# # output to static HTML file
# # output_file("square.html")
#  
# # p1 = figure()
# # p2 = figure()
# # p3 = figure()
#  
# # fig = plt.figure()
#  
# # fig1 for comp.windows.x
# fig1 = plt.figure(1).add_subplot(111)  # fig.add_subplot(1, 3, 1)
# fig1.set_title('class comp.windows_x')
#  
# # fig2 for rec.sport.hockey
# fig2 = plt.figure(2).add_subplot(111)  # fig.add_subplot(1, 3, 2)
# fig2.set_title('class sport_hockey')
#  
# # fig3 for all 2 class
# fig3 = plt.figure(3).add_subplot(111)  # fig.add_subplot(1, 3, 3)
# fig3.set_title('both classes without weighted activation')
#  
# # fig4 for all 2 class with weighted activations
# fig4 = plt.figure(4).add_subplot(111)
# fig4.set_title('both classes weighted activation')
# 
# # fig5 i.e. fig_keyword for keywords
# fig_keyword = plt.figure(5).add_subplot(111)
# fig_keyword.set_title('both classes weighted activation for keywords') 
#  
# 
# 
# 
#  
# for layer_i in range(1, 5, 1):
#     #     print('activated_neurons_comp_windows: ',
#     #           activated_neurons_comp_windows[layer_i])
#     #     print('activated_neurons_sport_hockey: ',
#     #           activated_neurons_sport_hockey[layer_i])
#  
#     activated_for_all_class = activated_neurons_comp_windows[
#         layer_i] & activated_neurons_sport_hockey[layer_i]
#     # print('activated_for_all_class: ', activated_for_all_class)
# #     print('hidden_layer %s' % (layer_i + 1))
# #     print('activated_for_all_class: ', activated_for_all_class)
# #     print('activated_neurons_only_for_comp_windows: ',
# #           activated_neurons_comp_windows[
# #               layer_i] - activated_for_all_class)
# #     print('activated_neurons_only_for_sport_hockey',
# #           activated_neurons_sport_hockey[layer_i] - activated_for_all_class)
# #     print('\n')
#     x = [layer_i ] * len(words)
#     x_keyword = [layer_i] * len(words_keywords)
#     y_keyword = [i for i in range(1, len(words_keywords) + 1, 1)]
#     itr_no = 0
#     bubble_size_keyword = [i * 10 for i in activations_over_all_itr_keyword[itr_no][layer_i]]
#  
#     # add a square renderer with a size, color, and alpha
#     y_windows = [0] * len(words)
#     y_hockey = [0] * len(words)
#     y_raw_all = [0] * len(words)
#     '''s_raw_all : size of the points'''
#     s_raw_all = [0] * len(words)
#  
#     for index in activated_neurons_comp_windows[layer_i]:
#         y_windows[index] = index
#  
#     # fig1.scatter(x, y, color=color[layer_i], marker='.')
#     # p1.square(x, y, size=2, color="olive", alpha=0.5)
#  
#     for index in activated_neurons_sport_hockey[layer_i]:
#         y_hockey[index] = index
#     # fig2.scatter(x, y, color=color[layer_i], marker='.')
#     # p2.square(x, y, size=2, color="olive", alpha=0.5)
#  
#  
#     for index, value in enumerate(activated_neurons_raw_sum_all[layer_i]):
#         if value > 0:
#             y_raw_all[index] = index
#             s_raw_all[index] = value * 50
#         else:
#             y_raw_all[index] = 0
#             s_raw_all[index] = 0
#  
# #     print('activated_neurons_raw_sum_all[layer_i]: ',
# #           len(y_raw_all), y_raw_all)
#  
#  
# #     if layer_i == 0:
# #         print()
# #         print('activated_neurons_raw_sum_all[layer_i]: ',
# #               activated_neurons_raw_sum_all[layer_i])
# #         # print('y_raw_all: ', y_raw_all)
# #         # print('s_raw_all: ', s_raw_all)
# #         print()
#     # fig3.scatter(x, y, color=color[layer_i], marker='.')
#     # p3.square(x, y, size=2, color="olive", alpha=0.5)
#  
#     '''using multiple figure'''
#     lbl = 'hidden_layer_' + str(layer_i)
#     fig1.scatter(x, y_windows, color=color[layer_i],
#                  marker='.', label=lbl)
#     fig2.scatter(x, y_hockey, color=color[layer_i],
#                  marker='.', label=lbl)
#  
#     '''using single figure'''
#     fig3.scatter(np.array(x) - .05, y_windows, color=color[0],
#                  marker='.')
#     fig3.scatter(np.array(x) + .05 , y_hockey, color=color[1],
#                  marker='.')
#  
#     '''using weighted bubble'''
#     x_ = [i - .1 for i in x]
#     fig4.scatter(x_, y_raw_all, s=s_raw_all, color=color[layer_i],
#                  marker='.', label=lbl)
#     
#     '''added in figure 4'''
#     x_keyword_ = [i + .1 for i in x_keyword]
#     fig4.scatter(x_keyword_, y_keyword, s=bubble_size_keyword,
#                         color=color[layer_i], marker='.', label=lbl)
#     
#     '''only keywords'''
#     fig_keyword.scatter(x_keyword, y_keyword, s=bubble_size_keyword,
#                         color=color[layer_i], marker='.', label=lbl)
#  
# # show the results
# # show(p1)
# # show(p2)
# # show(p3)
# y_ticks = [x for x in range(0, len(words))]
# x_ticks = [x for x in range(0, 5, 1)]
# fig1.set_xticks(x_ticks)
# fig2.set_xticks(x_ticks)
# fig3.set_xticks(x_ticks)
# fig4.set_xticks(x_ticks)
# fig_keyword.set_xticks(x_ticks)
#  
# # fig1.set_yticks(y_ticks)
# # fig2.set_yticks(y_ticks)
# # fig3.set_yticks(y_ticks)
# # fig4.set_yticks(y_ticks)
# # fig5.set_yticks(y_ticks)
#  
# fig1.set_xlabel('hidden_layers ')
# fig2.set_xlabel('hidden_layers')
# fig3.set_xlabel('hidden_layers')
# fig4.set_xlabel('hidden_layers')
# fig_keyword.set_xlabel('hidden_layers')
#  
# fig1.set_ylabel('n\'th neuron')
# fig2.set_ylabel('n\'th neuron')
# fig3.set_ylabel('n\'th neuron')
# fig4.set_ylabel('n\'th neuron')
# fig_keyword.set_xlabel('hidden_layers')
#  
# # fig4.set_label(['computer', 'guns', 'hockey', 'computer',
# #                 'guns', 'hockey', 'computer', 'guns', 'hockey'])
#  
# fig1.legend(loc='upper left')
# fig2.legend(loc='upper left')
# fig3.legend(['class comp.windows_x', 'class baseball'], loc='upper left')
# fig4.legend(loc='upper left')
# fig_keyword.legend(loc='upper left')
#  
# '''dump to pickle'''
# figure_file = '../../data/20news-18828/2_class/model/fig.pickle'
#  
# with open(figure_file, 'wb') as f:
#     pickle.dump([fig1, fig2, fig3, fig4, fig_keyword], f)
#  
# '''load figure from saved data'''
# '''after loading from pickle zooming is not working'''
# 
# print('figure display starts...') 
# plt.show()
# print('figure closed') 
# print('predict: ', clf.predict(X_test))
# predict_proba, activated_neurons = clf.predict_proba(X_test)
# print('predict_proba: ', predict_proba)
 
# for i, neurons in enumerate(activated_neurons):
#     print('layer %s: ' % (i + 1), neurons)



# part of analyze_activations():
#     print('\n\n\n')
#     print('pattern_baseball_np_array: ', pattern_baseball_np_array.shape)
#     print('concepts_baseball: ', np.array(concepts_baseball).shape)
#     print('\n\n\n')
#     
#     baseball_concept_set = set()
#     computer_concept_set = set()
#     
#     
#     
#     
#     
# #     print('pattern_computer_list: ', pattern_computer_list)
# #     print('pattern_computer_np: ', pattern_computer_np)
# #     print('concepts_baseball: \n', concepts_baseball)
# #     pattern_baseball_np_activated = pattern_baseball_np[pattern_baseball_np[:, 3] > 4]
# #     pattern_computer_np_activated = pattern_computer_np[pattern_computer_np[:, 3] > 4]
# # #     
# #     print('pattern_baseball_np_activated: \n', pattern_baseball_np_activated)
# #     print('pattern_computer_np_activated: \n', pattern_computer_np_activated)
#     # for all instance only layer 1
#     activations_all_instance_layer_0 = activations[0][:]
#     # print('len(activations_all_instance_layer_0): ', len(activations_all_instance_layer_0))
#     
#     # for all instance only layer 1
#     activations_all_instance_layer_1 = activations[1][:]
#     # print('len(activations_all_instance_layer_1): ', len(activations_all_instance_layer_1))
#     
#     # for all instance only layer 2
#     activations_all_instance_layer_2 = activations[2][:]
#     # print('len(activations_all_instance_layer_2): ', len(activations_all_instance_layer_2))
#     
#     # for all instance only layer 3
#     activations_all_instance_layer_3 = activations[3][:]
#     # print('len(activations_all_instance_layer_3): ', len(activations_all_instance_layer_3))
#     
#     # for instance j=1
#     j = 1
#     activations_single_instance = []
#     for layer_i in  activations[1:6]:
#         activations_single_instance.append(layer_i[j])
#     
#     activations_single_instance = np.array(activations_single_instance)
#     # print('activations_single_instance.shape: ', activations_single_instance.shape)
# #     plt.imshow(activations_single_instance.T, cmap='hot', interpolation='nearest')
# #     plt.show()



# part of def _activation_pattern_over_all_instance():

#                 neuron_activations[layer_index + 1] = layer_index + 1
#                 neuron_activations[neuron_index] = neuron_index
#                 neuron_activations['activated']
#                 , neuron_index, activated, no_of_times_activated]
                # neuron_activations.append(neuron_activation)
                
    
#     for layer_index, layer_i_activations in enumerate(activations):
#         no_of_times_activated = 0
#         for instance_index, instance_j_activations in enumerate(layer_i_activations):
#             # instance_j_activations contains the activation of neurons for a \
#             # particular instance for a layer
#             activation_mean_value = 0 #np.mean(instance_j_activations, axis=0)
# #             activated_neurons = [1 if value >= activation_mean_value else 0 for value in instance_j_activations ]
#             
#             activated = 0
#             for neuron_index, neuron_value in enumerate(instance_j_activations):
#                 if neuron_value > activation_mean_value:
#                     no_of_times_activated += 1
#                     activated = 1
#                 neuron_activation = [layer_index + 1, neuron_index, activated, no_of_times_activated]
#                 neuron_activations.append(neuron_activation)
#                 
#             # make it a pattern if activated_neuron
#             
