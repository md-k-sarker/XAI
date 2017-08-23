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



# x_text, y, y_label = dth.load_data_and_labels(util.train_data_folder, util.saving_data_file)
# words = classifier_2_class.make_word_embeddings(x_text, y, y_label)
# 
# documents, classes = classifier_2_class.create_document_and_classes_from_data(x_text, y, y_label)
# 
# '''documents is tuple
# documents = (text,label)
# classes is list of classes.
# classes = [0,0,0....1,1,1,]'''
# 
# # remove duplicates
# classes = list(set(classes))
# 
# # print(len(documents), " documents")
# # print(len(classes), " classes", classes)
# # print(len(words), " unique stemmed words")
# 
# # training data
# training, output = classifier_2_class.convert_training_documents_to_vector(documents, classes,words)
# X_training = np.array(training)
# y_training = np.array(output)
# 
# print('X_training.shape: ', X_training.shape)
# print('y_training.shape: ', y_training.shape)


####
# get data
def get_data():
    '''
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
    return X_training, X_test, y_training, y_test , X_training_keyword, \
        y_training_keyword, words, words_keywords

# test and train data        
X_train, X_test, y_train, y_test, X_training_keyword, \
    y_training_keyword, words, words_keywords = get_data()

# print('y_training_keyword: ', y_training_keyword)
print('words_keywords: ', words_keywords)

clf, activations_over_all_itr = classifier_2_class.train_NN(X_train, y_train, no_of_hidden_layer=5, max_iter=1, use_cache=False)


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
                _concept = concepts_of_layer_k[randint(0, no_of_concepts)]
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
        if index > len(concepts):
            continue
        else:
            no_of_concepts = len(last_layer_concepts)
            no_of_neurons = neurons[index]
        
            modified_concepts_list = []
            
            if random_prob:
                for i in range(0, neurons[index], 1):
                    _concept = last_layer_concepts[randint(0, no_of_concepts)]
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
    
    Parameters:
    ----------
    activations: activations of the neurons for all instances.
    --activations[layers][instances]
    
    Returns:
    ---------
    neuron_activations = list of neuron_activation for instances
    neuron_activation[layer,index,activated_or_not,no_of_times_activated]
    '''
    
    neuron_activations = []
    
    for layer_index, layer_i_activations in enumerate(activations[1, len(activations) - 1]):
        no_of_times_activated = 0
        for instance_index, instance_j_activations in enumerate(layer_i_activations):
            # instance_j_activations contains the activation of neurons for a \
            # particular instance for a layer
            activation_mean_value = np.mean(instance_j_activations, axis=0)
#             activated_neurons = [1 if value >= activation_mean_value else 0 for value in instance_j_activations ]
            
            for neuron_index, neuron_value in enumerate(instance_j_activations):
                activated = 0
                if neuron_value > activation_mean_value:
                    no_of_times_activated += 1
                    activated = 1
                neuron_activation = [layer_index + 1, neuron_index, activated, no_of_times_activated]
                neuron_activations.append(neuron_activation)
                
            # make it a pattern if activated_neuron
            
            
    return neuron_activations


def _activation_pattern_for_a_single_instance(activations, instance_data=None):
    '''
    
    Parameters:
    ----------
    activations: activations of the neurons for all instances.
    --activations[layers][instances]
    
    Returns:
    ---------
    neuron_activations = list of neuron_activation
    neuron_activation[layer,index,activated_or_not,no_of_times_activated]
    '''
    
    predict_proba, activated_neurons, activated_neurons_raw_sum = clf.predict_proba(
    instance_data)
    
            
    return activated_neurons
    


def analyze_activations(activations):
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
    
    '''
    
    # taking activation of last iteration
    activations = activations[-1]
    
    '''activations for all instance'''
    
    
    # for all instance only layer 1
    activations_all_instance_layer_0 = activations[0][:]
    print('len(activations_all_instance_layer_0): ', len(activations_all_instance_layer_0))
    
    # for all instance only layer 1
    activations_all_instance_layer_1 = activations[1][:]
    print('len(activations_all_instance_layer_1): ', len(activations_all_instance_layer_1))
    
    # for all instance only layer 2
    activations_all_instance_layer_2 = activations[2][:]
    print('len(activations_all_instance_layer_2): ', len(activations_all_instance_layer_2))
    
    # for all instance only layer 3
    activations_all_instance_layer_3 = activations[3][:]
    print('len(activations_all_instance_layer_3): ', len(activations_all_instance_layer_3))
    
    # for instance j=1
    j = 1
    activations_single_instance = []
    for layer_i in  activations[1:6]:
        activations_single_instance.append(layer_i[j])
    
    activations_single_instance = np.array(activations_single_instance)
    print('activations_single_instance.shape: ', activations_single_instance.shape)
    plt.imshow(activations_single_instance.T, cmap='hot', interpolation='nearest')
    plt.show()
    pass



'''
activations_over_all_itr[iterations][layers][instances]
'''
analyze_activations(activations_over_all_itr)

print('activations_over_all_itr[-1]: ', len(activations_over_all_itr[-1]))
print('activations_over_all_itr[-1][-2]: ', len(activations_over_all_itr[-1][-2]))
print('activations_over_all_itr[-1][-2][-1]: ', len(activations_over_all_itr[-1][-2][-1]))
print('activations_over_all_itr[-1][-1][-1]: ', (activations_over_all_itr[-1][-1]))

activations_over_all_itr_as_np = np.array(activations_over_all_itr[-1][-2])
print('activations_over_all_itr_as_np.shape: ', activations_over_all_itr_as_np.shape)
# plt.imshow(activations_over_all_itr_as_np.T, cmap='hot')
# plt.show()


# clf_keyword , activations_over_all_itr_keyword = classifier_2_class.train_NN(X_training_keyword, y_training_keyword, \
#                                            no_of_hidden_layer=5, max_iter=1, use_cache=False, for_keyword=True)

# for index, activations in enumerate(activations_over_all_itr_keyword):
#     for index_, activation in enumerate(activations):
#         # print('activations[%d]' % index_, activation[index_])
#         # pass
    
# visualize informations


'''keyword train phase:
 input_keyword = X_training_keyword
 l1_keyword = from ontoloy
 l2_keyword = from ontology
 l3_keyword = from ontology
 l4_keyword = from ontology
 
 explanation phase
 input_keyword = from ontology only related keywords others are 0
 l1_keyword = from ontoloy
 l2_keyword = from ontology
 l3_keyword = from ontology
 l4_keyword = from ontology
 mapping keywords with cluster of activation'''

'''for explanation''' 
l1_key = X_training_keyword
l2_key = ['sport', 'entertainment'] 
l3_key = ['intentional_process', 'motion', 'power_generation'] 
l4_key = [ 'process', 'object', 'physical_system'] 
l5_key = ['OWL_Thing'] 


l1_r_key = 'OWL_Thing'
l2_r_key = [ 'process', 'object', 'physical_system']
l3_r_key = ['intentional_process', 'motion', 'power_generation'] 
l4_r_key = ['sport', 'entertainment']  
l5_r_key = [''] 
 
 
print('predicting started...')
'''see difference in activated neurons for each class'''
'''class_names: dict'''
X_test_comp_windows = X_test[:2]
X_rec_sport_hockey = X_test[2:4]
 
 
predict_proba, activated_neurons_comp_windows, activated_neurons_raw_sum_comp_windows = clf.predict_proba(
    X_test_comp_windows)
predict_proba, activated_neurons_sport_hockey, activated_neurons_raw_sum_sport_hockey = clf.predict_proba(
    X_rec_sport_hockey)
 
 
predict_proba_all, activated_neurons_all, activated_neurons_raw_sum_all = clf.predict_proba(
    X_test)
print('len(X_test): ', len(X_test))
 
 
 
'''Plot for each class'''
# 3474
# output to static HTML file
# output_file("square.html")
 
# p1 = figure()
# p2 = figure()
# p3 = figure()
 
# fig = plt.figure()
 
# fig1 for comp.windows.x
fig1 = plt.figure(1).add_subplot(111)  # fig.add_subplot(1, 3, 1)
fig1.set_title('class comp.windows_x')
 
# fig2 for rec.sport.hockey
fig2 = plt.figure(2).add_subplot(111)  # fig.add_subplot(1, 3, 2)
fig2.set_title('class sport_hockey')
 
# fig3 for all 2 class
fig3 = plt.figure(3).add_subplot(111)  # fig.add_subplot(1, 3, 3)
fig3.set_title('both classes without weighted activation')
 
# fig4 for all 2 class with weighted activations
fig4 = plt.figure(4).add_subplot(111)
fig4.set_title('both classes weighted activation')

# fig5 i.e. fig_keyword for keywords
fig_keyword = plt.figure(5).add_subplot(111)
fig_keyword.set_title('both classes weighted activation for keywords') 
 
color = ['green', 'black', 'olive', 'cyan', 'green', 'blue']


 
for layer_i in range(1, 5, 1):
    #     print('activated_neurons_comp_windows: ',
    #           activated_neurons_comp_windows[layer_i])
    #     print('activated_neurons_sport_hockey: ',
    #           activated_neurons_sport_hockey[layer_i])
 
    activated_for_all_class = activated_neurons_comp_windows[
        layer_i] & activated_neurons_sport_hockey[layer_i]
    # print('activated_for_all_class: ', activated_for_all_class)
#     print('hidden_layer %s' % (layer_i + 1))
#     print('activated_for_all_class: ', activated_for_all_class)
#     print('activated_neurons_only_for_comp_windows: ',
#           activated_neurons_comp_windows[
#               layer_i] - activated_for_all_class)
#     print('activated_neurons_only_for_sport_hockey',
#           activated_neurons_sport_hockey[layer_i] - activated_for_all_class)
#     print('\n')
    x = [layer_i ] * len(words)
    x_keyword = [layer_i] * len(words_keywords)
    y_keyword = [i for i in range(1, len(words_keywords) + 1, 1)]
    itr_no = 0
    bubble_size_keyword = [i * 10 for i in activations_over_all_itr_keyword[itr_no][layer_i]]
 
    # add a square renderer with a size, color, and alpha
    y_windows = [0] * len(words)
    y_hockey = [0] * len(words)
    y_raw_all = [0] * len(words)
    '''s_raw_all : size of the points'''
    s_raw_all = [0] * len(words)
 
    for index in activated_neurons_comp_windows[layer_i]:
        y_windows[index] = index
 
    # fig1.scatter(x, y, color=color[layer_i], marker='.')
    # p1.square(x, y, size=2, color="olive", alpha=0.5)
 
    for index in activated_neurons_sport_hockey[layer_i]:
        y_hockey[index] = index
    # fig2.scatter(x, y, color=color[layer_i], marker='.')
    # p2.square(x, y, size=2, color="olive", alpha=0.5)
 
 
    for index, value in enumerate(activated_neurons_raw_sum_all[layer_i]):
        if value > 0:
            y_raw_all[index] = index
            s_raw_all[index] = value * 50
        else:
            y_raw_all[index] = 0
            s_raw_all[index] = 0
 
#     print('activated_neurons_raw_sum_all[layer_i]: ',
#           len(y_raw_all), y_raw_all)
 
 
#     if layer_i == 0:
#         print()
#         print('activated_neurons_raw_sum_all[layer_i]: ',
#               activated_neurons_raw_sum_all[layer_i])
#         # print('y_raw_all: ', y_raw_all)
#         # print('s_raw_all: ', s_raw_all)
#         print()
    # fig3.scatter(x, y, color=color[layer_i], marker='.')
    # p3.square(x, y, size=2, color="olive", alpha=0.5)
 
    '''using multiple figure'''
    lbl = 'hidden_layer_' + str(layer_i)
    fig1.scatter(x, y_windows, color=color[layer_i],
                 marker='.', label=lbl)
    fig2.scatter(x, y_hockey, color=color[layer_i],
                 marker='.', label=lbl)
 
    '''using single figure'''
    fig3.scatter(np.array(x) - .05, y_windows, color=color[0],
                 marker='.')
    fig3.scatter(np.array(x) + .05 , y_hockey, color=color[1],
                 marker='.')
 
    '''using weighted bubble'''
    x_ = [i - .1 for i in x]
    fig4.scatter(x_, y_raw_all, s=s_raw_all, color=color[layer_i],
                 marker='.', label=lbl)
    
    '''added in figure 4'''
    x_keyword_ = [i + .1 for i in x_keyword]
    fig4.scatter(x_keyword_, y_keyword, s=bubble_size_keyword,
                        color=color[layer_i], marker='.', label=lbl)
    
    '''only keywords'''
    fig_keyword.scatter(x_keyword, y_keyword, s=bubble_size_keyword,
                        color=color[layer_i], marker='.', label=lbl)
 
# show the results
# show(p1)
# show(p2)
# show(p3)
y_ticks = [x for x in range(0, len(words))]
x_ticks = [x for x in range(0, 5, 1)]
fig1.set_xticks(x_ticks)
fig2.set_xticks(x_ticks)
fig3.set_xticks(x_ticks)
fig4.set_xticks(x_ticks)
fig_keyword.set_xticks(x_ticks)
 
# fig1.set_yticks(y_ticks)
# fig2.set_yticks(y_ticks)
# fig3.set_yticks(y_ticks)
# fig4.set_yticks(y_ticks)
# fig5.set_yticks(y_ticks)
 
fig1.set_xlabel('hidden_layers ')
fig2.set_xlabel('hidden_layers')
fig3.set_xlabel('hidden_layers')
fig4.set_xlabel('hidden_layers')
fig_keyword.set_xlabel('hidden_layers')
 
fig1.set_ylabel('n\'th neuron')
fig2.set_ylabel('n\'th neuron')
fig3.set_ylabel('n\'th neuron')
fig4.set_ylabel('n\'th neuron')
fig_keyword.set_xlabel('hidden_layers')
 
# fig4.set_label(['computer', 'guns', 'hockey', 'computer',
#                 'guns', 'hockey', 'computer', 'guns', 'hockey'])
 
fig1.legend(loc='upper left')
fig2.legend(loc='upper left')
fig3.legend(['class comp.windows_x', 'class sport_hockey'], loc='upper left')
fig4.legend(loc='upper left')
fig_keyword.legend(loc='upper left')
 
'''dump to pickle'''
figure_file = '../../data/20news-18828/2_class/model/fig.pickle'
 
with open(figure_file, 'wb') as f:
    pickle.dump([fig1, fig2, fig3, fig4, fig_keyword], f)
 
'''load figure from saved data'''
'''after loading from pickle zooming is not working'''

print('figure display starts...') 
plt.show()
print('figure closed') 
# print('predict: ', clf.predict(X_test))
# predict_proba, activated_neurons = clf.predict_proba(X_test)
# print('predict_proba: ', predict_proba)
 
# for i, neurons in enumerate(activated_neurons):
#     print('layer %s: ' % (i + 1), neurons)