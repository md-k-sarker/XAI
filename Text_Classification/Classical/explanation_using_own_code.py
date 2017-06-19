'''
Created on Jun 8, 2017

@author: sarker
'''
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import numpy as np
import re
import itertools
from collections import Counter
import time
import datetime
import pickle

import tensorflow as tf
import numpy as np
stemmer = LancasterStemmer()

# Load data
root_data_folder = '../../data/20news-18828/test'
saving_data_file = '../../data/20news-18828/preloaded/20news_18828.dt'
saving_words_data_file = '../../data/20news-18828/preloaded/20news_all_words.dt'
saving_bag_of_words_data_file = '../../data/20news-18828/preloaded/20news_bag_of_words.dt'
test_file_dir = '/Users/sarker/WorkSpaces/EclipseNeon/XAI/data/20news-18828/test_backup'

classes = ['comp.graphics', 'rec.autos', 'soc.religion.christian']
words = []
ignore_words = ['?']

# If file is saved then just load the file
if os.path.isfile(saving_words_data_file):
    with open(os.path.abspath(saving_words_data_file), 'rb') as f:
        words = pickle.load(f)


def clean_up_document(document):
    # tokenize the pattern
    document_words = nltk.word_tokenize(document)
    # stem each word
    document_words = [stemmer.stem(word.lower()) for word in document_words]
    return document_words


'''return bag of words array: 0 or 1 for each word in the bag that exists in the document'''


def bow(document, words, show_details=False):
    # tokenize the pattern
    document_words = clean_up_document(document)
    if show_details:
        print('document_tokens: ', document_words)
        print('####################')

    # bag of words
    bag_0_1_vector = [0] * len(words)
    bag_words_vector = [0] * len(words)

    for s in document_words:
        for i, w in enumerate(words):
            if w == s:
                bag_0_1_vector[i] = 1
                bag_words_vector[i] = s
                if show_details:
                    print("found in bag: %s" % w)

    #print('bag_0_1_vector: ', bag_0_1_vector)

    bag_words = [p for p in bag_words_vector if (p != 0)]
    if show_details:
        print('bag_of_words: ', bag_words)
        print('####################')
        print('bag_words_vector', bag_words_vector)
        print('####################')

    return(np.array(bag_0_1_vector), np.array(bag_words_vector), np.array(bag_words))

# compute sigmoid nonlinearity


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# convert output of sigmoid function to its derivative


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


activation_threshold = 0.6

# As List
l0_activations = []
l1_activations = []
l2_activations = []
l3_activations = []


def estimate(document, show_details=False):
    bag_0_1_vector, bag_words_vector, bag_words = bow(
        document.lower(), words, show_details)

    # input
    x = bag_0_1_vector
    if show_details:
        print("document:", document, "\n bow:", x)

    '''Estimate the class'''
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer1
    l1 = sigmoid(np.dot(l0, synapse_0))
    # matrix multiplication of hidden layer1 and hidden layer2
    l2 = sigmoid(np.dot(l1, synapse_1))
    # output layer
    l3 = sigmoid(np.dot(l2, synapse_2))

    '''Try to get explanation'''
    '''analyze neurons'''
#     print('l0: ', l0.shape, np.amax(l0), np.argmax(l0))
#     print('l1: ', l1.shape, np.amax(l1), np.argmax(l1))
#     print('l2: ', l2.shape, np.amax(l2), np.argmax(l2))
#     print('l3: ', l3.shape, np.amax(l3), np.argmax(l3))
    l0_actv = [index for index, x in enumerate(
        l0) if x > (activation_threshold + .15)]
    l1_actv = [index for index, x in enumerate(l1) if x > activation_threshold]
    l2_actv = [index for index, x in enumerate(l2) if x > activation_threshold]
    l3_actv = [index for index, x in enumerate(l3) if x > activation_threshold]
#     print('l0_actv: ', l0_actv)
#     print('l1_actv: ', l1_actv)
#     print('l2_actv: ', l2_actv)
#     print('l3_actv: ', l3_actv)
    l0_activations.extend('\n')
    l0_activations.extend(l0_actv)

    l1_activations.extend('\n')
    l1_activations.extend(l1_actv)

    l2_activations.extend('\n')
    l2_activations.extend(l2_actv)

    l3_activations.extend('\n')
    l3_activations.extend(l3_actv)

    '''analyze synapses/weights'''
    '''Synapse_0_highest_weights'''
    synapse_0_higher_i, synapse_0_higher_j = np.where(
        synapse_0 >= np.mean(synapse_0))
    '''Synapse_1_highest_weights'''
    synapse_1_higher_i, synapse_1_higher_j = np.where(
        synapse_1 >= np.mean(synapse_1))
    '''Synapse_2_highest_weights'''
    synapse_2_higher_i, synapse_2_higher_j = np.where(
        synapse_2 >= np.mean(synapse_2))

    '''The words which is giving more influence in the input layer.
    The words are in the global word_vectors. These words may not appear in the input document'''
    # word_dict = {key, value}
    # key = weight, value = word
    words_dict = {}
    for i, j in zip(synapse_0_higher_i, synapse_0_higher_j):
        # if(bag_words_vector[i] != '0'):
        words_dict[synapse_0[i, j]] = bag_words_vector[i]
        # print(bag_words[i])

    sorted_x = sorted(words_dict.items(), key=words_dict.get(0), reverse=True)
    if show_details:
        print('influential words from global word_vector: ')
    with open('stats_.txt', 'w', encoding='utf-8') as f:
        for i, j in sorted_x:
            if show_details:
                print(' word: ', j, ' \tscore: ', i)
            f.write(' word: ' + str(j) + ' \tscore: ' + str(i) + '\n')

    return l3, l0_actv, l1_actv, l2_actv, l3_actv


# probability threshold
ERROR_THRESHOLD = 0

# load our calculated synapse values
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])
    synapse_2 = np.asarray(synapse['synapse2'])


def classify(document, show_details=False):
    results, l0_actv, l1_actv, l2_actv, l3_actv = estimate(
        document, show_details)

    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results = [[classes[r[0]], r[1]] for r in results]
    if show_details:
        print("first 5 words from document: %s \nclassification: %s" %
              (document.split(' ')[:5], return_results))
    return return_results, l0_actv, l1_actv, l2_actv, l3_actv


test_doc_1 = open(
    '/Users/sarker/WorkSpaces/EclipseNeon/XAI/data/20news-18828_3_class/comp.graphics/38343', encoding='utf-8').read()
test_doc_2 = open(
    '/Users/sarker/WorkSpaces/EclipseNeon/XAI/data/20news-18828_3_class/rec.autos/102800', encoding='utf-8').read()
test_doc_3 = open(
    '/Users/sarker/WorkSpaces/EclipseNeon/XAI/data/20news-18828_3_class/soc.religion.christian/20599', encoding='utf-8').read()


folder_names = ['comp.graphics', 'rec_autos', 'soc.religion.christian']
# As Set
l0_activations_set = {
    'com_graphics': set(), 'rec_autos': set(), 'soc.religion.christian': set()}
l1_activations_set = {
    'com_graphics': set(), 'rec_autos': set(), 'soc.religion.christian': set()}
l2_activations_set = {
    'com_graphics': set(), 'rec_autos': set(), 'soc.religion.christian': set()}
l3_activations_set = {
    'com_graphics': set(), 'rec_autos': set(), 'soc.religion.christian': set()}

for folder_name in os.listdir(test_file_dir):
    if not folder_name.startswith('.'):
        '''Do operations for each folder/class'''
        print('folder_name: ', folder_name)
        file_counter = 0
        for file_name in os.listdir(os.path.join(test_file_dir, folder_name)):
            if not file_name.startswith('.'):
                file_counter += 1
                document = open(os.path.join(test_file_dir, folder_name, file_name),
                                mode='r', encoding='utf-8', errors='ignore').read()

                result, l0_actv, l1_actv, l2_actv, l3_actv = classify(
                    document,  show_details=False)
                #print('classification: ', result)
                if file_counter == 1:
                    l0_activations_set[folder_name] = set(l0_actv)
                    l1_activations_set[folder_name] = set(l1_actv)
                    l2_activations_set[folder_name] = set(l2_actv)
                    l3_activations_set[folder_name] = set(l3_actv)
                else:
                    l0_activations_set[folder_name] &= set(l0_actv)
                    l1_activations_set[folder_name] &= set(l1_actv)
                    l2_activations_set[folder_name] &= set(l2_actv)
                    l3_activations_set[folder_name] &= set(l3_actv)

        # print activations
        print('l0_activations_set: ', l0_activations_set[folder_name])
        print('l1_activations_set: ', l1_activations_set[folder_name])
        print('l2_activations_set: ', l2_activations_set[folder_name])
        print('l3_activations_set: ', l3_activations_set[folder_name])
        print()

with open('stats.txt', 'w', encoding='utf-8') as f:
    f.write('l0:  ')
    for l0_a in l0_activations:
        # print(l0_a)
        if l0_a == '\n':
            f.write('\n')
        f.write(str(l0_a))
    f.write('\n\nl1:  ')
    for l1_a in l1_activations:
        # print(l1_a)
        if l1_a == '\n':
            f.write('\n')
        f.write(str(l1_a))
    f.write('\n\nl2:  ')
    for l2_a in l2_activations:
        # print(l2_a)
        if l2_a == '\n':
            f.write('\n')
        f.write(str(l2_a))
    f.write('\n\nl3:  ')
    for l3_a in l3_activations:
        # print(l3_a)
        if l3_a == '\n':
            f.write('\n')
        f.write(str(l3_a))
