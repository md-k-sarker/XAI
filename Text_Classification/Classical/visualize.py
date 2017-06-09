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
root_data_folder = '../../data/20news-18828'
saving_data_file = '../../data/preloaded/20news_18828.dt'
saving_words_data_file = '../../data/preloaded/20news_all_words.dt'
saving_bag_of_words_data_file = '../../data/preloaded/20news_bag_of_words.dt'


classes = ['comp.graphics', 'rec.autos', 'soc.religion.christian']
words = []
ignore_words = ['?']

# If file is saved then just load the file
if os.path.isfile(saving_words_data_file):
    with open(os.path.abspath(saving_words_data_file), 'rb') as f:
        words = pickle.load(f)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists
# in the sentence


def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    bag_words = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                bag_words[i] = s
                if show_details:
                    print("found in bag: %s" % w)

    return(np.array(bag), np.array(bag_words))

# compute sigmoid nonlinearity


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# convert output of sigmoid function to its derivative


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


activation_threshold = 0.6


def estimate(document, show_details=False):
    x, bag_words = bow(document.lower(), words, show_details)
    if show_details:
        print("document:", document, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer1
    l1 = sigmoid(np.dot(l0, synapse_0))
    # matrix multiplication of hidden layer1 and hidden layer2
    l2 = sigmoid(np.dot(l1, synapse_1))
    # output layer
    l3 = sigmoid(np.dot(l2, synapse_2))
#     print('l0: ', l0.shape, np.amax(l0), np.argmax(l0))
#     print('l1: ', l1.shape, np.amax(l1), np.argmax(l1))
#     print('l2: ', l2.shape, np.amax(l2), np.argmax(l2))
#     print('l3: ', l3.shape, np.amax(l3), np.argmax(l3))

    '''i, j = np.unravel_index(synapse_2.argmax(), synapse_2.shape)'''

    synapse_0_max_i, synapse_0_max_j = np.where(synapse_0 == synapse_0.max())
    synapse_1_max_i, synapse_1_max_j = np.where(synapse_1 == synapse_1.max())
    synapse_2_max_i, synapse_2_max_j = np.where(synapse_2 == synapse_2.max())

    synapse_0_higher_i, synapse_0_higher_j = np.where(
        synapse_0 >= np.mean(synapse_0))

    # print('influential words: ',  synapse_0_max_i, synapse_0_max_j,
    #     bag_words[synapse_0_max_i], )
    words_dict = {}
    for i, j in zip(synapse_0_higher_i, synapse_0_higher_j):
        if(bag_words[i] != '0'):
            words_dict[synapse_0[i, j]] = bag_words[i]
            # print(bag_words[i])

    sorted_x = sorted(words_dict.items(), key=words_dict.get(0), reverse=True)
    print('influential words: ')
    for i, j in sorted_x:
        print(' word: ', j, ' \tscore: ', i)

#     print('synapse_2: ', synapse_2.shape, synapse_2.ndim, np.amax(
#         synapse_2), np.argmax(synapse_2))
#     print('val: ', synapse_2[0, 2])
#     print('synapse_1: ', np.amax(synapse_1), np.argmax(synapse_1))
#     print('synapse_2: ', np.amax(synapse_2), np.argmax(synapse_2))
    l0_actv = [index for index, x in enumerate(l0) if x > activation_threshold]
    l1_actv = [index for index, x in enumerate(l1) if x > activation_threshold]
    l2_actv = [index for index, x in enumerate(l2) if x > activation_threshold]
#     print(l0_actv)
#     print(l1_actv)
#     print(l2_actv)
    return l3


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
    results = estimate(document, show_details)

    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results = [[classes[r[0]], r[1]] for r in results]
    print("%s \n classification: %s" %
          (document.split(' ')[:5], return_results))
    return return_results


test_doc_1 = open(
    '/Users/sarker/WorkSpaces/EclipseNeon/XAI/data/20news-18828_3_class/comp.graphics/38343', encoding='utf-8').read()
test_doc_2 = open(
    '/Users/sarker/WorkSpaces/EclipseNeon/XAI/data/20news-18828_3_class/rec.autos/102800', encoding='utf-8').read()
test_doc_3 = open(
    '/Users/sarker/WorkSpaces/EclipseNeon/XAI/data/20news-18828_3_class/soc.religion.christian/20599', encoding='utf-8').read()
classify(test_doc_1,  show_details=False)
# print()
# classify(test_doc_2,  show_details=False)
# print()
# classify(test_doc_3, show_details=False)
