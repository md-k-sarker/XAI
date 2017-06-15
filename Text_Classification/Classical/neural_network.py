# use natural language toolkit
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


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(root_data_folder, saved_file):
    """
    Loads 20news group dataset data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    # If file is saved then just load the file
    if os.path.isfile(saved_file):
        x_text, y, y_label = load_data(saved_file)
        return [x_text, y, y_label]

    else:
        # Load data from files
        x_text = []
        y_label = []
        y_textual_label = []
        counter = 0
        for folder_name in os.listdir(root_data_folder):
            if not folder_name.startswith('.'):
                for file_name in os.listdir(os.path.join(root_data_folder, folder_name)):
                    if not file_name.startswith('.'):
                        examples = open(os.path.join(root_data_folder, folder_name, file_name),
                                        mode='r', encoding='utf-8', errors='ignore').read().strip()

                        # Split by words
                        x_text.append(clean_str(examples))
                        label = [0] * 20
                        label[counter] = 1
                        y_label.append(label)
                        y_textual_label.append(folder_name)
                counter += 1

        y = np.concatenate([y_label], 0)
        save_data([x_text, y, y_textual_label], saved_file)
        return [x_text, y, y_textual_label]


def load_data(file_name):
    with open(os.path.abspath(file_name), 'rb') as f:
        x_text, y, y_label = pickle.load(f)
        return [x_text, y, y_label]


def save_data(data, file_name):
    with open(os.path.abspath(file_name), 'wb') as f:
        pickle.dump(data, f)



# Load data
root_data_folder = '../../data/20news-18828/train'
saving_data_file = '../../data/preloaded/20news_18828.dt'
saving_words_data_file = '../../data/preloaded/20news_all_words.dt'
saving_bag_of_words_data_file = '../../data/preloaded/20news_bag_of_words.dt'
x_text, y, y_label = load_data_and_labels(root_data_folder, saving_data_file)

words = []
ignore_words = ['?']

# If file is saved then just load the file
if os.path.isfile(saving_words_data_file):
    with open(os.path.abspath(saving_words_data_file), 'rb') as f:
        words = pickle.load(f)

else:
    # loop through each documents in our training data
    for text, y_, label in zip(x_text, y, y_label):
        # tokenize each word in the document
        w = nltk.word_tokenize(text)
        # add to our words list
        words.extend(w)

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = list(set(words))
    with open(os.path.abspath(saving_words_data_file), 'wb') as f:
        pickle.dump(words, f)

classes = []
documents = []


# loop through each documents in our training data
for text, label in zip(x_text, y_label):

    # add to documents in our corpus
    documents.append((text, label))
    # add to our classes list
    if label not in classes:
        classes.append(label)


# remove duplicates
classes = list(set(classes))

#
print(len(documents), " documents")
print(len(classes), " classes", classes)
print(len(words), " unique stemmed words")

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)


# If file is saved then just load the file
if os.path.isfile(saving_bag_of_words_data_file):
    with open(os.path.abspath(saving_bag_of_words_data_file), 'rb') as f:
        training = pickle.load(f)

    for doc in documents:
        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        output.append(output_row)

else:
    # training set, bag of words for each document
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        training.append(bag)

        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        output.append(output_row)

    with open(os.path.abspath(saving_bag_of_words_data_file), 'wb') as f:
        pickle.dump(training, f)


# sample training/output
i = 0
w = documents[i]
print([stemmer.stem(word.lower()) for word in w])
print(output[i])


# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# convert output of sigmoid function to its derivative


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


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
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return(np.array(bag))


def estimate(document, show_details=False):
    x = bow(document.lower(), words, show_details)
    if show_details:
        print("document:", document, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and input layer1
    l1 = sigmoid(np.dot(l0, synapse_0))
    # matrix multiplication of input layer1 and hidden layer1
    l2 = sigmoid(np.dot(l1, synapse_1))
    # matrix multiplication of hidden layer1 and hidden layer2
    # output layer
    l3 = sigmoid(np.dot(l2, synapse_2))
    # output layer
    #l4 = sigmoid(np.dot(l3, synapse_3))
    return l3


def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print("Training with %s neurons, alpha:%s, dropout:%s %s" % (
        hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
    print("Input matrix: %sx%s    Output matrix: %sx%s" %
          (len(X), len(X[0]), 1, len(classes)))
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2 * np.random.random((hidden_neurons, hidden_neurons)) - 1
    synapse_2 = 2 * np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)
    prev_synapse_2_weight_update = np.zeros_like(synapse_2)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
    synapse_2_direction_count = np.zeros_like(synapse_2)

    for j in iter(range(epochs + 1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))

#         if(dropout):
#             layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))],
# 1 - dropout_percent)[0] * (1.0 / (1 - dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        layer_3 = sigmoid(np.dot(layer_2, synapse_2))

        # error?
        layer_3_error = y - layer_3

        if (j % 1000) == 0 and j > 5000:
            # if this 1k iteration's error is greater than the last iteration,
            # break out
            if np.mean(np.abs(layer_3_error)) < last_mean_error:
                print("delta after " + str(j) + " iterations:" +
                      str(np.mean(np.abs(layer_3_error))))
                last_mean_error = np.mean(np.abs(layer_3_error))
            else:
                print("break:", np.mean(np.abs(layer_3_error)),
                      ">", last_mean_error)
                break

        # output_layer_delta
        layer_3_delta = layer_3_error * sigmoid_output_to_derivative(layer_3)

        # how much did each l2 value contribute to the l3 error (according to
        # the weights)?
        layer_2_error = layer_3_delta.dot(synapse_2.T)

        # in what direction is the target l2?
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to
        # the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_2_weight_update = (layer_2.T.dot(layer_3_delta))
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if(j > 0):
            synapse_0_direction_count += np.abs(
                ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(
                ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))
            synapse_2_direction_count += np.abs(
                ((synapse_2_weight_update > 0) + 0) - ((prev_synapse_2_weight_update > 0) + 0))

        synapse_2 += alpha * synapse_2_weight_update
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update
        prev_synapse_2_weight_update = synapse_2_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(),
               'synapse1': synapse_1.tolist(),
               'synapse2': synapse_2.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
               }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print("saved synapses to:", synapse_file)


X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1,
      epochs=1000000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print("processing time:", elapsed_time, "seconds")


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
print()
classify(test_doc_2,  show_details=False)
print()
classify(test_doc_3, show_details=False)
