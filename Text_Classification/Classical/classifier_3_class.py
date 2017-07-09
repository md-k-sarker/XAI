'''
Created on Jun 16, 2017

@author: sarker
'''

from collections import Counter
import datetime
import datetime
import itertools
import json
import os
import pickle
import re
import time

from bokeh.plotting import figure, output_file, show
import nltk
from nltk.stem.porter import *
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.neural_network import BaseMultilayerPerceptron
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import numpy as np
import tensorflow as tf


# Load Dataset
# use natural language toolkit
stemmer = PorterStemmer()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_document/blob/master/process_data.py
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


def load_data_and_labels(train_data_folder, saved_file):
    """
    Loads 20news group dataset data from files, splits the data into words and generates labels.
    Returns split documents and labels.
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
        for folder_name in os.listdir(train_data_folder):
            if not folder_name.startswith('.'):
                for file_name in os.listdir(os.path.join(train_data_folder, folder_name)):
                    if not file_name.startswith('.') and not file_name.endswith('keyword'):
                        examples = open(os.path.join(train_data_folder, folder_name, file_name),
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


def clean_up_document(document):
    # tokenize the pattern
    document_words = nltk.word_tokenize(document)
    # stem each word
#     document_words = [stemmer.stem(word.lower()) for word in document_words]
    return document_words


def bow(document, words, show_details=False):
    '''Take a document of text.
    return: 
    bag_0_1_vector: comparing with global words embedding make vector [0, 1, 0 ...]
    bag_words_vector: comparing with global words embedding make vector [0, word, 0 ...]
    bag_words: bag of words vector [word1, word2...]'''
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

    return(bag_0_1_vector, np.array(bag_words_vector), np.array(bag_words))


# Load data

train_data_folder = '../../data/20news-18828/train'
test_file_dir = '../../data/20news-18828/test'
saving_data_file = '../../data/20news-18828/3_class/20news_18828.dt'
saving_global_words_embedding_file = '../../data/20news-18828/3_class/20news_all_words.dt'
saving_bag_of_words_data_file = '../../data/20news-18828/3_class/20news_bag_of_words.dt'
saving_classifier_model_file = '../../data/20news-18828/3_class/model.dt'
saving_backup_classifier_model_file = '/Users/sarker/Google Drive/mdkamruzzamansarker@gmail.com/Education/Research/' + \
    'Projects/XAI-Freddy/experiments/backup/preloaded_Jun_17_2017/model.dt'
x_text, y, y_label = load_data_and_labels(train_data_folder, saving_data_file)

words = []
ignore_words = ['?', '#', '1', '2', '3', '4',
                '5', '6', '7', '8', '9', '0', '.', ',']

print("Program started")


def load_word_embeggings():
    '''word embedding. 
    vector of all words'''
    words = []
    # If file is saved then just load the file
    if os.path.isfile(saving_global_words_embedding_file):
        with open(os.path.abspath(saving_global_words_embedding_file), 'rb') as f:
            words = pickle.load(f)

    else:
        # loop through each documents in our training data
        for text, y_, label in zip(x_text, y, y_label):
            # tokenize each word in the document
            w = nltk.word_tokenize(text)
            # add to our words list
            words.extend(w)

        # stem and lower each word and remove duplicates
#         words = [stemmer.stem(w.lower())
#                  for w in words if w not in ignore_words]
        words = list(set(words))
        with open(os.path.abspath(saving_global_words_embedding_file), 'wb') as f:
            pickle.dump(words, f)

    return words


def load_document_and_classes():
    # loop through each documents in our training data
    classes = []
    documents = []
    for text, label in zip(x_text, y_label):

        # add to documents in our corpus
        documents.append((text, label))
        # add to our classes list
        if label not in classes:
            classes.append(label)

    return classes, documents


def convert_training_documents_to_vector():
    print("convert_training_documents_to_vector started")
    # create an empty array for our output
    training = []
    output = []
    output_empty = [0] * len(classes)
    # If file is saved then just load the file
    if os.path.isfile(saving_bag_of_words_data_file):
        print(saving_bag_of_words_data_file, ' found in disk. loading it.')
        with open(os.path.abspath(saving_bag_of_words_data_file), 'rb') as f:
            training = pickle.load(f)

        for doc in documents:
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            output.append(output_row)

    else:
        # training set, bag of words for each document
        print(saving_bag_of_words_data_file,
              ' not found in disk. doing operations....')
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

            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            output.append(output_row)

        print("dumping to pickle started")
        with open(os.path.abspath(saving_bag_of_words_data_file), 'wb') as f:
            pickle.dump(training, f)

    return training, output


words = load_word_embeggings()
classes, documents = load_document_and_classes()

# remove duplicates
classes = list(set(classes))

print(len(documents), " documents")
print(len(classes), " classes", classes)
print(len(words), " unique stemmed words")

class_names = {}


def load_test_documents(test_file_dir=test_file_dir):
    X_test = []
    y_test = []
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

                    bag_0_1_vector, bag_words_vector, bag_words = bow(
                        document.lower(), words, show_details=False)
                    y = [0, 0, 0]
                    if folder_name == 'comp.windows.x':
                        y = [1, 0, 0]
                    elif folder_name == 'rec.sport.hockey':
                        y = [0, 1, 0]
                    else:
                        y = [0, 0, 1]
                    X_test.append(bag_0_1_vector)
                    y_test.append(y)
    return np.array(X_test), np.array(y_test)


# training data

training, output = convert_training_documents_to_vector()
X_training = np.array(training)
y_training = np.array(output)
train = len(X_training)
print('X_training: ', X_training.shape)
print('y_training: ', y_training.shape)

# test data
X_test, y_test = load_test_documents()
print('X_Test: ', X_test.shape)
print('y_test: ', y_test.shape)
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
no_of_hidden_neurons = ((len(X_training[0])))


def train_NN():
    start_time = time.time()
    if os.path.isfile(saving_classifier_model_file):
        print('existing mlp model found. loading it.')
        clf = joblib.load(saving_classifier_model_file)
        print('clf: ', clf)
    else:
        print('mlp initilizing started')
        mlp = MLPClassifier(hidden_layer_sizes=(no_of_hidden_neurons,
                                                no_of_hidden_neurons,),
                            solver='adam', activation='relu',
                            learning_rate='adaptive', learning_rate_init=0.001,
                            max_iter=10,
                            verbose=True, tol=0.000000001)
        print('clf: ', mlp)
        print('mlp fitting started')
        mlp.fit(X_training, y_training)
        print('mlp fitting finished')

        '''save to disk'''
        joblib.dump(mlp, saving_classifier_model_file)
        clf = mlp
    end_time = time.time()
    print('trained in: ', end_time - start_time)
    return clf
# test data


print('train_shape: ', X_training.shape)
print('test_shape: ', X_test.shape)
clf = train_NN()
print('predicting started...')
print('##########Activated neurons##########')
'''see difference in activated neurons for each class'''
'''class_names: dict'''
X_test_comp_windows = X_test[:2]
X_rec_sport_hockey = X_test[2:4]
X_talk_politics_guns = X_test[4:]

predict_proba, activated_neurons_comp_windows, activated_neurons_raw_sum_comp_windows = clf.predict_proba(
    X_test_comp_windows)
predict_proba, activated_neurons_sport_hockey, activated_neurons_raw_sum_sport_hockey = clf.predict_proba(
    X_rec_sport_hockey)
predict_proba, activated_neurons_politics_guns, activated_neurons_raw_sum_politics_guns = clf.predict_proba(
    X_talk_politics_guns)

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

#fig = plt.figure()
fig1 = plt.figure(1).add_subplot(111)  # fig.add_subplot(1, 3, 1)
fig2 = plt.figure(2).add_subplot(111)  # fig.add_subplot(1, 3, 2)
fig3 = plt.figure(3).add_subplot(111)  # fig.add_subplot(1, 3, 3)
fig4 = plt.figure(4).add_subplot(111)
fig5 = plt.figure(5).add_subplot(111)
fig1.set_title('class comp.windows_x')
fig2.set_title('class sport_hockey')
fig3.set_title('class politics_guns')
color = ['green', 'black', 'olive']

for layer_i in range(3):
    #     print('activated_neurons_comp_windows: ',
    #           activated_neurons_comp_windows[layer_i])
    #     print('activated_neurons_sport_hockey: ',
    #           activated_neurons_sport_hockey[layer_i])
    #     print('activated_neurons_politics_guns: ',
    #           activated_neurons_politics_guns[layer_i])
    activated_for_all_class = activated_neurons_comp_windows[
        layer_i] & activated_neurons_sport_hockey[layer_i] & activated_neurons_politics_guns[layer_i]
    #print('activated_for_all_class: ', activated_for_all_class)
#     print('hidden_layer %s' % (layer_i + 1))
#     print('activated_for_all_class: ', activated_for_all_class)
#     print('activated_neurons_only_for_comp_windows: ',
#           activated_neurons_comp_windows[
#               layer_i] - activated_for_all_class)
#     print('activated_neurons_only_for_sport_hockey',
#           activated_neurons_sport_hockey[layer_i] - activated_for_all_class)
#     print('activated_neurons_only_for_politics_guns',
#           len(activated_neurons_politics_guns[layer_i] - activated_for_all_class))
#     print('\n')
    x = [layer_i + 1] * 3474

    # add a square renderer with a size, color, and alpha
    y_windows = [0] * 3474
    y_hockey = [0] * 3474
    y_guns = [0] * 3474
    y_raw_all = [0] * 3474
    s_raw_all = [0] * 3474

    for index in activated_neurons_comp_windows[layer_i]:
        y_windows[index] = index

    #fig1.scatter(x, y, color=color[layer_i], marker='.')
    #p1.square(x, y, size=2, color="olive", alpha=0.5)

    for index in activated_neurons_sport_hockey[layer_i]:
        y_hockey[index] = index
    #fig2.scatter(x, y, color=color[layer_i], marker='.')
    #p2.square(x, y, size=2, color="olive", alpha=0.5)

    for index in activated_neurons_politics_guns[layer_i]:
        y_guns[index] = index

    for index, value in enumerate(activated_neurons_raw_sum_all[layer_i]):
        if value > 2:
            y_raw_all[index] = index
            s_raw_all[index] = value * 10
        else:
            y_raw_all[index] = 0
            s_raw_all[index] = 0
#     print('activated_neurons_politics_guns[layer_i]: ',
#           activated_neurons_politics_guns[layer_i])
#     print('activated_neurons_raw_sum_all[layer_i]: ',
#           len(y_raw_all), y_raw_all)
#     print('y_guns: ', y_guns)

    if layer_i == 0:
        print()
        print('activated_neurons_raw_sum_all[layer_i]: ',
              activated_neurons_raw_sum_all[layer_i])
        #print('y_raw_all: ', y_raw_all)
        #print('s_raw_all: ', s_raw_all)
        print()
    #fig3.scatter(x, y, color=color[layer_i], marker='.')
    #p3.square(x, y, size=2, color="olive", alpha=0.5)
    '''using multiple figure'''
    lbl = 'hidden_layer_' + str(layer_i + 1)
    fig1.scatter(x, y_windows, color=color[layer_i],
                 marker='.', label=lbl)
    fig2.scatter(x, y_hockey, color=color[layer_i],
                 marker='.', label=lbl)
    fig3.scatter(x, y_guns, color=color[layer_i], marker='.', label=lbl)

    '''using single figure'''
    fig4.scatter(np.array(x) - .15, y_windows, color=color[0],
                 marker='.')
    fig4.scatter(np.array(x), y_hockey, color=color[1],
                 marker='.')
    fig4.scatter(np.array(x) + .15, y_guns,
                 color=color[2], marker='.')

    '''using weighted bubble'''
    fig5.scatter(x, y_raw_all, s=s_raw_all, color=color[layer_i],
                 marker='.', label=lbl)

# show the results
# show(p1)
# show(p2)
# show(p3)
y_ticks = [x for x in range(0, 3475)]
fig1.set_xticks([1, 2, 3])
fig2.set_xticks([1, 2, 3])
fig3.set_xticks([1, 2, 3])
fig4.set_xticks([1, 2, 3])
fig5.set_xticks([1, 2, 3])

# fig1.set_yticks(y_ticks)
# fig2.set_yticks(y_ticks)
# fig3.set_yticks(y_ticks)
# fig4.set_yticks(y_ticks)
# fig5.set_yticks(y_ticks)

fig1.set_xlabel('hidden_layers ')
fig2.set_xlabel('hidden_layers')
fig3.set_xlabel('hidden_layers')
fig4.set_xlabel('hidden_layers')
fig5.set_xlabel('hidden_layers')

fig1.set_ylabel('n\'th neuron')
fig2.set_ylabel('n\'th neuron')
fig3.set_ylabel('n\'th neuron')
fig4.set_ylabel('n\'th neuron')
fig5.set_ylabel('n\'th neuron')

# fig4.set_label(['computer', 'guns', 'hockey', 'computer',
#                 'guns', 'hockey', 'computer', 'guns', 'hockey'])

fig1.legend(loc='upper left')
fig2.legend(loc='upper left')
fig3.legend(loc='upper left')
fig4.legend(['class comp.windows_x', 'class sport_hockey',
             'class politics_guns'], loc='upper left')
fig5.legend(loc='upper left')

'''dump to pickle'''
figure_file = '../../data/20news-18828/preloaded/fig.dt'
with open(figure_file, 'wb') as f:
    pickle.dump([fig1, fig2, fig3, fig4], f)

'''load figure from saved data'''

plt.show()

#print('predict: ', clf.predict(X_test))
#predict_proba, activated_neurons = clf.predict_proba(X_test)
#print('predict_proba: ', predict_proba)

# for i, neurons in enumerate(activated_neurons):
#     print('layer %s: ' % (i + 1), neurons)