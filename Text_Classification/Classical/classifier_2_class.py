'''
2 class Classifier and explainer.
Classifier using sk-learn model 
'''
'''
Created on Jun 16, 2017

@author: sarker
'''
import re
import itertools
from collections import Counter
import time
import os
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

# from sklearn.neural_network import MLPClassifier
from Classical import multilayer_perceptron_custom
from multilayer_perceptron_custom import MLPClassifier_Custom
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf
import nltk
from nltk.stem.porter import *
# from bokeh.plotting import figure, output_file, show

from Classical import data_helpers as dth

print("Program started")

train_data_folder = '../../data/20news-18828/2_class/train'
test_file_dir = '../../data/20news-18828/2_class/test'
saving_data_file = '../../data/20news-18828/2_class/model/20news_18828.pickle'
saving_global_words_embedding_file = '../../data/20news-18828/2_class/model/20news_all_words_embedding.pickle'
saving_bag_of_words_data_file = '../../data/20news-18828/2_class/model/20news_bag_of_words.pickle'
saving_classifier_model_file = '../../data/20news-18828/2_class/model/model.pickle'
saving_backup_classifier_model_file = '/Users/sarker/Google Drive/mdkamruzzamansarker@gmail.com/Education/Research/' + \
    'Projects/XAI-Freddy/experiments/backup/preloaded_Jun_17_2017/model.dt'




def make_word_embeddings(x_text, y, y_label):
    '''word embedding. 
    vector of all words'''
    words = []
    # If file is saved then just load the file
    if os.path.isfile(saving_global_words_embedding_file):
        with open(os.path.abspath(saving_global_words_embedding_file), 'rb') as f:
            words = pickle.load(f)

    else:
        # loop through each documents in our training data
        for each_text, each_y, each_label in zip(x_text, y, y_label):
            # tokenize each word in the document
            w = nltk.word_tokenize(each_text)
            # add to our words list
            words.extend(w)

        # stem and lower each word and remove duplicates
        # words = [stemmer.stem(w.lower())
        #     for w in words if w not in ignore_words]

        # save word embeddings to cache
        words = list(set(words))
        with open(os.path.abspath(saving_global_words_embedding_file), 'wb') as f:
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


def convert_training_documents_to_vector(documents, classes):
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
    if os.path.isfile(saving_bag_of_words_data_file):
        print(saving_bag_of_words_data_file, ' found in disk. loading it.')
        with open(os.path.abspath(saving_bag_of_words_data_file), 'rb') as f:
            training = pickle.load(f)

        for doc in documents:
            # convert output to vector
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            output.append(output_row)

    else:
        # training set, bag of words for each document
        print(saving_bag_of_words_data_file,
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
        with open(os.path.abspath(saving_bag_of_words_data_file), 'wb') as f:
            pickle.dump(training, f)
    print("convert_training_documents_to_vector finished.")  
    return training, output


def train_NN(X_training, y_training):
    start_time = time.time()
    print('train_NN started...')
    if os.path.isfile(saving_classifier_model_file):
        print('existing mlp model found in cache. loading it.')
        clf = joblib.load(saving_classifier_model_file)
        print('clf: ', clf)
    else:
        print('mlp initilizing started')
        no_of_hidden_neurons = ((len(X_training[0])))
        mlp = MLPClassifier_Custom(hidden_layer_sizes=(no_of_hidden_neurons,
                                                no_of_hidden_neurons,),
                            solver='adam', activation='relu',
                            learning_rate='adaptive', learning_rate_init=0.001,
                            max_iter=100,
                            verbose=True, tol=0.000000001)
        print('clf: ', mlp)

        print('mlp fitting started...')
        mlp.fit(X_training, y_training)
        print('mlp fitting finished')

        '''save to disk'''
        joblib.dump(mlp, saving_classifier_model_file)
        clf = mlp
    end_time = time.time()
    print('trained in: ', end_time - start_time, ' seconds')
    print('train_NN finished')
    return clf


def load_test_documents(test_file_dir=test_file_dir):
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

x_text, y, y_label = dth.load_data_and_labels(train_data_folder, saving_data_file)
words = make_word_embeddings(x_text, y, y_label)

documents, classes = create_document_and_classes_from_data(x_text, y, y_label)
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
training, output = convert_training_documents_to_vector(documents, classes)
X_training = np.array(training)
y_training = np.array(output)

print('X_training.shape: ', X_training.shape)
print('y_training.shape: ', y_training.shape)
clf = train_NN(X_training, y_training)

# 
# test data
X_test, y_test = load_test_documents()
print('X_Test.shape: ', X_test.shape)
print('y_test.shape: ', y_test.shape)
 
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


color = ['green', 'black', 'olive']

for layer_i in range(2):
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
    x = [layer_i + 1] * len(words)

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
            s_raw_all[index] = value * 10
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
    lbl = 'hidden_layer_' + str(layer_i + 1)
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
    fig4.scatter(x, y_raw_all, s=s_raw_all, color=color[layer_i],
                 marker='.', label=lbl)

# show the results
# show(p1)
# show(p2)
# show(p3)
y_ticks = [x for x in range(0, len(words))]
fig1.set_xticks([1, 2])
fig2.set_xticks([1, 2])
fig3.set_xticks([1, 1, 2, 2])
fig4.set_xticks([1, 2])

# fig1.set_yticks(y_ticks)
# fig2.set_yticks(y_ticks)
# fig3.set_yticks(y_ticks)
# fig4.set_yticks(y_ticks)
# fig5.set_yticks(y_ticks)

fig1.set_xlabel('hidden_layers ')
fig2.set_xlabel('hidden_layers')
fig3.set_xlabel('hidden_layers')
fig4.set_xlabel('hidden_layers')

fig1.set_ylabel('n\'th neuron')
fig2.set_ylabel('n\'th neuron')
fig3.set_ylabel('n\'th neuron')
fig4.set_ylabel('n\'th neuron')

# fig4.set_label(['computer', 'guns', 'hockey', 'computer',
#                 'guns', 'hockey', 'computer', 'guns', 'hockey'])

fig1.legend(loc='upper left')
fig2.legend(loc='upper left')
fig3.legend(['class comp.windows_x', 'class sport_hockey'], loc='upper left')
fig4.legend(loc='upper left')

'''dump to pickle'''
figure_file = '../../data/20news-18828/2_class/model/fig.pickle'

with open(figure_file, 'wb') as f:
    pickle.dump([fig1,fig2,fig3,fig4], f)

'''load figure from saved data'''
'''after loading from pickle zooming is not working'''
    
plt.show()

# print('predict: ', clf.predict(X_test))
# predict_proba, activated_neurons = clf.predict_proba(X_test)
# print('predict_proba: ', predict_proba)

# for i, neurons in enumerate(activated_neurons):
#     print('layer %s: ' % (i + 1), neurons)
