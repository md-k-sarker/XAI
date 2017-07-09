'''
Created on Jul 9, 2017

@author: sarker
'''
from collections import Counter
import itertools
import json
import os
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


# from sklearn.neural_network import MLPClassifier
# from bokeh.plotting import figure, output_file, show
print("Program started")



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
training, output = classifier_2_class.convert_training_documents_to_vector(documents, classes,words)
X_training = np.array(training)
y_training = np.array(output)

print('X_training.shape: ', X_training.shape)
print('y_training.shape: ', y_training.shape)
clf = classifier_2_class.train_NN(X_training, y_training)



# 
# test data
X_test, y_test = classifier_2_class.load_test_documents(util.test_file_dir,words)
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
