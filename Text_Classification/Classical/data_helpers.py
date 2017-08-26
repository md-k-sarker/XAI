from collections import Counter
import datetime
import itertools
import os
import pickle
import re
import time

import nltk
from nltk.stem.porter import *

import numpy as np
import numpy as np
import tensorflow as tf
import pandas as pd

from Classical import utils as util


ignore_words = ['?', '#', '1', '2', '3', '4',
                '5', '6', '7', '8', '9', '0', '.', ',']

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


def load_concepts(file_name,saved_file,use_cache=False):
    '''
    Load the concepts from the csv file
    '''

    # io_name = 'file://'+root_folder+'concepts_baseball.xlsx'
    #io_name= os.path(file_name)
    concepts = pd.read_excel(io=file_name)   
    #save_data_to_cache(concepts,saved_file)
    
    #concepts = concepts.drop([0],axis=1)
    #print(type(concepts.fillna(0).values), ': ',concepts.fillna(0).values)  
    
    #concepts = concepts.drop(concepts.index[0]) #.drop(concepts.index[[0]])
#     print(concepts.shape)
#     print(concepts)
#     print(concepts.index[0])
    #concepts = concepts.iloc[1:,1:]
    '''
    for i in range(0,concepts.shape[1],1):
        #print(type(concepts.iloc[:,i]))
        if i>0:
            concepts.iloc[:,i] = concepts.iloc[:,i].fillna(concepts.iloc[:,i-1])
        #print(concepts.iloc[:,i])
    '''
    #print('concepts: ',concepts)
    return concepts

load_concepts(util.concepts_file_baseball,'',False)

def load_keywords_and_labels(train_data_folder, saved_file,use_cache = False):
    """
    Loads 20news group dataset data from files, splits the data into words and generates labels.
    Returns split documents and labels.
    """
    
    # If file is saved then just load the file
    if os.path.isfile(saved_file) and use_cache:
        x_text, y, y_label = load_data_from_cache(saved_file)
        return [x_text, y, y_label]

    else:
        # Load data from files
        x_keywords = []
        x_text = []
        y_label = []
        y_textual_label = []
        counter = 0
        for folder_name in os.listdir(train_data_folder):
            if not folder_name.startswith('.'):
                for file_name in os.listdir(os.path.join(train_data_folder, folder_name)):
                    if ((not file_name.startswith('.')) and ( not file_name.endswith('keyword'))):
                        examples = open(os.path.join(train_data_folder, folder_name, file_name),
                                        mode='r', encoding='utf-8', errors='ignore').read().strip()

                        # Split by words
                        words_for_keyword = [word for word in nltk.word_tokenize(clean_str(examples))]
                        words_for_text = clean_str(examples) 
                        
                        most_common_words = ""
                        for word, word_count in Counter(words_for_keyword).most_common(8):
                            most_common_words += " "
                            most_common_words += word 
                        x_keywords.append(most_common_words)
                        x_text.append(words_for_text)
#                         print('examples: ', examples)
#                         print('words_for_keyword: ', words_for_keyword)
#                         print('words_for_text: ', words_for_text)
#                         print('keywords: ', most_common_words)
#                         print()
                        label = [0] * 2
                        label[counter] = 1
                        y_label.append(label)
                        y_textual_label.append(folder_name)
                counter += 1
        
        # here np.concatenate and np.array both performing the same operations.
        # and making
        #  [[1 0]
        #   ....
        #   [0 1]
        #   [0 1]
        #   [0 1]]
        y = np.concatenate([y_label], axis=0)
        y_ = np.array(y_label)
        #save_data_to_cache([x_text, y, y_textual_label], saved_file)
        #print('x_keyword: ', x_keywords)
        #print('x_text: ', x_text)
        return [x_keywords, y, y_textual_label]



def load_data_and_labels(train_data_folder, saved_file):
    """
    Loads 20news group dataset data from files, splits the data into words and generates labels.
    Returns split documents and labels.
    """
    
    # If file is saved then just load the file
    if os.path.isfile(saved_file):
        x_text, y, y_label = load_data_from_cache(saved_file)
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
                        label = [0] * 2
                        label[counter] = 1
                        y_label.append(label)
                        y_textual_label.append(folder_name)
                counter += 1
        
        # here np.concatenate and np.array both performing the same operations.
        # and making
        #  [[1 0]
        #   ....
        #   [0 1]
        #   [0 1]
        #   [0 1]]
        y = np.concatenate([y_label], axis=0)
        y_ = np.array(y_label)
        save_data_to_cache([x_text, y, y_textual_label], saved_file)
        return [x_text, y, y_textual_label]


def load_data_from_cache(file_name):
    with open(os.path.abspath(file_name), 'rb') as f:
        x_text, y, y_label = pickle.load(f)
        return [x_text, y, y_label]


def save_data_to_cache(data, file_name):
    with open(os.path.abspath(file_name), 'wb') as f:
        pickle.dump(data, f)


def tokenize_document(document):
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
    document_words = tokenize_document(document)
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

    # print('bag_0_1_vector: ', bag_0_1_vector)

    bag_words = [p for p in bag_words_vector if (p != 0)]
    if show_details:
        print('bag_of_words: ', bag_words)
        print('####################')
        print('bag_words_vector', bag_words_vector)
        print('####################')

    return(bag_0_1_vector, np.array(bag_words_vector), np.array(bag_words))
