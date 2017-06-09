import numpy as np
import re
import itertools
from collections import Counter
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle


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


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
