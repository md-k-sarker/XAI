import numpy as np
import re
import itertools
from collections import Counter
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn


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


def load_data_and_labels(root_data_folder):
    """
    Loads 20news group dataset data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    x_text = []
    y_label = []
    counter = 0
    for folder_name in os.listdir(root_data_folder):
        if not folder_name.startswith('.'):
            for file_name in os.listdir(os.path.join(root_data_folder, folder_name)):

                examples = open(os.path.join(root_data_folder, folder_name, file_name),
                                mode='r', encoding='utf-8', errors='ignore').read().strip()

                # Split by words
                x_text.append(clean_str(examples))
                label = [0] * 20
                label[counter] = 1
                y_label.append(label)

            counter += 1

    y = np.concatenate([y_label], 0)
    return [x_text, y]

#                 examples = [s.strip() for s in positive_examples]
#                 negative_examples = list(
#                     open(negative_data_file, "r").readlines())
#                 negative_examples = [s.strip() for s in negative_examples]
#                 # Split by words
#                 x_text = positive_examples + negative_examples
#                 x_text = [clean_str(sent) for sent in x_text]
#                 # Generate labels
#                 positive_labels = [[0, 1] for _ in positive_examples]
#                 negative_labels = [[1, 0] for _ in negative_examples]
#                 y = np.concatenate([positive_labels, negative_labels], 0)

#     return [x_text, y]


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


# # Parameters
# # ==================================================
#
# # Data loading params
# tf.flags.DEFINE_float("dev_sample_percentage", .1,
#                       "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("root_data_folder", "./data/20news-18828",
#                        "Data source for the root folder.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
#                        "Data source for the negative data.")
#
# # Model Hyperparameters
# tf.flags.DEFINE_integer("embedding_dim", 128,
#                         "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_string("filter_sizes", "3,4,5",
#                        "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_integer(
#     "num_filters", 128, "Number of filters per filter size (default: 128)")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
#                       "Dropout keep probability (default: 0.5)")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.0,
#                       "L2 regularization lambda (default: 0.0)")
#
# # Training parameters
# tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer(
#     "num_epochs", 200, "Number of training epochs (default: 200)")
# tf.flags.DEFINE_integer("evaluate_every", 100,
#                         "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every", 100,
#                         "Save model after this many steps (default: 100)")
# tf.flags.DEFINE_integer("num_checkpoints", 5,
#                         "Number of checkpoints to store (default: 5)")
# # Misc Parameters
# tf.flags.DEFINE_boolean("allow_soft_placement", True,
#                         "Allow device soft device placement")
# tf.flags.DEFINE_boolean("log_device_placement", False,
#                         "Log placement of ops on devices")
#
# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")
#
#
# x_text, y = load_data_and_labels(FLAGS.root_data_folder)
# print(len(x_text), len(y))
# print(x_text[18827], y[18827])
