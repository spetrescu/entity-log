from re import S
#from src.placeholder_module.funcs import multiply, divide

import numpy as np
import tensorflow
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.utils import plot_model

import tensorflow as tf

import numpy as np
import os
import time
from sklearn.metrics import classification_report, confusion_matrix
import json

print(tf.__version__)

labels = set()

def addition(number_1, number_2):
    return number_1 + number_2

def get_labels_in_data(filepaths):
    """
    Get NER labels in data
    """
    labels = set()

    for filepath in filepaths:
        with open(filepath, "r") as file:
            data = json.load(file)
            for el in data:
                for elm in el["entities"][1]:
                    labels.add(elm)
    return labels


def prepare_raw_data_to_expected_format(file_name):
    """
    Read raw data file and return list of (char,class) pairs
    """
    labels = set()
    processed_data = []
    with open(file_name, "r") as file:
        data = json.load(file)
        for el in data:
            processed_data.append([el["entities"][0], el["entities"][1]])
            for elm in el["entities"][1]:
                labels.add(elm)
    print(
        f"Created {len(processed_data)} samples for {file_name.split('/')[-1].split('.')[0]} set"
    )
    return processed_data


def transform_raw_data_to_lstm_input(data, vocab, labels):
    char2idx = {u: i + 1 for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    label2idx = {u: i + 1 for i, u in enumerate(labels)}
    print("label2idx", label2idx)
    idx2label = np.array(labels)
    train_formatted = []
    for eg in data:
        tokens = eg[0]
        labels = eg[1]
        train_formatted.append(
            [
                [char2idx[x] for x in tokens[:-1]],
                np.array([label2idx[x] for x in labels[:-1]]),
            ]
        )

    return train_formatted


def split_char_labels(eg, char2idx, idx2char, label2idx, idx2label):
    """
    """

    tokens = eg[0]
    labels = eg[1]

    return [
        [char2idx[x] for x in tokens[:-1]],
        np.array([label2idx[x] for x in labels[:-1]]),
    ]


class Bi_LSTM_NER:
    def __init__(self, train_formatted, valid_formatted, test_formatted):
        self.train_formatted = train_formatted
        self.valid_formatted = valid_formatted
        self.test_formatted = test_formatted
        self.series = None
        self.series_valid = None
        self.series_test = None
        self.BATCH_SIZE = 128
        self.BUFFER_SIZE = 1000

        self.ds_series_batch = None
        self.ds_series_batch_valid = None
        self.ds_series_batch_test = None

        self.vocab_size = None
        self.label_size = None

        self.model = None

    def gen_train_series(self):
        for eg in self.train_formatted:
            yield eg[0], eg[1]

    def gen_valid_series(self):
        for eg in self.valid_formatted:
            yield eg[0], eg[1]

    def gen_test_series(self):
        for eg in self.test_formatted:
            yield eg[0], eg[1]

    def create_dataset_objects_train(self):
        self.series = tf.data.Dataset.from_generator(
            self.gen_train_series,
            output_types=(tf.int32, tf.int32),
            output_shapes=((None, None)),
        )

    def create_dataset_objects_valid(self):
        self.series_valid = tf.data.Dataset.from_generator(
            self.gen_valid_series,
            output_types=(tf.int32, tf.int32),
            output_shapes=((None, None)),
        )

    def create_dataset_objects_test(self):
        self.series_test = tf.data.Dataset.from_generator(
            self.gen_test_series,
            output_types=(tf.int32, tf.int32),
            output_shapes=((None, None)),
        )

    def create_padded_series_train(self):
        self.ds_series_batch = self.series.shuffle(self.BUFFER_SIZE).padded_batch(
            self.BATCH_SIZE, padded_shapes=([None], [None]), drop_remainder=True
        )

    def create_padded_series_valid(self):
        self.ds_series_batch_valid = self.series_valid.padded_batch(
            self.BATCH_SIZE, padded_shapes=([None], [None]), drop_remainder=True
        )

    def create_padded_series_test(self):
        self.ds_series_batch_test = self.series_test.padded_batch(
            self.BATCH_SIZE, padded_shapes=([None], [None]), drop_remainder=True
        )

    def print_example_batches(self):
        for (
            input_example_batch,
            target_example_batch,
        ) in self.ds_series_batch_valid.take(1):
            print(input_example_batch)
            print(target_example_batch)
    
    def get_vocab_size(self, vocab):
        self.vocab_size = len(vocab)
        return self.vocab_size
    
    def get_label_size(self, labels):
        self.label_size = len(labels)
        return self.label_size
    
    def build_model(self, vocab_size, label_size, embedding_dim, rnn_units, batch_size):
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None],mask_zero=True),
            Bidirectional(LSTM(rnn_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,recurrent_initializer='glorot_uniform')),
            tf.keras.layers.Dense(label_size)
            ])
        return self.model

    def loss(self, labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    def compute_accuracy(self):
        preds = np.array([])
        y_trues= np.array([])
        for input_example_batch, target_example_batch in self.ds_series_batch_test:
            pred=self.model.predict(input_example_batch, batch_size=self.BATCH_SIZE)
            pred_max=tf.argmax(tf.nn.softmax(pred),2).numpy().flatten()
            y_true=target_example_batch.numpy().flatten()
            preds=np.concatenate([preds,pred_max])
            y_trues=np.concatenate([y_trues,y_true])

        remove_padding = [(p,y) for p,y in zip(preds,y_trues) if y!=0]

        r_p = [x[0] for x in remove_padding]
        r_t = [x[1] for x in remove_padding]

        from collections import Counter
        print(Counter(r_p))
        print(Counter(r_t))

        print("set(r_t) - set(r_p)", set(r_t) - set(r_p))

        print(confusion_matrix(r_p,r_t))
        print(classification_report(r_p,r_t))

        self.model.save('saved_model/my_model_small_trainig')
