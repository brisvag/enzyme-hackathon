import numpy as np
import tensorflow as tf
import pandas as pd
from keras import backend as K
from keras import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import random


AA_LABELS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
             'T', 'V', 'W', 'Y']


def one_hot_dict(labels: list = None) -> dict:
    labels = AA_LABELS if labels is None else labels
    encoding = (LabelEncoder()
                .fit_transform(labels)
                .reshape(len(labels), 1))
    one_hot = OneHotEncoder(sparse=False, categories=[range(len(labels))]).fit_transform(encoding)
    return dict(zip(labels, one_hot))


def one_hot_encode(sequence: list or str, encoding: dict, max_len: int = None) -> np.array:
    one_hot_sequence = np.array([encoding[element] for element in sequence])
    if max_len is not None:
        padding = np.zeros(shape=(max_len - len(sequence), len(encoding[sequence[0]])))
        one_hot_sequence = np.concatenate([one_hot_sequence, padding])
    return one_hot_sequence


def one_hot_encode_sequences(sequences: list, encoding: dict) -> np.array:
    if len(set(len(seq) for seq in sequences)) > 1:
        raise ValueError('All sequences must have the same length!')
    return np.array([one_hot_encode(seq, encoding) for seq in sequences])


def one_hot_encode_screening_data(data: pd.DataFrame, seed: int = 123):
    """Prepares the train and test data used.

    :return: train and test data and values
    """
    one_hot_encoding = one_hot_dict()
    sequences = one_hot_encode_sequences(data['sequence'].values, one_hot_encoding)
    components = dict(x=train_test_split(sequences, random_state=seed), y=OrderedDict())
    for out in ['productivity', 'performance', 'stability']:
        values = data[out].values
        values = values.reshape(len(values), 1)
        mean, std = np.nanmean(values), np.nanstd(values)
        components['y'][out] = dict()
        components['y'][out]['data'] = train_test_split((values - mean) / std, random_state=seed)
        components['y'][out]['norm'] = mean, std
    return components


def nan_mse(y_true, y_pred):
    """Custom loss function which computes the mean squared error ignoring nan values.

    :param y_true: tf tensor containing the true response values
    :param y_pred: tf tensor containing the predicted response values
    :return: mean squared error for those values in which y_true is not nan
    """
    nan_mask = tf.logical_not(tf.is_nan(y_true))
    n_valid = K.sum(K.cast(nan_mask, tf.float32))
    zeros = K.zeros_like(y_true)
    y_true = tf.where(nan_mask, y_true, zeros)
    y_pred = tf.where(nan_mask, y_pred, zeros)
    return tf.divide(K.sum(K.square(y_true - y_pred)), n_valid)


def r_squared(y_true, y_pred):
    """Custom metric for keras. Computes R^2 (1 - (RSS/TSS))."""

    nan_mask = tf.logical_not(tf.is_nan(y_true))
    zeros = K.zeros_like(y_true)
    y_true = tf.where(nan_mask, y_true, zeros)
    y_pred = tf.where(nan_mask, y_pred, zeros)
    ss_res = K.sum(K.square(y_true - y_pred), axis=0)  # Compute residuals
    ss_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)), axis=0)  # Compute total sum squares
    r2 = 1 - ss_res / (ss_tot + K.epsilon())
    #  If residuals > TSS the result will be negative, make those values 0
    neg_indices = K.less_equal(r2, 0)
    zeros = tf.zeros(shape=tf.shape(r2), dtype=tf.float32)
    r2 = tf.where(neg_indices, zeros, r2)

    return r2  # In multitask cases, it will return the average r2 for each of the tasks


def create_fc_model(data: dict):
    """Create a multi-task fully connected neural network model

    :param data:
    :return:
    """
    sequence_input = Input(shape=data['x'][0][0].shape, name='sequence')
    sequence = Flatten()(sequence_input)

    sequence = Dense(50, activation='relu', kernel_regularizer=l2(1.1))(sequence)
    sequence = Dropout(.1)(sequence)

    outputs = []
    for task, _ in data['y'].items():
        t = Dense(20, activation='relu', kernel_regularizer=l2(1.1))(sequence)
        t = Dropout(.1)(t)
        t = BatchNormalization()(t)
        outputs.append(Dense(1, name=task)(t))

    model = Model(inputs=[sequence_input], outputs=outputs)
    model.compile(loss=nan_mse, optimizer=Adam(lr=.0001), metrics=[r_squared])
    return model


def predict(variant, model, data, one_hot_encoding):
    oh_variant = one_hot_encode_sequences([variant], one_hot_encoding)
    predicted = model.predict(oh_variant)
    return np.squeeze(np.array([(p * d['norm'][1]) + d['norm'][0]
                                for p, (_, d) in zip(predicted, data['y'].items())]))


def generate_variant(reference: str, model: Model, data: dict, alt_residues: list) -> pd.Series:
    one_hot_encoding = one_hot_dict()

    variant = list(reference)
    n_mutations = random.randint(2, len(reference) // 15)
    mutate_at = random.choices(list(range(len(reference))), k=n_mutations)
    for pos in mutate_at:
        variant[pos] = random.choice(alt_residues)
    return pd.Series(predict(variant, model, data, one_hot_encoding), index=list(data['y']))
