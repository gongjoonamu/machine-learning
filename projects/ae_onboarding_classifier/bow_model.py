"""Model used for classification of Anywhere Expert assessment transcripts

Note: This file is used for development and experimentation. Files used for production
are prepended with "sm_" signifying their use by AWS SageMaker.

This model implements a binary bag of words classifier using TensorFlow
"""
import numpy as np
import tensorflow as tf
import os
import sys
from utils.pre_processing import get_data

tf.logging.set_verbosity(tf.logging.INFO)

MAX_DOCUMENT_LENGTH = 300  # length of word vector consisting of word IDs
EMBEDDING_SIZE = 8  # input layer, ~=log_2(MAX_DOCUMENT_LENGTH)
WORDS_FEATURE = 'words'  # input tensor name
STEPS = 1e3
QUESTION = 1


def bag_of_words_model(num_buckets):
    bow_column = tf.feature_column.categorical_column_with_identity(
        WORDS_FEATURE, num_buckets=num_buckets)
    bow_embedding_column = tf.feature_column.embedding_column(
        bow_column, dimension=EMBEDDING_SIZE, combiner="sqrtn")
    model = tf.estimator.LinearClassifier(
        feature_columns=[bow_embedding_column],
        loss_reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    )
    return model

x_train, x_test, y_train, y_test = get_data("data", QUESTION, preprocess=True)

# Process vocabulary
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
    MAX_DOCUMENT_LENGTH)

x_transform_train = vocab_processor.fit_transform(x_train)
x_transform_test = vocab_processor.transform(x_test)

x_train = np.array(list(x_transform_train))
x_test = np.array(list(x_transform_test))

n_words = len(vocab_processor.vocabulary_)

classifier = bag_of_words_model(n_words)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={WORDS_FEATURE: x_train},
    y=y_train,
    batch_size=len(x_train),
    num_epochs=None,
    shuffle=True)


classifier.train(input_fn=train_input_fn, steps=STEPS)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={WORDS_FEATURE: x_train}, y=y_train, num_epochs=1, shuffle=False)


scores = classifier.evaluate(input_fn=test_input_fn)
print('Accuracy (train): {:.5f}'.format(scores['accuracy']))

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={WORDS_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False)


scores = classifier.evaluate(input_fn=test_input_fn)
print('Accuracy (test): {:.5f}'.format(scores['accuracy']))
