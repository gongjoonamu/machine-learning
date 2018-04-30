import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.corpus import stopwords

FILE_NAME = "anywhere_expert_transcripts.csv"
nltk.download('stopwords')

def _filter_words(words):
    stops = stopwords.words("english")
    # remove non-alphanumeric symbols
    words = re.sub('[^A-Za-z0-9]', ' ', words).lower().split(' ')
    for i in range(len(words)):
        if words[i] in stops:
            words[i] = ''
    return words

def preprocess_data(data):
    processed_data = []
    for transcript in data:
        processed_data.append(" ".join(_filter_words(transcript)))
    return processed_data

def _get_training_set(key, data):
    return data["x{}".format(key)], data["y{}".format(key)]

def get_data(training_dir, key, test_size=0.4, random_state=42, preprocess=True):
    train_data = os.path.join(training_dir, FILE_NAME)
    transcripts = pd.read_csv(
        train_data, delimiter=',', header=0, encoding='utf-8')
    X_train, Y_train = _get_training_set(key, transcripts)
    x_train, x_test, y_train, y_test = train_test_split(
        X_train, Y_train, test_size=test_size, random_state=random_state)

    if preprocess:
        x_train = preprocess_data(x_train)
        x_test = preprocess_data(x_test)
    y_train = y_train.fillna(0).astype(np.int32).values
    y_test = y_test.fillna(0).astype(np.int32).values

    return x_train, x_test, y_train, y_test

def get_model_name(question):
    return "model_question_{}".format(question)

def get_vocab_file(question):
    return "vocab_question_{}".format(question)
