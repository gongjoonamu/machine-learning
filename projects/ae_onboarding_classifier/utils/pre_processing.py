import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
import numpy as np

FILE_NAME = "anywhere_expert_transcripts.csv"
# dumped from NLTK stopwords
STOPS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def _filter_words(words):
    # remove non-alphanumeric symbols
    words = re.sub('[^A-Za-z0-9]', ' ', words).lower().split(' ')
    for i in range(len(words)):
        if words[i] in STOPS:
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
