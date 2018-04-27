import tensorflow as tf
import numpy as np
import os
from utils.pre_processing import preprocess_data, get_model_name, get_vocab_file

SAVED_MODELS_DIR = 'saved_models'
DATA_DIR = 'data'

def _create_feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=val))

def _create_example(features):
    return tf.train.Example(features=tf.train.Features(feature=features))

class Predict:
    def __init__(self, question):
        model_dir = os.path.join('.', SAVED_MODELS_DIR, get_model_name(question))
        vocab_file = os.path.join('.', DATA_DIR, get_vocab_file(question))
        self._vocab_processor = vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
            vocab_file)
        self._predictor = tf.contrib.predictor.from_saved_model(model_dir)

    def run(self, msg):
        features = {"words": _create_feature(self._transform(msg))}
        model_input = _create_example(features)
        model_input = model_input.SerializeToString()
        output = self._predictor({"inputs": [model_input]})
        return self._format_output(output)

    def _transform(self, msg):
        msg = preprocess_data([msg])
        msg_transform = list(self._vocab_processor.transform(msg))
        return msg_transform[0]

    def _format_output(self, output):
        scores = output['scores'].flatten()
        classes = output['classes'].flatten()
        max_score_idx = np.argmax(scores)
        return {
            'class': int(classes[max_score_idx]),
            'score': float(scores[max_score_idx])
        }
