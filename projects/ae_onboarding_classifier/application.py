from flask import Flask, request
import tensorflow as tf
import os
from sagemaker.tensorflow.predictor import tf_serializer, tf_deserializer
from sagemaker.predictor import RealTimePredictor
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from sagemaker.tensorflow.tensorflow_serving.apis import classification_pb2
from utils.pre_processing import preprocess_data

app = Flask(__name__)
ENDPOINT_NAME = 'sagemaker-tensorflow-2018-04-24-18-54-38-930'


def create_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def build_request(name, features, signature_name=DEFAULT_SERVING_SIGNATURE_DEF_KEY):
    req = classification_pb2.ClassificationRequest()
    req.model_spec.name = name
    req.model_spec.signature_name = signature_name

    example = tf.train.Example(features=tf.train.Features(feature=features))
    req.input.example_list.examples.extend([example])

    return req

def transform(msg, question):
    msg_transform = preprocess_data([msg])
    vocab_file = os.path.join("data", "vocab_question_{}".format(question))
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_file)
    msg_transform = vocab_processor.transform(msg_transform)
    return list(msg_transform)[0]


@app.route("/prediction/<int:question>", methods=['POST'])
def prediction(question):
    if request.is_json:
        msg = request.get_json()
        predictor = RealTimePredictor(endpoint=ENDPOINT_NAME,
                              serializer=tf_serializer,
                              content_type='application/octet-stream')
        features = {'words': create_feature(transform(msg, question))}
        req = build_request("generic_model", features)
        return predictor.predict(req).decode('utf-8')
