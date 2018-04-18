import tensorflow as tf
import os
from pre_processing import get_data

MAX_DOCUMENT_LENGTH = 300  # length of word vector consisting of word IDs
SAVE_DIR = '../data'

data_dir = os.path.join("..", "data")

for i in range(1, 4):
    x_train, x_test, y_train, y_test = get_data(data_dir, i, preprocess=True)
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)
    vocab_processor.fit(x_train)
    vocab_processor.save(os.path.join(SAVE_DIR, "vocab_question_{}".format(i)))
