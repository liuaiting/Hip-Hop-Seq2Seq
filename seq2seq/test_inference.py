# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import time

import tensorflow as tf

from . import config
from . import inference


FLAGS = tf.app.flags.FLAGS

# src_file = "seq2seq/data/plana/5.1.txt"
src_file = FLAGS.inference_input_file

if __name__ == "__main__":

    lines = codecs.open(src_file, "r", "utf-8").readlines()

    start_time = time.time()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = inference.load_model(sess)
    print("load infer model cost %.4f" % (time.time() - start_time))

    start_time = time.time()
    results = inference.inference_n(model, sess, lines, 3)
    print("inference one sample cost %.4f" % (
            (time.time() - start_time) / (len(lines) * 3)))




