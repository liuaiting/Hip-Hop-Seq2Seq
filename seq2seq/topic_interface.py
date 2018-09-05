# -*- coding: utf-8 -*-
import random
import codecs
import os
import time

import tensorflow as tf

from . import inference
import socket


data_path = "seq2seq/data/plana"
topic_list = ["1.0", "2.0", "2.1", "4.1", "5.1", "12.0"]

start_time = time.time()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model = inference.load_model(sess)
print("load infer model cost %.4f" % (time.time() - start_time))


def select_n_sample(topic, n):
    topic_file = os.path.join(data_path, topic + ".txt")
    topic_f = codecs.open(topic_file, "r", "utf-8")
    lines = topic_f.readlines()
    sample_ids = []
    samples = []
    while len(sample_ids) < n:
        sample_id = random.randint(0, len(lines) - 1)
        if sample_id not in sample_ids:
            sample_ids.append(sample_id)
            samples.append(lines[sample_id].strip())
    return samples


def topic_inference(model, sess, topic, hook_num, turn_num):
    samples = select_n_sample(topic, hook_num)
    results = inference.inference_n(model, sess, samples, turn_num)
    return results


if __name__ == "__main__":
    address = ("0.0.0.0", 11111)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(address)
    s.listen(4)

    while True:
        ss, addr = s.accept()
        input_x = ss.recv(512)
        origin_data = input_x.decode("utf-8")
        # origin_data = input()
        _, topic, _, hook_num, _ = origin_data.split("|")
        hook_num = int(hook_num)
        # turn_num = int(turn_num)


        turns = topic_inference(model, sess, topic, hook_num, 4)

        ss.send(bytes("\n\n".join(turns), "utf-8"))
        ss.close()
        print(turns)
    s.close()




