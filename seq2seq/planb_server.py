# -*- coding: utf-8 -*-
import random
import codecs
import os
import re


import socket


data_path = "seq2seq/data/planb"
topic_list = ["1.0", "2.0", "2.1", "3.0", "4.0", "4.1", "5.1", "12.0"]


def select_n_sample(topic, n):
    topic_file = os.path.join(data_path, topic + ".txt")
    topic_f = codecs.open(topic_file, "r", "utf-8")
    lines = "".join([re.sub("\r", "", l) for l in topic_f.readlines()]).split("\n\n")
    # print(lines)
    sample_ids = []
    samples = []
    while len(sample_ids) < n:
        sample_id = random.randint(0, len(lines) - 1)
        # print(sample_id)
        if sample_id not in sample_ids:
            sample_ids.append(sample_id)
            samples.append(lines[sample_id].strip())
    return samples



if __name__ == "__main__":
    address = ("0.0.0.0", 2222)
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
        print(topic, hook_num)

        turns = select_n_sample(topic, hook_num)

        ss.send(bytes("\n\n".join(turns), "utf-8"))
        ss.close()

    s.close()

    # turns = select_n_sample("12.0", 5)
    # print(turns)



