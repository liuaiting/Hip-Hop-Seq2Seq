from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import collections
import logging
import re
import codecs

import tensorflow as tf


logger = logging.getLogger(__name__)

Specials = ['<unk>', '<s>', '</s>']


def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(。，！？、—《》：；）（])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


def naive_tokenizer(line):
    return line.strip().lower().split()


def build_vocab(in_path, out_path, max_size=None, min_freq=1, specials=Specials, tokenizer=None):

    if not tf.gfile.Exists(out_path):
        print("Creating vocabulary {} from data {}".format(out_path, in_path))
        vocab = collections.Counter()
        with tf.gfile.GFile(in_path, mode='r') as f:
            for line in f:
                tokens = tokenizer(line) if tokenizer else naive_tokenizer(line)
                vocab.update(tokens)
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[0])
            sorted_vocab.sort(key=lambda x: x[1], reverse=True)
            itos = list(specials)
            for word, freq in sorted_vocab:
                if freq < min_freq or len(itos) == max_size:
                    break
                itos.append(word)
            with codecs.getwriter('utf-8')(tf.gfile.GFile(out_path, mode='wb')) as fw:
                for word in itos:
                    fw.write(str(word) + '\n')
                    # fw.write(str(word) + '\t' + str(freq) + '\n')


if __name__ == '__main__':
    build_vocab('data/v2/all.txt', 'data/v2/vocab.txt')




