"""Utility functions for building models."""
from __future__ import print_function

import collections
import time

import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from . import iterator_utils
from . import misc_utils as utils
from . import vocab_utils

__all__ = [
    "create_train_model", "create_infer_model",
    "create_or_load_model", "load_model"]


class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
    pass


def create_train_model(model_creator, src_file, tgt_file, flags, scope="train"):
    """Create train graph, model, and iterator."""
    src_file = flags.source_train_data
    tgt_file = flags.target_train_data
    src_vocab_file = flags.src_vocab_file
    tgt_vocab_file = flags.tgt_vocab_file

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "train"):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, share_vocab=flags.share_vocab)

        src_dataset = tf.data.TextLineDataset(src_file)
        tgt_dataset = tf.data.TextLineDataset(tgt_file)

        iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=flags.batch_size,
            sos=flags.sos,
            eos=flags.eos,
            random_seed=flags.random_seed,
            num_buckets=flags.num_buckets,
            src_max_len=flags.src_max_len,
            tgt_max_len=flags.tgt_max_len)

        model = model_creator(
            flags,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table,
            scope=scope)

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator)


class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model", "src_placeholder",
                            "batch_size_placeholder", "iterator"))):
    pass


def create_infer_model(model_creator, flags, scope=None):
    """Create inference model."""
    graph = tf.Graph()
    src_vocab_file = flags.src_vocab_file
    tgt_vocab_file = flags.tgt_vocab_file

    with graph.as_default(), tf.container(scope or "infer"):
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, flags.share_vocab)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
            tgt_vocab_file, default_value=vocab_utils.UNK)

        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        src_dataset = tf.data.Dataset.from_tensor_slices(
            src_placeholder)
        iterator = iterator_utils.get_infer_iterator(
            src_dataset,
            src_vocab_table,
            batch_size=batch_size_placeholder,
            eos=flags.eos,
            src_max_len=flags.src_max_len_infer)
        model = model_creator(
            flags,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.INFER,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table,
            reverse_target_vocab_table=reverse_tgt_vocab_table,
            scope=scope)
    return InferModel(
        graph=graph,
        model=model,
        src_placeholder=src_placeholder,
        batch_size_placeholder=batch_size_placeholder,
        iterator=iterator)


def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    utils.print_out(
        "  loaded %s model parameters from %s, time %.2fs" %
        (name, ckpt, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, name):
    """Create translation model and initialize or load parameters in session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        utils.print_out("  created %s model with fresh parameters, time %.2fs" %
                        (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return model, global_step



