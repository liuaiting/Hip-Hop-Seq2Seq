"""Generally useful utility functions."""
from __future__ import print_function

import codecs
import collections
import json
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf


def check_tensorflow_version():
    min_tf_version = "1.4.0-dev20171024"
    if tf.__version__ < min_tf_version:
        raise EnvironmentError("Tensorflow version must >= %s" % min_tf_version)


def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def print_time(s, start_time):
    """Take a start time, print elapsed duration, and return a new time."""
    print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
    sys.stdout.flush()
    return time.time()


def print_out(s, f=None, new_line=True):
    """Similar to print but with support to flush and output to a file."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    # stdout
    out_s = s.encode("utf-8")
    if not isinstance(out_s, str):
        out_s = out_s.decode("utf-8")
    print(out_s, end="", file=sys.stdout)

    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()


def debug_tensor(s, msg=None, summarize=10):
    """Print the shape and value of a tensor at test time. Return a new tensor."""
    if not msg:
        msg = s.name
    return tf.Print(s, [tf.shape(s), s], msg + " ", summarize=summarize)


def add_summary(summary_writer, global_step, tag, value):
    """Add a new summary to the current summary_writer.
  Useful to log things that are not part of the training graph, e.g., tag=BLEU.
  """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def get_config_proto(log_device_placement=False, allow_soft_placement=True,
                     num_intra_threads=0, num_inter_threads=0):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True

    # CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
    if num_inter_threads:
        config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto


def format_text(words):
    """Convert a sequence words into sentence."""
    if (not hasattr(words, "__len__") and  # for numpy array
            not isinstance(words, collections.Iterable)):
        words = [words]
    return b" ".join(words)

