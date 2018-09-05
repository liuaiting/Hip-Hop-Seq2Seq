# -*- coding: utf-8 -*-
"""To perform inference on test set given a trained model."""
from __future__ import print_function

import codecs
import time
import collections

import tensorflow as tf
import jieba
import re

from . import config
from . import hip_hop_model
from . import misc_utils as utils
from . import vocab_utils

FLAGS = tf.app.flags.FLAGS

__all__ = ["load_model", "inference_n"]


def del_chs_space(line):
    """delete space between Chinese characters in line."""
    pattern = u"((?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])|^\s+|\s+$)"
    res = re.sub(pattern, '', line)
    return res


def reverse_str(line):
    res = line.split(" ")
    res.reverse()
    return " ".join(res)


def tokenizer(line):
    return " ".join(jieba.cut(line))


def naive_tokenizer(line):
    return line.strip().split()


def convert_to_infer_data(line, src_vocab_table):
    src = [line]
    src = tf.convert_to_tensor(src)
    src = tf.string_split(src).values
    # Convert the word strings to ids
    src = tf.cast(src_vocab_table.lookup(src), tf.int32)
    src = tf.expand_dims(src, 0)
    src_length = tf.size(src)
    src_length = tf.expand_dims(src_length, 0)

    return src, src_length


def load_model(session, name="infer"):
    start_time = time.time()
    ckpt = tf.train.latest_checkpoint(FLAGS.out_dir)

    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        FLAGS.src_vocab_file, FLAGS.tgt_vocab_file, FLAGS.share_vocab)
    reverse_tgt_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(
        FLAGS.tgt_vocab_file, default_value=vocab_utils.UNK)
    model = hip_hop_model.Model(
        FLAGS,
        mode=tf.contrib.learn.ModeKeys.INFER,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table,
        reverse_target_vocab_table=reverse_tgt_vocab_table,
        scope=None)

    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    utils.print_out(
        "  loaded %s model parameters from %s, time %.2fs" %
        (name, ckpt, time.time() - start_time))
    return model


def get_translation(nmt_outputs, tgt_eos):
    """Given batch decoding outputs, select a sentence and turn to text."""
    if tgt_eos:  tgt_eos = tgt_eos.encode("utf-8")
    # Select a sentence
    output = nmt_outputs[0, :].tolist()

    # If there is an eos symbol in outputs, cut them at that point.
    if tgt_eos and tgt_eos in output:
        output = output[:output.index(tgt_eos)]

    translation = utils.format_text(output)
    return translation


def _inference(model, session, line, n):
    num_translations_per_input = max(min(FLAGS.num_translations_per_input, FLAGS.beam_width), 1)

    start_token = line.split()[0]

    if FLAGS.decoder_rule == "rhyme":
        line = reverse_str(line)

    results = []
    # get infer data for inference model
    source, source_sequence_length = convert_to_infer_data(line, model.source_vocab_table)

    for i in range(n - 1):  # inference n-1 turns for a specific source sequence.
        encoder_inputs, encoder_inputs_length = session.run([source, source_sequence_length])
        # infer one turn
        nmt_outputs = model.infer(session, encoder_inputs, encoder_inputs_length)

        for beam_id in range(num_translations_per_input):  # note: only support num=1
            translation = get_translation(nmt_outputs[beam_id], tgt_eos=FLAGS.eos)
            translation = translation.decode("utf-8")
            if FLAGS.decoder_rule == "rhyme":
                new_line = translation
                res = reverse_str(new_line)
                results.append(res)
            elif FLAGS.decoder_rule == "samefirst":
                new_line = start_token + " " + translation
                results.append(new_line)
            else:
                new_line = translation
                results.append(new_line)

            source, source_sequence_length = convert_to_infer_data(new_line, model.source_vocab_table)

    return results


def inference_n(loaded_infer_model, session, lines, n):
    """Perform translation."""
    assert type(loaded_infer_model) == hip_hop_model.Model
    start_time = time.time()
    results = []

    utils.print_out("# Start decoding")
    utils.print_out("  decoding to output %s." % FLAGS.inference_output_file)
    with codecs.open(FLAGS.inference_output_file, "w", "utf-8") as trans_f:
        trans_f.write("")  # Write empty string to ensure file is created.
        for line in lines:
            line = line.strip()

            hook = [line]
            inferences = _inference(loaded_infer_model, session, line, n)

            hook.extend(inferences)
            hook = list(map(del_chs_space, hook))

            print("\n".join(hook) + "\n")

            results.append("\n".join(hook))
            trans_f.write("\n".join(hook))
            trans_f.write("\n\n")

    print("done, %f" % (time.time() - start_time))

    return results

# if __name__ == "__main__":
#
#     sess = tf.Session(config=utils.get_config_proto())
#     loaded_model = load_model(sess)
#     inference_n(loaded_model, sess, ["坦然 你 的 内心"], 3)
