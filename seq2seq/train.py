# -*-coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time

import tensorflow as tf

from . import config
from . import model_helper
from . import hip_hop_model
from . import misc_utils as utils
from . import vocab_utils
from . import iterator_utils

FLAGS = tf.app.flags.FLAGS


if not tf.gfile.Exists(FLAGS.out_dir):
    utils.print_out("# Creating output directory %s ..." % FLAGS.out_dir)
    tf.gfile.MakeDirs(FLAGS.out_dir)


def run_internal_eval(model, global_step, sess, iterator, summary_writer, name):
    """Computing perplexity."""
    sess.run(iterator.initializer)

    total_loss = 0
    total_predict_count = 0
    start_time = time.time()

    while True:
        try:
            encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_outputs, decoder_inputs_length = \
                sess.run([
                    iterator.source,
                    iterator.source_sequence_length,
                    iterator.target_input,
                    iterator.target_output,
                    iterator.target_sequence_length])
            model.mode = "eval"
            loss, predict_count, batch_size = model.eval(sess, encoder_inputs, encoder_inputs_length,
                                                         decoder_inputs, decoder_inputs_length, decoder_outputs)
            total_loss += loss * batch_size
            total_predict_count += predict_count
        except tf.errors.OutOfRangeError:
            break

    perplexity = utils.safe_exp(total_loss / total_predict_count)
    utils.print_time("  eval %s: perplexity %.2f" % (name, perplexity),
                     start_time)
    utils.add_summary(summary_writer, global_step, "%s_ppl" % name, perplexity)

    result_summary = "%s_ppl %.2f" % (name, perplexity)

    return result_summary, perplexity


def init_stats():
    """Initialize statistics that we want to accumulate."""
    return {"step_time": 0.0, "loss": 0.0,
            "predict_count": 0.0, "total_count": 0.0}


def update_stats(stats, start_time, step_result):
    """Update stats: write summary and accumulate statistics."""
    (_, step_loss, step_summary, step_word_count, step_predict_count, batch_size, global_step) = step_result

    # Update statistics
    stats["step_time"] += (time.time() - start_time)
    stats["loss"] += (step_loss * batch_size)
    stats["predict_count"] += step_predict_count
    stats["total_count"] += float(step_word_count)

    return global_step, step_summary


def print_step_info(prefix, global_step, info, log_f, result_summary=""):
    """Print all info at the current global step."""
    utils.print_out(
        "%sstep %d step-time %.2fs wps %.2fK ppl %.2f %s, %s" %
        (prefix,
         global_step,
         info["avg_step_time"],
         info["speed"],
         info["train_ppl"],
         result_summary,
         time.ctime()),
        log_f)


def process_stats(stats, info, steps_per_stats):
    """Update info and check for overflow."""
    # Update info
    info["avg_step_time"] = stats["step_time"] / steps_per_stats
    info["train_ppl"] = utils.safe_exp(stats["loss"] / stats["predict_count"])
    info["speed"] = stats["total_count"] / (1000 * stats["step_time"])


def load_data(src_file, tgt_file, src_vocab_table, tgt_vocab_table):
    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)
    iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=FLAGS.batch_size,
            sos=FLAGS.sos,
            eos=FLAGS.eos,
            random_seed=FLAGS.random_seed,
            num_buckets=FLAGS.num_buckets,
            src_max_len=FLAGS.src_max_len,
            tgt_max_len=FLAGS.tgt_max_len)
    return iterator


def train():

    # Load train/dev/test data
    train_src_file = FLAGS.source_train_data
    train_tgt_file = FLAGS.target_train_data
    dev_src_file = FLAGS.source_dev_data
    dev_tgt_file = FLAGS.target_dev_data

    src_vocab_file = FLAGS.src_vocab_file
    tgt_vocab_file = FLAGS.tgt_vocab_file

    # Log and output files
    log_file = os.path.join(FLAGS.out_dir, "log_%d" % time.time())
    log_f = tf.gfile.GFile(log_file, mode="a")
    utils.print_out("# log_file=%s" % log_file, log_f)


    config_proto = utils.get_config_proto(
        log_device_placement=FLAGS.log_device_placement,
        num_intra_threads=FLAGS.num_intra_threads,
        num_inter_threads=FLAGS.num_inter_threads)

    with tf.Session(config=config_proto) as train_sess:
        # Vocabulary table
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, share_vocab=FLAGS.share_vocab)

        # Data iterator
        train_iterator = load_data(train_src_file, train_tgt_file, src_vocab_table, tgt_vocab_table)
        dev_iterator = load_data(dev_src_file, dev_tgt_file, src_vocab_table, tgt_vocab_table)

        # Model
        model = hip_hop_model.Model(
            FLAGS,
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table,
            scope=None)

        loaded_train_model, global_step = model_helper.create_or_load_model(
            model, FLAGS.out_dir, train_sess, "train")

        # Summary writer
        summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.out_dir, "train_log"))

        # Training loop
        stats = init_stats()
        info = {"train_ppl": 0.0, "speed": 0.0, "avg_step_time": 0.0}
        start_train_time = time.time()
        utils.print_out("# Start step %d, %s" %
                        (global_step, time.ctime()), log_f)

        # Initialize all of the iterators
        utils.print_out("# Init train iterator.")
        train_sess.run(train_iterator.initializer)

        epoch_idx = 0
        while epoch_idx < FLAGS.num_train_epochs:
            start_time = time.time()
            try:
                encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_outputs, decoder_inputs_length = \
                    train_sess.run([
                        train_iterator.source,
                        train_iterator.source_sequence_length,
                        train_iterator.target_input,
                        train_iterator.target_output,
                        train_iterator.target_sequence_length])
                loaded_train_model.mode = "train"
                step_result = loaded_train_model.train(train_sess, encoder_inputs, encoder_inputs_length,
                                                       decoder_inputs, decoder_inputs_length, decoder_outputs)
                FLAGS.epoch_step += 1
            except tf.errors.OutOfRangeError:
                FLAGS.epoch_step = 0
                epoch_idx += 1
                utils.print_out(
                    "# Finished epoch %d, step %d." % (epoch_idx, global_step))

                train_sess.run(train_iterator.initializer)
                continue

            # Process step_result, accumulate stats, and write summary
            global_step, step_summary = update_stats(
                stats, start_time, step_result)
            summary_writer.add_summary(step_summary, global_step)

            # Once in a while, we print statistics.
            if global_step % FLAGS.steps_per_stats == 0:
                process_stats(stats, info, FLAGS.steps_per_stats)
                print_step_info("  ", global_step, info, log_f)

                # Reset statistics
                stats = init_stats()

            # Evaluate on dev/test
            if global_step % FLAGS.steps_per_eval == 0:
                utils.print_out("# Save eval, global step %d" % global_step)
                utils.add_summary(summary_writer, global_step, "train_ppl",
                                  info["train_ppl"])

                run_internal_eval(
                    loaded_train_model, global_step, train_sess, dev_iterator, summary_writer, "dev")

            # Save model
            if global_step % FLAGS.steps_per_save == 0:
                loaded_train_model.saver.save(
                    train_sess,
                    os.path.join(FLAGS.out_dir, "translate.ckpt"),
                    global_step=global_step)
        # Done training
        loaded_train_model.saver.save(
            train_sess,
            os.path.join(FLAGS.out_dir, "translate.ckpt"),
            global_step=global_step)

        result_summary, ppl = run_internal_eval(
                    loaded_train_model, global_step, train_sess, dev_iterator, summary_writer, "dev")

        print_step_info("# Final, ", global_step, info, log_f, result_summary)
        utils.print_time("# Done training!", start_train_time)

        summary_writer.close()


if __name__ == "__main__":
    train()
