# -*-coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from . import misc_utils as utils


class Model(object):
    def __init__(self,
                 flags,
                 mode,
                 source_vocab_table,
                 target_vocab_table,
                 reverse_target_vocab_table=None,
                 scope=None):
        self.flags = flags
        self.mode = mode

        self.source_vocab_table = source_vocab_table
        self.target_vocab_table = target_vocab_table
        self.reverse_target_vocab_table = reverse_target_vocab_table

        self.eos = flags.eos
        self.sos = flags.sos

        self.src_vocab_size = flags.src_vocab_size
        self.tgt_vocab_size = flags.tgt_vocab_size
        self.share_vocab = flags.share_vocab
        self.embed_size = flags.embed_size

        self.num_encoder_layers = flags.num_encoder_layers
        self.num_decoder_layers = flags.num_decoder_layers
        self.unit_type = flags.unit_type
        self.forget_bias = flags.forget_bias
        self.num_units = flags.num_units
        self.dropout = flags.dropout
        self.encoder_type = flags.encoder_type
        self.base_gpu = 0
        self.num_gpus = flags.num_gpus

        # Initializer
        self.init_op = flags.init_op
        self.init_weight = flags.init_weight
        self.random_seed = flags.random_seed
        initializer = self.build_initializer()
        tf.get_variable_scope().set_initializer(initializer)

        # Training
        self.optimizer = flags.optimizer
        self.learning_rate = flags.learning_rate
        self.max_gradient_norm = flags.max_gradient_norm

        # Inference
        self.infer_batch_size = flags.infer_batch_size
        self.beam_width = flags.beam_width
        self.tgt_max_len_infer = flags.tgt_max_len_infer
        self.sampling_temperature = flags.sampling_temperature
        self.decoder_rule = flags.decoder_rule
        if self.decoder_rule == "rhyme":
            self.table = np.load(flags.rhyme_table_file)

        self.global_step = tf.Variable(0, trainable=False)

        # Build graph
        self.build_graph(scope)


        # Saver
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=flags.num_keep_ckpts)

        # Print trainable variables
        utils.print_out("# Trainable variables")
        for param in tf.trainable_variables():
            utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                              param.op.device))

    def build_initializer(self):
        """Create an initializer. init_weight is only for uniform."""
        utils.print_out("# build %s initializer ..." % self.init_op)
        if self.init_op == "uniform":
            assert self.init_weight
            return tf.random_uniform_initializer(
                -self.init_weight, self.init_weight, seed=self.random_seed)
        elif self.init_op == "glorot_normal":
            return tf.keras.initializers.glorot_normal(
                seed=self.random_seed)
        elif self.init_op == "glorot_uniform":
            return tf.keras.initializers.glorot_uniform(
                seed=self.random_seed)
        else:
            raise ValueError("Unknown init_op %s" % self.init_op)

    def init_embeddings(self):
        """Init embeddings."""
        with tf.variable_scope("embeddings", dtype=tf.float32):
            if self.share_vocab:
                utils.print_out("# Use the same embedding for source and target.")
                self.embedding_encoder = tf.get_variable(
                    name="embedding_share", shape=[self.src_vocab_size, self.embed_size], dtype=tf.float32)
                self.embedding_decoder = self.embedding_encoder
            else:
                self.embedding_encoder = tf.get_variable(
                    name="embedding_encoder", shape=[self.src_vocab_size, self.embed_size], dtype=tf.float32)
                self.embedding_decoder = tf.get_variable(
                    name="embedding_decoder", shape=[self.tgt_vocab_size, self.embed_size], dtype=tf.float32)

    def init_placeholders(self):
        # encoder_inputs: [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32, shape=(None, None), name="encoder_inputs")
        # encoder_inputs_length: [batch_size]
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name="encoder_inputs_length")

        # get dynamic batch_size
        self.batch_size = tf.size(self.encoder_inputs_length)

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            # decoder_inputs: [batch_size, max_time_steps]
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32, shape=(None, None), name="decoder_inputs")
            # decoder_inputs_length: [batch_size]
            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32, shape=(None,), name="decoder_inputs_length")
            # decoder_outputs: [batch_size, max_time_steps]
            self.decoder_outputs = tf.placeholder(
                dtype=tf.int32, shape=(None, None), name="decoder_outputs")

    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, decoder_outputs, mode):

        input_feed = dict()

        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if mode != tf.contrib.learn.ModeKeys.INFER:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length
            input_feed[self.decoder_outputs.name] = decoder_outputs

        return input_feed

    def build_optimizer(self):
        utils.print_out("# setting optimizer ...")
        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()

        # Optimizer
        if self.optimizer == "adam":
            opt = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optimizer == "adadelta":
            opt = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif self.optimizer == "rmsprop":
            opt = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)

        gradients = tf.gradients(self.loss, params)

        clipped_grads, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        # Update the model
        self.update = opt.apply_gradients(
            zip(clipped_grads, params), global_step=self.global_step)

    def build_graph(self, scope=None):
        utils.print_out("# creating %s graph ..." % self.mode)
        dtype = tf.float32

        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
            # Embeddings
            self.init_embeddings()

            # Placeholders
            self.init_placeholders()

            # Encoder
            self.build_encoder()

            # Decoder
            self._build_decoder()

            # Merge all the training summaries
            self.summary_op = tf.summary.merge_all()

    def _build_single_cell(self, device_str=None):
        """Create an instance of a single RNN cell."""
        # dropout (= 1 - keep_prob) is set to 0 during eval and infer
        dropout = self.dropout if self.mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

        # Cell Type
        if self.unit_type == "lstm":
            utils.print_out("  LSTM, forget_bias=%g" % self.forget_bias, new_line=False)
            single_cell = tf.contrib.rnn.BasicLSTMCell(
                self.num_units,
                forget_bias=self.forget_bias)
        elif self.unit_type == "gru":
            utils.print_out("  GRU", new_line=False)
            single_cell = tf.contrib.rnn.GRUCell(self.num_units)
        else:
            raise ValueError("Unknown unit type %s!" % self.unit_type)

        # Dropout (= 1 - keep_prob)
        if dropout > 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - dropout))
            utils.print_out("  %s, dropout=%g " % (type(single_cell).__name__, dropout),
                            new_line=False)

        # Device Wrapper
        if device_str:
            single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
            utils.print_out("  %s, device=%s" %
                            (type(single_cell).__name__, device_str), new_line=False)

        return single_cell

    @staticmethod
    def get_device_str(device_id, num_gpus):
        """Return a device string for multi-GPU setup."""
        if num_gpus == 0:
            return "/cpu:0"
        device_str_output = "/gpu:%d" % (device_id % num_gpus)
        return device_str_output

    def _build_rnn_cell(self, num_layers):
        """Create multi-layer RNN cell."""
        cell_list = []
        for i in range(num_layers):
            utils.print_out("  cell %d" % i, new_line=False)
            single_cell = self._build_single_cell(self.get_device_str(i + self.base_gpu, self.num_gpus))
            utils.print_out("")
            cell_list.append(single_cell)

        if len(cell_list) == 1:  # Single layer.
            return cell_list[0]
        else:  # Multi layers
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def _build_encoder_cell(self, num_layers):
        return self._build_rnn_cell(num_layers)

    def build_encoder(self):
        """Build an encoder."""
        num_layers = self.num_encoder_layers

        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype
            # Embeded_inputs: [batch_size, time_step, embed_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_encoder, self.encoder_inputs)

            # Encoder_outputs: [max_time, batch_size, num_units]
            if self.encoder_type == "uni":
                utils.print_out("  num_layers = %d" % num_layers)
                self.encoder_cell = self._build_encoder_cell(num_layers)

                self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                    self.encoder_cell,
                    self.encoder_inputs_embedded,
                    dtype=dtype,
                    sequence_length=self.encoder_inputs_length,
                    time_major=False,
                    swap_memory=True)
            elif self.encoder_type == "bi":
                num_bi_layers = int(num_layers / 2)
                utils.print_out("  num_bi_layers = %d" % num_bi_layers)

                fw_cell = self._build_encoder_cell(num_bi_layers)
                bw_cell = self._build_encoder_cell(num_bi_layers)
                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    self.encoder_inputs_embedded,
                    dtype=dtype,
                    sequence_length=self.encoder_inputs_length,
                    time_major=False,
                    swap_memory=True)

                self.encoder_outputs, bi_encoder_state = tf.concat(bi_outputs, -1), bi_state

                if num_bi_layers == 1:
                    self.encoder_state = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    self.encoder_state = []
                    for layer_id in range(num_bi_layers):
                        self.encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                        self.encoder_state.append(bi_encoder_state[1][layer_id])  # backward
                        self.encoder_state = tuple(self.encoder_state)
            else:
                raise ValueError("Unknown encoder_type %s" % self.encoder_type)

    def _build_decoder_cell(self, encoder_state):
        """Build an RNN cell that can be used by decoder."""
        cell = self._build_rnn_cell(self.num_decoder_layers)

        # For beam search, we need to replicate encoder infos beam_width times
        if self.mode == tf.contrib.learn.ModeKeys.INFER and self.beam_width:
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                encoder_state, multiplier=self.beam_width)
        else:
            decoder_initial_state = encoder_state

        return cell, decoder_initial_state

    def _get_infer_maximum_iterations(self, source_sequence_length):
        """Maximum decoding steps at inference time."""
        if self.tgt_max_len_infer:
            maximum_iterations = self.tgt_max_len_infer
            utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
        else:
            # TODO(thangluong): add decoding_length_factor flag
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))
        return maximum_iterations

    def _compute_loss(self, logits):
        """Compute optimization loss."""
        decoder_outputs = self.decoder_outputs
        max_time = tf.shape(decoder_outputs)[1]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=decoder_outputs, logits=logits)
        target_weights = tf.sequence_mask(
            self.decoder_inputs_length, max_time, dtype=logits.dtype)

        loss = tf.reduce_sum(
            crossent * target_weights) / tf.to_float(self.batch_size)
        return loss


    def training_decoder(self, scope):
        # decoder_inputs_embeded: [batch_size, max_time, num_units]
        self.decoder_inputs_embeded = tf.nn.embedding_lookup(
            self.embedding_decoder, self.decoder_inputs)

        # Helper
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            self.decoder_inputs_embeded, self.decoder_inputs_length,
            time_major=False)
        # Decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_cell,
            training_helper,
            self.decoder_initial_state, )

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            output_time_major=False,
            swap_memory=True,
            scope=scope)

        logits = self.output_layer(outputs.rnn_output)
        self.loss = self._compute_loss(logits)

        self.word_count = tf.reduce_sum(
            self.encoder_inputs_length) + tf.reduce_sum(self.decoder_inputs_length)
        self.predict_count = tf.reduce_sum(
            self.decoder_inputs_length)

        # Training summary
        tf.summary.scalar("loss", self.loss)

        # Construct graphs for minimizing loss
        self.build_optimizer()

    def first_token_issue(self):
        if self.decoder_rule == "rhyme":
            tgt_sos_id = tf.cast(self.target_vocab_table.lookup(tf.constant(self.sos)), tf.int32)
            tgt_eos_id = tf.cast(self.target_vocab_table.lookup(tf.constant(self.eos)), tf.int32)

            # maximum_iterations: The maximum decoding steps.
            maximum_iterations = self._get_infer_maximum_iterations(self.encoder_inputs_length)

            # Control the inference sentence and the source sentence rhyme.
            to_be_rhymed = self.encoder_inputs[:, 0]
            to_be_rhymed = tf.expand_dims(to_be_rhymed, axis=1)
            rhyme_range = tf.gather_nd(self.table, to_be_rhymed)
            rhyme_range = tf.cast(rhyme_range, dtype=tf.int32)
            left = tf.slice(rhyme_range, [0, 0], [1, 1])
            left = tf.reshape(left, shape=[self.infer_batch_size])
            right = tf.slice(rhyme_range, [0, 1], [1, 1])
            right = tf.reshape(right, shape=[self.infer_batch_size])
            mask = magic_slice(left, right, self.tgt_vocab_size)
            mask = tf.cast(mask, dtype=tf.float32)

            first_inputs = tf.nn.embedding_lookup(
                self.embedding_decoder,
                tf.fill([self.batch_size], tgt_sos_id))
            # run one step first
            first_outputs, first_states = self.decoder_cell(
                first_inputs, self.decoder_initial_state)
            # get predictions logits of first token [batch_size, vocab_size]
            first_predictions = self.output_layer(first_outputs)
            first_logits = tf.multiply(first_predictions, mask)
            start_tokens = tf.argmax(first_logits, axis=1)
            start_tokens = tf.cast(start_tokens, dtype=tf.int32)

            end_token = tgt_eos_id
        elif self.decoder_rule == "samefirst":
            tgt_eos_id = tf.cast(self.target_vocab_table.lookup(tf.constant(self.eos)), tf.int32)

            # maximum_iterations: The maximum decoding steps.
            maximum_iterations = self._get_infer_maximum_iterations(self.encoder_inputs_length)

            # Control the inference first token is the same as source sequence.
            start_tokens = self.encoder_inputs[:, 0]
            end_token = tgt_eos_id
        else:
            tgt_sos_id = tf.cast(self.target_vocab_table.lookup(tf.constant(self.sos)), tf.int32)
            tgt_eos_id = tf.cast(self.target_vocab_table.lookup(tf.constant(self.eos)), tf.int32)

            # maximum_iterations: The maximum decoding steps.
            maximum_iterations = self._get_infer_maximum_iterations(self.encoder_inputs_length)

            start_tokens = tf.fill([self.batch_size], tgt_sos_id)
            end_token = tgt_eos_id

        return maximum_iterations, start_tokens, end_token


    def helper_and_dynamic_decoding(self, maximum_iterations, start_tokens, end_token, decoder_scope):
        if self.beam_width > 0:
            inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=self.decoder_cell,
                embedding=self.embedding_decoder,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=self.decoder_initial_state,
                beam_width=self.beam_width,
                output_layer=self.output_layer)
        else:
            # Helper
            # sampling_temperature = self.sampling_temperature
            if self.sampling_temperature > 0.0:
                helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                    embedding=self.embedding_decoder,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    softmax_temperature=self.sampling_temperature,
                    seed=self.random_seed)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self.embedding_decoder,
                    start_tokens=start_tokens,
                    end_token=end_token)

            # Decoder
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=helper,
                initial_state=self.decoder_initial_state,
                output_layer=self.output_layer)  # applied per timestep

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=inference_decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=False,
            swap_memory=True,
            scope=decoder_scope)

        return outputs

    def process_decoder_outputs(self, start_tokens, outputs):
        if self.decoder_rule == "rhyme":
            if self.beam_width > 0:
                # sample_id: shape = [batch_size, time, beam_width]
                sample_id = outputs.predicted_ids

                # Concat start_token with the inference sequence.
                start_tokens = tf.contrib.seq2seq.tile_batch(start_tokens, multiplier=self.beam_width)
                start_tokens = tf.expand_dims(start_tokens, 0)
                start_tokens = tf.expand_dims(start_tokens, 0)
                sample_id = tf.concat([start_tokens, sample_id], axis=1)
                self.sample_id = sample_id

                self.sample_id = tf.transpose(self.sample_id, [2, 0, 1])

            else:
                # sample_id: shape = [batch_size, time, 1]
                sample_id = outputs.sample_id

                # Concat start_token with the inference sequence.
                start_tokens = tf.expand_dims(start_tokens, 1)
                sample_id = tf.concat([start_tokens, sample_id], axis=1)

                self.sample_id = tf.expand_dims(sample_id, 0)

            # sample_words: shape = [beam_width, batch_size, time]
            self.sample_words = self.reverse_target_vocab_table.lookup(
                tf.to_int64(self.sample_id))
        else:
            if self.beam_width > 0:
                # sample_id: shape = [batch_size, time, beam_width]
                sample_id = outputs.predicted_ids
                # Don't concat start_token with the inference sequence
                self.sample_id = sample_id
                self.sample_id = tf.transpose(self.sample_id, [2, 0, 1])

            else:
                # sample_id: shape = [batch_size, time, 1]
                sample_id = outputs.sample_id
                # Don't concat start_token with the inference sequence
                self.sample_id = tf.expand_dims(sample_id, 0)

            # sample_words: shape = [beam_width, batch_size, time]
            self.sample_words = self.reverse_target_vocab_table.lookup(
                tf.to_int64(self.sample_id))



    def _build_decoder(self):
        """Build and run a RNN decoder with a final projection layer."""
        # Decoder.
        with tf.variable_scope("decoder") as decoder_scope:
            self.output_layer = tf.layers.Dense(
                self.tgt_vocab_size, use_bias=False, name="output_projection")

            self.decoder_cell, self.decoder_initial_state = self._build_decoder_cell(self.encoder_state)

            # Train or eval
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                self.training_decoder(decoder_scope)

            # Inference
            else:
                maximum_iterations, start_tokens, end_token = self.first_token_issue()
                outputs = self.helper_and_dynamic_decoding(maximum_iterations, start_tokens, end_token, decoder_scope)
                self.process_decoder_outputs(start_tokens, outputs)



    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length, decoder_outputs):
        """Run a train step of the model feeding the given inputs."""
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length,
                                      decoder_outputs, tf.contrib.learn.ModeKeys.TRAIN)
        output_feed = [self.update,
                       self.loss,
                       self.summary_op,
                       self.word_count,
                       self.predict_count,
                       self.batch_size,
                       self.global_step]
        outputs = sess.run(output_feed, input_feed)
        return outputs

    def eval(self, sess, encoder_inputs, encoder_inputs_length,
             decoder_inputs, decoder_inputs_length, decoder_outputs):
        """Run a evaluation step of the model feeding the given inputs."""
        # assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length,
                                      decoder_outputs, tf.contrib.learn.ModeKeys.EVAL)

        output_feed = [self.loss,
                       self.predict_count,
                       self.batch_size]
        outputs = sess.run(output_feed, input_feed)
        return outputs

    def infer(self, sess, encoder_inputs, encoder_inputs_length):
        """Decode a batch."""
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs=None, decoder_inputs_length=None,
                                      decoder_outputs=None, mode=tf.contrib.learn.ModeKeys.INFER)

        output_feed = [self.sample_words]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0]

    # def decode(self, sess, encoder_inputs, encoder_inputs_length):
    #     """Decode a batch."""
    #     _, infer_summary, _, sample_words = self.infer(sess, encoder_inputs, encoder_inputs_length)
    #
    #     # make sure outputs is of shape [batch_size, time] or [beam_width,
    #     # batch_size, time] when using beam search.
    #     if sample_words.ndim == 3:
    #         # beam search output in [batch_size, time, beam_width] shape.
    #         sample_words = sample_words.transpose([2, 0, 1])
    #     return sample_words, infer_summary


# (lilei) sequence rhyme mask tensor
def magic_slice(left, right, vocab_size):
    # assert left.shape[0] == right.shape[0]
    ret = []
    # print(left.shape[0])
    for i in range(left.shape[0]):
        # tmp = tf.zeros(vocab_size, dtype=tf.int32)
        # ttt = tf.assign(tmp[left[i]:right[i] + 1], tf.ones([right[i] + 1 - left[i]], dtype=tf.int32))

        ttl = tf.zeros([left[i]], dtype=tf.int32)
        ttm = tf.ones([right[i] + 1 - left[i]], dtype=tf.int32)
        ttr = tf.zeros([vocab_size - right[i] - 1], dtype=tf.int32)
        ttt = tf.concat([ttl, ttm, ttr], axis=0)

        ret.append(ttt)

    return tf.stack(ret)


# a = np.array([[1, 2, 3]])
# table = np.array([[0, 2], [0, 2], [0, 2], [3, 4], [3, 4]])
# output = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
# output = tf.convert_to_tensor(output)
# b = tf.expand_dims(a[:, 0], 1)
# # d = np.take(table, b, axis=0)
# # left = np.take(d, 0, axis=1)
# # right = np.take(d, 1, axis=1)
# d = tf.gather_nd(table, b)
# d = tf.cast(d, dtype=tf.int32)
# left = tf.slice(d, [0, 0], [1, 1])
# left = tf.reshape(left, shape=[1])
# right = tf.slice(d, [0, 1], [1, 1])
# right = tf.reshape(right, shape=[1])
# left = tf.cast(left, dtype=tf.int32)
# right = tf.cast(right, dtype=tf.int32)
# mask = magic_slice(left, right, 5)
# mask = tf.cast(mask, dtype=tf.float64)
# first_logits = tf.multiply(output, mask)
# start_tokens = tf.argmax(first_logits, axis=1)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run([d, left, right, start_tokens]))
