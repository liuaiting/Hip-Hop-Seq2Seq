import tensorflow as tf

# data
tf.app.flags.DEFINE_string("source_train_data", "seq2seq/data/v2/train.src", "Path to source training data.")
tf.app.flags.DEFINE_string("target_train_data", "seq2seq/data/v2/train.tgt", "Path to target training data.")
tf.app.flags.DEFINE_string("source_dev_data", "seq2seq/data/v2/dev.src", "Path to source validation data.")
tf.app.flags.DEFINE_string("target_dev_data", "seq2seq/data/v2/dev.tgt", "Path to target validation data.")
tf.app.flags.DEFINE_string("source_test_data", "seq2seq/data/v2/test.src", "Path to source test data.")
tf.app.flags.DEFINE_string("target_test_data", "seq2seq/data/v2/test.tgt", "Path to target test data.")
# rhyme
# tf.app.flags.DEFINE_string("src_vocab_file", "seq2seq/data/v2/vocab.tgt", "Source vocab file.")
# tf.app.flags.DEFINE_string("tgt_vocab_file", "seq2seq/data/v2/vocab.tgt", "Target vocab file.")
# tf.app.flags.DEFINE_integer("src_vocab_size", 23442, "Source vocab size.")
# tf.app.flags.DEFINE_integer("tgt_vocab_size", 23442, "Target vocab size.")
# samefirst
tf.app.flags.DEFINE_string("src_vocab_file", "seq2seq/data/v2/vocab.src", "Source vocab file.")
tf.app.flags.DEFINE_string("tgt_vocab_file", "seq2seq/data/v2/vocab.tgt", "Target vocab file.")
tf.app.flags.DEFINE_integer("src_vocab_size", 43836, "Source vocab size.")
tf.app.flags.DEFINE_integer("tgt_vocab_size", 43836, "Target vocab size.")
tf.app.flags.DEFINE_bool("share_vocab", True, "Whether to use the source vocab and embeddings "
                                              "for both source and target.")
tf.app.flags.DEFINE_string("sos", "<s>", "Start-of-sentence symbol.")
tf.app.flags.DEFINE_string("eos", "</s>", "End-of-sentence symbol.")
tf.app.flags.DEFINE_integer("num_buckets", 3, "Put data into similar-length buckets.")
tf.app.flags.DEFINE_integer("src_max_len", 30, "Max length of src sequences during training.")
tf.app.flags.DEFINE_integer("tgt_max_len", 30, "Max length of tgt sequences during training.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size.")

# Session config
tf.app.flags.DEFINE_bool("log_device_placement", False, "Debug GPU allocation.")
tf.app.flags.DEFINE_integer("num_inter_threads", 0, "number of inter_op_parallelism_threads")
tf.app.flags.DEFINE_integer("num_intra_threads", 0, "number of intra_op_parallelism_threads")

# Network
tf.app.flags.DEFINE_integer("embed_size", 256, "Embedding size.")
tf.app.flags.DEFINE_string("unit_type", "lstm", "lstm | gru | layer_norm_lstm | nas")
tf.app.flags.DEFINE_float("forget_bias", 1.0, "Forget bias for BasicLSTMCell.")
tf.app.flags.DEFINE_float("dropout", 0.2, "Dropout rate (not keep_prob)")
tf.app.flags.DEFINE_integer("num_units", 256, "Network size.")
# tf.app.flags.DEFINE_integer("num_layers", 2, "Network depth.")
tf.app.flags.DEFINE_integer("num_encoder_layers", 2, "Encoder depth, equal to num_layers if None.")
tf.app.flags.DEFINE_integer("num_decoder_layers", 2, "Decoder depth, equal to num_layers if None.")
tf.app.flags.DEFINE_string("encoder_type", "bi", "uni | bi. For bi, we build"
                                                 " num_encoder_layers/2 bi-directional layers.")
# Training
tf.app.flags.DEFINE_string("init_op", "uniform", "uniform | glorot_normal | glorot_uniform")
tf.app.flags.DEFINE_float("init_weight", 0.1, "for uniform init_op, initializer weights "
                                              "between [-this, this].")
tf.app.flags.DEFINE_integer("random_seed", None, "Random seed (>0, set a specific seed).")

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate. Adam: 0.001 | 0.0001")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam | adadelta | rmsprop")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_string("decoder_rule", "samefirst", "naive | rhyme | samefirst")
tf.app.flags.DEFINE_string("rhyme_table_file", "seq2seq/data/v3/table_23442.npy", "rhyme table")

# Inference
tf.app.flags.DEFINE_integer("infer_batch_size", 1, "Batch size for inference mode.")
tf.app.flags.DEFINE_integer("src_max_len_infer", None, "Max length of src sequences during inference.")
tf.app.flags.DEFINE_integer("tgt_max_len_infer", None, "Max length of tgt sequences during inference."
                                                       "Also use to restrict the maximum decoding length.")
tf.app.flags.DEFINE_integer("beam_width", 0, "Beam width when using beam search decoder. "
                                             "If 0 (default), use standard decoder with greedy helper.")
tf.app.flags.DEFINE_float("length_penalty_weight", 0.0, "Length penalty for beam search.")
tf.app.flags.DEFINE_float("sampling_temperature", 0.0, "Softmax sampling temperature for inference decoding."
                                                       "0.0 means greedy decoding. This option is ignored"
                                                       "when using beam search.")
tf.app.flags.DEFINE_integer("num_translations_per_input", 1, "Number of translations generated for each sentence."
                                                             "This is only used for inference.")
tf.app.flags.DEFINE_string("ckpt", "", "Checkpoint file to load a model for inference.")
tf.app.flags.DEFINE_string("inference_input_file", "seq2seq/data/plana/4.1.txt", "Set to the text to decode.")
tf.app.flags.DEFINE_string("inference_output_file", "seq2seq/model1/output",
                           "Output file to store decoding results.")

# Misc
tf.app.flags.DEFINE_string("out_dir", "seq2seq/model1", "Store log/model files.")
tf.app.flags.DEFINE_integer("num_gpus", 1, "Number of gpus in each worker.")
# tf.app.flags.DEFINE_integer("num_train_steps", 1000000, "Num steps to train.")
tf.app.flags.DEFINE_integer("num_train_epochs", 60, "Num epochs to train.")
tf.app.flags.DEFINE_integer("epoch_step", 0, "record where we were within an epoch.")
tf.app.flags.DEFINE_integer("num_keep_ckpts", 5, "Max number of checkpoints to keep.")
tf.app.flags.DEFINE_integer("steps_per_stats", 100, "How many training steps to do per stats logging."
                                                    "Save checkpoint every 10x steps_per_stats")
tf.app.flags.DEFINE_integer("steps_per_eval", 10000, "How many training steps to do per "
                                                     "internal evaluation.  "
                                                     "Automatically set based on data if None")
tf.app.flags.DEFINE_integer("steps_per_save", 1000, "How many training steps to do per "
                                                    "internal evaluation.  "
                                                    "Automatically set based on data if None")

