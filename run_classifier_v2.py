# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
from SpmTextEncoder import BOS_ID, EOS_ID, PAD_ID
import tensorflow as tf
from tensorflow.python.client import device_lib


flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")


flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")


flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("num_classes", 2, "number of class")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_steps", 100000,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer(
    "num_warmup_steps", 1500,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("max_sequence_length", 128,
                     "maximum sequence length")


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def file_based_input_fn_builder(input_files, is_training, batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "src_ids": tf.VarLenFeature(tf.int64),
        "tgt_ids": tf.VarLenFeature(tf.int64),
        "label": tf.FixedLenFeature([1], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example["src_ids"].values, example["tgt_ids"].values, example["label"][0]

    def input_fn():
        """The actual input function."""
        bos_id = tf.constant(BOS_ID, tf.int32)
        eos_id = tf.constant(EOS_ID, tf.int32)

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_files)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.map(lambda record: _decode_record(record, name_to_features))

        d = d.map(lambda src_ids, tgt_ids, label: (
            tf.concat([[bos_id], src_ids, [eos_id]], 0),
            tf.concat([tgt_ids, [eos_id]], 0),
            label))

        d = d.map(lambda src_ids, tgt_ids, label: (
            src_ids[:FLAGS.max_sequence_length],
            tgt_ids[:FLAGS.max_sequence_length],
            label
        ))


        d = d.map(lambda src_ids, tgt_ids, label: (
            tf.concat([src_ids, tgt_ids], 0),
            tf.concat([tf.zeros_like(src_ids), tf.ones_like(tgt_ids)], 0),
            label
        ))

        def batching_func(x):
            return x.padded_batch(
                batch_size,
                # The entry is the source line rows;
                # this has unknown-length vectors.  The last entry is
                # the source row size; this is a scalar.
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([None]),  # tgt
                    tf.TensorShape([])),  # src_len
                # Pad the source sequences with eos tokens.
                # (Though notice we don't generally need to do this since
                # later on we will be masking out calculations past the true sequence.
                padding_values=(
                    PAD_ID,  # src
                    PAD_ID,
                    0))  # src_len -- unused

        batched_dataset = batching_func(d)
        features = batched_dataset.map(lambda input_ids, segment_ids, label:
        {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "label_ids": label

        })

        return features

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return loss, per_example_loss, logits, probabilities


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, None, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op
            )
        elif mode == tf.estimator.ModeKeys.EVAL:

            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions)
            loss = tf.metrics.mean(values=per_example_loss)
            eval_metrics = {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities}
            )
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    mirrored_strategy = tf.contrib.distribute.MirroredStrategy(
        devices=get_available_gpus(),
    )

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=20,
        train_distribute=mirrored_strategy
    )

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=FLAGS.num_classes,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_one_hot_embeddings=False)

    # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    if FLAGS.do_train:
        train_files = os.listdir(FLAGS.data_dir)
        train_files = [os.path.join(FLAGS.data_dir, path) for path in train_files if "train"in path]
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", FLAGS.num_train_steps)

        train_input_fn = file_based_input_fn_builder(
            input_files=train_files,
            is_training=True,
            batch_size=FLAGS.train_batch_size)

        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            max_steps=FLAGS.num_train_steps
        )

        eval_files = os.listdir(FLAGS.data_dir)
        eval_files = [os.path.join(FLAGS.data_dir, path) for path in eval_files if "dev" in path]
        # This tells the estimator to run through the entire set.

        # However, if running eval on the TPU, you will need to specify the
        # number of steps.

        eval_input_fn = file_based_input_fn_builder(
            input_files=eval_files,
            is_training=False,
            batch_size=FLAGS.eval_batch_size)

        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            steps=None
        )

        tf.estimator.train_and_evaluate(
            estimator,
            train_spec,
            eval_spec
        )

    if FLAGS.do_predict:

        predict_files = os.listdir(FLAGS.data_dir)
        predict_files = [os.path.join(FLAGS.data_dir, path) for path in predict_files if "pred" in path]

        tf.logging.info("***** Running prediction*****")

        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_files=predict_files,
            is_training=False,
            batch_size=FLAGS.predict_batch_size)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()