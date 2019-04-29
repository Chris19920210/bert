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
from SpmTextEncoder import BOS_ID, EOS_ID, PAD_ID
import tensorflow as tf
from run_classifier_v2 import FLAGS, model_fn_builder


def serving_input_fn_builder():
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "src_ids": tf.VarLenFeature(tf.int64),
        "tgt_ids": tf.VarLenFeature(tf.int64),
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

        return example["src_ids"].values, example["tgt_ids"].values, tf.constant(0, dtype=tf.int32)

    def input_fn():
        """The actual input function."""

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        # d = tf.data.TFRecordDataset(input_files)

        bos_id = tf.constant(BOS_ID, tf.int32)
        eos_id = tf.constant(EOS_ID, tf.int32)

        serialized_example = tf.placeholder(
            dtype=tf.string, shape=[None], name="serialized_example")
        d = tf.data.Dataset.from_tensor_slices(serialized_example)

        batch_size = tf.shape(serialized_example, out_type=tf.int64)[0]

        d = d.map(lambda record: _decode_record(record, name_to_features))

        d = d.map(lambda src_ids, tgt_ids, label_ids: (
            tf.concat([[bos_id], src_ids, [eos_id]], 0),
            tf.concat([tgt_ids, [eos_id]], 0),
            label_ids
        ))

        d = d.map(lambda src_ids, tgt_ids, label_ids: (
            tf.concat([src_ids, tgt_ids], 0),
            tf.concat([tf.zeros_like(src_ids), tf.ones_like(tgt_ids)], 0),
            label_ids
        ))

        d = d.map(lambda input_ids, segment_ids, label_ids: (
            input_ids,
            segment_ids,
            tf.ones_like(input_ids),
            label_ids
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
                    tf.TensorShape([None]),
                    tf.TensorShape([])
                ),
                # Pad the source sequences with eos tokens.
                # (Though notice we don't generally need to do this since
                # later on we will be masking out calculations past the true sequence.
                padding_values=(
                    PAD_ID,  # src
                    PAD_ID,
                    PAD_ID,
                    0
                ))

        batched_dataset = batching_func(d)
        input_ids, segment_ids, input_mask, label_ids = tf.contrib.data.get_single_element(batched_dataset)

        features = {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "input_mask": input_mask,
            "label_ids": label_ids
        }

        #return features
        return tf.estimator.export.ServingInputReceiver(
            features=features, receiver_tensors=serialized_example)

    return input_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
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

    serving_input_fn = serving_input_fn_builder()

    exporter = tf.estimator.FinalExporter(
        "exporter",
        lambda: serving_input_fn(),
        as_text=True)

    ckpt_dir = os.path.expanduser(FLAGS.output_dir)
    checkpoint_path = tf.train.latest_checkpoint(ckpt_dir)
    export_dir = os.path.join(ckpt_dir, "export")

    exporter.export(
        estimator,
        export_dir,
        checkpoint_path=checkpoint_path,
        eval_result=None,
        is_the_final_export=True)


if __name__ == "__main__":
    tf.app.run()


