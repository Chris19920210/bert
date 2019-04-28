# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Utilities for serving tensor2tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import grpc

from tensor2tensor.data_generators import text_encoder
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from mosestokenizer import MosesTokenizer, MosesDetokenizer
from SpmTextEncoder import SpmTextEncoder
import jieba
import os


def _make_example(ids,
                  feature_name="sources"):
    """Make a tf.train.Example for the problem.

    features[input_feature_name] = input_ids

    Also fills in any other required features with dummy values.

    Args:
        ids: list<int>.
        feature_name:

    Returns:
      tf.train.Example
    """
    features = {
        feature_name:
            tf.train.Feature(int64_list=tf.train.Int64List(value=ids))
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def _create_stub(server):
    channel = grpc.insecure_channel(server)
    return prediction_service_pb2_grpc.PredictionServiceStub(channel)


def _encode(inputs, encoder, add_eos=True):
    input_ids = encoder.encode(inputs)
    if add_eos:
        input_ids.append(text_encoder.EOS_ID)
    return input_ids


def _decode(output_ids, output_decoder):
    return output_decoder.decode(output_ids, strip_extraneous=True)


def make_request_fn(server, servable_name, timeout_secs):
    request_fn = make_grpc_request_fn(
        servable_name=servable_name,
        server=server,
        timeout_secs=timeout_secs)
    return request_fn


def make_grpc_request_fn(servable_name, server, timeout_secs):
    """Wraps function to make grpc requests with runtime args."""
    stub = _create_stub(server)

    def _make_grpc_request(src_examples, tgt_examples):
        """Builds and sends request to TensorFlow model server."""
        request = predict_pb2.PredictRequest()
        request.model_spec.name = servable_name
        assert len(src_examples) == len(tgt_examples)
        request.inputs["src_ids"].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                [ex.SerializeToString() for ex in src_examples], shape=[len(src_examples)]))

        request.inputs["tgt_ids"].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                [ex.SerializeToString() for ex in tgt_examples], shape=[len(tgt_examples)]))

        response = stub.Predict(request, timeout_secs)
        outputs = tf.make_ndarray(response.outputs["outputs"])
        return outputs

    return _make_grpc_request


def predict(src_ids_list, tgt_ids_list, request_fn):
    """Encodes inputs, makes request to deployed TF model, and decodes outputs."""
    assert isinstance(src_ids_list, list)
    assert isinstance(tgt_ids_list, list)
    src_examples = [_make_example(src_ids, "src_ids")
                    for src_ids in src_ids_list]

    tgt_examples = [_make_example(tgt_ids, "tgt_ids")
                    for tgt_ids in tgt_ids_list]

    predictions = request_fn(src_examples, tgt_examples)

    return predictions


class BertAlignClient(object):
    def __init__(self,
                 data_dir,
                 bert_config_file,
                 output_dir,
                 bert_usr_dir,
                 user_dict,
                 src_vacob_model,
                 tgt_vocab_model,
                 server,
                 servable_name,
                 timeout_secs
                 ):
        tf.logging.set_verbosity(tf.logging.INFO)
        usr_dir.import_usr_dir(bert_usr_dir)
        self.src_encoder = SpmTextEncoder(src_vacob_model)
        self.tgt_encoder = SpmTextEncoder(tgt_vocab_model)

        self.en_tokenizer = MosesTokenizer('en')
        self.zh_detokenizer = MosesDetokenizer("ko")
        jieba.load_userdict(user_dict)
        self.request_fn = make_request_fn(server, servable_name, timeout_secs)

    def src_encode(self, s):
        tokens = self.en_tokenizer(s)
        return _encode(" ".join(tokens), self.src_encoder, add_eos=False)

    def tgt_encode(self, s):
        tokens = jieba.lcut(s)
        return _encode(" ".join(tokens), self.tgt_encoder, add_eos=False)

    def tgt_decode(self, s):
        tokens = _decode(s, self.tgt_encoder)
        return self.zh_detokenizer(tokens.split(" "))

    def query(self, msg):
        """

        :param msg: msg
        :return: msg
        """
        src_ids_list = list(map(lambda x: self.src_encode(x['origin']), msg["data"]))
        tgt_ids_list = list(map(lambda x: self.tgt_encode(x['translate']), msg["data"]))
        results = predict(src_ids_list, tgt_ids_list, self.request_fn)

        msg['data']['results'] = results
        return msg
