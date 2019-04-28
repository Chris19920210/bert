#!/usr/bin/python
# -*- coding: utf-8 -*-
from celery import Celery
import celery
from serving_utils import BertAlignClient
import numpy as np
import json
import logging
import os
import traceback

"""Celery asynchronous task"""


class BertAlignTask(celery.Task):
    servers = os.environ['SERVERS'].split(" ")
    servable_names = os.environ['SERVABLE_NAMES'].split(" ")
    data_dir = os.environ["DATA_DIR"]
    bert_config_file = os.environ["BERT_CONFIG_FILE"]
    src_vocab_model = os.environ["SRC_VOCAB_MODEL"]
    tgt_vocab_model = os.environ["TGT_VOCAB_MODEL"]
    timeout_secs = os.environ["TIMEOUT_SECS"]
    user_dict = os.environ["USER_DICT"]
    index = np.random.randint(len(servable_names))
    server = servers[index]
    servable_name = servable_names[index]

    _align_clients = []
    num_servers = len(servable_names)

    for server, servable_name in zip(servers, servable_names):
        _align_clients.append(BertAlignClient(
            data_dir,
            bert_config_file,
            user_dict,
            src_vocab_model,
            tgt_vocab_model,
            server,
            servable_name,
            int(timeout_secs)
        ))

    @property
    def align_clients(self):

        return self._align_clients

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logging.error('{0!r} failed: {1!r}'.format(task_id, exc))


# set up the broker
app = Celery("tasks_align_bert",
             broker="amqp://{user:s}:{password:s}@{host:s}:{port:s}"
             .format(
                 user=os.environ['MQ_USER'],
                 password=os.environ['MQ_PASSWORD'],
                 host=os.environ['MQ_HOST'],
                 port=os.environ['MQ_PORT']),
             backend='amqp',
             task_serializer='json',
             result_serializer='json',
             accept_content=['json'],
             result_persistent=False
             )
app.config_from_object("celeryconfig_align_bert")


@app.task(name="tasks_align_bert.alignment", base=BertAlignTask, bind=True, max_retries=int(os.environ['MAX_RETRIES']))
def alignment(self, msg):
    try:
        source = json.loads(msg, strict=False)
        target = json.dumps(alignment.align_clients[os.getpid() % self.num_servers]
                            .query(source),
                            ensure_ascii=False).replace("</", "<\\/")
        return target

    except Exception:
        return json.dumps({"error": traceback.format_exc()}, ensure_ascii=False)


if __name__ == '__main__':
    app.start()
