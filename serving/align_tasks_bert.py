from celery import Celery
import celery
from serving_utils import BertAlignClient
import numpy as np
import json
import logging
import os
import traceback

"""Celery asynchronous task"""


class AlignTask(celery.Task):
    servers = os.environ['SERVERS'].split(" ")
    servable_names = os.environ['SERVABLE_NAMES'].split(" ")
    problem = os.environ["PROBLEM"]
    data_dir = os.environ["DATA_DIR"]
    timeout_secs = os.environ["TIMEOUT_SECS"]
    t2t_usr_dir = os.environ["T2T_USR_DIR"]
    user_dict = os.environ["USER_DICT"]
    index = np.random.randint(len(servable_names))
    server = servers[index]
    servable_name = servable_names[index]

    _align_clients = []
    num_servers = len(servable_names)

    for server, servable_name in zip(servers, servable_names):
        _align_clients.append(BertAlignClient(
            t2t_usr_dir,
            problem,
            data_dir,
            user_dict,
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


@app.task(name="tasks_align_bert.alignment", base=AlignTask, bind=True, max_retries=int(os.environ['MAX_RETRIES']))
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
