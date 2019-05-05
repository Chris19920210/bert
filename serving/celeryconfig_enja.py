from kombu import Exchange, Queue


exchange = Exchange("tasks_align_bert_enja")
queue = Queue("tasks_align_bert_enja", exchange, routing_key="tasks_align_bert_enja")

CELERY_QUEUES = (
   queue,
)

CELERY_ROUTES = {
    'tasks_align_bert_enja.alignment': {"queue": "tasks_align_bert_enja", "routing_key": "tasks_align_bert_enja"},
}
