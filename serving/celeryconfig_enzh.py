from kombu import Exchange, Queue


exchange = Exchange("tasks_align_bert_enzh")
queue = Queue("tasks_align_bert_enzh", exchange, routing_key="tasks_align_bert_enzh")

CELERY_QUEUES = (
   queue,
)

CELERY_ROUTES = {
    'tasks_align_bert_enzh.alignment': {"queue": "tasks_align_bert_enzh", "routing_key": "tasks_align_bert_enzh"},
}
