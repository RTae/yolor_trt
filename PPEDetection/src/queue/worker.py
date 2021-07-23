import os
from celery import Celery

#broker = "amqp://localhost"
#backend = "redis://localhost"

broker=os.environ['BROKER_URI']
backend=os.environ['BACKEND_URI']

app = Celery(
    'celery_app',
    broker=broker,
    backend=backend,
    include=['src.queue.tasks']
)