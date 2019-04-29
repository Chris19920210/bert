#!/bin/bash
BASE_DIR=/bert
WORK_DIR=$BASE_DIR/serving
VOCAB_DIR=/vocab/sp_32k
TASK=bert_align_tasks
export PYTHONPATH=$BASE_DIR:$BASE_DIR/data_preprocess:$BASE_DIR/serving:$PYTHONPATH
export MQ_USER=nmt
export MQ_PASSWORD=dip_gpu
export MQ_HOST=127.0.0.1
export MQ_PORT=5672
export DATA_DIR=/export
export BERT_CONFIG_FILE=$BASE_DIR/bert_config.json
export USER_DICT=$BASE_DIR/dicts.txt
export SRC_VOCAB_MODEL=$VOCAB_DIR/vocab.sp_enzh_ai32k.32000.subwords.en.model
export TGT_VOCAB_MODEL=$VOCAB_DIR/vocab.sp_enzh_ai32k.32000.subwords.zh.model
export SERVERS="127.0.0.1:9000"
export SERVABLE_NAMES="bert_model"
export TIMEOUT_SECS=1000
export MAX_RETRIES=4
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6379
export NUM_ALIGNER=4

celery -A $TASK worker --loglevel=INFO --concurrency=4
