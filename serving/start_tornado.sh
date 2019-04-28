#!/bin/bash
BASE_DIR=/bert
WORK_DIR=$BASE_DIR/serving
VOCAB_DIR=/vocab/sp_32k
HOST=172.17.0.2
PORT=7002
export PYTHONPATH=$WORK_DIR:$BASE_DIR/data_preprocess:$BASE_DIR/serving:$PYTHONPATH
export CONFIG_PROPERTIES=$WORK_DIR/config.properties
export MQ_USER=nmt
export MQ_PASSWORD=dip_gpu
export MQ_HOST=127.0.0.1
export MQ_PORT=5672
export SERVERS="127.0.0.1:9000"
export SERVABLE_NAMES="bertaligner"
export TIMEOUT_SECS=1000
export MAX_RETRIES=4
export DATA_DIR=/export
export BERT_CONFIG_FILE=$BASE_DIR/bert_config.json
export USER_DICT=$BASE_DIR/dicts.txt
export SRC_VOCAB_MODEL=$VOCAB_DIR/vocab.sp_enzh_ai32k.32000.subwords.en.model
export TGT_VOCAB_MODEL=$VOCAB_DIR/vocab.sp_enzh_ai32k.32000.subwords.zh.model

python3.6 $WORK_DIR/bert_app.py --host $HOST --port $PORT
