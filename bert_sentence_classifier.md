# 基于Bert的词对齐模型 (Training)

### Encoder Preparation (sentencepieces model)

```
#!/bin/bash
USR_DIR=/home/chenrihan/bert/data_preprocess
PROBLEM=sp_enzh_ai32k
DATA_DIR=/home/chenrihan/nmt_datasets_spm/sp_encoder
TMP_DIR=/home/chenrihan/nmt_datasets_final/all
mkdir -p $DATA_DIR $TMP_DIR
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/home/chenrihan/DipML/preprocess:$PYTHONPATH

python $USR_DIR/bpe_vocab_generator.py \
  --method sspm \
  --tmp-dir  $TMP_DIR \
  --data-dir $DATA_DIR \
  --problem $PROBLEM
```
Generate the vocabs for encode/decode sentences.

### Generate Samples

```
#!/bin/bash
USR_DIR=/home/chenrihan/bert/data_preprocess
DATA_DIR=/home/chenrihan/nmt_datasets_spm/sp_all_32k
TMP_DIR=/home/chenrihan/nmt_datasets_final/all
mkdir -p $DATA_DIR $TMP_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTHONPATH=/home/chenrihan/DipML/preprocess:$PYTHONPATH

python $USR_DIR/tfrecord_data_generate.py \
  --data_dir=$DATA_DIR \
  --src en \
  --tgt zh \
  --tmp_dir=$TMP_DIR \
  --vocab=vocab.sp_enzh_ai32k.32000.subwords

```
Generate the tfrecords for data

### Train the model
```
#!/bin/bash

export PYTHONPATH=/home/chenrihan/bert:/home/chenrihan/bert/data_preprocess:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2,3

python /home/chenrihan/bert/run_classifier_v2.py \
       --data_dir=/home/chenrihan/nmt_datasets_spm/sp_all_32k \
       --bert_config_file=bert_config.json \
       --output_dir=/home/chenrihan/nmt/sp_train \
       --do_train=true \
       --do_eval=true \
       --num_train_steps=2000000 \
       --train_batch_size=64 \
       --learning_rate=2e-5 \
       --label_smoothing=0.01

```


### bert_config

```
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 512,
  "max_position_embeddings": 512,
  "num_attention_heads": 16,
  "num_hidden_layers": 8,
  "type_vocab_size": 2,
  "vocab_size": 32000
  "trainable_pos_embedding":false
}
```


### Modification

```
Compared with original Bert, there are totally three modifications.

1. trainable position embedding: we can choose the sinusoidal position embeddings
   or trainable position embedding.
2. label_smoothing: we can choose the extent of label_smoothing.
3. Multi-gpus training: we can choose the number of gpus by setting CUDA_VISIBLE_DEVICES in bash script.
```
