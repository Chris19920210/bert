export PYTHONPATH=/home/wudong/bert:/home/wudong/bert/data_preprocess:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2,3

/home/wudong/t2t_aan/python3.6/bin/python3.6 /home/wudong/bert/run_classifier_v2.py \
       --data_dir=/home/wudong/sp_all_32k \
       --bert_config_file=bert_config.json \
       --output_dir=/home/wudong/bert_outputs/sp_train \
       --do_train=true \
       --do_eval=true \
       --num_train_steps=2000000 \
       --train_batch_size=64 \
       --learning_rate=2e-5 \
       --label_smoothing=0.01
