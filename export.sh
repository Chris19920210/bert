export PYTHONPATH=/home/wudong/bert:/home/wudong/bert/data_preprocess:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7

/home/wudong/t2t_aan/python3.6/bin/python3.6 /home/wudong/bert/serving/export.py \
       --data_dir=/home/wudong/sp_all_32k \
       --bert_config_file=/home/wudong/bert/bert_config.json \
       --output_dir=/home/wudong/sp_train/sp_train_2 \
       --do_train=true \
       --do_eval=true \
       --num_train_steps=2000000 \
       --train_batch_size=64 \
       --learning_rate=2e-5 \
       --label_smoothing=0.01
