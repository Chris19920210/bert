export PYTHONPATH=/home/wudong/bert:/home/wudong/bert/data_preprocess:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=6

/home/wudong/t2t_aan/python3.6/bin/python3.6 /home/wudong/bert/run_classifier_v2.py \
       --data_dir=/home/wudong/sp_all_32k \
       --bert_config_file=bert_config.json \
       --output_dir=/home/wudong/bert_outputs/sp_train/sp_train_2 \
       --do_train=False \
       --do_eval=False \
       --do_predict=True
