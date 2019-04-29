# 1. export serving model from host CentOS
```
cd path_to_bert
./export.sh
```
then copy the export file and vocab models of sentencepiece into docker

# 2. enter into docker and setup its env
## install python3.6.7
```
apt-get install libssl-dev openssl
# cd to python source dir
./configure --with-ssl && make && make install
```

## python requirements
```
cd path_to_bert/serving
python3.6 -m pip install -r requirements.txt
```

## rabbitmq
```
apt install rabbitmq-server
rabbitmqctl add_user nmt dip_gpu
rabbitmqctl add_vhost myvhost
rabbitmqctl set_permissions -p myvhost nmt ".*" ".*" ".*"
```

## add ubuntu user: nmt
```
apt install sudo
adduser nmt
usermod -aG sudo nmt
su nmt
```

# 3. start up services
## tf serving
```
tensorflow_model_server \
  --port=9000 \
  --model_name=my_bert \
  --model_base_path=/export
```

## tornado and celery
```
nohup sh start_tornado.sh > tornado.log &
nohup sh sync_celery.sh > celery.log &
```
