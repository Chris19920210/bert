# 1. setup env
## install python3.6.7 (

```
apt-get install libssl-dev openssl
# cd to python source dir
./configure --with-ssl && make && make install
```

##
```
cd path_to_bert/serving
python3.6 -m pip install -r requirements.txt
```
