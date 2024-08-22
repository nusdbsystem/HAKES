# Embed Worker (SGX)

Remember to change the `common/config.h` (especially output size amd buffer size) the `MODULE_OBJS_DIR` and `Enclave.config.xml`.

## build

```sh
# for tvm-default
docker build -t hakes-embedworker-nosgx:v1 --build-arg INFERENCERT="TVMCRT_DEFAULT" --build-arg MODULE_OBJS_DIR=./data/tvm-mb/mobilenet1.0 -f docker/embed-worker/no-sgx/Dockerfile .
# for tflm-default
docker build -t hakes-embedworker-nosgx-tflm:v1 --build-arg INFERENCERT="TFLM_DEFAULT" -f docker/embed-worker/no-sgx/Dockerfile .
# for tvm-imageembed
docker build -t hakes-embedworker-mbnetembed-nosgx:v1 --build-arg INFERENCERT="TVMCRT_IMAGEEMBED" --build-arg MODULE_OBJS_DIR=./data/tvm-embed-mb/mbnetembed -f docker/embed-worker/no-sgx/Dockerfile .
docker build -t hakes-embedworker-resnetembed-nosgx:v1 --build-arg INFERENCERT="TVMCRT_IMAGEEMBED" --build-arg MODULE_OBJS_DIR=./data/tvm-embed-rs/resnetembed -f docker/embed-worker/no-sgx/Dockerfile .
# for tvm-bert
docker build -t hakes-embedworker-bert-nosgx:v1 --build-arg INFERENCERT="TVMCRT_BERT" --build-arg MODULE_OBJS_DIR=./data/tvm-embed-bert/bert -f docker/embed-worker/no-sgx/Dockerfile .
```

## run

```sh
docker run --name embed-worker-test -p 2053:8080 -v $PWD/embed-worker/tmp/:/mounted_store hakes-embedworker-nosgx:v1 
```

load raw embed worker

```sh
docker run --name embed-worker-test -p 2053:8080 -v $PWD/embed-worker/tmp/:/mounted_store hakes-embedworker-nosgx:v1 /install/bin/embed_server_no_sgx 8080 /mounted_store 0
```
