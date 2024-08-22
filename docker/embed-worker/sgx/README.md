# Embed Worker (SGX)

Remember to change the `common/config.h` (especially output size and buffer size) the `MODULE_OBJS_DIR` and `Enclave.config.xml`.

## build

```sh
# for tvm-default
docker build -t hakes-embedworker:v1 --build-arg INFERENCERT="TVMCRT_DEFAULT" --build-arg MODULE_OBJS_DIR=./data/tvm-mb/mobilenet1.0 -f docker/embed-worker/sgx/Dockerfile .
# for tflm-default
docker build -t hakes-embedworker-tflm:v1 --build-arg INFERENCERT="TFLM_DEFAULT" -f docker/embed-worker/sgx/Dockerfile .
# for tvm image embed
docker build -t hakes-embedworker-mbnetembed:v1 --build-arg INFERENCERT="TVMCRT_IMAGEEMBED" --build-arg MODULE_OBJS_DIR=./data/tvm-embed-mb/mbnetembed -f docker/embed-worker/sgx/Dockerfile .
docker build -t hakes-embedworker-resnetembed:v1 --build-arg INFERENCERT="TVMCRT_IMAGEEMBED" --build-arg MODULE_OBJS_DIR=./data/tvm-embed-rs/resnetembed -f docker/embed-worker/sgx/Dockerfile .
# for tvm bert
docker build -t hakes-embedworker-bert:v1 --build-arg INFERENCERT="TVMCRT_BERT" --build-arg MODULE_OBJS_DIR=./data/tvm-embed-bert/bert -f docker/embed-worker/sgx/Dockerfile .
```

## run

```sh
docker run --name embed-worker-test -p 2053:8080 -v $PWD/embed-worker/tmp/:/mounted_store --device /dev/sgx_enclave:/dev/sgx/enclave -v /var/run/aesmd:/var/run/aesmd hakes-embedworker:v1 
```

load raw embed worker

```sh
docker run --name embed-worker-test -p 2053:8080 -v $PWD/embed-worker/tmp/:/mounted_store --device /dev/sgx_enclave:/dev/sgx/enclave -v /var/run/aesmd:/var/run/aesmd hakes-embedworker:v1 /install/bin/embed_server 8080 /mounted_store 0 /install/lib/Worker_Enclave.signed.so
```

send a request

```sh
curl -X POST localhost:2053/run -H 'Content-Type: application/json' -d @data/tvm-mb/ow_run_req.json
```
