# Worker

Worker creates an enclave to perform embedding model inference. It read the requests and model to use.

## Build the worker

### tflm_default

```sh
INFERENCERT=TFLM_DEFAULT make mrproper
INFERENCERT=TFLM_DEFAULT make all
INFERENCERT=TFLM_DEFAULT make install
```

### tvm_default

```sh
INFERENCERT=TVMCRT_DEFAULT MODULE_OBJS_DIR=../data/tvm-mb/mobilenet1.0 make mrproper
INFERENCERT=TVMCRT_DEFAULT MODULE_OBJS_DIR=../data/tvm-mb/mobilenet1.0 make all
INFERENCERT=TVMCRT_DEFAULT MODULE_OBJS_DIR=../data/tvm-mb/mobilenet1.0 make install
```

### tvm_image_embed

```sh
INFERENCERT=TVMCRT_IMAGEEMBED MODULE_OBJS_DIR=../data/tvm-embed-mb/mbenetembed make mrproper
INFERENCERT=TVMCRT_IMAGEEMBED MODULE_OBJS_DIR=../data/tvm-embed-mb/mbenetembed make all
INFERENCERT=TVMCRT_IMAGEEMBED MODULE_OBJS_DIR=../data/tvm-embed-mb/mbenetembed make install
```

### tvm_bert_embed

```sh
INFERENCERT=TVMCRT_BERT MODULE_OBJS_DIR=../data/tvm-embed-bert/bert make mrproper
INFERENCERT=TVMCRT_BERT MODULE_OBJS_DIR=../data/tvm-embed-bert/bert make all
INFERENCERT=TVMCRT_BERT MODULE_OBJS_DIR=../data/tvm-embed-bert/bert make install
```

## Test worker locally

Launch the key service and stores keys and access control info to the key service via client (if the a cached directory with sealed keys already exist, then this step can be skipped). Then launch the worker

```bash
# the cache directory will be set to <file-store-directory>/cache.
./App <file-store-directory> <inference json request>
```

## send a request with curl

```sh
curl -X POST localhost:8080/run -H 'Content-Type: application/json' -d @examples/tvmcrt/mobilenet1.0/ow_run_req.json
```
