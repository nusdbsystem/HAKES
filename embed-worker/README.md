# Worker

Worker creates an enclave to perform model inference securely. It connects to KeyServer to obtain the keys to read the encrypted requests and encrypted model to use.

The application is developed in reference to mtclient in ratls library.

Use sgx_emmt to profile and set the heap to suitable larger values to run other workloads.

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

### configure enclave memory

During inference, models, inputs and intermediate results are all hold inside enclave. Set suitable heap memory size inside `trusted/Worker_Enclave.config.xml` is important to ensure enough space and minimize memory consumption.

Refer to [tflm memory configuration guide](../models/README.md#configure-tflm-interpreter-memory) to set a heap size 10x of that. Then set the expected max concurrency (set the macro in `App.cc`) and build worker.

```cpp
#define INFERENCE_COUNT 8
```

Launch worker in `sgx-gdb` with `sgx_emmt` enabled. Supply a sample request to profile the peak heap usage. Now we can adjust the heap size in enclave config file to just slightly above the measured peak usage.

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
