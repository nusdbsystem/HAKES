# Search Worker (SGX)

## build

```sh
docker build -t hakes-searchworker:v1 -f docker/search-worker/sgx/Dockerfile .
```

## run

```sh
docker run --name search-worker-test -p 2053:8080 -v $PWD/data/searchworker/sample-index/index_opq_ivfpq_glove-200_OPQ100_M50_nlist1024_hakesindex_base_index_withdata:/mounted_store/index --device /dev/sgx_enclave:/dev/sgx/enclave -v /var/run/aesmd:/var/run/aesmd hakes-searchworker:v1 
```
