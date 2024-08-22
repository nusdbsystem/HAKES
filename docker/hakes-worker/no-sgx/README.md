# HAKES Worker (SGX)

## build

```sh
docker build -t hakes-hakesworker-nosgx:v1 -f docker/hakes-worker/no-sgx/Dockerfile .
```

## run

```sh
docker run --name hakes-worker-test -p 2053:8080 -v $PWD/data/hakesworker/sample-config.json:/mounted_store/config hakes-hakesworker-nosgx:v1 
```
