# Search Worker (SGX)

## build

```sh
docker build -t hakes-searchworker:v1 -f docker/search-worker/Dockerfile .
```

## run

```sh
docker run --name search-worker-test -p 2053:8080 -v $PWD/data/searchworker-sampledata/:/mounted_store/index hakes-searchworker:v1
```
