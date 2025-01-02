# HAKES Worker

## build

```sh
docker build -t hakes-hakesworker-nosgx:v1 -f docker/hakes-worker/no-sgx/Dockerfile .
```

## prepare the config file

The config file mounted for hakes-worker container controls how it interacts with other components in HAKES to serve user requests. A sample config is show below.

```json
{
  "embed_endpoint_type": "hakes-embed",
  "embed_endpoint_config": "",
  "embed_endpoint_addr": "10.10.10.28:2700",
  "search_worker_addrs": ["10.10.10.57:2800"],
  "preferred_search_worker": 0,
  "store_addr": "10.10.10.26:2600"
}
```

* `embed_endpoint_type`, `embed_endpoint_config` and `embed_endpoint_addr` dictate the embed service that converts raw data in user requests to embedding vectors. When the `embed_endpoint_type` is `hakes-embed`, it uses an embed-worker server or function. We also support external embedding services through the [embed-endpoint](../../../embed-endpoint/) module, such as `openai` and `huggingface` for open-ai embedding APIs and huggingface inrference endpoints.
* `search_worker_addrs` points to the search-workers that maintains the vector index and `preferred_search_worker` indicates the request routing to a specific search-worker that the hakes-worker will contact.
* `store_addr` points to the hakes-store service that persists the data.

## run

```sh
docker run --name hakes-worker-test -p 2053:8080 -v $PWD/data/hakesworker/sample-config.json:/mounted_store/config hakes-hakesworker-nosgx:v1 
```
