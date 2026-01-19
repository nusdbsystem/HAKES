# FnPacker

## build

```sh
docker build -t hakes-fnpacker:v1 -f docker/fnpacker/Dockerfile .
```

## run

```sh
# these env settings are not used yet. It still take the .wskprops
# docker run -it -e OW_SERVICE_ADDRESS=<ow-address> -e OW_SERVICE_PORT=<ow-port> -e OW_SERVICE_AUTH=<ow-auch> -e OW_FUNCTION_MANAGER_POOL_SIZE=3 --name fnpacker --net host hakes-fnpacker:v1 /fnpacker -port 7322
docker run -it -e OW_FUNCTION_MANAGER_POOL_SIZE=3 -v <.wskprops>:/root/.wskprops --name fnpacker -net host hakes-fnpacker:v1 /fnpacker -port 7310
```

They all have default values but the address and port should always be set manually to connect to service outside the container.
