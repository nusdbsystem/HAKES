# README

## Control plane

Zookeeper use the official image. In a kubernetes managed deployment, we only need one instance of the zookeeper to be running and let kubernetes to relaunch when crashed. The other components in HAKES-Store shall periodically send heatbeat to the zookeeper such that the directory structure is always recreated. Sharding config map on the other hand shall live as a config file in Kubernetes.

```sh
docker run --name some-zookeeper --network host zookeeper
```

Use a client to create the necessary data structure. (Alternatively, it can also be created by the first node, if the HAKES-Store service is managed by kubernetes.)

```sh
docker run -it --rm --network host zookeeper zkCli.sh
create /rgs
create /rgs/regionA
```

## build the hakes-store server

```sh
docker build -t hakes-store-server -f docker/hakes-store/hakes-store/Dockerfile .
```

```sh
docker run -v $PWD/conf/hakes-store/sample_rs0.yaml:/app/conf/server_config.yaml -v $PWD/conf/hakes-store/sample_kv.yaml:/app/conf/kv_config.yaml -p 2500:2500 -p 2600:2600 hakes-store-server
docker run -v $PWD/conf/hakes-store/sample_rs1.yaml:/app/conf/server_config.yaml -v $PWD/conf/hakes-store/sample_kv.yaml:/app/conf/kv_config.yaml -p 2501:2501 -p 2601:2601 hakes-store-server
docker run -v $PWD/conf/hakes-store/sample_rs2.yaml:/app/conf/server_config.yaml -v $PWD/conf/hakes-store/sample_kv.yaml:/app/conf/kv_config.yaml -p 2502:2502 -p 2602:2602 hakes-store-server
```

Note that for using local storage, additional volume mount shall be provided.

## build the store-daemon server

```sh
docker build -t hakes-store-daemon -f docker/hakes-store/store-daemon/Dockerfile .
```

```sh
docker run -p 2220:2220 -v $PWD/conf/docker/store_daemon.yaml:/app/conf/store_daemon_config.yaml hakes-store-daemon
docker run -p 2221:2220 -v $PWD/conf/docker/store_daemon1.yaml:/app/conf/store_daemon_config.yaml hakes-store-daemon
```

Note that for using local storage, additional volume mount shall be provided.
