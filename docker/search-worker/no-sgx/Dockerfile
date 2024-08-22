FROM hakes_es_base_nosgx:v1 as search_worker_builder

USER root

# need openssl headers
RUN apt-get update && apt-get install -y libssl-dev

COPY --from=hakes_es_base_nosgx:v1 /hakes-es/deps /hakes-es/deps
COPY ./search-worker/ /hakes-es/search-worker
COPY ./message/ /hakes-es/message
COPY ./ratls-channel/ /hakes-es/ratls-channel
COPY ./secret/ /hakes-es/secret
COPY ./server/ /hakes-es/server
COPY ./store-client/ /hakes-es/store-client
COPY ./utils/ /hakes-es/utils

RUN cd /hakes-es/search-worker \
  && make mrproper && make no_sgx && make install && cd ..

# image
FROM ubuntu:20.04

USER root

# default libuv thread pool size to 8.
ARG UV_THREADPOOL_SIZE=8
RUN echo UV_THREADPOOL_SIZE=${UV_THREADPOOL_SIZE}
ENV UV_THREADPOOL_SIZE ${UV_THREADPOOL_SIZE}

RUN apt-get update && apt-get install -y libgomp1

COPY --from=search_worker_builder /hakes-es/search-worker/install /install

CMD ["/install/bin/search_server_no_sgx", "8080", "/mounted_store/index", "1", "0"]
