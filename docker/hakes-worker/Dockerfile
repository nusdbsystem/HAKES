FROM hakes_base:v1 as hakes_worker_builder

USER root

# need openssl headers
RUN apt-get update && apt-get install -y libssl-dev libcurl4-openssl-dev

COPY --from=hakes_base:v1 /hakes-es/deps /hakes-es/deps
COPY ./hakes-worker/ /hakes-es/hakes-worker
COPY ./embed-endpoint/ /hakes-es/embed-endpoint
COPY ./message/ /hakes-es/message
COPY ./server/ /hakes-es/server
COPY ./store-client/ /hakes-es/store-client
COPY ./utils/ /hakes-es/utils

RUN cd /hakes-es/hakes-worker \
  && make mrproper && make all && make install && cd ..

# image
FROM ubuntu:20.04

USER root

# default libuv thread pool size to 8.
ARG UV_THREADPOOL_SIZE=8
RUN echo UV_THREADPOOL_SIZE=${UV_THREADPOOL_SIZE}
ENV UV_THREADPOOL_SIZE ${UV_THREADPOOL_SIZE}

RUN apt-get update && apt-get install -y curl

COPY --from=hakes_worker_builder /hakes-es/hakes-worker/install /install

# default ow action so last argument 1
CMD ["/install/bin/hakes_server", "8080", "/mounted_store/config", "1" ]
