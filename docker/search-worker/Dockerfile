FROM intel/oneapi:2025.0.0-0-devel-ubuntu24.04 as search_worker_builder

USER root

# need openssl headers
RUN apt-get update && apt-get install -y libssl-dev

COPY --from=hakes_base:v1 /hakes-es/deps /hakes-es/deps
COPY ./search-worker/ /hakes-es/search-worker
COPY ./message/ /hakes-es/message
COPY ./server/ /hakes-es/server
COPY ./utils/ /hakes-es/utils

RUN cd /hakes-es/search-worker \
  && make mrproper && make all && make install && cd ..

# image
FROM intel/oneapi:2025.0.0-0-devel-ubuntu24.04

USER root

# default libuv thread pool size to 8.
ARG UV_THREADPOOL_SIZE=8
RUN echo UV_THREADPOOL_SIZE=${UV_THREADPOOL_SIZE}
ENV UV_THREADPOOL_SIZE ${UV_THREADPOOL_SIZE}

RUN apt-get update && apt-get install -y libgomp1

COPY --from=search_worker_builder /hakes-es/search-worker/install /install

CMD ["/install/bin/search_server", "8080", "/mounted_store/index", "1", "0"]
