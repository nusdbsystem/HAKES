FROM hakes_es_base_sgx:v1 as search_worker_builder

USER root

COPY --from=hakes_es_base_sgx:v1 /hakes-es/deps /hakes-es/deps
COPY ./search-worker/ /hakes-es/search-worker
COPY ./message/ /hakes-es/message
COPY ./ratls-channel/ /hakes-es/ratls-channel
COPY ./secret/ /hakes-es/secret
COPY ./server/ /hakes-es/server
COPY ./store-client/ /hakes-es/store-client
COPY ./utils/ /hakes-es/utils

RUN cd /hakes-es/search-worker \
  && make mrproper && make sgx_u sgx_t && make install && cd ..

# image
FROM sgx_dcap_ssl_2.14_1.11:v1

USER root

# default libuv thread pool size to 8.
ARG UV_THREADPOOL_SIZE=8
RUN echo UV_THREADPOOL_SIZE=${UV_THREADPOOL_SIZE}
ENV UV_THREADPOOL_SIZE ${UV_THREADPOOL_SIZE}
ENV SGX_AESM_ADDR=1

COPY --from=search_worker_builder /hakes-es/search-worker/install /install

CMD ["/install/bin/search_server", "8080", "/mounted_store/index", "/install/lib/Enclave.signed.so", "1", "0"] 
