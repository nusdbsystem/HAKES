FROM hakes_es_base_sgx:v1 as hakes_worker_builder

USER root

COPY --from=hakes_es_base_sgx:v1 /hakes-es/deps /hakes-es/deps
COPY ./hakes-worker/ /hakes-es/hakes-worker
COPY ./message/ /hakes-es/message
COPY ./ratls-channel/ /hakes-es/ratls-channel
COPY ./secret/ /hakes-es/secret
COPY ./server/ /hakes-es/server
COPY ./store-client/ /hakes-es/store-client
COPY ./utils/ /hakes-es/utils

RUN cd /hakes-es/hakes-worker \
  && make mrproper && make sgx_u sgx_t && make install && cd ..

# image
FROM sgx_dcap_ssl_2.14_1.11:v1

USER root

# default libuv thread pool size to 8.
ARG UV_THREADPOOL_SIZE=8
RUN echo UV_THREADPOOL_SIZE=${UV_THREADPOOL_SIZE}
ENV UV_THREADPOOL_SIZE ${UV_THREADPOOL_SIZE}
ENV SGX_AESM_ADDR=1

COPY --from=hakes_worker_builder /hakes-es/hakes-worker/install /install

# default ow action so second last argument 1
CMD ["/install/bin/hakes_server", "8080", "/mounted_store/config", "1", "/install/lib/Worker_Enclave.signed.so"] 
