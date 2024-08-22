FROM hakes_es_base_sgx:v1 as embed_worker_builder

USER root

COPY --from=hakes_es_base_sgx:v1 /hakes-es/deps /hakes-es/deps
COPY ./embed-worker/ /hakes-es/embed-worker
COPY ./message/ /hakes-es/message
COPY ./ratls-channel/ /hakes-es/ratls-channel
COPY ./secret/ /hakes-es/secret
COPY ./server/ /hakes-es/server
COPY ./store-client/ /hakes-es/store-client
COPY ./utils/ /hakes-es/utils

# ENV SGX_SDK=/opt/intel/sgxsdk

ARG INFERENCERT
ARG MODULE_OBJS_DIR=docker/embed-worker/sgx 
# MODULE_OBJS_DIR is the folder that contains model_c
# give a default exist value to copy for tflm

RUN [ ${INFERENCERT} = "TFLM_DEFAULT" ] && echo "use tflm rt" \
  || (([ ${INFERENCERT} = "TVMCRT_DEFAULT" ] || [ ${INFERENCERT} = "TVMCRT_IMAGEEMBED" ] || [ ${INFERENCERT} = "TVMCRT_BERT" ]) && ([ -n "${MODULE_OBJS_DIR}" ] \
  && echo "use tvm c rt"))


COPY ./${MODULE_OBJS_DIR} /hakes-es/model

ENV INFERENCERT ${INFERENCERT}
ENV MODULE_OBJS_DIR /hakes-es/model

RUN cd /hakes-es/embed-worker \
  && make mrproper && make sgx_u sgx_t && make install && cd ..

# image
FROM sgx_dcap_ssl_2.14_1.11:v1

USER root

# default libuv thread pool size to 1.
ARG UV_THREADPOOL_SIZE=1
RUN echo UV_THREADPOOL_SIZE=${UV_THREADPOOL_SIZE}
ENV UV_THREADPOOL_SIZE ${UV_THREADPOOL_SIZE}
ENV SGX_AESM_ADDR=1

COPY --from=embed_worker_builder /hakes-es/embed-worker/install /install

# default ow action so second last argument 1
CMD ["/install/bin/embed_server", "8080", "/mounted_store", "1", "/install/lib/Worker_Enclave.signed.so"] 
