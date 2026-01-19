FROM hakes_base:v1 as embed_worker_builder

USER root

# need openssl headers
RUN apt-get update && apt-get install -y libssl-dev

COPY --from=hakes_base:v1 /hakes-es/deps /hakes-es/deps
COPY ./embed-worker/ /hakes-es/embed-worker
COPY ./message/ /hakes-es/message
COPY ./server/ /hakes-es/server
COPY ./store-client/ /hakes-es/store-client
COPY ./utils/ /hakes-es/utils

ARG INFERENCERT
ARG MODULE_OBJS_DIR=docker/embed-worker
# MODULE_OBJS_DIR is the folder that contains model_c
# give a default exist value to copy for tflm

RUN [ ${INFERENCERT} = "TFLM_DEFAULT" ] && echo "use tflm rt" \
  || (([ ${INFERENCERT} = "TVMCRT_DEFAULT" ] || [ ${INFERENCERT} = "TVMCRT_IMAGEEMBED" ] || [ ${INFERENCERT} = "TVMCRT_BERT" ]) && ([ -n "${MODULE_OBJS_DIR}" ] \
  && echo "use tvm c rt"))


COPY ./${MODULE_OBJS_DIR} /hakes-es/model

ENV INFERENCERT ${INFERENCERT}
ENV MODULE_OBJS_DIR /hakes-es/model

RUN cd /hakes-es/embed-worker \
  && make mrproper && make all && make install && cd ..

# image
FROM ubuntu:20.04

USER root

# default libuv thread pool size to 1.
ARG UV_THREADPOOL_SIZE=1
RUN echo UV_THREADPOOL_SIZE=${UV_THREADPOOL_SIZE}
ENV UV_THREADPOOL_SIZE ${UV_THREADPOOL_SIZE}

# need libcrypto.so.1.1
RUN apt-get update && apt-get install -y openssl

COPY --from=embed_worker_builder /hakes-es/embed-worker/install /install

# default ow action so last argument 1
CMD ["/install/bin/embed_server", "8080", "/mounted_store", "1"] 
