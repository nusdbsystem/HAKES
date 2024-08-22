FROM sgx_dcap_ssl_2.14_1.11:v1 as hakes_es_base_sgx

USER root

ENV SGX_SDK=/opt/intel/sgxsdk

# avoid tzdata interactive config during apt install.
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG SPID
ARG EPID_SUBSCRIPTION_KEY
ARG QUOTE_TYPE
ARG USE_ECDSA

RUN ([ ${USE_ECDSA} = 1 ] && echo "use ecdsa") || (test -n "$SPID" \
  && test -n "$EPID_SUBSCRIPTION_KEY" \
  && test -n "$QUOTE_TYPE" \
  && echo ${SPID} ${EPID_SUBSCRIPTION_KEY} ${QUOTE_TYPE})

# RUN test -n "$SPID" \
#   && test -n "$EPID_SUBSCRIPTION_KEY" \
#   && test -n "$QUOTE_TYPE" \
#   && echo ${SPID} ${EPID_SUBSCRIPTION_KEY} ${QUOTE_TYPE}
ENV USE_ECDSA ${USE_ECDSA}
ENV SPID ${SPID}
ENV EPID_SUBSCRIPTION_KEY ${EPID_SUBSCRIPTION_KEY}
ENV QUOTE_TYPE ${QUOTE_TYPE}

# to install gcc-5 required by wolfssl in ratls.
RUN printf "deb http://dk.archive.ubuntu.com/ubuntu/ xenial main \n\
  deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe" > /etc/apt/sources.list.d/xenial.list

RUN apt-get update && apt-get install -y \
  # install deps (libuv/http-parser)
  cmake \
  clang \
  # install deps (ratls)
  autoconf \
  libtool \
  gcc-5 \
  git \
  # install deps (tflm)
  curl \
  python3 \
  python3-pip \
  unzip \
  wget \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install --upgrade Pillow

ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION v16.15.0
RUN mkdir -p /usr/local/nvm
RUN wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
RUN /bin/bash -c "source $NVM_DIR/nvm.sh && nvm install $NODE_VERSION && nvm use --delete-prefix $NODE_VERSION"
ENV NODE_PATH $NVM_DIR/versions/node/$NODE_VERSION/bin
ENV PATH $NODE_PATH:$PATH

RUN node --version && npm --version

COPY ./deps /hakes-es/deps
COPY ./Makefile /hakes-es/
RUN cd /hakes-es/ && make mrproper && make deps

