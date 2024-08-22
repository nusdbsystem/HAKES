FROM ubuntu:20.04 as hakes_es_base_nosgx

USER root

# avoid tzdata interactive config during apt install.
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
  # install deps (libuv/http-parser)
  cmake \
  clang \
  libwolfssl-dev \
  # install deps (tflm)
  curl \
  git \
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
RUN cd /hakes-es/ && make mrproper && make server_deps

