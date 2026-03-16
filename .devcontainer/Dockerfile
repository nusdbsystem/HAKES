FROM ubuntu:22.04 as hakes_base

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
  libssl-dev \
  libsasl2-dev \
  libsnappy-dev \
  pkg-config \
  libcurl4-openssl-dev \
  git \
  python3 \
  python3-pip \
  unzip \
  wget \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install --upgrade Pillow

RUN pip install --no-cache-dir \
  ipykernel \
  jupyter \
  notebook

# MongoDB C++ Driver
RUN wget https://github.com/mongodb/mongo-cxx-driver/releases/download/r4.1.4/mongo-cxx-driver-r4.1.4.tar.gz \
  && tar xzf mongo-cxx-driver-r4.1.4.tar.gz \
  && cd mongo-cxx-driver-r4.1.4/build \
  && cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. \
  && make -j$(nproc) \
  && make install \
  && ldconfig


ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION v16.15.0
RUN mkdir -p /usr/local/nvm
RUN wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
RUN /bin/bash -c "source $NVM_DIR/nvm.sh && nvm install $NODE_VERSION && nvm use --delete-prefix $NODE_VERSION"
ENV NODE_PATH $NVM_DIR/versions/node/$NODE_VERSION/bin
ENV PATH $NODE_PATH:$PATH

RUN node --version && npm --version
