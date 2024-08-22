FROM hakes_es_base_sgx:v1 as ksbuilder

USER root

COPY --from=hakes_es_base_sgx:v1 /hakes-es/deps /hakes-es/deps
COPY ./key-service/ /hakes-es/key-service
COPY ./message/ /hakes-es/message
COPY ./ratls-channel/ /hakes-es/ratls-channel
COPY ./secret/ /hakes-es/secret
COPY ./store-client/ /hakes-es/store-client
COPY ./utils/ /hakes-es/utils

RUN cd /hakes-es/key-service && make mrproper && make all && make install

FROM sgx_dcap_ssl_2.14_1.11:v1

USER root

COPY --from=ksbuilder /hakes-es/key-service/install /install

CMD SGX_AESM_ADDR=1 /install/bin/key_server /tmp /install/lib/KeyServer_Enclave.signed.so
