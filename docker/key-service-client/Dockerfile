FROM hakes_es_base_sgx:v1 as clientbuilder

COPY --from=hakes_es_base_sgx:v1 /hakes-es/deps /hakes-es/deps
COPY ./key-service-client/ /hakes-es/key-service-client
COPY ./message/ /hakes-es/message
COPY ./ratls-channel/ /hakes-es/ratls-channel
COPY ./secret/ /hakes-es/secret
COPY ./store-client/ /hakes-es/store-client
COPY ./utils/ /hakes-es/utils

RUN cd /hakes-es/key-service-client && make clean && make all

# the verification still depends on some SGX libraries.
# for simplicity we used the same base image.
FROM sgx_dcap_ssl_2.14_1.11:v1

COPY --from=clientbuilder /hakes-es/key-service-client/keyservice_client ./keyservice_client

CMD ./keyservice_client

# docker build -t hakes-keyservice-client:v1 -f docker/key-service-client/Dockerfile .
