FROM golang:1.21-bullseye

WORKDIR /app

ENV GOMEMLIMIT=8GiB

COPY ./hakes-store /app

RUN apt update \
    && apt install -y tini \
    && go mod tidy


RUN CGO_ENABLED=1 GOOS=linux go build -o /app/server /app/hakes-store/http

CMD ["/app/server", "-config", "/app/conf/server_config.yaml","-hakeskvConfig", "/app/conf/kv_config.yaml"]
