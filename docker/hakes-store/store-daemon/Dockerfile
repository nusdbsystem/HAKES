FROM golang:1.21-bullseye

WORKDIR /app

ENV GOMEMLIMIT=8GiB

COPY ./hakes-store /app

RUN apt update \
    && apt install -y tini \
    && go mod tidy

RUN CGO_ENABLED=1 GOOS=linux go build -o /app/server /app/store-daemon/sample/main.go

CMD ["/app/server", "-port", "2220", "-config", "/app/conf/store_daemon_config.yaml"]
