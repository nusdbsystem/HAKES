FROM golang:1.21-bullseye

ENV GOMEMLIMIT=2GiB

RUN apt update \
  && apt install -y tini

COPY ./fnpacker /hakes-es/fnpacker

RUN cd /hakes-es/fnpacker \
  && go mod tidy \
  && CGO_ENABLED=1 GOOS=linux go build -o /fnpacker /hakes-es/fnpacker

CMD ["/fnpacker", "-port", "7310"]
