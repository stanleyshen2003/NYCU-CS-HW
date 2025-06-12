FROM alpine:3.2
LABEL maintainer="N0BALL"

RUN apk update
RUN apk add iproute2 tcpdump mtr

ENTRYPOINT ["/bin/ash", "-c", "while sleep 3600; do :; done"]