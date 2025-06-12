FROM ubuntu:22.04

RUN apt-get update -y \
&&  apt-get install -y traceroute \
&&  apt-get install -y net-tools \
&&  apt-get install -y iproute2 \
&&  apt-get install -y iputils-ping


CMD ["sleep","infinity"]