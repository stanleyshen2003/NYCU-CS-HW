# Use the official ONOS image
FROM onosproject/onos:2.7-latest

# Set the working directory
WORKDIR /root/onos

# Copy the .oar file into the ONOS container
# Ensure you have your .oar file in the same directory as this Dockerfile or adjust the path accordingly
# COPY ./your-app.oar /opt/onos/apps/
# COPY ./bridge-1.0-SNAPSHOT.oar /root/onos/apps/
# COPY ./ProxyArp-1.0-SNAPSHOT.oar /root/onos/apps/
# COPY ./ProxyNdp-1.0-SNAPSHOT.oar /root/onos/apps/

# Set environment variables for ONOS apps
ENV ONOS_APPS=drivers,fpm,openflow,hostprovider,lldpprovider,gui2,route-service,optical-model,bridge,ProxyArp
ENV ONOS_IP=192.168.0.1

# Install the app and start ONOS
# Use the entry point to install the app when the container starts
# CMD ["sh", "-c", "onos"]
# RUN onos-app install! apps/bridge-1.0-SNAPSHOT.oar
# RUN onos-app install! apps/ProxyArp-1.0-SNAPSHOT.oar
# RUN onos-app install! apps/ProxyNdp-1.0-SNAPSHOT.oar


# Expose required ports for ONOS
EXPOSE 2620 6653 8101 8181

ENTRYPOINT ["./bin/onos-service"]
CMD ["server"]
