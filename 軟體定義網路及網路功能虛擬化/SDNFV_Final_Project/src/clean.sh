#!/bin/bash

# Remove network namespaces for all containers with the specified label
for container in $(docker ps --filter "label=app=sdntest" --format '{{.Names}}'); do
    rm -f /var/run/netns/"$container"
done

# Stop all containers with the specified label
docker ps -q --filter "label=app=sdntest" | xargs -r docker stop 2>/dev/null || true
docker ps -aq --filter "label=app=sdntest" | xargs -r docker rm 2>/dev/null || true

# Delete all veth interfaces matching the specific pattern
for veth in $(basename -a /sys/class/net/veth* 2>/dev/null); do # | grep -E '([Rbh][0-9]){2}'); do
    ip link delete "$veth" 2>/dev/null || true
done

ovs-vsctl del-br br1
ovs-vsctl del-br br2
# ip link delete bond0 type bond

wg-quick down wg0

# docker network rm sdn_network

echo "Cleanup complete."
