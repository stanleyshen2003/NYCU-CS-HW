#!/bin/bash
#set -x

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

# Creates a veth pair
# params: endpoint1 endpoint2
function create_veth_pair {
    echo "Creating veth pair $1 $2"
    ip link add $1 type veth peer name $2
    ip link set $1 mtu 3000
    ip link set $2 mtu 3000
    ip link set $1 up
    ip link set $2 up
    ip addr flush dev $1
    ip addr flush dev $2
}

# Add a container with a certain image
# params: image_name container_name
function add_container {
	docker run -dit --network=none --label app=sdntest --privileged --cap-add NET_ADMIN --cap-add SYS_MODULE \
		 --hostname $2 --name $2 ${@:3} $1
	pid=$(docker inspect -f '{{.State.Pid}}' $(docker ps -aqf "name=$2"))
	mkdir -p /var/run/netns
	ln -s /proc/$pid/ns/net /var/run/netns/$pid
}

function add_container_custom_net {
	docker run -dit --label app=sdntest --privileged --cap-add NET_ADMIN --cap-add SYS_MODULE \
		 --hostname $2 --name $2 ${@:3} $1
	pid=$(docker inspect -f '{{.State.Pid}}' $(docker ps -aqf "name=$2"))
	mkdir -p /var/run/netns
	ln -s /proc/$pid/ns/net /var/run/netns/$pid
}

# Set container interface's ip address and gateway
# params: container_name infname [ipaddress] [gw addr]
function set_intf_container {
    pid=$(docker inspect -f '{{.State.Pid}}' $(docker ps -aqf "name=$1"))
    ifname=$2
    ipaddr=$3
    echo "Add interface $ifname with ip $ipaddr to container $1"

    ip link set "$ifname" netns "$pid"
    if [ $# -ge 3 ]
    then
        ip netns exec "$pid" ip addr add "$ipaddr" dev "$ifname"
    fi
    ip netns exec "$pid" ip link set "$ifname" up
    if [ $# -ge 4 ]
    then
        ip netns exec "$pid" route add default gw $4
    fi
}

# Set container interface's ipv6 address and gateway
# params: container_name infname [ipaddress] [gw addr]
function set_v6intf_container {
    pid=$(docker inspect -f '{{.State.Pid}}' $(docker ps -aqf "name=$1"))
    ifname=$2
    ipaddr=$3
    echo "Add interface $ifname with ip $ipaddr to container $1"

    if [ $# -ge 3 ]
    then
        ip netns exec "$pid" ip addr add "$ipaddr" dev "$ifname"
    fi
    ip netns exec "$pid" ip link set "$ifname" up
    if [ $# -ge 4 ]
    then
        ip netns exec "$pid" route -6 add default gw $4
    fi
}

# Connects the bridge and the container
# params: bridge_name container_name [ipaddress] [gw addr]
function build_bridge_container_path {
    br_inf="veth$1$2"
    container_inf="veth$2$1"
    create_veth_pair $br_inf $container_inf
    brctl addif $1 $br_inf
    set_intf_container $2 $container_inf $3 $4
}

# Connects two ovsswitches
# params: ovs1 ovs2
function build_ovs_path {
    inf1="veth$1$2"
    inf2="veth$2$1"
    create_veth_pair $inf1 $inf2
    ovs-vsctl add-port $1 $inf1
    ovs-vsctl add-port $2 $inf2
}

# Connects a container to an ovsswitch
# params: ovs container [ipaddress] [mac addr] [gateway]
function build_ovs_container_path {
    ovs_inf="veth$1$2"
    container_inf="veth$2$1"
    create_veth_pair $ovs_inf $container_inf
    if [ $# -ge 4 ]
    then
        ifconfig $container_inf hw ether $4
    fi
    ovs-vsctl add-port $1 $ovs_inf
    if [ $# -ge 5 ]
    then
        echo $5
        set_intf_container $2 $container_inf $3 $5
    else
        set_intf_container $2 $container_inf $3
    fi
    
}

# Connects a container to an ovsswitch
# params: ovs container veth_id [ipaddress] [ipv6address] [mac address]
function build_ovs_router_path_custom {
    ovs_inf="veth$1$2$3"
    container_inf="veth$2$1$3"
    create_veth_pair $ovs_inf $container_inf
    ifconfig $container_inf hw ether $6

    ovs-vsctl add-port $1 $ovs_inf
    set_intf_container $2 $container_inf $4
    
    set_v6intf_container $2 $container_inf $5

}

# Connects a container to an ovsswitch
# params: ovs container veth_id [ipaddress] [ipv6address] [gw addr] [gw v6 addr]
function build_ovs_host_path_custom {
    ovs_inf="veth$1$2$3"
    container_inf="veth$2$1$3"
    create_veth_pair $ovs_inf $container_inf
    ovs-vsctl add-port $1 $ovs_inf
    set_intf_container $2 $container_inf $4
    set_v6intf_container $2 $container_inf $5
}

# Connects two containers
# params: host_container1 router_container2 [ipaddress1] [ipaddress2] [ipv6address1] [ipv6address2] [gw addr] [gw v6 addr]
function connect_containers_v4v6 {
    inf1="veth$1$2"
    inf2="veth$2$1"
    create_veth_pair $inf1 $inf2
    set_intf_container $1 $inf1 $3 $7
    set_intf_container $2 $inf2 $4
    set_v6intf_container $1 $inf1 $5 $8
    set_v6intf_container $2 $inf2 $6
}

function add_onos {

    ip link add name vethonos type bridge
    ip link set vethonos up
    ip addr flush dev vethonos
    ip link set vethonos mtu 3000
    ip addr add 192.168.100.1/24 dev vethonos

    docker run -dit --net=host --name onos --hostname onos --privileged \
        -p 2620:2620 -p 6653:6653 -p 8101:8101 -p 8181:8181 \
        --tty --label app=sdntest sdnfv-final-onos
    pid=$(docker inspect -f '{{.State.Pid}}' $(docker ps -aqf "name=onos"))
    mkdir -p /var/run/netns
    ln -s /proc/$pid/ns/net /var/run/netns/$pid

## ----------------??????????????????????????????????????????????????????????????????
    ## 1. create dummy
    ## 2. make router and host
    ## 3. add --add-host host.docker.internal:host-gateway for R1?
}

function install_onos_apps {
    onos_apps=$(ls containers/onos | grep .oar)
    for app in $onos_apps; do
        while true; do
            status=$(/home/stanley/onos/tools/package/runtime/bin/onos-app localhost install! containers/onos/$app)
            status=$(echo $status | grep "{")
            if [ -z "$status" ]; then
                sleep 5
                  # Break the loop if not 503
                  # Wait 5 seconds before retrying
            else
                echo "Installed $app"
                break
            fi
        done
        
    done
}

# Create a veth pair for a bond
# params: container_name ovs_name veth_id bond_name
function create_veth_pair_for_bond {
    link_name="veth$1$2$3"
    peer_name="veth$2$1$3"
    ip link add $link_name type veth peer name $peer_name
    ip link set $link_name mtu 3000
    ip link set $peer_name mtu 3000
    ip link set $link_name master $4
    ip link set $link_name up
    ip link set $peer_name up
    ip addr flush dev $link_name
    ip addr flush dev $peer_name
}



ID="53"

HOSTIMAGE="sdnfv-final-host"
ROUTERIMAGE="sdnfv-final-frr"
ONOSIMAGE="sdnfv-final-onos"


HOST1Name="host1"
HOST2Name="host2"
ROUTER1Name="r1"
ROUTER2Name="r2"
BONDName="bond0"
OVS1Name="br1"
OVS2Name="br2"

DockerNetworkName="sdn_network"


# Build host base image
docker_images=$(docker images)

if [[ $docker_images != *"$HOSTIMAGE"* ]]; then
    docker build containers/host -t "$HOSTIMAGE"
fi

if [[ $docker_images != *"$ROUTERIMAGE"* ]]; then
    docker build containers/frr -t "$ROUTERIMAGE"
fi

if [[ $docker_images != *"$ONOSIMAGE"* ]]; then
    docker build containers/onos -t "$ONOSIMAGE"
fi

add_onos

wg-quick up wg0

add_container $HOSTIMAGE $HOST1Name
add_container $ROUTERIMAGE $ROUTER1Name -v ./config/R1/frr.conf:/etc/frr/frr.conf -v ./config/daemons:/etc/frr/daemons

# Setting AS65530
####
echo "Adding bridges"
ovs-vsctl add-br $OVS1Name
ovs-vsctl set bridge $OVS1Name protocols=OpenFlow14
ovs-vsctl set-controller $OVS1Name tcp:192.168.100.1:6653
ovs-vsctl set bridge $OVS1Name other-config:datapath-id=0000000000000001
ovs-vsctl set interface $OVS1Name mtu_request=3000
ovs-vsctl add-br $OVS2Name
ovs-vsctl set bridge $OVS2Name protocols=OpenFlow14
ovs-vsctl set-controller $OVS2Name tcp:192.168.100.1:6653
ovs-vsctl set bridge $OVS2Name other-config:datapath-id=0000000000000002
ovs-vsctl set interface $OVS1Name mtu_request=3000


####
# echo "Connect router1 to ovs1"
# ip link add $BONDName type bond
# ip link set $BONDName mtu 3000
# ip link set $BONDName up
# ip addr flush dev $BONDName

# echo "Adding veth pairs for bond"
# create_veth_pair_for_bond $ROUTER1Name $OVS1Name 0 $BONDName
# create_veth_pair_for_bond $ROUTER1Name $OVS1Name 1 $BONDName
# create_veth_pair_for_bond $ROUTER1Name $OVS1Name 2 $BONDName

echo "Adding bond to ovs"
# ovs-vsctl add-port $OVS1Name bond0
# set_intf_container $ROUTER1Name "veth${OVS1Name}${ROUTER1Name}0" "172.16.${ID}.69/24"
# set_v6intf_container $ROUTER1Name "veth${OVS1Name}${ROUTER1Name}0" "2a0b:4e07:c4:${ID}::69/64"
# set_intf_container $ROUTER1Name "veth${OVS1Name}${ROUTER1Name}1" "192.168.70.${ID}/24"
# set_v6intf_container $ROUTER1Name "veth${OVS1Name}${ROUTER1Name}1" "fd70::${ID}/64"
# set_intf_container $ROUTER1Name "veth${OVS1Name}${ROUTER1Name}2" "192.168.63.1/24"
# set_v6intf_container $ROUTER1Name "veth${OVS1Name}${ROUTER1Name}2" "fd63::1/64"
build_ovs_router_path_custom $OVS1Name $ROUTER1Name 0 "172.16.${ID}.69/24" "2a0b:4e07:c4:${ID}::69/64" "00:00:00:00:00:01"
build_ovs_router_path_custom $OVS1Name $ROUTER1Name 1 "192.168.70.${ID}/24" "fd70::${ID}/64" "00:00:00:00:00:02"
build_ovs_router_path_custom $OVS1Name $ROUTER1Name 2 "192.168.63.1/24" "fd63::1/64" "00:00:00:00:00:03"


####
echo "Add 192.168.100.3 to router 1 and connect to vethonos"
ip link add vethtoonos type veth peer name vethtoonospeer
ip link set vethtoonospeer mtu 3000
ip link set vethtoonos mtu 3000
ip link set vethtoonos up
ip link set vethtoonospeer up
ip addr flush dev vethtoonos
ip addr flush dev vethtoonospeer
ip link set vethtoonospeer master vethonos
temp=$(docker inspect -f '{{.State.Pid}}' $(docker ps -aqf "name=$ROUTER1Name"))
ip link set vethtoonos netns $temp
ip netns exec $temp ip addr add 192.168.100.3/24 dev vethtoonos
ip netns exec $temp ip link set vethtoonos up
echo "find gatesy?"
ip netns exec $temp route add default gw 192.168.100.1/24

####
echo "Connecting two ovs"
build_ovs_path $OVS1Name $OVS2Name
echo "Connecting ovs to host"
build_ovs_container_path $OVS2Name $HOST1Name "172.16.${ID}.2/24" "00:00:00:00:00:05" "172.16.${ID}.1"
set_v6intf_container $HOST1Name "veth${HOST1Name}${OVS2Name}" "2a0b:4e07:c4:${ID}::2/64" "2a0b:4e07:c4:${ID}::69"

####
echo "Adding containers for AS65531"
add_container $ROUTERIMAGE  $ROUTER2Name -v ./config/R2/frr.conf:/etc/frr/frr.conf -v ./config/daemons:/etc/frr/daemons
add_container $HOSTIMAGE  $HOST2Name

####
echo "Connecting ovs to router"
build_ovs_container_path $OVS1Name $ROUTER2Name "192.168.63.2/24" "00:00:00:00:00:04"

set_v6intf_container $ROUTER2Name "veth${ROUTER2Name}${OVS1Name}" "fd63::2/64"

####
echo "Connecting two containers"
connect_containers_v4v6 $HOST2Name $ROUTER2Name "172.17.${ID}.2/24" "172.17.${ID}.1/24" "2a0b:4e07:c4:1${ID}::2/64" "2a0b:4e07:c4:1${ID}::1/64" "172.17.${ID}.1" "2a0b:4e07:c4:1${ID}::1" 

####
echo "Configuring router2 to TA's router"
ip route add 10.0.0.0/24 dev wg0

ovs-vsctl add-port $OVS2Name vxlan0 -- set interface vxlan0 type=vxlan options:remote_ip=192.168.60.53 options:dst_port=4789
#$ADD
build_ovs_router_path_custom $OVS1Name $ROUTER1Name 3 "192.168.27.1/24" "fd27::1/64" "00:00:00:00:00:06"
ovs-vsctl add-port $OVS2Name vxlan1 -- set interface vxlan1 type=vxlan options:remote_ip=192.168.61.52 options:dst_port=4789

ip link add veth2onos type veth peer name veth2onospeer
ip link set veth2onos mtu 3000
ip link set veth2onospeer mtu 3000
ip link set veth2onos up
ip link set veth2onospeer up
ip addr flush dev veth2onos
ip addr flush dev veth2onospeer
ip link set veth2onos master vethonos
ovs-vsctl add-port $OVS2Name veth2onospeer




docker exec r2 sysctl -w net.ipv6.conf.all.forwarding=1
docker exec r1 sysctl -w net.ipv4.conf.all.forwarding=1
# install_onos_apps
echo "Done"