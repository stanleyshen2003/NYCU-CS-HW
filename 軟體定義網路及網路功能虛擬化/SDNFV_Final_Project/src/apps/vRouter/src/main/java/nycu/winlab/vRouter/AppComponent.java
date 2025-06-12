/*
 * Copyright 2024-present Open Networking Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// 0327
package nycu.winlab.vRouter;

import static org.onosproject.net.config.NetworkConfigEvent.Type.CONFIG_ADDED;
import static org.onosproject.net.config.NetworkConfigEvent.Type.CONFIG_UPDATED;
import static org.onosproject.net.config.basics.SubjectFactories.APP_SUBJECT_FACTORY;
import org.onosproject.net.config.NetworkConfigEvent;
import org.onosproject.net.config.NetworkConfigListener;
import org.onosproject.net.config.NetworkConfigRegistry;
import org.onosproject.net.config.NetworkConfigService;
import org.onosproject.net.config.ConfigFactory;

import org.onlab.packet.IpAddress;
import org.onlab.packet.IpPrefix;
import org.onosproject.cfg.ComponentConfigService;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import org.onosproject.net.intf.InterfaceService;
import org.onosproject.net.intf.Interface;
import org.onosproject.routeservice.*;

import org.onosproject.net.host.*;


import org.onosproject.core.ApplicationId;
import org.onosproject.core.CoreService;
import org.onosproject.net.flow.DefaultFlowRule;
import org.onosproject.net.flow.DefaultTrafficSelector;
import org.onosproject.net.flow.TrafficSelector;

import org.onosproject.net.flow.DefaultTrafficTreatment;
import org.onosproject.net.flow.FlowRule;
import org.onosproject.net.flow.TrafficTreatment;
import org.onosproject.net.flowobjective.DefaultForwardingObjective;
import org.onosproject.net.flowobjective.FlowObjectiveService;
import org.onosproject.net.flowobjective.ForwardingObjective;
import org.onosproject.net.packet.PacketPriority;
import org.onosproject.net.packet.PacketService;
import org.onosproject.net.packet.PacketProcessor;
import org.onosproject.net.packet.PacketContext;
import org.onosproject.net.packet.DefaultOutboundPacket;
import org.onosproject.net.packet.InboundPacket;
import org.onosproject.net.packet.OutboundPacket;
import org.onlab.packet.Ethernet;
import org.onlab.packet.ICMP6;
import org.onlab.packet.IPv6;
import org.onlab.packet.IPv4;
import org.onlab.packet.MacAddress;
import org.onlab.packet.ndp.NeighborAdvertisement;
import org.onlab.packet.ndp.NeighborSolicitation;
import org.onlab.packet.Ip4Address;
import org.onlab.packet.Ip4Prefix;
import org.onlab.packet.Ip6Prefix;
import org.onlab.packet.ARP;
import org.onlab.packet.EthType;
import org.onlab.packet.Ip6Address;

import org.onosproject.net.PortNumber;
import org.onosproject.net.DeviceId;
import org.onosproject.net.Device;
import org.onosproject.net.Port;
import org.onosproject.net.ConnectPoint;

import org.onlab.packet.ndp.NeighborSolicitation;
import org.onlab.packet.ndp.NeighborAdvertisement;

import org.onosproject.net.intent.*;
import org.onosproject.net.FilteredConnectPoint;
import org.onosproject.net.Host;
import org.onosproject.net.ConnectPoint;
import org.onosproject.net.edge.EdgePortService;
import org.onosproject.net.device.DeviceService;
import org.onosproject.net.flow.FlowRuleService;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import org.apache.commons.lang3.tuple.Pair;
import java.util.HashMap;
import java.util.Map;
import java.util.HashSet;
import java.util.Set;

/**
 * Skeletal ONOS application component.
 */
@Component(immediate = true)
public class AppComponent {
    private final NameConfigListener cfgListener = new NameConfigListener();
    private final Logger log = LoggerFactory.getLogger(getClass());

    private final ConfigFactory<ApplicationId, NameConfig> factory = new ConfigFactory<ApplicationId, NameConfig>(
        APP_SUBJECT_FACTORY, NameConfig.class, "router") {
        @Override
        public NameConfig createConfig() {
            return new NameConfig();
        }
    };

    /** Some configurable property. */

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected NetworkConfigRegistry cfgService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected CoreService coreService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected PacketService packetService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected FlowRuleService flowRuleService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected FlowObjectiveService flowObjectiveService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected DeviceService deviceService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected EdgePortService edgePortService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected InterfaceService interfaceService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected IntentService intentService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected RouteService routeService;

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected HostService hostService;

    private LearningBridgeProcessor processor = new LearningBridgeProcessor();
    private ProxyArpHandler arpHandler = new ProxyArpHandler();

    private ApplicationId appId;
    private Map<Ip4Address, MacAddress> macTable = new HashMap<>();
    private Map<Ip6Address, MacAddress> macTable6 = new HashMap<>();
    private Map<DeviceId, Map<Pair<MacAddress, Ip4Address>, PortNumber>> bridgeTable = new HashMap<>();
    private Map<DeviceId, Map<Pair<MacAddress, Ip6Address>, PortNumber>> bridgeTable6 = new HashMap<>();
    private Map<Ip4Prefix, Ip4Address> routeTable = new HashMap<>();
    private Map<Ip6Prefix, Ip6Address> routeTable6 = new HashMap<>();
    private Set<Pair<DeviceId, PortNumber>> vrouter = new HashSet<>();
    private Boolean BGPintent = false;

    private Map<Ip4Address, MacAddress> arpTable = new HashMap<>();
    private Map<MacAddress, ConnectPoint> portTable = new HashMap<>();



    private class NameConfigListener implements NetworkConfigListener {
        @Override
        public void event(NetworkConfigEvent event) {
            if ((event.type() == CONFIG_ADDED || event.type() == CONFIG_UPDATED)
                && event.configClass().equals(NameConfig.class)) {
                NameConfig config = cfgService.getConfig(appId, NameConfig.class);
                if (config != null) {
                    log.info("vrrouting: {}", config.vrrouting());
                }
                

                for (Interface inf : interfaceService.getInterfaces()) {
                    vrouter.add(Pair.of(inf.connectPoint().deviceId(), inf.connectPoint().port()));
                }

            }
        }
    }

    
    private void refreshTable() {
        NameConfig config = cfgService.getConfig(appId, NameConfig.class);

        for (Interface inf : interfaceService.getInterfaces()) {
            for (InterfaceIpAddress ip : inf.ipAddressesList()) {
                if (ip.ipAddress().isIp4()) {
                    arpTable.put(ip.ipAddress().getIp4Address(), inf.mac());
                    // log.info("write ip4: {}, mac: {}", ip.ipAddress().getIp4Address(), inf.mac());
                }
                else {
                    macTable6.put(ip.ipAddress().getIp6Address(), inf.mac());
                    // log.info("write ip6: {}, mac: {}", ip.ipAddress().getIp6Address(), inf.mac());
                }
            }
        }

        arpTable.put(Ip4Address.valueOf(config.gatewayIp4()), MacAddress.valueOf(config.gatewayMac()));
        // log.info("write ip4: {}, mac: {}", config.gatewayIp4(), config.gatewayMac());
        macTable6.put(Ip6Address.valueOf(config.gatewayIp6()), MacAddress.valueOf(config.gatewayMac()));
        // log.info("write ip6: {}, mac: {}", config.gatewayIp6(), config.gatewayMac());
    }
    

    @Activate
    protected void activate() {

        // register your app
        appId = coreService.registerApplication("nycu.winlab.vRouter");
        cfgService.addListener(cfgListener);
        cfgService.registerConfigFactory(factory);

        // vrouter = Collections.emptySet();
        // for (Interface inf : interfaceService.getInterfaces()) {
        //     vrouter.add(Pair.of(inf.connectPoint().deviceId(), inf.connectPoint().port()));
        // }
        NameConfig config = cfgService.getConfig(appId, NameConfig.class);
        installBGPIntents(config);

        refreshTable();

                log.info("************");
        // log.info("I want to /find srcip:{} dstip:{}", srcIp, dstIp);
        // log.info("Travel macTable6");
        // for (Map.Entry<Ip6Address, MacAddress> entry : macTable6.entrySet()) {
        //     Ip6Prefix prefix = entry.getKey();
        //     Ip6Address nextHop = entry.getValue();
        //     log.info("ip6: {}, macA: {}", prefix, nextHop);
        // }
        log.info("************");
        

        // add a packet processor to packetService
        packetService.addProcessor(processor, PacketProcessor.director(2));
        packetService.addProcessor(arpHandler, PacketProcessor.director(3));

        TrafficSelector.Builder selector0 = DefaultTrafficSelector.builder()
            .matchEthType(Ethernet.TYPE_ARP).matchInPort(PortNumber.portNumber(4));

        // drop the packet
        TrafficTreatment treatment0 = DefaultTrafficTreatment.builder()
            .drop()
            .build();

            // Create the FlowRule
        FlowRule flowRule = DefaultFlowRule.builder()
                .forDevice(DeviceId.deviceId("of:0000000000000002"))
                .withSelector(selector0.build())
                .withTreatment(treatment0)
                .withPriority(50000)
                .makePermanent()
                .fromApp(appId)
                .build();

        // Submit the FlowRule
        flowRuleService.applyFlowRules(flowRule);

        TrafficSelector.Builder selector1 = DefaultTrafficSelector.builder()
            .matchEthType(Ethernet.TYPE_IPV6)
            .matchIPProtocol((byte) 58) 
            .matchInPort(PortNumber.portNumber(4))
            .matchIcmpv6Code((byte) 135);

        // drop the packet
        TrafficTreatment treatment1 = DefaultTrafficTreatment.builder().drop().build();

        // Create the FlowRule
        FlowRule flowRule1 = DefaultFlowRule.builder()
            .forDevice(DeviceId.deviceId("of:0000000000000002"))
            .withSelector(selector1.build())
            .withTreatment(treatment1)
            .withPriority(50000)
            .makePermanent()
            .fromApp(appId)
            .build();

        // Submit the FlowRule
        flowRuleService.applyFlowRules(flowRule1);

        

        // install a flowrule for packet-in
        TrafficSelector.Builder selector = DefaultTrafficSelector.builder();
        selector.matchEthType(Ethernet.TYPE_ARP);
        packetService.requestPackets(selector.build(), PacketPriority.REACTIVE, appId);


        TrafficSelector.Builder selector2 = DefaultTrafficSelector.builder();
        selector2.matchEthType(Ethernet.TYPE_IPV6);
        packetService.requestPackets(selector2.build(), PacketPriority.REACTIVE, appId);

        TrafficSelector.Builder selector3 = DefaultTrafficSelector.builder();
        selector3.matchEthType(Ethernet.TYPE_IPV4);
        packetService.requestPackets(selector3.build(), PacketPriority.REACTIVE, appId);

        TrafficSelector.Builder selector4 = DefaultTrafficSelector.builder()
            .matchEthType(Ethernet.TYPE_IPV6)
            .matchIPProtocol((byte) 58) 
            .matchIcmpv6Code((byte) 135)
            .matchIPv6Src(IpPrefix.valueOf("fd27::2/128"));
        
        
        ConnectPoint cp = new ConnectPoint(DeviceId.deviceId("of:0000000000000002"), PortNumber.portNumber(4));
        ConnectPoint cp2 = new ConnectPoint(DeviceId.deviceId("of:0000000000000001"), PortNumber.portNumber(6));

        FilteredConnectPoint fcp = new FilteredConnectPoint(cp);
        FilteredConnectPoint fcp2 = new FilteredConnectPoint(cp2);

        PointToPointIntent intent = PointToPointIntent.builder()
            .appId(appId)
            .key(Key.of("teamate ndp", appId))
            .filteredIngressPoint(fcp)
            .filteredEgressPoint(fcp2)
            .selector(selector4.build())
            .priority(60000)
            .build();

        intentService.submit(intent);

        TrafficSelector selector5 = DefaultTrafficSelector.builder()
            .matchEthType(Ethernet.TYPE_ARP)
            .matchArpSpa(Ip4Address.valueOf("192.168.27.2"))
            .build();

        
        PointToPointIntent intent2 = PointToPointIntent.builder()
            .appId(appId)
            .key(Key.of("teamate arp", appId))
            .filteredIngressPoint(fcp)
            .filteredEgressPoint(fcp2)
            .selector(selector5)
            .priority(60000)
            .build();

        intentService.submit(intent2);


        log.info("Started");
    }

    @Deactivate
    protected void deactivate() {

        // remove flowrule installed by your app
        flowRuleService.removeFlowRulesById(appId);
        // remove your packet processor
        packetService.removeProcessor(processor);
        packetService.removeProcessor(arpHandler);
        processor = null;

        // remove flowrule you installed for packet-in
        TrafficSelector.Builder selector = DefaultTrafficSelector.builder();
        selector.matchEthType(Ethernet.TYPE_ARP);
        packetService.cancelPackets(selector.build(), PacketPriority.REACTIVE, appId);

        TrafficSelector.Builder selector2 = DefaultTrafficSelector.builder();
        selector2.matchEthType(Ethernet.TYPE_IPV6);
        packetService.cancelPackets(selector2.build(), PacketPriority.REACTIVE, appId);

        TrafficSelector.Builder selector3 = DefaultTrafficSelector.builder();
        selector3.matchEthType(Ethernet.TYPE_IPV4);
        packetService.cancelPackets(selector3.build(), PacketPriority.REACTIVE, appId);

        intentService.getIntentsByAppId(appId).forEach(intentService::withdraw);

        log.info("Stopped");
    }

    
    private Integer findNDP(Ethernet packet){
        
        if (packet.getEtherType() != Ethernet.TYPE_IPV6) {
            return 0;
        }

        IPv6 ipv6Packet = (IPv6) packet.getPayload();
        
        // Check if the packet uses ICMPv6 (IPv6 Next Header == ICMPv6 protocol number)
        if (ipv6Packet.getNextHeader() != IPv6.PROTOCOL_ICMP6) {
            return 0;
        }
        ICMP6 icmp6Packet = (ICMP6) ipv6Packet.getPayload();
        byte icmpType = icmp6Packet.getIcmpType();
        
        // Check if the ICMPv6 type is Neighbor Solicitation (135) or Neighbor Advertisement (136)
        if (icmpType == (byte)135) {
            return 1;
        } else if (icmpType == (byte)136) {
            return 2;
        }
        return 0;
        
    }

    private void processNDPSol(PacketContext context, NeighborSolicitation ndp) {
         // get payload
        Ethernet ethPkt = context.inPacket().parsed();
        IPv6 ipv6Packet = (IPv6) ethPkt.getPayload();
        Ip6Address srcIp = Ip6Address.valueOf(ipv6Packet.getSourceAddress());
        MacAddress srcMac = ethPkt.getSourceMAC();
        Ip6Address dstIp = Ip6Address.valueOf(ndp.getTargetAddress());

        // write the srcIP if it is not written
        if (macTable6.get(srcIp) == null) {
            macTable6.put(srcIp, srcMac);
            log.info("Add new entry. IP = {}, MAC = {}", srcIp, srcMac);
        }

        if (macTable6.get(dstIp) == null){
            log.info("TABLE MISS. Missed IP = {}", dstIp);
        } else {
            log.info("TABLE HIT. Requested MAC = {}, Required IP = {}", macTable6.get(dstIp), dstIp);
        }
    }

    private void processNDPAdv(PacketContext context, NeighborAdvertisement ndp) {
        // get payload
        Ethernet ethPkt = context.inPacket().parsed();
        IPv6 ipv6Packet = (IPv6) ethPkt.getPayload();
        Ip6Address srcIp = Ip6Address.valueOf(ipv6Packet.getSourceAddress());
        MacAddress srcMac = ethPkt.getSourceMAC();

        if (macTable6.get(srcIp) == null) {
            macTable6.put(srcIp, srcMac);
            log.info("Add new entry. IP = {}, MAC = {}", srcIp, srcMac);
        }
    }

    private void installBGPIntents(NameConfig config) {
        List<String> v4Peers = config.v4Peers();
        List<String> v6Peers = config.v6Peers();
        String devicePort = config.vrrouting();
        DeviceId devID = DeviceId.deviceId(devicePort.split("/")[0]);
        PortNumber port; // = PortNumber.portNumber(devicePort.split("/")[1]);
        
        for (int i = 0; i < v4Peers.size(); i+=2) {
            Ip4Address peerIP1 = Ip4Address.valueOf(v4Peers.get(i));
            Ip4Address peerIP2 = Ip4Address.valueOf(v4Peers.get(i+1));
            // install flow rule for ARP packets
            TrafficSelector.Builder selector1 = DefaultTrafficSelector.builder();
            selector1.matchIPSrc(peerIP1.toIpPrefix()).matchIPDst(peerIP2.toIpPrefix()).matchEthType(Ethernet.TYPE_IPV4);

            TrafficSelector.Builder selector2 = DefaultTrafficSelector.builder();
            selector2.matchIPSrc(peerIP2.toIpPrefix()).matchIPDst(peerIP1.toIpPrefix()).matchEthType(Ethernet.TYPE_IPV4);

            TrafficSelector.Builder selector3 = DefaultTrafficSelector.builder();
            selector3.matchArpTpa(peerIP2).matchEthType(Ethernet.TYPE_ARP);

            TrafficSelector.Builder selector4 = DefaultTrafficSelector.builder();
            selector4.matchArpTpa(peerIP1).matchEthType(Ethernet.TYPE_ARP);
            
            if (i == 0){
                port = PortNumber.portNumber("2");
            }
            else if (i == 2 ) {
                port = PortNumber.portNumber("3");
            }
            else {
                port = PortNumber.portNumber("6");
            }
            ConnectPoint src = new ConnectPoint(devID, port);
            ConnectPoint dst = interfaceService.getMatchingInterface(IpAddress.valueOf(v4Peers.get(i))).connectPoint();

            log.info("Created intent from {}/{} to {}/{}", src.deviceId(), src.port(), dst.deviceId(), dst.port());

            FilteredConnectPoint fsrc = new FilteredConnectPoint(src);
            FilteredConnectPoint fdst = new FilteredConnectPoint(dst);


            PointToPointIntent intent1 = PointToPointIntent.builder()
                .appId(appId)
                .key(Key.of(peerIP1.toString() + "-" + peerIP2.toString(), appId))
                .selector(selector1.build())
                .filteredIngressPoint(fsrc)
                .filteredEgressPoint(fdst)
                .priority(39999)
                .build();

            PointToPointIntent intent2 = PointToPointIntent.builder()
                .appId(appId)
                .key(Key.of(peerIP2.toString() + "-" + peerIP1.toString(), appId))
                .selector(selector2.build())
                .filteredIngressPoint(fdst)
                .filteredEgressPoint(fsrc)
                .priority(39999)
                .build();

            PointToPointIntent intent3 = PointToPointIntent.builder()
                .appId(appId)
                .key(Key.of(peerIP1.toString() + "-" + peerIP2.toString() + "ARP", appId))
                .selector(selector3.build())
                .filteredIngressPoint(fsrc)
                .filteredEgressPoint(fdst)
                .priority(49999)
                .build();

            PointToPointIntent intent4 = PointToPointIntent.builder()
                .appId(appId)
                .key(Key.of(peerIP2.toString() + "-" + peerIP1.toString() + "ARP", appId))
                .selector(selector4.build())
                .filteredIngressPoint(fdst)
                .filteredEgressPoint(fsrc)
                .priority(49999)
                .build();

            intentService.submit(intent1);
            intentService.submit(intent2);
            intentService.submit(intent3);
            intentService.submit(intent4);
            
        }

        for (int i = 0; i < v6Peers.size(); i+=2) {
            Ip6Address peerIP1 = Ip6Address.valueOf(v6Peers.get(i));
            Ip6Address peerIP2 = Ip6Address.valueOf(v6Peers.get(i+1));
            // install flow rule for ARP packets
            TrafficSelector.Builder selector1 = DefaultTrafficSelector.builder();
            selector1.matchEthType(Ethernet.TYPE_IPV6).matchIPv6Src(peerIP1.toIpPrefix());

            TrafficSelector.Builder selector2 = DefaultTrafficSelector.builder();
            selector2.matchEthType(Ethernet.TYPE_IPV6).matchIPv6Src(peerIP2.toIpPrefix());
            
            if (i == 0){
                port = PortNumber.portNumber("2");
            }
            else if (i == 2 ) {
                port = PortNumber.portNumber("3");
            }
            else {
                port = PortNumber.portNumber("6");
            }
            
            ConnectPoint src = new ConnectPoint(devID, port);
            
            ConnectPoint dst = interfaceService.getMatchingInterface(IpAddress.valueOf(v6Peers.get(i))).connectPoint();

            log.info("Created intent from {}/{} to {}/{}", src.deviceId(), src.port(), dst.deviceId(), dst.port());

            FilteredConnectPoint fsrc = new FilteredConnectPoint(src);
            FilteredConnectPoint fdst = new FilteredConnectPoint(dst);

            PointToPointIntent intent1 = PointToPointIntent.builder()
                .appId(appId)
                .key(Key.of(peerIP1.toString() + "-" + peerIP2.toString(), appId))
                .selector(selector1.build())
                .filteredIngressPoint(fsrc)
                .filteredEgressPoint(fdst)
                .priority(39999)
                .build();

            PointToPointIntent intent2 = PointToPointIntent.builder()
                .appId(appId)
                .key(Key.of(peerIP2.toString() + "-" + peerIP1.toString(), appId))
                .selector(selector2.build())
                .filteredIngressPoint(fdst)
                .filteredEgressPoint(fsrc)
                .priority(39999)
                .build();

            intentService.submit(intent1);
            intentService.submit(intent2);
            
        }
    }

    
    private class ProxyArpHandler implements PacketProcessor {
        @Override
        public void process(PacketContext context) {
            if (context.isHandled()) {
                return;
            }
            log.info("In ProxyArpHandler");

            InboundPacket pkt = context.inPacket();
            Ethernet ethPkt = pkt.parsed();

            if (ethPkt.getEtherType() != Ethernet.TYPE_ARP) {
                return;
            }

            refreshTable();

            ARP arpPkt = (ARP) ethPkt.getPayload();

            if (arpPkt.getProtocolType() != ARP.PROTO_TYPE_IP) {
                return;
            }

            Ip4Address dstIp = Ip4Address.valueOf(arpPkt.getTargetProtocolAddress());
            Ip4Address srcIp = Ip4Address.valueOf(arpPkt.getSenderProtocolAddress());
            MacAddress srcMac = ethPkt.getSourceMAC();
            ConnectPoint inPort = pkt.receivedFrom();


            if (arpTable.get(srcIp) == null) {
                log.info("in arpprocessof, add new entry. IP = {}, MAC = {}", srcIp, srcMac);
                arpTable.put(srcIp, srcMac);
            }
            if (portTable.get(srcMac) == null) {
                portTable.put(srcMac, inPort);
            }

            MacAddress dstMac = arpTable.get(dstIp);
            ConnectPoint outPort = portTable.get(dstMac);

            if (arpPkt.getOpCode() == ARP.OP_REQUEST) {
                if (dstMac == null) {
                    log.info("TABLE MISS. Send request to edge ports");
                    flood(ethPkt, inPort);
                } else {
                    log.info("TABLE HIT. Requested MAC = {}", dstMac);
                    sendArpReply(ethPkt, dstIp, dstMac, inPort);
                }
            } else if (arpPkt.getOpCode() == ARP.OP_REPLY) {
                log.info("RECV REPLY. Requested MAC = {}", srcMac);
                sendPacket(ethPkt, outPort);
            }
        }

        private void flood(Ethernet ethPkt, ConnectPoint inPort) {
            for (ConnectPoint cp : edgePortService.getEdgePoints()) {
                if (!cp.equals(inPort)) {
                    sendPacket(ethPkt, cp);
                }
            }
        }

        private void sendArpReply(Ethernet ethPkt, Ip4Address dstIp,
         MacAddress dstMac, ConnectPoint inPort) {

            Ethernet arpReply = ARP.buildArpReply(dstIp, dstMac, ethPkt);
            sendPacket(arpReply, inPort);
        }

        private void sendPacket(Ethernet ethPkt, ConnectPoint cp) {

            TrafficTreatment treatment = DefaultTrafficTreatment.builder().setOutput(cp.port()).build();
            OutboundPacket outPacket = new DefaultOutboundPacket(cp.deviceId(), treatment,
                ByteBuffer.wrap(ethPkt.serialize()));

            packetService.emit(outPacket);
        }
    }

    private class LearningBridgeProcessor implements PacketProcessor {

        @Override
        public void process(PacketContext context) {
            // Stop processing if the packet has been handled, since we
            // can't do any more to it.
            if (context.isHandled()) {
                return;
            }
            
            
            

            // Collection<RouteInfo> routes6 = routeService.getRoutes(new RouteTableId("ipv6"));
            // for (RouteInfo route : routes6) {
            //     for (ResolvedRoute resRoute : route.allRoutes()) {
            //         routeTable6.put(resRoute.prefix().getIp6Prefix(), resRoute.nextHop().getIp6Address());
            //     }
            // }


            
            InboundPacket pkt = context.inPacket();
            Ethernet ethPkt = pkt.parsed();
            PortNumber recPort = pkt.receivedFrom().port();
            DeviceId recDevId = pkt.receivedFrom().deviceId();
            
            if (ethPkt == null) {
                return;
            }
            short ethType = ethPkt.getEtherType();

            if (ethType == Ethernet.TYPE_IPV6) {
                Integer ndpType = findNDP(ethPkt);
                if (ndpType == 1){
                    processNDPSol(context, (NeighborSolicitation) ethPkt.getPayload().getPayload().getPayload());
                }
                else if (ndpType == 2){
                    processNDPAdv(context, (NeighborAdvertisement) ethPkt.getPayload().getPayload().getPayload());
                }
            }

        /*
            // ARP packet
            // else if ( ethType == Ethernet.TYPE_ARP) {
            //     ARP arpPacket = (ARP) ethPkt.getPayload();

            //     // get payload
            //     Ip4Address srcIp = Ip4Address.valueOf(arpPacket.getSenderProtocolAddress());
            //     MacAddress srcMac = MacAddress.valueOf(arpPacket.getSenderHardwareAddress());
            //     Ip4Address dstIp = Ip4Address.valueOf(arpPacket.getTargetProtocolAddress());

                    
            //     // if it is a request packet
            //     if (arpPacket.getOpCode() == ARP.OP_REQUEST){

            //         // write the srcIP if it is not written
            //         if (macTable.get(srcIp) == null) {
            //             macTable.put(srcIp, srcMac);
            //             log.info("Add new entry from request. IP = {}, MAC = {}", srcIp, srcMac);
            //         }

            //         if (macTable.get(dstIp) == null){
            //             log.info("TABLE MISS on IP = {}", dstIp);
            //             flood_to_all(ethPkt, recDevId, recPort);
            //         } else {
            //             log.info("TABLE HIT. Requested MAC = {}", macTable.get(dstIp));
            //             controller_reply(ethPkt, dstIp, macTable.get(dstIp), recDevId, recPort);
            //         }
            //     }
            //     else if (arpPacket.getOpCode() == ARP.OP_REPLY) {
            //         if (macTable.get(srcIp) == null) {
            //             macTable.put(srcIp, srcMac);
            //             log.info("Add new entry from request. IP = {}, MAC = {}", srcIp, srcMac);
            //         }
            //         log.info("Add new entry from reply. IP = {}, MAC = {}", srcIp, srcMac);

            //     }
            //     context.block();
            //     return;
            // }
        */

            if (ethType == Ethernet.TYPE_IPV4) {
                MacAddress srcMac = ethPkt.getSourceMAC();
                Ip4Address srcIp = Ip4Address.valueOf(((IPv4) ethPkt.getPayload()).getSourceAddress());
                MacAddress dstMac = ethPkt.getDestinationMAC();
                Ip4Address dstIp = Ip4Address.valueOf(((IPv4) ethPkt.getPayload()).getDestinationAddress());
                if (bridgeTable.get(recDevId) == null) {
                    bridgeTable.put(recDevId, new HashMap<>());
                }


                bridgeTable.get(recDevId).put(Pair.of(srcMac, srcIp), recPort);
                // if (oldPort != null){
                //     log.info("port of {} {} changed from {} to {} in {}", srcMac, srcIp, oldPort, recPort, recDevId);
                // }
                // if (bridgeTable.get(recDevId).get(Pair.of(srcMac, srcIp)) == null) {
                //     // the mapping of pkt's src mac and receivedfrom port wasn't store in the table of the rec device
                //     log.info("Add an entry to the port table of `{}`. MAC address: `{}`, IPv4 Address: `{}` => Port: `{}`.",
                //             recDevId, srcMac, srcIp, recPort);
                //     bridgeTable.get(recDevId).put(Pair.of(srcMac, srcIp), recPort);
                // }

                Boolean installed = buildMacChange(srcMac, dstMac, srcIp, dstIp, recDevId, recPort);

                if (installed) {
                    context.block();
                    return;
                }
                if (bridgeTable.get(recDevId).get(Pair.of(dstMac, dstIp)) == null) {
                    // the mapping of dst mac and forwarding port wasn't store in the table of the rec device
                    flood(context, dstMac, recDevId);
                } 
                else {
                    // there is a entry store the mapping of dst mac and forwarding port
                    installRule(context, srcMac, dstMac, srcIp, dstIp, recDevId, bridgeTable.get(recDevId).get(Pair.of(dstMac, dstIp)));
                    packetOut(context, bridgeTable.get(recDevId).get(Pair.of(dstMac, dstIp)));
                }



                context.block();
            }


            if (ethType == Ethernet.TYPE_IPV6) {
                MacAddress srcMac = ethPkt.getSourceMAC();
                Ip6Address srcIp = Ip6Address.valueOf(((IPv6) ethPkt.getPayload()).getSourceAddress());
                MacAddress dstMac = ethPkt.getDestinationMAC();
                Ip6Address dstIp = Ip6Address.valueOf(((IPv6) ethPkt.getPayload()).getDestinationAddress());

                log.info("111111111111111111111");
                log.info("srcIp: {}, srcMac: {}\ndstIp: {}, dstMac: {}", srcIp, srcMac, dstIp, dstMac);
                log.info("111111111111111111111");

                if (bridgeTable6.get(recDevId) == null) {
                    log.info("Got new recDev in ip6");
                    bridgeTable6.put(recDevId, new HashMap<>());
                }

                if (bridgeTable6.get(recDevId).get(Pair.of(srcMac, srcIp)) == null) {
                    // the mapping of pkt's src mac and receivedfrom port wasn't store in the table of the rec device
                    log.info("Add an entry to the port table of `{}`. MAC address: `{}`, IPv6 Address: `{}` => Port: `{}`.",
                            recDevId, srcMac, srcIp, recPort);
                    bridgeTable6.get(recDevId).put(Pair.of(srcMac, srcIp), recPort);
                }

                if (bridgeTable6.get(recDevId).get(Pair.of(dstMac, dstIp)) == null) {
                    // the mapping of dst mac and forwarding port wasn't store in the table of the rec device
                    log.info("flood6");
                    flood(context, dstMac, recDevId);
    
                } 
                else if (bridgeTable6.get(recDevId).get(Pair.of(dstMac, dstIp)) != null) {
                    // there is a entry store the mapping of dst mac and forwarding port
                    log.info("installRule6");
                    installRule6(context, srcMac, dstMac, srcIp, dstIp, recDevId, bridgeTable6.get(recDevId).get(Pair.of(dstMac, dstIp)));
                    packetOut(context, bridgeTable6.get(recDevId).get(Pair.of(dstMac, dstIp)));
                }
                log.info("buildMac6Change");
                // installRule6(context, srcMac, dstMac, srcIp, dstIp, recDevId, bridgeTable6.get(recDevId).get(Pair.of(dstMac, dstIp)));
                buildMac6Change(srcMac, dstMac, srcIp, dstIp, recDevId);
                    
                context.block();
            }

        }
    }

    private void flood_to_all(Ethernet ethPkt, DeviceId devID, PortNumber inPort) {
        // all devices
        for (ConnectPoint edgePort : edgePortService.getEdgePoints()) {
            DeviceId outDevID = edgePort.deviceId();
            PortNumber outPort = edgePort.port();
    
            // Check if the current edge port is not the receiving port
            if (!(outDevID.equals(devID) && outPort.equals(inPort))) {
                // send packet to edge ports only
                TrafficTreatment treatment = DefaultTrafficTreatment.builder()
                        .setOutput(outPort)
                        .build();
    
                // build and send the outbound packet
                OutboundPacket outboundPacket = new DefaultOutboundPacket(
                        outDevID, treatment, ByteBuffer.wrap(ethPkt.serialize())
                );
                packetService.emit(outboundPacket);
            }
        }
    }

    public void printMap(Map<Ip4Address, MacAddress> map) {
        // Print the entire map using log
        // Loop through the map and print each entry
        for (Map.Entry<Ip4Address, MacAddress> entry : map.entrySet()) {
            log.info("Key: {}, Value: {}", entry.getKey(), entry.getValue());
        }
    }

    private void controller_reply(Ethernet ethPkt, Ip4Address dstIP, MacAddress dstMac,
                                 DeviceId devID, PortNumber outPort) {

        // create Ethernet frame for ARP reply
        Ethernet ethReply = ARP.buildArpReply(dstIP, dstMac, ethPkt);

        // set port
        TrafficTreatment treatment = DefaultTrafficTreatment.builder()
                .setOutput(outPort)
                .build();
        
        // send to devices
        OutboundPacket outboundPacket = new DefaultOutboundPacket(
                devID,
                treatment,
                ByteBuffer.wrap(ethReply.serialize())
        );
        packetService.emit(outboundPacket);
        
    }


    private void flood(PacketContext context, MacAddress dstMac, DeviceId recDevId) {
        // log.info("MAC address `{}` is missed on `{}`. Flood the packet.", dstMac, recDevId);
        packetOut(context, PortNumber.FLOOD);
    }

    private void packetOut(PacketContext context, PortNumber portNumber) {
        context.treatmentBuilder().setOutput(portNumber);
        context.send();
    }

    private MacAddress getPeerMac(Ip4Address IP) {
        // find peer's ipv4 address
        NameConfig config = cfgService.getConfig(appId, NameConfig.class);
        List<String> v4Peers = config.v4Peers();

        for (int i = 0; i < v4Peers.size(); i+=2) {
            if (IP.toString().equals(v4Peers.get(i))) {
                return arpTable.get(Ip4Address.valueOf(v4Peers.get(i+1)));
            }
            else if (IP.toString().equals(v4Peers.get(i+1))) {
                return arpTable.get(Ip4Address.valueOf(v4Peers.get(i)));
            }
        }
        return null;
    }

    private MacAddress getPeerMac6(Ip6Address IP) {
        // find peer's ipv4 address
        NameConfig config = cfgService.getConfig(appId, NameConfig.class);
        List<String> v6Peers = config.v6Peers();

        for (int i = 0; i < v6Peers.size(); i+=2) {
            if (interfaceService.getMatchingInterface(IP).equals(interfaceService.getMatchingInterface(IpAddress.valueOf(v6Peers.get(i))))) {
                return macTable6.get(Ip6Address.valueOf(v6Peers.get(i+1)));
            }
            else if (interfaceService.getMatchingInterface(IP).equals(interfaceService.getMatchingInterface(IpAddress.valueOf(v6Peers.get(i+1))))) {
                return macTable6.get(Ip6Address.valueOf(v6Peers.get(i)));
            }
        }
        return null;
    }

    private void installMacChanging(Ip4Prefix srcIp, Ip4Prefix dstIp, MacAddress srcMac, MacAddress dstMac, ConnectPoint srcCP, ConnectPoint dstCP) {
        TrafficSelector.Builder selectorBuilder = DefaultTrafficSelector.builder()
            .matchEthType(Ethernet.TYPE_IPV4)
            .matchIPDst(dstIp)
            .matchIPSrc(srcIp);

        TrafficTreatment treatment = DefaultTrafficTreatment.builder()
            .setEthSrc(srcMac)
            .setEthDst(dstMac)
            .build();

        FilteredConnectPoint fsrc = new FilteredConnectPoint(srcCP);
        FilteredConnectPoint fdst = new FilteredConnectPoint(dstCP);

        PointToPointIntent intent = PointToPointIntent.builder()
            .appId(appId)
            .key(Key.of(srcIp.toString() + "-" + dstIp.toString(), appId))
            .selector(selectorBuilder.build())
            .filteredIngressPoint(fsrc)
            .filteredEgressPoint(fdst)
            .priority(39998)
            .treatment(treatment)
            .build();

        intentService.submit(intent);
    }

    private void installMac6Changing(Ip6Prefix srcIp, Ip6Prefix dstIp, MacAddress srcMac, MacAddress dstMac, ConnectPoint srcCP, ConnectPoint dstCP) {
        TrafficSelector.Builder selectorBuilder = DefaultTrafficSelector.builder()
            .matchEthType(Ethernet.TYPE_IPV6)
            .matchIPv6Dst(dstIp)
            .matchIPv6Src(srcIp);

        TrafficTreatment treatment = DefaultTrafficTreatment.builder()
            .setEthSrc(srcMac)
            .setEthDst(dstMac)
            .build();

        FilteredConnectPoint fsrc = new FilteredConnectPoint(srcCP);
        FilteredConnectPoint fdst = new FilteredConnectPoint(dstCP);

        PointToPointIntent intent = PointToPointIntent.builder()
            .appId(appId)
            .key(Key.of(srcIp.toString() + "-" + dstIp.toString(), appId))
            .selector(selectorBuilder.build())
            .filteredIngressPoint(fsrc)
            .filteredEgressPoint(fdst)
            .priority(39998)
            .treatment(treatment)
            .build();

        intentService.submit(intent);
    }


    private ConnectPoint getIpConnectPoint(Ip4Address ip) {
        for (Host host : hostService.getHostsByIp(ip)) {
            return host.location();
        }
        return null;
    }

    private ConnectPoint getIp6ConnectPoint(Ip6Address ip) {
        for (Host host : hostService.getHostsByIp(ip)) {
            return host.location();
        }
        return null;
    }

    private Boolean buildMacChange(MacAddress srcMac, MacAddress dstMac, Ip4Address srcIp, Ip4Address dstIp, DeviceId recDevId, PortNumber recPort) {
        NameConfig config = cfgService.getConfig(appId, NameConfig.class);

        Collection<RouteInfo> routes = routeService.getRoutes(new RouteTableId("ipv4"));
        routeTable.clear();
        for (RouteInfo route : routes) {
            for (ResolvedRoute resRoute : route.allRoutes()) {
                routeTable.put(resRoute.prefix().getIp4Prefix(), resRoute.nextHop().getIp4Address());
            }
        }

        Boolean ipFromOther = false, ipToOther = false;
        Ip4Address srcIpOther = null, dstIpOther = null;
        Ip4Prefix srcPrefixOther = null, dstPrefixOther = null;

        log.info("[Function Called] buildMacChange:964");
        log.info("DST IP: {}", dstIp);



        // check if ip from/to other
        for (Map.Entry<Ip4Prefix, Ip4Address> entry: routeTable.entrySet()){
            Ip4Prefix prefix = entry.getKey();
            log.info("Prefix: {}", prefix);
            if (prefix.contains(srcIp)) {
                ipFromOther = true;
                log.info("src IP: {}", srcIp);
                srcIpOther = entry.getValue();
                srcPrefixOther = prefix;
            }
            if (prefix.contains(dstIp)) {
                ipToOther = true;
                log.info("dst IP: {}", dstIp);
                dstIpOther = entry.getValue();
                dstPrefixOther = prefix;
            }
        }

        if (ipFromOther) {
            log.info("IP from other: {}", srcIpOther);
            log.info("Real dst IP = {}", srcIp);
        }

        if (ipToOther) {
            log.info("IP to other: {}", dstIpOther);
            log.info("Real dst IP = {}", dstIp);
        }

        if (ipFromOther) {
            if (ipToOther) {
                // cross domain
                MacAddress newSrcMac = getPeerMac(dstIpOther);
                MacAddress newDstMac = arpTable.get(dstIpOther);

                ConnectPoint srcCP = new ConnectPoint(recDevId, recPort);
                ConnectPoint dstCP = getIpConnectPoint(dstIpOther);

                if (srcCP == null || dstCP == null) {
                    return false;
                }

                installMacChanging(srcPrefixOther, dstPrefixOther, newSrcMac, newDstMac, srcCP, dstCP);
            }
            else {
                // other -> local
                MacAddress newSrcMac = MacAddress.valueOf(config.gatewayMac());
                MacAddress newDstMac = arpTable.get(dstIp);
                if (newDstMac == null){
                    return false;
                }
                ConnectPoint srcCP = new ConnectPoint(recDevId, recPort);
                ConnectPoint dstCP = getIpConnectPoint(dstIp);

                if (srcCP == null || dstCP == null) {
                    return false;
                }
                log.info("srcPrefixOther: {}", srcPrefixOther);
                log.info("dstPrefixOther: {}", dstIp.toIpPrefix().getIp4Prefix());
                log.info("newSrcMac: {}", newSrcMac);
                log.info("newDstMac: {}", newDstMac);
                log.info("srcCP: {}", srcCP);
                log.info("dstCP: {}", dstCP);
                
                installMacChanging(srcPrefixOther, dstIp.toIpPrefix().getIp4Prefix(), newSrcMac, newDstMac, srcCP, dstCP);
            }
            return true;
        }
        else {
            if (ipToOther) {
                // local -> other
                MacAddress newSrcMac = getPeerMac(dstIpOther);
                MacAddress newDstMac = arpTable.get(dstIpOther);
                
                ConnectPoint srcCP = new ConnectPoint(recDevId, recPort);
                ConnectPoint dstCP = getIpConnectPoint(dstIpOther);

                if (srcCP == null || dstCP == null) {
                    return false;
                }

                installMacChanging(srcIp.toIpPrefix().getIp4Prefix(), dstPrefixOther, newSrcMac, newDstMac, srcCP, dstCP);
                return true;
            }
        }
        return false;
    }


    private Boolean buildMac6Change(MacAddress srcMac, MacAddress dstMac, Ip6Address srcIp, Ip6Address dstIp, DeviceId recDevId) {
        NameConfig config = cfgService.getConfig(appId, NameConfig.class);

        Collection<RouteInfo> routes = routeService.getRoutes(new RouteTableId("ipv6"));
        routeTable6.clear();
        for (RouteInfo route : routes) {
            for (ResolvedRoute resRoute : route.allRoutes()) {
                routeTable6.put(resRoute.prefix().getIp6Prefix(), resRoute.nextHop().getIp6Address());
            }
        }

        // 遍歷 routeTable6
        log.info("************");
        log.info("I want to find srcip:{} dstip:{}", srcIp, dstIp);
        log.info("Travel routeTable6");
        for (Map.Entry<Ip6Prefix, Ip6Address> entry : routeTable6.entrySet()) {
            Ip6Prefix prefix = entry.getKey();
            Ip6Address nextHop = entry.getValue();
            log.info("Prefix: {}, Next Hop: {}", prefix, nextHop);
        }
        log.info("************");

        Boolean ipFromOther = false, ipToOther = false;
        Ip6Address srcIpOther = null, dstIpOther = null;
        Ip6Prefix srcPrefixOther = null, dstPrefixOther = null;

        if (ipFromOther) {
            log.info("IP from other: {}", srcIpOther);
            log.info("Real dst IP = {}", srcIp);
        }

        if (ipToOther) {
            log.info("IP to other: {}", dstIpOther);
            log.info("Real dst IP = {}", dstIp);
        }

        // check if ip from/to other
        for (Map.Entry<Ip6Prefix, Ip6Address> entry: routeTable6.entrySet()){
            Ip6Prefix prefix = entry.getKey();
            log.info("Prefix: {}", prefix);
            if (prefix.contains(srcIp)) {
                ipFromOther = true;
                log.info("src IP: {}", srcIp);
                srcIpOther = entry.getValue();
                srcPrefixOther = prefix;
            }
            if (prefix.contains(dstIp)) {
                ipToOther = true;
                log.info("dst IP: {}", dstIp);
                dstIpOther = entry.getValue();
                dstPrefixOther = prefix;
            }
        }

        if (ipFromOther) {
            if (ipToOther) {
                // cross domain
                MacAddress newSrcMac = getPeerMac6(dstIpOther);
                MacAddress newDstMac = macTable6.get(dstIpOther);
                // MacAddress newDstMac = macTable6.get(dstIpOther);

                ConnectPoint srcCP = getIp6ConnectPoint(srcIpOther);
                ConnectPoint dstCP = getIp6ConnectPoint(dstIpOther);

                log.info("+++++++++++");
                log.info("From other but To other: srcCp:{}/dstCp:{}", srcCP, dstCP);
                log.info("+++++++++++");

                if (srcCP == null || dstCP == null || dstIpOther == null) {
                    
                    return false;
                }

                installMac6Changing(srcPrefixOther, dstPrefixOther, newSrcMac, newDstMac, srcCP, dstCP);
            }
            else {
                // other -> local
                log.info("From other, not to other");

                MacAddress newSrcMac = MacAddress.valueOf(config.gatewayMac());
                MacAddress newDstMac = macTable6.get(dstIp);
                if (newDstMac == null){
                    log.info("There is no newDstMac");
                    return false;
                }
                ConnectPoint srcCP = getIp6ConnectPoint(srcIpOther);
                ConnectPoint dstCP = getIp6ConnectPoint(dstIp);

                log.info("+++++++++++");
                log.info("From other but Not to other: srcCp:{}/dstCp:{}", srcCP, dstCP);
                log.info("+++++++++++");

                if (srcCP == null || dstCP == null || newDstMac == null) {
                    return false;
                }
                
                installMac6Changing(srcPrefixOther, dstIp.toIpPrefix().getIp6Prefix(), newSrcMac, newDstMac, srcCP, dstCP);
            }
            return true;
        }
        else {
            if (ipToOther) {
                // local -> other
                MacAddress newSrcMac = getPeerMac6(dstIpOther);
                MacAddress newDstMac = macTable6.get(dstIpOther);
                // MacAddress newDstMac = macTable6.get(dstIpOther);
                
                ConnectPoint srcCP = getIp6ConnectPoint(srcIp);
                ConnectPoint dstCP = getIp6ConnectPoint(dstIpOther);

                log.info("+++++++++++");
                log.info("Not from other: srcCp:{}/dstCp:{}", srcCP, dstCP);
                log.info("+++++++++++");
                log.info("srcIp:{}, dstIp:{}", srcIp, dstIp);
                log.info("newSrcMac:{}, newDstMac:{}", newSrcMac, newDstMac);
                log.info("srcCP:{}, dstCP:{}", srcCP, dstCP);

                if (srcCP == null || dstCP == null || newDstMac == null) {
                    return false;
                }

                installMac6Changing(srcIp.toIpPrefix().getIp6Prefix(), dstPrefixOther, newSrcMac, newDstMac, srcCP, dstCP);
                return true;
            }
        }
        return false;
    }

    private void installRule(PacketContext context, MacAddress srcMac, MacAddress dstMac, Ip4Address srcIp, Ip4Address dstIp,
                             DeviceId recDevId, PortNumber outPort) {
        log.info("MAC address `{}` is matched on `{}`. Install a flow rule.", dstMac, recDevId);

        // buildMacChange(srcMac, dstMac, srcIp, dstIp, recDevId);
        TrafficSelector.Builder selectorBuilder = DefaultTrafficSelector.builder()
                    .matchEthType(Ethernet.TYPE_IPV4)
                    .matchEthDst(dstMac)
                    .matchEthSrc(srcMac)
                    .matchIPSrc(srcIp.toIpPrefix())
                    .matchIPDst(dstIp.toIpPrefix());

        TrafficTreatment treatment = DefaultTrafficTreatment.builder()
                    .setOutput(outPort)
                    .build();

        ForwardingObjective forwardingObjective = DefaultForwardingObjective.builder()
                    .withSelector(selectorBuilder.build())
                    .withTreatment(treatment)
                    .withPriority(30)
                    .withFlag(ForwardingObjective.Flag.VERSATILE)
                    .fromApp(appId)
                    .makeTemporary(300)
                    .add();

        flowObjectiveService.forward(recDevId, forwardingObjective);

        packetOut(context, outPort);
    }


    private void installRule6(PacketContext context, MacAddress srcMac, MacAddress dstMac, Ip6Address srcIp, Ip6Address dstIp,
                             DeviceId recDevId, PortNumber outPort) {
        log.info("MAC address `{}` is matched on `{}`. Install a flow rule.", dstMac, recDevId);

        // buildMac6Change(srcMac, dstMac, srcIp, dstIp, recDevId);
        TrafficSelector.Builder selectorBuilder = DefaultTrafficSelector.builder()
                    .matchEthType(Ethernet.TYPE_IPV6)
                    .matchEthDst(dstMac)
                    .matchEthSrc(srcMac)
                    .matchIPv6Dst(dstIp.toIpPrefix())
                    .matchIPv6Src(srcIp.toIpPrefix());

        TrafficTreatment treatment = DefaultTrafficTreatment.builder()
                    .setOutput(outPort)
                    .build();

        ForwardingObjective forwardingObjective = DefaultForwardingObjective.builder()
                    .withSelector(selectorBuilder.build())
                    .withTreatment(treatment)
                    .withPriority(30)
                    .withFlag(ForwardingObjective.Flag.VERSATILE)
                    .fromApp(appId)
                    .makeTemporary(300)
                    .add();

        flowObjectiveService.forward(recDevId, forwardingObjective);

        packetOut(context, outPort);
    }

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

}