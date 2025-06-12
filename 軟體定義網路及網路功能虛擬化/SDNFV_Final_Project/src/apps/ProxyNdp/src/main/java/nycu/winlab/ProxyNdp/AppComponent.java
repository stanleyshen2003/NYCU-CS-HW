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
package nycu.winlab.ProxyNdp;

import org.onosproject.cfg.ComponentConfigService;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.nio.ByteBuffer;

import org.onosproject.core.ApplicationId;
import org.onosproject.core.CoreService;
import org.onosproject.net.flow.DefaultTrafficSelector;
import org.onosproject.net.flow.TrafficSelector;

import org.onosproject.net.flow.DefaultTrafficTreatment;
import org.onosproject.net.flow.TrafficTreatment;

import org.onosproject.net.flowobjective.FlowObjectiveService;

import org.onosproject.net.packet.PacketPriority;
import org.onosproject.net.packet.PacketService;
import org.onosproject.net.packet.PacketProcessor;
import org.onosproject.net.packet.PacketContext;
import org.onosproject.net.packet.DefaultOutboundPacket;
import org.onosproject.net.packet.InboundPacket;
import org.onosproject.net.packet.OutboundPacket;
import org.onlab.packet.Ethernet;
import org.onlab.packet.ICMP6;
import org.onlab.packet.MacAddress;
import org.onlab.packet.Ip6Address;
import org.onlab.packet.ARP;

import org.onosproject.net.PortNumber;
import org.onosproject.net.DeviceId;
import org.onosproject.net.Device;
import org.onosproject.net.Port;
import org.onosproject.net.ConnectPoint;


import org.onosproject.net.edge.EdgePortService;
import org.onosproject.net.device.DeviceService;
import org.onosproject.net.flow.FlowRuleService;

import java.util.Objects;
import java.util.Optional;
import java.util.stream.Stream;

import javax.naming.Context;

import org.onlab.packet.IPv6;
import org.onlab.packet.ndp.NeighborSolicitation;
import org.onlab.packet.ndp.NeighborAdvertisement;
import org.onlab.packet.IPacket;

/**
 * Skeletal ONOS application component.
 */
@Component(immediate = true)
public class AppComponent {

    private final Logger log = LoggerFactory.getLogger(getClass());

    /** Some configurable property. */

    @Reference(cardinality = ReferenceCardinality.MANDATORY)
    protected ComponentConfigService cfgService;

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

    private LearningBridgeProcessor processor = new LearningBridgeProcessor();
    private ApplicationId appId;
    private Map<Ip6Address, MacAddress> macTable = new HashMap<>();

    @Activate
    protected void activate() {

        // register your app
        appId = coreService.registerApplication("nycu.winlab.ProxyNdp");

        // add a packet processor to packetService
        packetService.addProcessor(processor, PacketProcessor.director(2));

        // install a flowrule for packet-in
        TrafficSelector.Builder selector = DefaultTrafficSelector.builder();
        selector.matchEthType(Ethernet.TYPE_IPV6);
        packetService.requestPackets(selector.build(), PacketPriority.REACTIVE, appId);

        TrafficSelector.Builder selector2 = DefaultTrafficSelector.builder();
        selector2.matchIcmpv6Type((byte)135);
        packetService.requestPackets(selector2.build(), PacketPriority.REACTIVE, appId);

        TrafficSelector.Builder selector3 = DefaultTrafficSelector.builder();
        selector3.matchIcmpv6Type((byte)136);
        packetService.requestPackets(selector3.build(), PacketPriority.REACTIVE, appId);


        log.info("Started");
    }

    @Deactivate
    protected void deactivate() {

        // remove flowrule installed by your app
        flowRuleService.removeFlowRulesById(appId);

        // remove your packet processor
        packetService.removeProcessor(processor);
        processor = null;

        // remove flowrule you installed for packet-in
        TrafficSelector.Builder selector = DefaultTrafficSelector.builder();
        selector.matchEthType(Ethernet.TYPE_IPV6);
        packetService.cancelPackets(selector.build(), PacketPriority.REACTIVE, appId);

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
        // log.info("ICMPv6 type: `{}`", icmpType);
        // log.info("reference: {}, {}", (byte)135, (byte)136);
        
        // Check if the ICMPv6 type is Neighbor Solicitation (135) or Neighbor Advertisement (136)
        if (icmpType == (byte)135) {
            return 1;
        } else if (icmpType == (byte)136) {
            return 2;
        }
        return 0;
        
    }

    private class LearningBridgeProcessor implements PacketProcessor {

        @Override
        public void process(PacketContext context) {
            // Stop processing if the packet has been handled, since we
            // can't do any more to it.
            if (context.isHandled()) {
                return;
            }
            InboundPacket pkt = context.inPacket();
            Ethernet ethPkt = pkt.parsed();

            Integer ndpType = findNDP(ethPkt);
            if (ndpType == 0) {
                return;
            }
            if (ndpType == 1){
                log.info("NDP SOLICITATION");
                processNDPSol(context, (NeighborSolicitation) ethPkt.getPayload().getPayload().getPayload());
            }
            if (ndpType == 2){
                log.info("NDP ADVERTISEMENT");
                processNDPAdv(context, (NeighborAdvertisement) ethPkt.getPayload().getPayload().getPayload());
            }
           


        }
    }

    private void processNDPSol(PacketContext context, NeighborSolicitation ndp) {
         // get payload
        Ethernet ethPkt = context.inPacket().parsed();
        IPv6 ipv6Packet = (IPv6) ethPkt.getPayload();
        DeviceId devID = context.inPacket().receivedFrom().deviceId();
        PortNumber recPort = context.inPacket().receivedFrom().port();
        Ip6Address srcIp = Ip6Address.valueOf(ipv6Packet.getSourceAddress());
        MacAddress srcMac = ethPkt.getSourceMAC();
        Ip6Address dstIp = Ip6Address.valueOf(ndp.getTargetAddress());

        // write the srcIP if it is not written
        if (macTable.get(srcIp) == null) {
            macTable.put(srcIp, srcMac);
        }


        if (macTable.get(dstIp) == null){
            log.info("TABLE MISS. Send request to edge ports");
            // log.info("table miss devID: {}, src {} / {}, dst: {}, {}", devID, srcIp, srcMac, dstIp, dstMac);
            flood(ethPkt, devID, recPort);
        } else {
            log.info("TABLE HIT. Requested MAC = {}", macTable.get(dstIp));
            // log.info("table hit devID: {}, src {} / {}, dst: {}, {}", devID, srcIp, srcMac, dstIp, dstMac);

            controller_reply(ethPkt, dstIp, macTable.get(dstIp), devID, recPort);
        }
    }

    private void processNDPAdv(PacketContext context, NeighborAdvertisement ndp) {
        // get payload
        Ethernet ethPkt = context.inPacket().parsed();
        IPv6 ipv6Packet = (IPv6) ethPkt.getPayload();
        Ip6Address srcIp = Ip6Address.valueOf(ipv6Packet.getSourceAddress());
        MacAddress srcMac = ethPkt.getSourceMAC();

        macTable.put(srcIp, srcMac);

    }

    private void flood(Ethernet ethPkt, DeviceId devID, PortNumber inPort) {
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


    private void controller_reply(Ethernet ethPkt, Ip6Address dstIP, MacAddress dstMac,
                                 DeviceId devID, PortNumber outPort) {
        log.info("Controller reply");
        log.info("dstIP: {}, dstMac: {}", dstIP, dstMac);
        log.info("devID: {}, outPort: {}", devID, outPort);
         // create Ethernet frame for ARP reply
        Ethernet ethReply = NeighborAdvertisement.buildNdpAdv(dstIP, dstMac, ethPkt);
        // flood(ethReply, devID, outPort);
        
        

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

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

}