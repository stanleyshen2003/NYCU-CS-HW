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
package nycu.winlab.groupmeter;

import static org.onosproject.net.config.NetworkConfigEvent.Type.CONFIG_ADDED;
import static org.onosproject.net.config.NetworkConfigEvent.Type.CONFIG_UPDATED;
import static org.onosproject.net.config.basics.SubjectFactories.APP_SUBJECT_FACTORY;

import org.onosproject.core.ApplicationId;
import org.onosproject.core.CoreService;
import org.onosproject.net.config.ConfigFactory;
import org.onosproject.net.config.NetworkConfigEvent;
import org.onosproject.net.config.NetworkConfigListener;
import org.onosproject.net.config.NetworkConfigRegistry;
import org.onosproject.net.config.NetworkConfigService;

import org.onosproject.net.packet.PacketService;
import org.onosproject.net.packet.InboundPacket;
import org.onosproject.net.packet.DefaultOutboundPacket;
import org.onosproject.net.packet.PacketPriority;

import org.onosproject.net.flow.*;


import org.onosproject.net.PortNumber;
import org.onosproject.net.group.*;
import org.onosproject.net.meter.*;
import org.onosproject.net.intent.*;
import org.onosproject.net.DeviceId;
import org.onosproject.net.packet.PacketProcessor;
import org.onosproject.net.ConnectPoint;
import org.onosproject.net.FilteredConnectPoint;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.onosproject.net.packet.OutboundPacket;
import org.onlab.packet.Ethernet;
import org.onlab.packet.MacAddress;
import org.onlab.packet.Ip4Address;
import org.onlab.packet.ARP;

import org.onosproject.net.packet.PacketContext;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import javax.crypto.Mac;

import org.onosproject.core.GroupId;

/** Sample Network Configuration Service Application. **/
@Component(immediate = true)
public class AppComponent {

  private final Logger log = LoggerFactory.getLogger(getClass());
  private final NameConfigListener cfgListener = new NameConfigListener();


  private final ConfigFactory<ApplicationId, NameConfig> factory = new ConfigFactory<ApplicationId, NameConfig>(
      APP_SUBJECT_FACTORY, NameConfig.class, "informations") {
    @Override
    public NameConfig createConfig() {
      return new NameConfig();
    }
  };

  private ApplicationId appId;

  @Reference(cardinality = ReferenceCardinality.MANDATORY)
  protected NetworkConfigRegistry cfgService;

  @Reference(cardinality = ReferenceCardinality.MANDATORY)
  protected CoreService coreService;

  @Reference(cardinality = ReferenceCardinality.MANDATORY)
  protected GroupService groupService;

  @Reference(cardinality = ReferenceCardinality.MANDATORY)
  protected MeterService meterService;

  @Reference(cardinality = ReferenceCardinality.MANDATORY)
  protected PacketService packetService;

  @Reference(cardinality = ReferenceCardinality.MANDATORY)
  protected IntentService intentService;

  @Reference(cardinality = ReferenceCardinality.MANDATORY)
  protected FlowRuleService flowRuleService;

  private ARPProcessor processor = new ARPProcessor();
  private GroupKey groupKey = new DefaultGroupKey(ByteBuffer.allocate(4).putInt(1).array());
  private GroupId groupId = GroupId.valueOf(1);
  private MeterId meterId = MeterId.meterId(1);


  @Activate
  protected void activate() {
    appId = coreService.registerApplication("nycu.winlab.groupmeter");
    cfgService.addListener(cfgListener);
    cfgService.registerConfigFactory(factory);

    packetService.addProcessor(processor, PacketProcessor.director(2));
    TrafficSelector.Builder selector = DefaultTrafficSelector.builder();
    selector.matchEthType(Ethernet.TYPE_ARP);
    packetService.requestPackets(selector.build(), PacketPriority.CONTROL, appId);

    TrafficSelector.Builder ipv4Selector = DefaultTrafficSelector.builder();
    ipv4Selector.matchEthType(Ethernet.TYPE_IPV4);
    packetService.requestPackets(ipv4Selector.build(), PacketPriority.CONTROL, appId);

    createFailoverGroup();
    createMeter();

    // createP2PIntents();
    installFlowRule1();
    installFlowRule2();

    log.info("Started");
  }

  @Deactivate
  protected void deactivate() {
    cfgService.removeListener(cfgListener);
    cfgService.unregisterConfigFactory(factory);

    flowRuleService.removeFlowRulesById(appId);
    
    packetService.removeProcessor(processor);
    TrafficSelector.Builder selector = DefaultTrafficSelector.builder();
    selector.matchEthType(Ethernet.TYPE_ARP);
    packetService.cancelPackets(selector.build(), PacketPriority.CONTROL, appId);

    TrafficSelector.Builder ipv4Selector = DefaultTrafficSelector.builder();
    ipv4Selector.matchEthType(Ethernet.TYPE_IPV4);
    packetService.cancelPackets(ipv4Selector.build(), PacketPriority.CONTROL, appId);

    deleteFailoverGroup();
    deleteMeter();

    log.info("Stopped");
  }

  private void createP2PIntents() {
    NameConfig config = cfgService.getConfig(appId, NameConfig.class);

    TrafficSelector trafficSelector_h1 = DefaultTrafficSelector.builder()
            .matchEthDst(MacAddress.valueOf(config.mac1()))
            .build();
    TrafficSelector trafficSelector_h2 = DefaultTrafficSelector.builder()
            .matchEthDst(MacAddress.valueOf(config.mac2())) // Match packets destined for h2's MAC address
            .build();

    String[] parts = config.host2().split("/"); 
    String device2 = parts[0];
    Integer port2 = Integer.parseInt(parts[1]);
    createP2PIntent("of:0000000000000002", 1, device2, port2, trafficSelector_h2);
    createP2PIntent("of:0000000000000005", 3, device2, port2, trafficSelector_h2);

    parts = config.host1().split("/"); 
    String device1 = parts[0];
    Integer port1 = Integer.parseInt(parts[1]);
    createP2PIntent("of:0000000000000005", 1, "of:0000000000000005", 3, trafficSelector_h1);
    createP2PIntent("of:0000000000000004", 2, device1, port1, trafficSelector_h1);
  }

  private void createP2PIntent(String deviceId1,Integer port1, String deviceId2, Integer port2, TrafficSelector selector) {
    // Define the source and destination connect points
    ConnectPoint src = new ConnectPoint(DeviceId.deviceId(deviceId1), PortNumber.portNumber(port1));
    ConnectPoint dst = new ConnectPoint(DeviceId.deviceId(deviceId2), PortNumber.portNumber(port2));

    FilteredConnectPoint fsrc = new FilteredConnectPoint(src);
    FilteredConnectPoint fdst = new FilteredConnectPoint(dst);
    // Create the point-to-point intent
    PointToPointIntent p2pIntent = PointToPointIntent.builder()
            .appId(appId)
            .key(Key.of(deviceId1 +String.valueOf(port1) + "-" + deviceId2 + String.valueOf(port2), appId))
            .filteredIngressPoint(fsrc)
            .filteredEgressPoint(fdst)
            .selector(selector)
            .priority(40005)
            .build();

    // Submit the intent
    intentService.submit(p2pIntent);
    log.info("Intent `{}`, port `{}` => `{}`, port `{}` is submitted.", deviceId1, port1, deviceId2, port2);
  }

  private void createFailoverGroup(){
    DeviceId deviceId = DeviceId.deviceId("of:0000000000000001");  // Replace with actual device ID    

    // define treatment for each bucket
    TrafficTreatment treatmentBucket1 = DefaultTrafficTreatment.builder()
            .setOutput(PortNumber.portNumber(2))
            .build();
    TrafficTreatment treatmentBucket2 = DefaultTrafficTreatment.builder()
            .setOutput(PortNumber.portNumber(3))
            .build();

    // create the two buckets
    GroupBucket bucket1 = DefaultGroupBucket.createFailoverGroupBucket(treatmentBucket1, PortNumber.portNumber(2), null);
    GroupBucket bucket2 = DefaultGroupBucket.createFailoverGroupBucket(treatmentBucket2, PortNumber.portNumber(3), null);

    // add buckets to group
    GroupBuckets groupBuckets = new GroupBuckets(Arrays.asList(bucket1, bucket2));

    // create group description
    GroupDescription groupDescription = new DefaultGroupDescription(
            deviceId,
            GroupDescription.Type.FAILOVER,
            groupBuckets,
            groupKey,
            1,
            appId
    );

    // Add group to device
    groupService.addGroup(groupDescription);
  }

  private void deleteFailoverGroup(){
    DeviceId deviceId = DeviceId.deviceId("of:0000000000000001");
    groupService.removeGroup(deviceId, groupKey, appId);
  }

  private void createMeter() {
    DeviceId deviceId = DeviceId.deviceId("of:0000000000000004");

    // create band
    Band dropBand = DefaultBand.builder()
            .ofType(Band.Type.DROP)
            .withRate(512)  // Rate in KB_PER_SEC
            .burstSize(1024)  // Burst size in KB
            .build();

    // Create meter request
    MeterRequest meterRequest = DefaultMeterRequest.builder()
            .forDevice(deviceId)
            .fromApp(appId)
            .withUnit(Meter.Unit.KB_PER_SEC)
            .withBands(Collections.singleton(dropBand))
            .burst()
            .add();

    // Submit meter request to MeterService
    Meter temp = meterService.submit(meterRequest);
    meterId = temp.id();
  }

  private void deleteMeter() {
    DeviceId deviceId = DeviceId.deviceId("of:0000000000000004");

    // create band
    Band dropBand = DefaultBand.builder()
            .ofType(Band.Type.DROP)
            .withRate(512)  // Rate in KB_PER_SEC
            .burstSize(1024)  // Burst size in KB
            .build();

    // Create meter request
    MeterRequest meterRequest = DefaultMeterRequest.builder()
            .forDevice(deviceId)
            .fromApp(appId)
            .withUnit(Meter.Unit.KB_PER_SEC)
            .withBands(Collections.singleton(dropBand))
            .burst()
            .add();

    meterService.withdraw(meterRequest, meterId);
  }

  private class NameConfigListener implements NetworkConfigListener {
    @Override
    public void event(NetworkConfigEvent event) {
      if ((event.type() == CONFIG_ADDED || event.type() == CONFIG_UPDATED)
          && event.configClass().equals(NameConfig.class)) {
        NameConfig config = cfgService.getConfig(appId, NameConfig.class);
        if (config != null) {
          log.info("ConnectPoint_h1: {}, ConnectPoint_h2: {}", config.host1(), config.host2());
          log.info("MacAddress_h1: {}, MacAddress _h2: {}", config.mac1(), config.mac2());
          log.info("IpAddress_h1: {}, IpAddress_h2: {}", config.ip1(), config.ip2());
        }
      }
    }
  }

  private void installFlowRule1() {
    DeviceId recDevId = DeviceId.deviceId("of:0000000000000001");
    Group group = groupService.getGroup(recDevId, groupKey);
    // Create a TrafficSelector to match the source and destination MAC addresses
    TrafficSelector selector = DefaultTrafficSelector.builder()
            .matchInPort(PortNumber.portNumber(1))
            .matchEthType(Ethernet.TYPE_IPV4)
            .build();

    // Create a TrafficTreatment to define the output action
    TrafficTreatment treatment = DefaultTrafficTreatment.builder()
            .group(group.id())
            .build();

    // Create a FlowRule to specify the behavior on matching packets
    FlowRule flowRule = DefaultFlowRule.builder()
            .forDevice(recDevId)
            .withSelector(selector)
            .withTreatment(treatment)
            .withPriority(40005) // Set the priority of the flow rule
            .fromApp(appId) // Specify the application ID
            .makePermanent() // Set the flow rule to be permanent
            .build();

    flowRuleService.applyFlowRules(flowRule);
  }

  private void installFlowRule2() {
    DeviceId recDevId = DeviceId.deviceId("of:0000000000000004");
    NameConfig config = cfgService.getConfig(appId, NameConfig.class);
    TrafficSelector selector = DefaultTrafficSelector.builder()
            .matchEthSrc(MacAddress.valueOf(config.mac1()))
            .build();

    // Create a TrafficTreatment that specifies the actions to take
    TrafficTreatment treatment = DefaultTrafficTreatment.builder()
            .setOutput(PortNumber.portNumber(2))
            .meter(meterId)
            .build();

    // Create a FlowRule to specify the behavior on matching packets
    FlowRule flowRule = DefaultFlowRule.builder()
            .forDevice(recDevId)
            .withSelector(selector)
            .withTreatment(treatment)
            .withPriority(40005)         // Set the priority of the flow rule
            .fromApp(appId)           // Specify the application ID
            .makePermanent()          // Make the flow rule permanent
            .build();

    // Install the flow rule using the FlowRuleService
    flowRuleService.applyFlowRules(flowRule);
  }



  private class ARPProcessor implements PacketProcessor {

    @Override
    public void process(PacketContext context) {
      
      if (context.isHandled()) {
        return;
      }

      NameConfig config = cfgService.getConfig(appId, NameConfig.class);

      InboundPacket pkt = context.inPacket();
      Ethernet ethPkt = pkt.parsed();

      PortNumber recPort = pkt.receivedFrom().port();
      DeviceId devID = pkt.receivedFrom().deviceId();
      if (ethPkt == null) {
          return;
      }

      if (ethPkt.getEtherType() == Ethernet.TYPE_ARP) {
        ARP arpPacket = (ARP) ethPkt.getPayload();
      
        // get payload
        Ip4Address dstIp = Ip4Address.valueOf(arpPacket.getTargetProtocolAddress());
  
        // if it is a request packet
        if (arpPacket.getOpCode() == ARP.OP_REQUEST){
  
          if (dstIp.equals(Ip4Address.valueOf(config.ip1()))) {
            controller_reply(ethPkt, dstIp, MacAddress.valueOf(config.mac1()), devID, recPort);
          } else {
            controller_reply(ethPkt, dstIp, MacAddress.valueOf(config.mac2()), devID, recPort);
          }
        }
      }
      else {
        MacAddress dst_mac = ethPkt.getDestinationMAC();
        TrafficSelector trafficSelector = DefaultTrafficSelector.builder()
            .matchEthDst(dst_mac)
            .build();
        if (dst_mac.equals(MacAddress.valueOf(config.mac1()))){
          String[] parts = config.host1().split("/"); 
          String device1 = parts[0];
          Integer port1 = Integer.parseInt(parts[1]);
          createP2PIntent(devID.toString(), Integer.parseInt(recPort.toString()), device1, port1, trafficSelector);
        }
        else if (dst_mac.equals(MacAddress.valueOf(config.mac2()))){
          String[] parts = config.host2().split("/"); 
          String device2 = parts[0];
          Integer port2 = Integer.parseInt(parts[1]);
          createP2PIntent(devID.toString(), Integer.parseInt(recPort.toString()), device2, port2, trafficSelector);
        }
      }

      
      context.block();

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
  }
}

