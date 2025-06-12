/*
* Copyright 2020-present Open Networking Foundation
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
package nycu.winlab.vRouter;

import java.util.List;
import java.util.function.Function;

import org.onosproject.core.ApplicationId;
import org.onosproject.net.config.Config;

public class NameConfig extends Config<ApplicationId> {
    // Constants for configuration keys
    public static final String VR_ROUTING = "vrrouting";
    public static final String VR_ROUTING_MAC = "vrrouting-mac";
    public static final String GATEWAY_IP4 = "gateway-ip4";
    public static final String GATEWAY_IP6 = "gateway-ip6";
    public static final String GATEWAY_MAC = "gateway-mac";
    public static final String V4_PEERS = "v4-peers";
    public static final String V6_PEERS = "v6-peers";

    // Getter methods to retrieve each field within the configuration
    public String vrrouting() {
        return get(VR_ROUTING, null);
    }

    public String vrroutingMac() {
        return get(VR_ROUTING_MAC, null);
    }

    public String gatewayIp4() {
        return get(GATEWAY_IP4, null);
    }

    public String gatewayIp6() {
        return get(GATEWAY_IP6, null);
    }

    public List<String> v4Peers() {
        return getList(V4_PEERS, Function.identity(), null);
    }

    public List<String> v6Peers() {
        return getList(V6_PEERS, Function.identity(), null);
    }

    public String gatewayMac() {
        return get(GATEWAY_MAC, null);
    }

    @Override
    public boolean isValid() {
        // Validate if all required fields are present
        return hasOnlyFields(GATEWAY_MAC, VR_ROUTING, VR_ROUTING_MAC, GATEWAY_IP4, GATEWAY_IP6, V4_PEERS, V6_PEERS);
    }


}