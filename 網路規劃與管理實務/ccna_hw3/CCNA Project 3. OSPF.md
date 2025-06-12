---
title: CCNA Project 3. OSPF

---

# CCNA Project 3. OSPF

![image](https://hackmd.io/_uploads/r1-mnf7aJe.png)

### 注意事項

- 寫作業的過程中建議定時存檔，以免 Packet Tracer 突然 crash
    - 特別是在 Simulation 模式下如果紀錄太多封包可能會因為記憶體用量過大而 crash，請多加注意
    - 定時存檔包括保存 pka 以及 switch 的設定
- 在 Packet Tracer 底下測試網路時，時常有「前幾次測試沒通，後來就通了」的情況
    - 可以嘗試多戳幾次或讓時間加速一點，避免被 Packet Tracer 雷
- 請勿增加任何不必要的設定以免影響自動評分腳本

:::danger
:warning: 某些部分在設定不當時會有**倒扣**情況，請同學仔細確認設定。
:::

### 設定要求
- OSPF `Process ID` 皆需設為 100。
- 請使用 `network` 指令宣告 OSPF 網段，不要使用 per-interface 設定。
- 請使用 `router-id` 指令設定 router id。
- 設定 DR/BDR interface priority，請在有標明要修改 priority 的 interface 上設定能符合要求的最大值。
    - 例如：假設設定 255 以及 254 皆可達成要求，那標準答案為 255。

----

## Basic Config

- 請根據拓樸來對每台 `switch` / `router` 進行 OSPF 網段宣告 (10%)
    - 可參考 `Appendix B`
    - Note: 只有在紅色區域內的 `switch` / `router` 有 `area 0` 設定
    - 只有 `CT-xxx-Core` & `YM-xxx-Core` 是 Area 0 的`ABR`

- 請根據 `Appendix A` 來完成 `router id` 的設定 (5%)

## Advance Config

- `NYCU IT` 與 google (8.8.8.8) 有連線，請利用 `network` 與 `per-interface` 以外的方式來讓其他裝置也能與 `8.8.8.8` 連線 (2.5%)

- 在現有拓樸不會改變的情況下，讓 `Area 10` 和 `Area 20` 中不在管轄範圍內的 device 無法收到 ospf hello packet (5%)
    - 看不到的設備都當不在管轄範圍內
    - Note: 需要用**最少的** config 數量來完成

- `Dorm B` 宿舍內的裝置過於老舊、僅支援 `RIP`，請寫出正確的 config 讓 `140.113.12.0` 能夠與外界聯繫 (10%)

- 請在 `CS-core1`, `CS-core2` 上進行設定，讓兩台設備在 `show ip ospf neighbor` 時沒有身份為 `DR` 的 neighbor (5%)
    - Note: 需要用**最少的** config 數量來完成

- 請運用你在上課中學到的知識來讓 
    - **所有 `CS-lab`, `CS-colo` 的流量**從 `CS-Core1` 進出 (2.5%)
    - **所有 `CS-intra` 的流量**從 `CS-Core2` 進出 (2.5%)
    - 此項會在 demo 時進行評分

- 請確保僅有 `140.113.150.0/24` 這個網段可以正常存取 `140.113.151.0/24` 上的服務 (2.5%)
    - `140.113.150.0/24` 以外的網段都無法存取 private 網段
    - ❗️請用 Routing 技巧完成，使用 ACL、NAT 不算分

- 在 `area 20` 中做設定使得 area 中的 ospf Database 有著最少資訊 (5%)

- 現在 `Area 20` 有一些錯誤設定，請修正他們並記錄下原因與解法 (5%)
    - demo 時將會有相關問題

- 請透過設定相關機器來讓 `Med-server-core` 能夠與 `YM-Dep-Core` 成為 ospf neighbor (5%)

### 連通性

除了 `private` 網段內的機器，其餘所有裝置都應該能夠相互連接
:::danger
:bulb: 評分時會**隨機測試**連通性，若是連通性出現錯誤將會予以扣分
:::

----

## 作業繳交
- 如果有任何對 Spec 有不清楚的地方需要請助教解釋或是需要助教幫忙，請以下方式擇一：
    - TA Time 時候來系計中（EC320）
    - 寄信到 npta@cs.nctu.edu.tw
- **如果你寄信到助教的私人信箱或 E3 信箱，那麼有可能會被忽略！**
- 本次作業僅需要上傳一個檔案至 E3 作業繳交區：
    -  pka 檔上傳（佔總分 60%）
        - 檔名請命名為 `HW3_<學號>.pka`
        - ex. `HW3_112550013.pka`
- 請確定你有保存 switch 的 config，demo 時我們將重開 switch
- Deadline: ==**2025/05/01 23:59**==
    - 允許在 Demo 前補交，分數將為 $$ (pka\ score) * 0.8^{\ late\ days}$$
- Demo: ==**2025/05/02**==
    - Demo 時將會問一些問題（佔總分 40%）
    - Schedule 將在之後公布

-----
### Appendix A.

:::spoiler **Router-id**
- NYCU IT: `140.113.0.1`
- YM-Dorm-Core: `140.113.0.2`
- Dorm A1: `140.113.0.11`
- Dorm B1: `140.113.0.12`
- YM-Dep-Core: `140.113.0.3`
- Meb-Lab1: `140.113.0.21`
- Med-Lab2: `140.113.0.22`
- Med-server-core: `140.113.0.23`
- CT-Dep-Core: `140.113.0.4`
- CS-core1: `140.113.0.31`
- CS-core2: `140.113.0.32`
- CS-Lab: `140.113.0.33`
- CS-colo: `140.113.0.34`
- CS-intra: `140.113.0.35`
- CT-Dorm-Core: `140.113.0.5`
- Dorm A: `140.113.0.41`
- Dorm B: `140.113.0.42`
:::

### Appendix B.

:::spoiler **Device Interface ip**
- NYCU IT
    - `1/0/1` : `10.0.0.1` | YM-Dorm-Core 
    - `1/0/2` : `10.0.1.1` | YM-Dep-Core
    - `1/0/3` : `10.0.2.1` | CT-Dep-Core
    - `1/0/4` : `10.0.3.1` | CT-Dorm-Core
    - `1/0/24` : `1.1.1.2/24` | google
- YM-Dorm-Core
    - `0/0` : `10.0.0.2` | NYCU IT
    - `0/1` : `10.1.1.1` | Dorm A1
    - `0/2` : `10.1.2.1` | Dorm B1
- Dorm A1
    - `0/0` : `10.1.1.2` | YM-Dorm-Core
    - `0/2` : `140.113.21.254` | edge
- Dorm B1
    - `0/0` : `10.1.2.2` | YM-Dorm-Core
    - `0/2` : `140.113.22.254` | edge
- YM-Dep-Core
    - `0/0` : `10.0.1.2` | NYCU IT
    - `0/1` : `10.2.1.1` | Meb-Lab1
    - `0/2` : `10.2.2.1` | Meb-Lab2
- Meb-Lab1
    - `0/0` : `10.2.1.2` | YM-Dep-Core
    - `0/1` : `10.2.3.1` | Med-server-core
    - `0/2` : `10.2.4.1` | Meb-Lab2
- Med-Lab2
    - `0/0` : `10.2.2.2` | YM-Dep-Core
    - `0/1` : `140.113.201.254` | edge
    - `0/2` : `10.2.4.2` | Meb-Lab1
- Med-server-core
    - `0/0` : `10.2.3.2` | Meb-Lab1
    - `0/1` : `140.113.251.254` | edge
    - `0/2` : `140.113.252.254` | edge
- CT-Dep-Core
    - `0/0` : `10.0.2.2` | NYCU IT
    - `0/1` : `10.3.1.1` | CS-core1
    - `0/2` : `10.3.2.1` | CS-core1
- CS-core1
    - `1/0/1` : `10.3.1.2` | CT-Dep-Core
    - `1/0/2` : `10.3.11.1` | CS-lab
    - `1/0/3` : `10.3.12.1` | CS-colo
    - `1/0/4` : `10.3.13.1` | CS-intra
- CS-core2
    - `1/0/1` : `10.3.2.2` | CT-Dep-Core
    - `1/0/2` : `10.3.21.1` | CS-lab
    - `1/0/3` : `10.3.22.1` | CS-colo
    - `1/0/4` : `10.3.23.1` | CS-intra
- CS-Lab
    - `0/0` : `10.3.11.2` | CS-core1
    - `0/1` : `10.3.21.2` | CS-core2
    - `0/2` : `140.113.110.254` | edge
- CS-colo
    - `0/0` : `10.3.12.2` | CS-core1
    - `0/1` : `10.3.22.2` | CS-core2
    - `0/2` : `140.113.120.254` | edge
- CS-intra
    - `1/0/1` : `10.3.13.2` | CS-core1
    - `1/0/2` : `10.3.23.2` | CS-core2
    - `1/0/3` : `140.113.150.254` | edge
    - `1/0/4` : `140.113.151.254` | edge
- CT-Dorm-Core
    - `0/0` : `10.0.3.2` | NYCU IT
    - `0/1` : `10.4.1.1` | Dorm A
    - `0/2` : `10.4.2.1` | Dorm B
- Dorm A
    - `0/0` : `10.4.1.2` | CT-Dorm-Core
    - `0/2` : `140.113.11.254` | edge
- Dorm B
    - `0/0` : `10.4.2.2` | CT-Dorm-Core
    - `0/2` : `10.4.3.1` | RIP
:::
