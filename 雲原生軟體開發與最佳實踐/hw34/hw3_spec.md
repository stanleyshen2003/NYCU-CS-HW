# 雲原生 2025 - HW 3
使用 Github Import Repo 的功能，將 testing-lab repo 匯入自己的帳號底下，設為 Private 並開啟權限給講師 （ID: linroex），最後，請務必到 E3 HW3 填寫你的 Repo 連結

## 作業要求
- 替前端、後端撰寫 `Dockerfile`。 (10)
- 替該專案撰寫 `docker-compose.yaml`，使該專案功能可以正常運作。 (30)
    - `docker-compose.yaml` 的前端、後端，Image 來源不能使用任何 Container Registry 的 Image，而需要在 `docker compose up` 時即時、自動編譯 Image。
    - 資料庫的 Image 可以從 Container Registry 取得。
- 替該專案撰寫 K8s 相關設定檔，包含 Deployment、Service、Ingress 等，讓該專案功能可以在 K8s 正常運作。
    - 讓講師可以透過 VM IP (80 port) 連線到前端，並且相關功能（後端、資料庫）皆可正常運作。 (50)
    - 針對前、後端與 DB 加上 Liveness Probe。 (10)
    - 前後端的 Image 請從自己的 Github Container Registry 取得，會檢查 Image 來源是否和 Repo 的 Username 相同。
## 檔案配置
- 前後端的 `Dockerfile` 不指定位置，視需要安排即可。
- `docker-compose.yaml` 放置在專案根目錄。
- 請在專案根目錄建立 k8s 資料夾，K8s 相關檔案放置如下
    - `deploys.yaml`: 前、後端與資料庫的 Deployment、PVC、Config Map (若有需要)等皆放置在同一個檔案內。
    - `services.yaml`
    - `ingress.yaml`
```bash
.
├── README.md
├── docker-compose.yaml
├── backend
├── frontend
└── k8s
    ├── deploys.yaml
    ├── ingress.yaml
    └── services.yaml
```
## 評分方式
評分流程供同學參考，但給分以作業要求為準。

* 評分者會將你的 Github Repo Clone 到檢測伺服器。
* 評分者會執行進到專案根目錄，執行 `docker-compose up -d`
* 評分者會透過瀏覽器進到 `http://localhost:5173` 並測試相關功能是否正常運作
* 評分者會進到 `k8s` 資料夾，並依序執行如下指令，將專案部署到講師準備的 k3d v5.6.3 Server。
    1. `kubectl apply -f deploys.yaml`
    2. `kubectl apply -f services.yaml`
    3. `kubectl apply -f ingress.yaml`
* 評分者會在檢測伺服器上，透過瀏覽器進到 `http://{k3d server ip}` 來測試相關功能
    * 注意，沒有 Port Forwarding，並且是在 80 Port。
## 注意事項
* 程式碼可以依照需求略做修改，但不能改變大架構（需要有前端、後端、資料庫），也不能改變程式行為、API、使用者流程與操作界面等等。
* 講師會隨機抽檢程式碼內容，若有不符合作業要求，或是作弊嫌疑，會再和同學確認（例如： 把後端 Mock 掉）