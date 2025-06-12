# 雲原生 2025 - HW 4

請使用和 HW 3 相同的 Repo（但不需要完成 HW 3 也沒關係），替 HW 3 的 Repo 加上以下 Github Action 的功能：

## 作業要求

當 Github Repo 有新的 Commit 時 （CI）：
- 執行 Repo 內的單元測試，全部測試通過該 Job 才算成功，否則失敗。 (10)
- 產生測試覆蓋率報表，若本次覆蓋率低於上次的覆蓋率則該 Job 失敗，反之則成功。 (30)
    - 覆蓋率報表需要上傳至 Artifact 方便事後檢視。
- 使用任何 JavaScript Formatter 工具對本次 Commit 涵蓋的程式碼進行自動格式化。 (30)
    - 如果程式碼已經是格式化過的狀態，則不要再次 Format (也不要產生 Commit)。
    - 需要自動產生新的 Commit 來記錄格式化結果。

當 PR Merge 到 Main Branch 時（CD）：
- 將最新的程式碼打包成 Image 並發佈到 ghcr.io (30)

## 繳交方式

請將 Repo 的網址繳交至 E3 HW4。
若有完成自動佈署的部份，除了 Repo 網址外，請換行並提供 **Github Action cd.yaml 執行成功的網址**。

範例：
```
https://github.com/linroex/hw
https://github.com/linroex/hw/actions/runs/8909981547
```

## 檔案配置

請放在 Repo 的 .github/workflows 資料夾底下，並分成以下兩個檔案

- ci.yaml：當 Github 收到 Commit 時執行的 Jobs
- cd.yaml：當 PR Merge 到 Main Branch 時執行的 Jobs

## 評分方式

該作業將同時採用以下方式評分：

- 評分者會到你的 Repo 內提交一筆 PR，藉此測試 CI 功能正常運作以及符合要求。
- 評分者會人工檢視你的 ci.yaml 與 cd.yaml 以確認符合作業要求。

## 注意事項

- 雖然 Repo 網址和 HW 3 相同，但請還是將 Repo 網址再次繳交至 HW 4 的連結。
- 作弊的話 HW 4 會直接 0 分哦😆
    - 例如自動佈署的部份，如果發現你的執行成功記錄和你的 cd.yaml 完全不可能對應起來 ^__^
        - 例如 cd.yaml 執行根本不可能成功，但 Github Action 卻成功了
        - 這部分原則上不會雞蛋裡挑骨頭，單純的執行時錯誤導致自動佈署失敗會 Pass，這條只針對**明顯蓄意的作弊行為**。