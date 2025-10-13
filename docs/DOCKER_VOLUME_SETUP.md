# Docker Volume Configuration for Local File Access

本文檔說明如何配置 Docker 以訪問本地電腦上的任何資料夾。

## 概述

LLM Evaluation Hub 需要訪問本地資料夾來：
1. 讀取輸入文檔
2. 寫入生成的 personas、documents、testsets
3. 保存評估結果

## 配置方法

### 1. Mac / Linux 用戶

在 `docker-compose.yml` 中取消註解以下行：

```yaml
volumes:
  # ...其他 volumes...
  - $HOME:/app/host_files
```

這將掛載你的整個用戶目錄到容器的 `/app/host_files`。

**路徑映射示例**：
- 本地路徑: `/Users/username/Documents/my_project`
- 容器路徑: `/app/host_files/Documents/my_project`

**使用方式**：
在 API 請求中，使用容器路徑：
```json
{
  "output_folder": "/app/host_files/Documents/my_project/output"
}
```

### 2. Windows 用戶

在 `docker-compose.yml` 中取消註解並修改以下行：

```yaml
volumes:
  # ...其他 volumes...
  - D:/:/app/host_d_drive
```

將 `D:/` 替換為你要掛載的磁碟機代號。

**路徑映射示例**：
- 本地路徑: `D:\workplace\project\output`
- 容器路徑: `/app/host_d_drive/workplace/project/output`

**使用方式**：
在 API 請求中，使用容器路徑：
```json
{
  "output_folder": "/app/host_d_drive/workplace/project/output"
}
```

### 3. 自訂資料夾掛載

如果你只想掛載特定資料夾，可以添加自訂掛載：

```yaml
volumes:
  # ...其他 volumes...
  - /path/to/your/folder:/app/custom_data
```

**範例**：
```yaml
volumes:
  - /Users/jeff/Projects/data:/app/project_data
  - D:/MyData:/app/my_data
```

## 內建掛載點

系統已經預設掛載了以下目錄：

| 容器路徑 | 本地路徑 | 用途 |
|---------|---------|------|
| `/app/data` | `./data` | 測試文檔 |
| `/app/results` | `./results` | 評估結果 |
| `/app/artifacts` | `./artifacts` | 模型 artifacts |
| `/app/outputs` | `./outputs` | 輸出文件 |
| `/app/host_github` | `.` (專案根目錄) | 專案代碼 |

## 路徑轉換邏輯

系統會自動處理路徑轉換：

### Windows 路徑轉換
```python
# 輸入: D:\workplace\project_management\my_github\data
# 輸出: /app/host_github/data
```

### Mac/Linux 路徑轉換
```python
# 輸入: /Users/jeff/project/data
# 輸出: /app/host_files/project/data
```

## 使用建議

### 方案 1: 使用專案內部路徑（推薦用於開發）

```json
{
  "output_folder": "./outputs/my_scenario"
}
```

這會將文件保存到專案的 `outputs` 目錄。

### 方案 2: 使用絕對路徑（推薦用於生產）

#### Mac/Linux:
```json
{
  "output_folder": "/Users/jeff/Documents/evaluation_results"
}
```

系統會自動映射到 `/app/host_files/Documents/evaluation_results`。

#### Windows:
```json
{
  "output_folder": "D:\\Projects\\evaluation_results"
}
```

系統會自動映射到 `/app/host_d_drive/Projects/evaluation_results`。

## 權限問題

### Mac/Linux

如果遇到權限問題，確保 Docker 有權限訪問該目錄：

1. 打開 Docker Desktop
2. Settings → Resources → File Sharing
3. 添加要掛載的目錄

### Windows

1. 打開 Docker Desktop
2. Settings → Resources → File Sharing
3. 添加要掛載的磁碟機或目錄

## 安全性考慮

1. **最小權限原則**: 只掛載必要的目錄
2. **避免掛載根目錄**: 不要掛載 `/` 或 `C:\`
3. **使用唯讀掛載**: 對於輸入文件，可以使用唯讀模式：
   ```yaml
   - /path/to/input:/app/input:ro
   ```

## 故障排除

### 問題: 找不到文件或目錄

**解決方案**:
1. 確認 docker-compose.yml 中已正確配置 volume
2. 重啟 Docker 容器: `docker-compose down && docker-compose up -d`
3. 檢查路徑映射是否正確

### 問題: 權限被拒絕

**解決方案**:
1. 確認 Docker Desktop 有權限訪問該目錄
2. Mac/Linux: 檢查目錄權限 `ls -la`
3. Windows: 確認 Docker Desktop 在 Settings 中有該磁碟機的訪問權限

### 問題: Windows 路徑格式錯誤

**常見錯誤**:
```json
{
  "output_folder": "D:\workplace\project"  // 錯誤：反斜杠需要轉義
}
```

**正確格式**:
```json
{
  "output_folder": "D:\\workplace\\project"  // 正確：雙反斜杠
}
```

或使用正斜杠：
```json
{
  "output_folder": "D:/workplace/project"  // 也正確
}
```

## 範例配置

### 完整的 docker-compose.yml volumes 配置

```yaml
services:
  app:
    # ...其他配置...
    volumes:
      # 專案內部目錄
      - ./data:/app/data
      - ./results:/app/results
      - ./artifacts:/app/artifacts
      - ./outputs:/app/outputs
      - .:/app/host_github
      
      # 用戶目錄掛載 (Mac/Linux)
      - $HOME:/app/host_files
      
      # Windows 磁碟機掛載
      - D:/:/app/host_d_drive
      
      # 自訂目錄掛載
      - /Users/jeff/Projects/ML:/app/ml_projects
      - D:/MyData:/app/my_data:ro  # 唯讀
```

## 更新配置後

每次修改 `docker-compose.yml` 後，需要重啟容器：

```bash
# 停止並移除容器
docker-compose down

# 重新啟動
docker-compose up -d

# 查看日誌確認啟動成功
docker-compose logs -f app
```

## 相關文檔

- [Docker Volume Documentation](https://docs.docker.com/storage/volumes/)
- [Docker Compose Volumes](https://docs.docker.com/compose/compose-file/compose-file-v3/#volumes)
- [DOCKER_SETUP.md](../DOCKER_SETUP.md) - 完整 Docker 設置指南

