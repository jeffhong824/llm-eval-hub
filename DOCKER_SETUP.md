# Docker Setup Guide

## 問題解決

### 1. Docker Desktop 未運行

**錯誤信息**：
```
error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.47/images/llm-eval-hub-app/json": open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

**解決方案**：
1. 啟動 Docker Desktop 應用程序
2. 等待 Docker Desktop 完全啟動（狀態欄顯示 "Docker Desktop is running"）
3. 重新運行命令

### 2. Docker Compose 版本警告

**錯誤信息**：
```
the attribute `version` is obsolete, it will be ignored
```

**解決方案**：
已修復！移除了 docker-compose.yml 中的 `version: '3.8'` 字段。

## 啟動選項

### 選項 1：使用 Docker Compose（推薦）

```bash
# 1. 確保 Docker Desktop 正在運行
# 2. 啟動服務
docker compose up -d

# 3. 查看日誌
docker compose logs -f

# 4. 停止服務
docker compose down
```

### 選項 2：本地開發（無需 Docker）

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 配置環境變量
cp env.example .env
# 編輯 .env 文件設定 API keys

# 3. 啟動應用
python start-local.py
```

### 選項 3：手動啟動

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 配置環境變量
cp env.example .env

# 3. 啟動 FastAPI
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 驗證安裝

### 檢查 Docker 狀態
```bash
docker --version
docker compose version
docker info
```

### 檢查應用狀態
```bash
# 訪問健康檢查
curl http://localhost:8000/health

# 訪問 Web 界面
# 瀏覽器打開：http://localhost:8000
```

## 常見問題

### Q: Docker Desktop 啟動失敗
A: 
1. 確保 Windows 功能中啟用了 Hyper-V 或 WSL2
2. 重啟 Docker Desktop
3. 檢查 Windows 更新

### Q: 端口被佔用
A: 
```bash
# 檢查端口使用情況
netstat -ano | findstr :8000

# 修改端口（在 docker-compose.yml 中）
ports:
  - "8001:8000"  # 改為 8001
```

### Q: 環境變量未加載
A: 
1. 確保 .env 文件在項目根目錄
2. 檢查 .env 文件格式（無空格，無引號）
3. 重啟應用

## 開發建議

- **開發環境**：使用本地啟動（選項 2 或 3）
- **生產環境**：使用 Docker Compose（選項 1）
- **調試**：使用 `--reload` 參數自動重載

## 下一步

1. 啟動應用後訪問 http://localhost:8000
2. 查看 API 文檔：http://localhost:8000/docs
3. 開始使用 Web 界面進行 LLM 評估



