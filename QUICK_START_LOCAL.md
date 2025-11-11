# 快速本地啟動指南

## 問題診斷

你遇到的問題：
1. ✅ **Docker Compose 版本警告** - 已修復
2. ❌ **Docker Desktop 未運行** - 需要啟動 Docker Desktop

## 解決方案

### 方案 1：啟動 Docker Desktop（推薦）

1. **啟動 Docker Desktop**
   - 在 Windows 開始菜單中找到 "Docker Desktop"
   - 點擊啟動，等待完全加載

2. **驗證 Docker 運行**
   ```bash
   docker info
   ```

3. **啟動應用**
   ```bash
   docker compose up -d
   ```

4. **訪問應用**
   - Web 界面：http://localhost:3010
   - API 文檔：http://localhost:3010/docs

### 方案 2：本地開發（無需 Docker）

1. **安裝 Python 依賴**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置環境變量**
   ```bash
   cp env.example .env
   # 編輯 .env 文件，設定你的 API keys
   ```

3. **啟動應用**
   ```bash
   python start-local.py
   ```

4. **訪問應用**
   - Web 界面：http://localhost:3010
   - API 文檔：http://localhost:3010/docs

## 環境變量配置

編輯 `.env` 文件：

```bash
# 必需的 API Keys
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here

# 其他配置
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://postgres:password@localhost:5432/llm_eval_hub
```

## 功能測試

啟動後，你可以測試以下功能：

### 1. 情境轉文檔
- 訪問 Web 界面
- 點擊 "Scenario to Docs"
- 輸入情境描述
- 生成文檔

### 2. RAG 測試集生成
- 準備文檔資料夾
- 點擊 "RAG Testset"
- 配置參數
- 生成測試集

### 3. Agent 測試集生成
- 點擊 "Agent Testset"
- 輸入 JSON 格式的情境
- 生成 Excel 測試集

### 4. 評估
- 點擊 "Evaluation"
- 選擇測試集和 judge 模型
- 執行評估

## 故障排除

### Docker Desktop 問題
```bash
# 檢查 Docker 狀態
docker --version
docker compose version

# 如果 Docker Desktop 未運行
# 1. 啟動 Docker Desktop
# 2. 等待完全加載
# 3. 重新運行命令
```

### 端口被佔用
```bash
# 檢查端口使用
netstat -ano | findstr :3010

# 修改端口（在 docker-compose.yml 中）
ports:
  - "3011:3010"
```

### 依賴安裝問題
```bash
# 升級 pip
python -m pip install --upgrade pip

# 安裝依賴
pip install -r requirements.txt

# 如果遇到權限問題
pip install --user -r requirements.txt
```

## 下一步

1. 選擇啟動方案（Docker 或本地）
2. 配置 API keys
3. 啟動應用
4. 訪問 http://localhost:3010
5. 開始使用 LLM 評估功能

## 支援

如果遇到問題：
1. 檢查 Docker Desktop 是否運行
2. 確認 API keys 已配置
3. 查看應用日誌
4. 參考 DOCKER_SETUP.md 獲取詳細說明



