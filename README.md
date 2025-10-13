# 🧠 LLM Evaluation Hub

> **一個完整的 RAG & Agent 系統自動化測試平台**  
> 從情境描述到評估，只需簡單點擊！

## 📖 這個專案是什麼？

LLM Evaluation Hub 是一個專為 **RAG（檢索增強生成）** 和 **Agent（對話代理）** 系統設計的全自動化測試平台。

### 🎯 主要目標

解決 LLM 應用開發中最痛的問題：**如何測試你的 RAG/Agent 系統？**

傳統方式需要：
- ❌ 手動編寫大量測試問題
- ❌ 自己想像各種用戶角色
- ❌ 人工檢查每個回答
- ❌ 難以模擬真實用戶行為

**我們的解決方案：**
- ✅ **自動生成真實用戶角色**：只需描述使用情境，系統自動生成 10+ 個超詳細的用戶角色
- ✅ **自動生成測試集**：基於角色特徵生成真實的問題或任務
- ✅ **自動化評估**：使用 LLM Judge 和 RAGAS 指標自動評估系統表現
- ✅ **智能對話模擬**：Agent 測試中，LLM 自動扮演用戶進行多輪對話

### 🔄 完整工作流程

```
情境描述
    ↓
自動生成 10+ 個真實用戶角色
    ↓
自動生成知識文件
    ↓
選擇測試模式（RAG / Agent）
    ↓
自動生成測試集（QA Pairs / 任務場景）
    ↓
自動化評估（RAGAS 指標 / 對話測試）
```

### 💡 使用範例

**情境**: "一個房地產媒合平台"

**系統自動生成**:
- 👤 10+ 個真實用戶角色
  - 陳小美（32歲軟體工程師，養貓，想在內湖買房，預算2000萬...）
  - 王大強（45歲企業主，需賣屋換屋，重視學區...）
  - ...
- 📚 15+ 份知識文件
  - 購房流程指南、各區房價分析、貸款申請指南...
- 🔍 30+ 個真實問題
  - "我和我的貓咪想找靠近內湖科技園區的房子..."（陳小美的問題）
  - "我想賣掉現有房子換更大的，小孩要上小學了..."（王大強的問題）

**然後自動評估你的系統回答質量！**

## 🚀 快速開始（3 步驟）

### 步驟 1: 準備環境

確保已安裝：
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- OpenAI API Key（必須）
- Gemini API Key（選填）

### 步驟 2: 啟動服務

```bash
# 1. Clone 專案
git clone <repository-url>
cd llm-eval-hub

# 2. 創建 .env 文件（填入你的 API keys）
cp env.example .env
# 編輯 .env，填入你的 OPENAI_API_KEY

# 3. 啟動 Docker 服務
docker-compose up -d

# 4. 查看日誌（確認啟動成功）
docker-compose logs -f app
```

### 步驟 3: 開始使用

打開瀏覽器訪問：

**🎨 Web 界面（推薦新手）**
- 完整工作流程: http://localhost:8000/static/workflow.html
- 主控台: http://localhost:8000

**📚 API 文檔**
- Swagger UI: http://localhost:8000/docs

## 🎬 使用方式

### 方式 1: Web 界面（最簡單）

1. 訪問 http://localhost:8000/static/workflow.html
2. **階段 1**: 輸入使用情境 → 自動生成角色
3. **階段 2**: 自動生成知識文件
4. **階段 3**: 選擇 RAG 或 Agent 模式
5. **階段 4**: 自動生成測試集
6. **階段 5**: 執行評估

### 方式 2: API 調用

```python
import requests

BASE_URL = "http://localhost:8000/api/v1/testset"

# 生成角色
response = requests.post(f"{BASE_URL}/workflow/generate-personas", json={
    "scenario_description": "你的使用情境描述...",
    "num_personas": 10,
    "output_folder": "/app/outputs/my_test",
    "model_provider": "openai",
    "model_name": "gpt-4"
})

# 繼續下一個階段...
```

## ✨ 核心功能

| 功能 | 說明 | 適用場景 |
|------|------|----------|
| **🎭 智能角色生成** | 根據情境自動生成超詳細用戶角色 | 了解潛在用戶是誰 |
| **📚 文件生成** | 自動生成多樣化知識文件 | 建立 RAG 知識庫 |
| **🔍 RAG 測試集** | 生成角色化的問答對 | 測試 RAG 系統 |
| **🤖 Agent 測試集** | 生成任務場景 | 測試對話 Agent |
| **💬 智能對話** | LLM 扮演用戶多輪對話 | Agent 壓力測試 |
| **📊 自動評估** | RAGAS 指標 + LLM Judge | 量化系統表現 |

## 📁 輸出結構

```
outputs/
└── your_scenario/
    ├── 01_personas/              # 角色檔案
    │   ├── persona_001_陳小美.md
    │   ├── personas_summary.xlsx
    │   └── personas_full.json
    ├── 02_documents/             # 知識文件
    │   ├── doc_001_購房指南.txt
    │   └── metadata.json
    ├── 03_rag_testset/          # RAG 測試集
    │   └── rag_testset.xlsx
    ├── 03_agent_testset/        # Agent 測試集
    │   └── agent_testset.xlsx
    └── 04_evaluation/           # 評估結果
        ├── results.xlsx
        └── conversation_logs/
```

## 📚 文檔

- 📖 **[完整工作流程指南](docs/WORKFLOW_GUIDE.md)** - 詳細使用說明
- 🐳 **[Docker 配置指南](docs/DOCKER_VOLUME_SETUP.md)** - 本地資料夾訪問設定
- 📝 **[實施總結](docs/IMPLEMENTATION_SUMMARY.md)** - 技術細節

## 🔧 常見問題

### Q: 需要哪些 API Keys？
A: 至少需要 OpenAI API Key。Gemini 和 Hugging Face 是選填。

### Q: 生成一次測試集要多少錢？
A: 使用 GPT-4 完整流程（10 角色 + 10 文件 + 30 測試）約 $3.50

### Q: 支援中文嗎？
A: 完全支持繁體中文和簡體中文

### Q: 可以用本地模型嗎？
A: 支持 Ollama 本地模型

### Q: 如何訪問本地資料夾？
A: 參考 [Docker Volume 配置指南](docs/DOCKER_VOLUME_SETUP.md)

## 🛠️ 管理命令

```bash
# 啟動服務
docker-compose up -d

# 停止服務
docker-compose down

# 查看日誌
docker-compose logs -f app

# 重啟服務
docker-compose restart

# 重建並啟動（當更新代碼後）
docker-compose up -d --build

# 查看服務狀態
docker-compose ps

# 進入容器（debug 用）
docker-compose exec app bash
```

## 💻 本地開發（不使用 Docker）

如果你想在本地直接運行（不推薦新手）：

```bash
# 1. 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Mac/Linux
# 或 venv\Scripts\activate  # Windows

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 啟動服務
python scripts/start.py
```

## 🔌 API 範例

### 完整工作流程 API

```python
import requests

BASE_URL = "http://localhost:8000/api/v1/testset"

# 階段 1: 生成角色
personas_response = requests.post(f"{BASE_URL}/workflow/generate-personas", json={
    "scenario_description": "一個房地產媒合平台...",
    "num_personas": 10,
    "output_folder": "/app/outputs/real_estate",
    "model_provider": "openai",
    "model_name": "gpt-4"
})
personas_result = personas_response.json()

# 階段 2: 生成文件
documents_response = requests.post(f"{BASE_URL}/workflow/generate-documents", json={
    "scenario_description": "一個房地產媒合平台...",
    "num_documents": 10,
    "output_folder": "/app/outputs/real_estate",
    "scenario_name": personas_result["scenario_name"],
    "model_provider": "openai",
    "model_name": "gpt-4"
})
documents_result = documents_response.json()

# 階段 3 & 4: 生成 RAG 測試集
rag_response = requests.post(f"{BASE_URL}/workflow/generate-rag-testset", json={
    "documents_folder": documents_result["documents_folder"],
    "personas_json_path": personas_result["json_file"],
    "output_folder": "/app/outputs/real_estate",
    "scenario_name": personas_result["scenario_name"],
    "model_provider": "openai",
    "model_name": "gpt-3.5-turbo",
    "chunk_size": 5000,
    "chunk_overlap": 200,
    "qa_per_chunk": 3
})
rag_result = rag_response.json()
print(f"生成了 {rag_result['total_qa_pairs']} 個 QA pairs")
```

### 查看 API 文檔

訪問 http://localhost:8000/docs 查看完整的 API 文檔（Swagger UI）

## 🤝 貢獻

歡迎貢獻！請查看 [貢獻指南](CONTRIBUTING.md) 或直接提交 Pull Request。

## 📄 License

MIT License - 詳見 [LICENSE](LICENSE) 文件

## 🙏 致謝

- [RAGAS](https://github.com/explodinggradients/ragas) - RAG 評估框架
- [LangChain](https://github.com/langchain-ai/langchain) - LLM 應用框架
- [LangSmith](https://smith.langchain.com/) - LLM 追蹤和監控

---

**如有問題或建議，歡迎開 Issue 或聯繫我們！**

### 舊版 API 使用範例（僅供參考）

<details>
<summary>點擊展開查看舊版 API 範例</summary>

### LLM Judge Evaluation

```bash
# Evaluate with OpenAI judge
curl -X POST "http://localhost:8000/api/v1/evaluation/judge" \
  -H "Content-Type: application/json" \
  -d '{
    "testset_data": [
      {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is a subset of AI...",
        "llm_response": "Machine learning is a method of data analysis..."
      }
    ],
    "llm_endpoint": "https://api.openai.com/v1/chat/completions",
    "judge_model_type": "openai",
    "judge_model": "gpt-4-turbo-preview",
    "evaluation_criteria": ["accuracy", "factual_correctness"]
  }'

# Evaluate with Gemini judge
curl -X POST "http://localhost:8000/api/v1/evaluation/judge" \
  -H "Content-Type: application/json" \
  -d '{
    "testset_data": [
      {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is a subset of AI...",
        "llm_response": "Machine learning is a method of data analysis..."
      }
    ],
    "llm_endpoint": "https://api.openai.com/v1/chat/completions",
    "judge_model_type": "gemini",
    "judge_model": "gemini-pro",
    "evaluation_criteria": ["accuracy", "factual_correctness"]
  }'

# Evaluate with Ollama judge
curl -X POST "http://localhost:8000/api/v1/evaluation/judge" \
  -H "Content-Type: application/json" \
  -d '{
    "testset_data": [
      {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is a subset of AI...",
        "llm_response": "Machine learning is a method of data analysis..."
      }
    ],
    "llm_endpoint": "https://api.openai.com/v1/chat/completions",
    "judge_model_type": "ollama",
    "judge_model": "llama2",
    "evaluation_criteria": ["accuracy", "factual_correctness"]
  }'
```

## Multi-Model Judge Support

The platform supports multiple LLM models as judges for evaluation:

### Supported Judge Models

1. **OpenAI Models**
   - `gpt-4-turbo-preview`
   - `gpt-4`
   - `gpt-3.5-turbo`

2. **Google Gemini Models**
   - `gemini-pro`
   - `gemini-pro-vision`

3. **Ollama Models** (Local)
   - `llama2`
   - `codellama`
   - `mistral`
   - `neural-chat`

4. **Hugging Face Models**
   - `microsoft/DialoGPT-medium`
   - `facebook/blenderbot-400M-distill`

### Judge Model Selection

You can specify the judge model in your evaluation requests:

```json
{
  "judge_model_type": "gemini",
  "judge_model": "gemini-pro",
  "evaluation_criteria": ["accuracy", "factual_correctness"]
}
```

## Available Metrics

### RAG Metrics
- `accuracy`: Overall accuracy compared to ground truth
- `factual_correctness`: Factual accuracy of responses
- `precision`: Precision of information
- `recall`: Recall of relevant information
- `f1`: Harmonic mean of precision and recall
- `response_relevancy`: Relevance of responses to questions
- `faithfulness`: Faithfulness to provided context
- `context_precision`: Precision of retrieved context
- `context_recall`: Recall of relevant context
- `answer_relevancy`: Relevance of answers
- `answer_correctness`: Correctness of answers
- `answer_similarity`: Similarity to ground truth
- `semantic_similarity`: Semantic similarity
- `bleu_score`: BLEU score for text similarity
- `rouge_score`: ROUGE score for text similarity
- `exact_match`: Exact match with ground truth

### Agent Metrics
- `average_turn`: Average number of turns per conversation
- `success_rate`: Success rate of agent tasks
- `tool_call_accuracy`: Accuracy of tool calls
- `agent_goal_accuracy`: Accuracy in achieving goals
- `topic_adherence`: Adherence to conversation topic

## Project Structure

```
llm-eval-hub/
├── ai/                     # AI evaluation modules
│   ├── core/              # Core evaluation logic
│   │   ├── evaluator.py   # Main evaluator using RAGAS
│   │   └── llm_judge.py   # LLM judge system
│   └── testset/           # Testset generation
│       └── generator.py   # Testset generator service
├── api/                   # FastAPI application
│   ├── routes/           # API routes
│   │   ├── evaluation.py # Evaluation endpoints
│   │   ├── testset.py    # Testset endpoints
│   │   └── health.py     # Health check endpoints
│   ├── middleware.py     # Custom middleware
│   └── main.py          # FastAPI app
├── configs/             # Configuration
│   └── settings.py     # Application settings
├── data/               # Data storage
│   ├── raw/           # Raw data
│   └── processed/     # Processed data
├── docs/              # Documentation
├── tutorials/         # Tutorial examples
├── examples/          # Usage examples
├── outputs/          # Inference results
├── artifacts/        # Model artifacts
├── results/          # Evaluation results
├── tests/            # Test files
├── docker-compose.yml # Docker configuration
├── Dockerfile        # Docker image
├── requirements.txt  # Python dependencies
├── pyproject.toml   # Project configuration
└── Makefile         # Build commands
```

## Development

### Running Tests

```bash
make test
```

### Code Formatting

```bash
make format
```

### Linting

```bash
make lint
```

### Building Docker Image

```bash
make build
```

## Configuration

The application can be configured through environment variables. See `env.example` for all available options.

### Key Configuration Options

- `LANGSMITH_API_KEY`: LangSmith API key for evaluation tracking
- `OPENAI_API_KEY`: OpenAI API key for LLM judge
- `GEMINI_API_KEY`: Google Gemini API key for judge evaluation
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `HUGGINGFACE_API_KEY`: Hugging Face API key for judge evaluation
- `SECRET_KEY`: Secret key for JWT token generation and session management
- `DATABASE_URL`: Database connection string
- `DEFAULT_EVALUATION_TIMEOUT`: Timeout for evaluations (seconds)
- `MAX_CONCURRENT_EVALUATIONS`: Maximum concurrent evaluations

## Monitoring

The application includes comprehensive monitoring:

- **Health Checks**: `/health`, `/health/detailed`, `/health/ready`, `/health/live`
- **Metrics**: Prometheus-compatible metrics endpoint
- **Structured Logging**: JSON-formatted logs with request tracing
- **Error Tracking**: Comprehensive error handling and logging

</details>