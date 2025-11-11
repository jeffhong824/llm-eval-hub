# 完整工作流程指南

本文檔詳細說明 LLM Evaluation Hub 的完整工作流程，從情境描述到最終評估。

## 概述

LLM Evaluation Hub 提供一個全自動化的 RAG 和 Agent 系統評估平台，只需簡單的點擊和貼上，即可完成從角色生成到評估的完整流程。

## 工作流程架構

```
階段1: 情境 → 角色生成
    ↓
階段2: 情境 → 文件生成
    ↓
階段3: 選擇模式 (RAG / Agent)
    ↓
階段4: 角色 + 文件 → 測試集生成
    ↓
階段5: 測試集 → 自動化評估
```

## 階段詳細說明

### 階段 1: 角色生成

**目的**: 根據使用情境自動生成多個超級詳細的用戶角色。

**輸入**:
- 使用情境描述（例如：房地產媒合平台）
- 角色數量（建議 10-20 個）
- LLM 模型選擇（建議使用 GPT-4）
- 輸出資料夾

**輸出**:
```
output_folder/
└── scenario_name/
    └── 01_personas/
        ├── persona_001_張小明.md
        ├── persona_002_李美華.md
        ├── ...
        ├── scenario_name_personas_summary.xlsx
        └── scenario_name_personas_full.json
```

**角色細節包含**:
1. **基本資訊**: 姓名、年齡、性別、職業、教育、收入、居住地
2. **生活狀態**: 婚姻、家庭、寵物、住房、通勤方式
3. **需求與目標**: 痛點、核心需求、短期/長期目標、預算、決策標準
4. **行為模式**: 日常作息、科技使用、資訊搜尋、決策風格
5. **心理特徵**: 個性、價值觀、態度、焦慮點、動機
6. **情境專屬**: 根據具體情境添加的專屬需求
7. **溝通風格**: 問問題方式、語言特色、常見疑問類型
8. **使用場景**: 使用時段、頻率、環境、裝置偏好

**範例 - 房地產平台角色**:

角色 1: 陳小美 (32歲，軟體工程師)
- 核心需求: 靠近科技園區、有貓友善、30坪、2房2廳
- 預算: 1500-2000萬
- 通勤: 捷運 + 步行 < 30分鐘
- 決策風格: 數據驅動型，會詳細比較多個選項
- 溝通特色: 直接、重視數據、會問很多細節問題

角色 2: 王大強 (45歲，企業主)
- 核心需求: 賣出老屋換新房、學區優先、停車位2個
- 預算: 3000-4000萬
- 通勤: 開車
- 決策風格: 注重投資報酬率
- 溝通特色: 簡潔明瞭、重視效率

### 階段 2: 文件生成

**目的**: 根據情境生成多份知識文件，作為 RAG 系統的知識庫。

**輸入**:
- 使用情境描述（與階段1相同）
- 文件數量（建議 10-20 份）
- LLM 模型選擇
- 輸出資料夾（與階段1相同）

**輸出**:
```
output_folder/
└── scenario_name/
    └── 02_documents/
        ├── doc_001_購房流程指南.txt
        ├── doc_002_各地區房價分析.txt
        ├── doc_003_貸款申請指南.txt
        ├── ...
        └── scenario_name_documents_metadata.json
```

**文件類型多樣性**:
1. 概述性文件
2. 操作指南
3. 常見問題解答
4. 專業知識深入探討
5. 案例分析
6. 比較分析
7. 注意事項和最佳實踐
8. 術語解釋
9. 流程說明
10. 政策或規則說明

### 階段 3: 選擇評估模式

**目的**: 根據您要評估的系統類型，選擇 RAG 或 Agent 模式。

#### RAG 模式
- **適用**: 問答系統、知識檢索系統
- **測試方式**: 單次問答
- **生成內容**: QA Pairs（問題-答案對）
- **評估重點**: 答案準確度、相關性、忠實度

#### Agent 模式
- **適用**: 對話系統、任務導向 Agent
- **測試方式**: 多輪對話
- **生成內容**: 任務場景（角色-目標對）
- **評估重點**: 目標達成率、對話質量、輪數效率

### 階段 4A: RAG 測試集生成

**目的**: 使用角色和文件生成問答對測試集。

**配置參數**:
- 文件分塊大小 (chunk_size): 5000 字元
- 分塊重疊 (chunk_overlap): 200 字元
- 每個分塊的 QA 數量 (qa_per_chunk): 3

**生成邏輯**:
1. 將文件切分成多個 chunks
2. 對每個 chunk，選擇不同的角色
3. 根據角色的特徵生成問題
4. 問題反映角色的需求、痛點、溝通風格
5. 生成對應的期望答案

**輸出**:
```
output_folder/
└── scenario_name/
    └── 03_rag_testset/
        └── rag_testset.xlsx
```

**Excel 結構**:
```
| question | answer | context | difficulty | question_type | persona_name | persona_id | chunk_id |
|----------|--------|---------|------------|---------------|--------------|------------|----------|
| ...      | ...    | ...     | ...        | ...           | ...          | ...        | ...      |
```

**範例 QA Pair**:

角色: 陳小美 (軟體工程師，養貓)
```
問題: "我和我的貓咪想找靠近內湖科技園區的房子，2房就好，預算2000萬以內。
       請問有什麼推薦的區域嗎？我希望通勤時間不要超過30分鐘。"

期望答案: "根據您的需求，推薦以下區域：
           1. 內湖區碧湖周邊：...
           2. 南港區：...
           3. 汐止區：...
           
           這些區域都有寵物友善的社區，且符合您的預算..."

情境: [相關文件片段]
難度: medium
問題類型: decision_making
角色: 陳小美
```

### 階段 4B: Agent 測試集生成

**目的**: 使用角色和文件生成任務場景測試集。

**配置參數**:
- 文件分塊大小 (chunk_size): 5000 字元
- 分塊重疊 (chunk_overlap): 200 字元
- 每個分塊的任務數量 (tasks_per_chunk): 3

**生成邏輯**:
1. 將文件切分成多個 chunks
2. 對每個 chunk，選擇不同的角色
3. 根據角色的目標和需求生成任務場景
4. 定義清晰的成功標準
5. 包含角色的完整背景資訊

**輸出**:
```
output_folder/
└── scenario_name/
    └── 03_agent_testset/
        └── agent_testset.xlsx
```

**Excel 結構**:
```
| task_id | task_type | user_role | user_background | task_context | initial_request | expected_outcome | success_criteria | difficulty | context_used |
|---------|-----------|-----------|-----------------|--------------|-----------------|------------------|------------------|------------|--------------|
| ...     | ...       | ...       | ...             | ...          | ...             | ...              | ...              | ...        | ...          |
```

**範例任務場景**:

角色: 王大強 (45歲企業主，需賣屋換屋)
```
任務類型: problem_resolution
背景: 企業主，需要賣出現有房屋並購買更大的房子，重視學區
初始請求: "我想賣掉目前的房子然後換一個更大的，小孩要上小學了，
           學區很重要。你能幫我規劃一下怎麼做最合適嗎？"
期望結果: Agent 協助制定完整的賣屋換屋計劃，包括：
           1. 現有房屋估價
           2. 推薦學區優質區域
           3. 換屋時間規劃
           4. 財務建議
成功標準: 
           - 提供至少3個學區優質區域選項
           - 給出合理的房屋估價範圍
           - 制定完整的時間表
           - 提供具體的下一步行動建議
難度: hard
```

### 階段 5A: RAG 評估

**目的**: 使用 RAGAS 指標評估 RAG 系統。

**配置**:
- 被測試 LLM API endpoint（OpenAI 格式）
- Judge LLM 模型選擇
- 評估指標選擇

**評估流程**:
1. 載入 RAG testset
2. 對每個 QA pair:
   - 調用被測試系統的 API，發送問題
   - 獲取系統回應
   - 使用 Judge LLM 評估回應質量
   - 計算各項指標分數
3. 生成評估報告

**RAGAS 指標**:
- **Answer Relevancy**: 答案與問題的相關性
- **Faithfulness**: 答案對上下文的忠實度
- **Context Precision**: 檢索上下文的精確度
- **Context Recall**: 檢索上下文的召回率
- **Answer Correctness**: 答案的正確性
- **Answer Similarity**: 答案與 ground truth 的相似度

**輸出**:
```
output_folder/
└── scenario_name/
    └── 04_rag_evaluation/
        ├── evaluation_results.xlsx
        └── evaluation_summary.json
```

### 階段 5B: Agent 評估

**目的**: 評估 Agent 系統在多輪對話中達成目標的能力。

**配置**:
- 被測試 Agent API endpoint（OpenAI 格式）
- Judge LLM 模型選擇
- Conversation LLM 模型選擇（用於生成用戶回應）
- 最大對話輪數（default: 30）

**評估流程**:
1. 載入 Agent testset
2. 對每個任務場景:
   - 使用角色的 initial_request 開始對話
   - 進入對話循環:
     a. 調用被測試 Agent API
     b. 獲取 Agent 回應
     c. 使用 Judge LLM 判斷目標是否達成
     d. 如果未達成且未超過最大輪數:
        - 使用 Conversation LLM 生成下一個用戶問題
        - 問題基於角色特徵和對話歷史
        - 繼續對話
   - 記錄對話輪數和達成狀態
3. 生成評估報告

**關鍵特性**:
- **智能對話生成**: 使用 LLM 根據角色特徵生成真實的用戶回應
- **自動目標判斷**: 使用 LLM 判斷是否達成目標，而非簡單關鍵詞匹配
- **限定輪數**: 測試 Agent 在有限輪數內解決問題的能力

**評估指標**:
- **Task Completion Score**: 任務完成度評分
- **Conversation Quality Score**: 對話質量評分
- **User Satisfaction Score**: 用戶滿意度評分
- **Goal Achievement Score**: 目標達成度評分
- **Average Turns**: 平均對話輪數
- **Success Rate**: 成功率（在限定輪數內達成目標的比例）

**輸出**:
```
output_folder/
└── scenario_name/
    └── 04_agent_evaluation/
        ├── evaluation_results.xlsx
        ├── evaluation_summary.json
        └── conversation_logs/
            ├── task_001_conversation.json
            ├── task_002_conversation.json
            └── ...
```

## 輸出資料夾結構

完整的輸出結構如下：

```
output_folder/
└── scenario_name/
    ├── workflow_metadata.json          # 工作流程元數據
    ├── 01_personas/                    # 階段1輸出
    │   ├── persona_001_xxx.md
    │   ├── ...
    │   ├── scenario_personas_summary.xlsx
    │   └── scenario_personas_full.json
    ├── 02_documents/                   # 階段2輸出
    │   ├── doc_001_xxx.txt
    │   ├── ...
    │   └── scenario_documents_metadata.json
    ├── 03_rag_testset/                 # 階段4A輸出（RAG模式）
    │   └── rag_testset.xlsx
    ├── 03_agent_testset/               # 階段4B輸出（Agent模式）
    │   └── agent_testset.xlsx
    ├── 04_rag_evaluation/              # 階段5A輸出（RAG模式）
    │   ├── evaluation_results.xlsx
    │   └── evaluation_summary.json
    └── 04_agent_evaluation/            # 階段5B輸出（Agent模式）
        ├── evaluation_results.xlsx
        ├── evaluation_summary.json
        └── conversation_logs/
            ├── task_001_conversation.json
            └── ...
```

## API 使用示例

### 使用 Python 腳本

```python
import requests
import json

# 基礎 URL
BASE_URL = "http://localhost:3010/api/v1/testset"

# 階段 1: 生成角色
personas_request = {
    "scenario_description": "一個房地產媒合平台，連接購屋者和賣屋者...",
    "num_personas": 10,
    "output_folder": "/app/outputs/real_estate_test",
    "scenario_name": "real_estate_platform",
    "model_provider": "openai",
    "model_name": "gpt-4",
    "language": "繁體中文"
}

response = requests.post(
    f"{BASE_URL}/workflow/generate-personas",
    json=personas_request
)
personas_result = response.json()
print(f"生成了 {personas_result['total_personas']} 個角色")

# 階段 2: 生成文件
documents_request = {
    "scenario_description": personas_request["scenario_description"],
    "num_documents": 10,
    "output_folder": personas_request["output_folder"],
    "scenario_name": personas_result["scenario_name"],
    "model_provider": "openai",
    "model_name": "gpt-4",
    "language": "繁體中文"
}

response = requests.post(
    f"{BASE_URL}/workflow/generate-documents",
    json=documents_request
)
documents_result = response.json()
print(f"生成了 {documents_result['total_documents']} 份文件")

# 階段 3 & 4: 生成 RAG 測試集
rag_testset_request = {
    "documents_folder": documents_result["documents_folder"],
    "personas_json_path": personas_result["json_file"],
    "output_folder": personas_request["output_folder"],
    "scenario_name": personas_result["scenario_name"],
    "model_provider": "openai",
    "model_name": "gpt-3.5-turbo",
    "language": "繁體中文",
    "chunk_size": 5000,
    "chunk_overlap": 200,
    "qa_per_chunk": 3
}

response = requests.post(
    f"{BASE_URL}/workflow/generate-rag-testset",
    json=rag_testset_request
)
rag_result = response.json()
print(f"生成了 {rag_result['total_qa_pairs']} 個 QA pairs")
```

### 使用 cURL

```bash
# 階段 1: 生成角色
curl -X POST http://localhost:3010/api/v1/testset/workflow/generate-personas \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_description": "一個房地產媒合平台...",
    "num_personas": 10,
    "output_folder": "/app/outputs/real_estate_test",
    "model_provider": "openai",
    "model_name": "gpt-4",
    "language": "繁體中文"
  }'
```

## 最佳實踐

### 1. 情境描述撰寫
- **詳細且具體**: 描述目標用戶、主要功能、使用場景
- **包含關鍵資訊**: 用戶需求、痛點、使用情境
- **範例**: "一個線上學習平台，提供程式設計課程。用戶包括完全新手、轉職者、在職工程師。平台提供影片教學、互動練習、專案實作。用戶主要痛點是不知道如何選擇適合的課程、學習進度追蹤困難、缺乏實戰經驗。"

### 2. 角色數量選擇
- **10-15個**: 適合中小型項目
- **20-30個**: 適合大型項目或需要高度多樣性
- **注意**: 更多角色 = 更長生成時間 + 更高 API 成本

### 3. 文件數量選擇
- **5-10份**: 適合簡單場景
- **10-20份**: 適合一般場景（推薦）
- **20+份**: 適合複雜場景

### 4. LLM 模型選擇
- **角色生成**: 使用 GPT-4（質量更高）
- **文件生成**: 使用 GPT-4
- **測試集生成**: 可使用 GPT-3.5-turbo（成本較低）
- **評估**: 使用 GPT-4 作為 Judge（評分更準確）

### 5. 分塊參數調整
- **chunk_size**: 
  - 5000: 適合大多數情況
  - 3000: 適合短文檔
  - 10000: 適合長文檔和複雜情境
- **chunk_overlap**: 
  - 200: 標準值
  - 500: 需要更強上下文連續性時

## 故障排除

### 問題: API 調用超時

**原因**: 生成大量內容時可能超時

**解決方案**:
1. 減少一次生成的數量
2. 使用更快的模型（如 GPT-3.5-turbo）
3. 增加超時設置

### 問題: 生成的角色不夠詳細

**原因**: 模型選擇或 prompt 問題

**解決方案**:
1. 使用 GPT-4 而非 GPT-3.5
2. 在情境描述中提供更多細節
3. 檢查 temperature 設置（建議 0.8）

### 問題: Excel 文件無法打開

**原因**: 文件路徑映射問題

**解決方案**:
1. 檢查 Docker volume 配置
2. 使用容器內路徑（/app/...）
3. 參考 [DOCKER_VOLUME_SETUP.md](./DOCKER_VOLUME_SETUP.md)

### 問題: 生成的問題重複性高

**原因**: 角色多樣性不足或 chunk 太少

**解決方案**:
1. 增加角色數量
2. 確保角色之間有明顯差異
3. 調整 chunk_size 和 qa_per_chunk

## 進階功能

### 自訂 LLM Provider

支持的 providers:
- `openai`: OpenAI API
- `gemini`: Google Gemini
- `ollama`: 本地 Ollama 模型

### Import 角色功能

如果已經有角色 JSON 檔案，可以直接使用：

```json
{
  "documents_folder": "./documents",
  "personas_json_path": "./existing_personas.json",
  ...
}
```

### 階段性輸出

每個階段完成後都會立即保存結果，可以：
- 檢查中間產物
- 從任意階段重新開始
- 使用不同參數重新生成某個階段

## 相關文檔

- [Docker Volume Setup](./DOCKER_VOLUME_SETUP.md) - Docker 本地資料夾訪問配置
- [API Documentation](../README.md#api-usage) - 完整 API 文檔
- [RAGAS Metrics](https://docs.ragas.io/en/latest/concepts/metrics/index.html) - RAGAS 評估指標說明

## 技術架構

### 核心模組

1. **PersonaGenerator** (`ai/testset/persona_generator.py`)
   - 角色生成邏輯
   - 支持多種 LLM providers

2. **ScenarioDocumentGenerator** (`ai/testset/scenario_generator.py`)
   - 文件生成邏輯
   - 多樣化文件類型

3. **RAGTestsetGenerator** (`ai/testset/rag_generator.py`)
   - RAG 測試集生成
   - 與角色系統整合

4. **AgentTestsetGenerator** (`ai/testset/agent_generator.py`)
   - Agent 測試集生成
   - 任務場景構建

5. **OutputManager** (`ai/testset/output_manager.py`)
   - 階段性輸出管理
   - 工作流程追蹤

6. **AgentEvaluator** (`ai/evaluation/agent_evaluator.py`)
   - Agent 評估邏輯
   - 智能對話生成
   - 目標達成判斷

## 貢獻指南

歡迎貢獻！請參考以下方向：

1. **新的角色生成策略**: 針對特定領域的角色生成優化
2. **評估指標擴展**: 新的評估維度和指標
3. **UI 改進**: 更好的前端體驗
4. **效能優化**: 加速生成和評估過程
5. **文檔完善**: 更多使用範例和最佳實踐

## License

MIT License

