# RAGAS 評估指標完整指南

## 📦 版本資訊

- **RAGAS 版本**: 0.3.6
- **升級日期**: 2025-10-13
- **支援的指標**: 10+ 個核心與進階指標

## 🎯 可用指標清單

### 核心指標（推薦用於基本評估）

#### 1. Answer Accuracy（答案準確率）
- **類別**: `AnswerAccuracy`
- **用途**: 評估答案是否正確回答問題
- **適用場景**: 所有 RAG 系統
- **說明**: 比較生成的答案與標準答案的準確性

#### 2. Factual Correctness（事實正確性）
- **類別**: `FactualCorrectness`
- **用途**: 檢測生成回應是否與文件一致、無臆造資訊
- **適用場景**: 需要高度準確性的應用（醫療、法律等）
- **說明**: 評估答案中的事實陳述是否正確

#### 3. Context Precision（上下文精確率）
- **類別**: `ContextPrecision`
- **用途**: 評估檢索到的上下文中相關資訊的比例
- **適用場景**: 優化檢索系統
- **說明**: 回應中正確資訊所佔比例

#### 4. Context Recall（上下文召回率）
- **類別**: `ContextRecall`
- **用途**: 評估標準答案中的資訊有多少被檢索到
- **適用場景**: 優化檢索系統
- **說明**: 標準答案中的資訊有多少被正確涵蓋

#### 5. Answer Correctness（綜合正確性）
- **類別**: `AnswerCorrectness`
- **用途**: Precision 與 Recall 的綜合平衡指標（F1 分數）
- **適用場景**: 全面評估答案質量
- **說明**: 結合準確率和召回率的綜合指標

### 進階指標（用於深度評估）

#### 6. Faithfulness（忠實度）
- **類別**: `Faithfulness`
- **用途**: 評估答案是否基於提供的上下文，沒有幻覺
- **適用場景**: 防止 LLM 幻覺
- **說明**: 確保答案不包含未在上下文中出現的資訊

#### 7. Answer Relevancy（答案相關性）
- **類別**: `AnswerRelevancy`
- **用途**: 評估答案與問題的相關程度
- **適用場景**: 優化答案品質
- **說明**: 確保答案直接回答用戶問題

#### 8. Response Relevancy（回應相關性）
- **類別**: `ResponseRelevancy`
- **用途**: 評估整體回應的相關性
- **適用場景**: 多輪對話系統
- **說明**: 評估回應在對話上下文中的相關性

#### 9. Context Relevance（上下文相關性）
- **類別**: `ContextRelevance`
- **用途**: 評估檢索到的上下文是否與問題相關
- **適用場景**: 優化檢索策略
- **說明**: 確保檢索的文檔與問題相關

#### 10. Answer Similarity（答案相似度）/ Semantic Similarity（語義相似度）
- **類別**: `AnswerSimilarity`
- **用途**: 評估生成答案與標準答案的語義相似度
- **適用場景**: 評估答案的語義一致性
- **說明**: 使用語義嵌入計算相似度

## 🔧 使用方式

### 後端 API 使用

```python
from ragas.metrics import (
    AnswerAccuracy,
    AnswerCorrectness,
    AnswerRelevancy,
    AnswerSimilarity,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy
)

# 實例化指標
metrics = {
    'answer_accuracy': AnswerAccuracy(),
    'answer_correctness': AnswerCorrectness(),
    'factual_correctness': FactualCorrectness(),
    'context_precision': ContextPrecision(),
    'context_recall': ContextRecall(),
    'faithfulness': Faithfulness(),
    'answer_relevancy': AnswerRelevancy(),
    'answer_similarity': AnswerSimilarity(),
    'context_relevance': ContextRelevance(),
    'response_relevancy': ResponseRelevancy()
}

# 使用 RAGAS 評估
from ragas import evaluate
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[metrics['answer_accuracy'], metrics['faithfulness']],
    llm=evaluator_llm
)
```

### 前端 UI 使用

在評估表單中，用戶可以選擇以下指標：

**核心指標（預設勾選）**:
- ✅ Answer Accuracy (答案準確率)
- ✅ Factual Correctness (事實正確性)
- ✅ Context Precision (精確率)
- ✅ Context Recall (召回率)
- ✅ Answer Correctness (綜合正確性 - 含 F1)

**進階指標（可選）**:
- ⬜ Faithfulness (忠實度)
- ⬜ Answer Relevancy (答案相關性)
- ⬜ Response Relevancy (回應相關性)
- ⬜ Context Relevance (上下文相關性)
- ⬜ Semantic Similarity (語義相似度)

## 📊 評估結果解讀

### 分數範圍
- 所有指標的分數範圍: **0.0 - 1.0**
- 1.0 = 完美
- 0.0 = 最差

### 推薦閾值

| 指標 | 優秀 | 良好 | 需改進 |
|------|------|------|--------|
| Answer Accuracy | ≥ 0.9 | 0.7-0.9 | < 0.7 |
| Factual Correctness | ≥ 0.95 | 0.85-0.95 | < 0.85 |
| Context Precision | ≥ 0.8 | 0.6-0.8 | < 0.6 |
| Context Recall | ≥ 0.8 | 0.6-0.8 | < 0.6 |
| Answer Correctness (F1) | ≥ 0.85 | 0.7-0.85 | < 0.7 |
| Faithfulness | ≥ 0.9 | 0.75-0.9 | < 0.75 |
| Answer Relevancy | ≥ 0.85 | 0.7-0.85 | < 0.7 |

## 🎓 指標選擇建議

### 基礎評估（快速測試）
選擇 3-5 個核心指標：
- Answer Accuracy
- Factual Correctness
- Answer Correctness

### 全面評估（生產環境）
選擇 5-7 個指標：
- Answer Accuracy
- Factual Correctness
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy

### 深度評估（研究與優化）
選擇所有 10 個指標，進行全面分析

## 🔍 常見問題

### Q1: 為什麼某些指標需要更長的評估時間？
**A**: 某些指標（如 Faithfulness、FactualCorrectness）需要 LLM 進行深度分析，因此評估時間較長。

### Q2: 如何選擇合適的 Judge LLM？
**A**: 
- 推薦使用 GPT-4 或 GPT-4-turbo 以獲得最佳評估質量
- Gemini Pro 也是不錯的選擇
- 本地 Ollama 模型可用於快速測試，但準確性可能較低

### Q3: Context Precision 和 Context Recall 的區別？
**A**:
- **Precision**: 檢索到的內容中有多少是相關的（質量）
- **Recall**: 相關內容中有多少被檢索到（覆蓋率）

### Q4: 什麼時候使用 Faithfulness？
**A**: 當您需要確保 LLM 不會產生幻覺（hallucination），特別是在醫療、法律等需要高度準確性的領域。

## 📝 升級記錄

### v0.3.6 (2025-10-13)
- ✅ 升級 RAGAS 從 0.1.0 到 0.3.6
- ✅ 新增 AnswerAccuracy 指標
- ✅ 新增 FactualCorrectness 指標
- ✅ 新增 ResponseRelevancy 指標
- ✅ 新增 ContextRelevance 指標
- ✅ 改進指標實例化方式
- ✅ 更新前端 UI 以支援所有新指標
- ✅ 添加詳細的指標說明和使用建議

## 🔗 相關資源

- [RAGAS 官方文檔](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [RAGAS 指標詳解](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)

## 💡 最佳實踐

1. **開始時使用核心指標**: 先用 5 個核心指標進行基礎評估
2. **逐步增加指標**: 根據需求逐步添加進階指標
3. **定期評估**: 建立定期評估流程，追蹤系統性能變化
4. **結合多個指標**: 不要只依賴單一指標，綜合多個指標進行判斷
5. **保存評估結果**: 系統會自動保存 Excel 報告，便於後續分析和比較

