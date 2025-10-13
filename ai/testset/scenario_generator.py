"""
Scenario Document Generator Module

Generates comprehensive documents based on scenario descriptions and personas.
These documents serve as the knowledge base for RAG systems.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from langchain_openai import ChatOpenAI
from configs.settings import settings

logger = logging.getLogger(__name__)


class ScenarioDocumentGenerator:
    """Generates documents for a given scenario and personas."""
    
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4"):
        """
        Initialize scenario document generator.
        
        Args:
            model_provider: LLM provider (openai, gemini, ollama)
            model_name: Specific model to use
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.llm = self._create_llm()
    
    def _create_llm(self):
        """Create LLM instance based on provider and model."""
        if self.model_provider == "openai":
            return ChatOpenAI(
                api_key=settings.openai_api_key,
                model=self.model_name,
                temperature=0.7
            )
        elif self.model_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=settings.gemini_api_key,
                temperature=0.7
            )
        elif self.model_provider == "ollama":
            return ChatOpenAI(
                api_key="ollama",
                model=self.model_name,
                base_url=settings.ollama_base_url,
                temperature=0.7
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    def generate_documents(
        self,
        scenario_description: str,
        num_documents: int = 10,
        language: str = "繁體中文"
    ) -> List[Dict[str, str]]:
        """
        Generate comprehensive documents for a scenario.
        
        Args:
            scenario_description: Description of the use case/scenario
            num_documents: Number of documents to generate
            language: Language for document content
            
        Returns:
            List of document dictionaries with content and metadata
        """
        logger.info(f"Generating {num_documents} documents for scenario")
        
        prompt = f"""作為一個專業的內容創作者，請根據以下使用情境生成{num_documents}份以使用者為核心的知識文件。

使用情境：
{scenario_description}

這些文件將用於 RAG (檢索增強生成) 系統，請專注於使用者最常用的主要功能和核心情境。

1. **以使用者為核心的內容要求**：
   - 專注於使用者會實際使用的主要功能
   - 描述使用者如何完成核心任務
   - 包含實用的操作步驟和具體指引
   - 避免過度發散到邊緣場景
   - 內容簡潔但完整（500-1000字）

2. **核心情境優先**：
   - 使用者主要操作流程（如何開始使用、基本操作）
   - 最常見的使用場景（使用者最需要的功能）
   - 常見問題和解決方案（使用者經常遇到的問題）
   - 實用建議和最佳實踐（幫助使用者更好地使用系統）

3. **避免過度多樣性**：
   - 不要生成使用者很少會用到的邊緣功能說明
   - 不要過度細分類別（如：不需要分別說明不同地區、不同類型等細節）
   - 專注於使用者視角的核心流程和主要功能
   - 每個文件應該圍繞使用者的一個主要任務或需求

4. **格式要求**：
   - 每份文件有清晰的標題
   - 以使用者的角度描述（例如："如何使用XXX"、"您可以如何..."）
   - 包含實際操作步驟
   - 使用簡潔清晰的語言

請以JSON格式返回（注意：content 字段中的換行符請使用 \\n 表示）：

```json
[
  {{
    "document_id": "doc_001",
    "title": "文件標題（以使用者為中心）",
    "category": "文件類型類別",
    "content": "完整的文件內容（以使用者視角撰寫）...使用\\n表示換行...",
    "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"],
    "target_audience": "目標受眾描述",
    "complexity_level": "simple|moderate|advanced"
  }}
]
```

**重要**：content 字段內的所有換行必須用 \\n 轉義，不要直接使用換行符。

請確保：
1. 每份文件都圍繞使用者的核心需求
2. 專注於使用者主要使用情境，不要過度發散
3. 內容實用、可操作，避免過於理論
4. 使用{language}語言
5. 如果只生成1份文件，請涵蓋使用者最核心的使用流程

現在請生成{num_documents}份以使用者為核心的文件。只返回JSON，不要有其他文字。"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean up response
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON with strict=False to be more lenient
            try:
                documents = json.loads(content, strict=False)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.error(f"Content preview: {content[:500]}...")
                # Try to fix common issues
                import re
                # Remove control characters except newline and tab
                fixed_content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', content)
                try:
                    documents = json.loads(fixed_content, strict=False)
                    logger.info("Successfully parsed JSON after fixing control characters")
                except:
                    raise e
            
            if not isinstance(documents, list):
                raise ValueError("Response is not a list of documents")
            
            # Validate documents
            for i, doc in enumerate(documents):
                if not doc.get("document_id"):
                    doc["document_id"] = f"doc_{i+1:03d}"
                if not doc.get("title"):
                    raise ValueError(f"Document {i+1} missing title")
                if not doc.get("content"):
                    raise ValueError(f"Document {i+1} missing content")
            
            logger.info(f"Successfully generated {len(documents)} documents")
            return documents
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content: {content[:500]}...")
            raise
        except Exception as e:
            logger.error(f"Error generating documents: {e}")
            raise
    
    def save_documents(
        self,
        documents: List[Dict[str, str]],
        output_folder: str,
        scenario_name: str = "scenario"
    ) -> Dict[str, Any]:
        """
        Save documents to individual text files.
        
        Args:
            documents: List of document dictionaries
            output_folder: Output directory path
            scenario_name: Name of the scenario for file naming
            
        Returns:
            Dictionary with paths and metadata
        """
        try:
            # Create output folder structure
            output_path = Path(output_folder)
            docs_folder = output_path / "02_documents"
            docs_folder.mkdir(parents=True, exist_ok=True)
            
            # Save individual document files
            document_files = []
            for doc in documents:
                doc_id = doc.get("document_id", "unknown")
                title = doc.get("title", "Untitled").replace("/", "_").replace("\\", "_")
                
                # Create text file for each document
                txt_file = docs_folder / f"{doc_id}_{title}.txt"
                
                # Format document content
                content = f"""# {doc.get('title', 'Untitled')}

分類: {doc.get('category', 'N/A')}
目標受眾: {doc.get('target_audience', 'N/A')}
複雜度: {doc.get('complexity_level', 'N/A')}
關鍵詞: {', '.join(doc.get('keywords', []))}

---

{doc.get('content', '')}

---
文件ID: {doc_id}
生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                document_files.append(str(txt_file))
                logger.info(f"Saved document file: {txt_file}")
            
            # Save metadata JSON
            metadata_file = docs_folder / f"{scenario_name}_documents_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(documents)} document files to {docs_folder}")
            
            return {
                "success": True,
                "documents_folder": str(docs_folder),
                "document_files": document_files,
                "metadata_file": str(metadata_file),
                "total_documents": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error saving documents: {e}")
            return {
                "success": False,
                "error": str(e)
            }

