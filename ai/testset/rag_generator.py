"""
RAG Testset Generator

Generates RAG test sets from documents by chunking them and creating QA pairs.
"""

# Import pydantic patch FIRST, before any langchain imports
import ai.core.pydantic_patch  # noqa: F401

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI
from configs.settings import settings

logger = logging.getLogger(__name__)


class RAGTestsetGenerator:
    """Generate RAG test sets from documents."""
    
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
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
            # Use OpenAI-compatible API for Ollama
            return ChatOpenAI(
                api_key="ollama",  # Dummy key for Ollama
                model=self.model_name,
                base_url=settings.ollama_base_url,
                temperature=0.7
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    def load_documents_from_folder(self, folder_path: str) -> List[str]:
        """Load all text documents from a folder."""
        try:
            # Handle Windows absolute paths in Docker
            if folder_path.startswith('D:\\'):
                # D:\workplace\project_management\my_github\test -> /app/host_github/test
                folder_path = folder_path.replace('D:\\workplace\\project_management\\my_github', '/app/host_github')
                folder_path = folder_path.replace('\\', '/')
                logger.info(f"Mapped Windows path to Docker path: {folder_path}")
            elif folder_path.startswith('D:'):
                # D:\workplace\project_management\my_github\test -> /app/host_github/test
                folder_path = folder_path.replace('D:\\workplace\\project_management\\my_github', '/app/host_github')
                folder_path = folder_path.replace('\\', '/')
                logger.info(f"Mapped Windows path to Docker path: {folder_path}")
            
            folder_path = Path(folder_path)
            logger.info(f"Looking for documents in: {folder_path}")
            logger.info(f"Path exists: {folder_path.exists()}")
            
            if not folder_path.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            documents = []
            for file_path in folder_path.rglob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            documents.append(content)
                            logger.info(f"Loaded document: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
            
            logger.info(f"Loaded {len(documents)} documents from {folder_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from {folder_path}: {e}")
            raise
    
    def chunk_documents(self, documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split documents into chunks."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = []
            for doc in documents:
                doc_chunks = text_splitter.split_text(doc)
                chunks.extend(doc_chunks)
            
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise
    
    def generate_qa_pairs_for_chunk(
        self, 
        chunk: str, 
        num_pairs: int = 3, 
        language: str = "繁體中文",
        personas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, str]]:
        """Generate QA pairs for a single chunk, optionally using specific personas."""
        try:
            # Load system prompt
            from ai.resources.prompt_manager import prompt_manager
            system_prompt = prompt_manager.get_qa_generation_prompt()
            
            # Build persona context if provided
            persona_context = ""
            if personas and len(personas) > 0:
                persona_context = "\n## 用戶角色參考\n\n請根據以下用戶角色來生成問題，確保問題能夠反映這些真實用戶的需求、背景和問問題的方式：\n\n"
                for i, persona in enumerate(personas[:num_pairs], 1):
                    persona_context += f"""
### 角色 {i}: {persona.get('persona_name', 'Unknown')}
- **職業背景**: {persona.get('basic_info', {}).get('occupation', 'N/A')}
- **年齡**: {persona.get('basic_info', {}).get('age', 'N/A')}
- **核心需求**: {', '.join(persona.get('needs_and_goals', {}).get('core_needs', [])[:2])}
- **痛點**: {', '.join(persona.get('needs_and_goals', {}).get('pain_points', [])[:2])}
- **決策風格**: {persona.get('behavior_patterns', {}).get('decision_style', 'N/A')}
- **溝通方式**: {persona.get('communication_style', {}).get('questioning_approach', 'N/A')}
- **問問題特色**: {persona.get('communication_style', {}).get('language_characteristics', 'N/A')}

"""
                persona_context += "\n**重要**: 每個問題都應該明確對應一個角色，並體現該角色的特徵、需求和提問方式。\n"
            
            # Prepare requirement 2
            requirement_2 = 'Each question MUST correspond to one of the personas listed above, reflecting their background, needs, pain points, and communication style' if personas else 'Each question should represent a different user persona/role'
            
            # Prepare requirement 9
            requirement_9 = '9. Include the persona name in the output to show which persona is asking' if personas else ''
            
            # Prepare diversity guidelines
            if personas:
                diversity_section = 'Use the personas provided above as the basis for generating questions. Each question should naturally emerge from a specific personas needs, pain points, and communication style.'
            else:
                diversity_section = '''Diversity Guidelines:
- Beginner: "I'm new to this, can you explain...?"
- Expert: "What are the advanced techniques for...?"
- Problem-solver: "I'm having trouble with..."
- Decision-maker: "Should I choose... or...?"
- Learner: "I want to understand..."
- Practical user: "How do I actually do...?"
- Researcher: "What are the latest trends in...?"
- Concerned user: "Is... safe?"'''
            
            # Prepare output format fields
            if personas:
                trailing_comma = ","
                persona_name_field = '"persona_name": "Name of the persona asking this question",'
                persona_id_field = '"persona_id": "persona_xxx"'
            else:
                trailing_comma = ""
                persona_name_field = ""
                persona_id_field = ""
            
            prompt = f"""{system_prompt}

## Task
Generate {num_pairs} diverse, authentic questions from different user perspectives about this content:

Text Content:
{chunk}

Language: {language}
{persona_context}

Requirements:
1. Generate exactly {num_pairs} unique questions
2. {requirement_2}
3. Vary the expertise level (beginner to expert)
4. Mix different question types and intents
5. Use diverse language patterns and expressions that match the persona's communication style
6. Include personal context and real-world scenarios from the persona's life
7. Avoid repetitive or generic questions
8. Make each question sound like a different person asking
{requirement_9}

{diversity_section}

Output Format:
```json
[
  {{
    "question": "A unique question from a specific user perspective",
    "answer": "Accurate answer based on the text",
    "context": "Relevant text fragment",
    "difficulty": "easy|medium|hard",
    "question_type": "practical|problem_solving|decision_making|learning|use_case"{trailing_comma}
    {persona_name_field}
    {persona_id_field}
  }}
]
```

Return only the JSON array.
"""
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean up the response content - remove markdown code blocks
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]   # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
            
            # Try to parse JSON response
            import json
            try:
                qa_pairs = json.loads(content)
                if not isinstance(qa_pairs, list):
                    raise ValueError("Response is not a list")
                
                # Validate structure
                required_fields = ["question", "answer", "context", "difficulty", "question_type"]
                for pair in qa_pairs:
                    if not all(key in pair for key in required_fields):
                        raise ValueError(f"Missing required fields in QA pair: {required_fields}")
                    
                    # Validate question_type
                    valid_types = ["practical", "problem_solving", "decision_making", "learning", "use_case"]
                    if pair.get("question_type") not in valid_types:
                        logger.warning(f"Invalid question_type: {pair.get('question_type')}, setting to 'practical'")
                        pair["question_type"] = "practical"
                
                # Validate the number of QA pairs
                if len(qa_pairs) != num_pairs:
                    logger.warning(f"Expected {num_pairs} QA pairs, got {len(qa_pairs)}")
                    if len(qa_pairs) < num_pairs:
                        # Generate additional fallback pairs to reach the target number
                        fallback_questions = [
                            "我想了解更多關於這個內容的資訊",
                            "這個內容的主要重點是什麼？",
                            "我該如何應用這些資訊？",
                            "這個內容對我有什麼幫助？",
                            "我需要知道什麼重要細節？"
                        ]
                        for i in range(len(qa_pairs), num_pairs):
                            question_idx = (i - len(qa_pairs)) % len(fallback_questions)
                            qa_pairs.append({
                                "question": fallback_questions[question_idx],
                                "answer": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                                "context": chunk,
                                "difficulty": "medium",
                                "question_type": "practical"
                            })
                        logger.info(f"Added fallback pairs to reach target count: {len(qa_pairs)}")
                    elif len(qa_pairs) > num_pairs:
                        # Truncate to the requested number
                        qa_pairs = qa_pairs[:num_pairs]
                        logger.info(f"Truncated to target count: {len(qa_pairs)}")
                
                logger.info(f"Generated {len(qa_pairs)} QA pairs for chunk")
                return qa_pairs
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {content}")
                # Return fallback QA pairs with the requested number
                fallback_questions = [
                    "我想了解更多關於這個內容的資訊",
                    "這個內容的主要重點是什麼？",
                    "我該如何應用這些資訊？",
                    "這個內容對我有什麼幫助？",
                    "我需要知道什麼重要細節？",
                    "這個內容適合我嗎？",
                    "我該從哪裡開始？",
                    "有什麼需要注意的地方嗎？"
                ]
                fallback_pairs = []
                for i in range(num_pairs):
                    question_idx = i % len(fallback_questions)
                    fallback_pairs.append({
                        "question": fallback_questions[question_idx],
                        "answer": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                        "context": chunk,
                        "difficulty": "medium",
                        "question_type": "practical"
                    })
                logger.warning(f"Using fallback QA pairs: {len(fallback_pairs)} pairs")
                return fallback_pairs
                
        except Exception as e:
            logger.error(f"Error generating QA pairs for chunk: {e}")
            # Return fallback QA pairs with the requested number
            fallback_questions = [
                "我想了解更多關於這個內容的資訊",
                "這個內容的主要重點是什麼？",
                "我該如何應用這些資訊？",
                "這個內容對我有什麼幫助？",
                "我需要知道什麼重要細節？",
                "這個內容適合我嗎？",
                "我該從哪裡開始？",
                "有什麼需要注意的地方嗎？"
            ]
            fallback_pairs = []
            for i in range(num_pairs):
                question_idx = i % len(fallback_questions)
                fallback_pairs.append({
                    "question": fallback_questions[question_idx],
                    "answer": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    "context": chunk,
                    "difficulty": "medium",
                    "question_type": "practical"
                })
            logger.warning(f"Using fallback QA pairs due to error: {len(fallback_pairs)} pairs")
            return fallback_pairs
    
    def generate_testset(
        self, 
        documents_folder: str, 
        output_folder: str, 
        chunk_size: int = 5000, 
        chunk_overlap: int = 200, 
        qa_per_chunk: int = 3, 
        language: str = "繁體中文",
        personas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate complete RAG testset."""
        try:
            logger.info(f"Starting RAG testset generation from {documents_folder}")
            
            # Load documents
            documents = self.load_documents_from_folder(documents_folder)
            if not documents:
                raise ValueError("No documents found in the specified folder")
            
            # Chunk documents
            chunks = self.chunk_documents(documents, chunk_size, chunk_overlap)
            if not chunks:
                raise ValueError("No chunks created from documents")
            
            # Generate QA pairs for each chunk
            all_qa_pairs = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                # Select different personas for each chunk if available
                chunk_personas = None
                if personas and len(personas) > 0:
                    # Rotate through personas for variety
                    start_idx = (i * qa_per_chunk) % len(personas)
                    chunk_personas = []
                    for j in range(qa_per_chunk):
                        persona_idx = (start_idx + j) % len(personas)
                        chunk_personas.append(personas[persona_idx])
                
                qa_pairs = self.generate_qa_pairs_for_chunk(chunk, qa_per_chunk, language, chunk_personas)
                
                # Add chunk metadata
                for pair in qa_pairs:
                    pair["chunk_id"] = i + 1
                    pair["chunk_size"] = len(chunk)
                    pair["source_folder"] = documents_folder
                
                all_qa_pairs.extend(qa_pairs)
            
            # Create temporary output directory in /app/outputs
            temp_output_path = Path("/app/outputs/rag_testset_temp")
            temp_output_path.mkdir(parents=True, exist_ok=True)
            
            # Save to Excel in temp location
            temp_excel_path = temp_output_path / "rag_testset.xlsx"
            df = pd.DataFrame(all_qa_pairs)
            df.to_excel(temp_excel_path, index=False)
            
            logger.info(f"RAG testset saved to temp location: {temp_excel_path}")
            
            # Move file from temp location to user-specified output folder
            import shutil
            
            # Determine final target path
            if output_folder.startswith('D:\\'):
                # D:\workplace\project_management\my_github\ -> /app/host_github/
                final_output_path = Path(output_folder.replace('D:\\workplace\\project_management\\my_github', '/app/host_github'))
                final_output_path = Path(str(final_output_path).replace('\\', '/'))
                logger.info(f"Mapped Windows output path to Docker path: {final_output_path}")
            elif output_folder.startswith('D:'):
                # D:\workplace\project_management\my_github\ -> /app/host_github/
                final_output_path = Path(output_folder.replace('D:\\workplace\\project_management\\my_github', '/app/host_github'))
                final_output_path = Path(str(final_output_path).replace('\\', '/'))
                logger.info(f"Mapped Windows output path to Docker path: {final_output_path}")
            elif os.path.isabs(output_folder):
                # Unix absolute path - use as is
                final_output_path = Path(output_folder)
                logger.info(f"Unix absolute path used as is: {final_output_path}")
            else:
                # Relative path - map to host_github volume mount for Docker compatibility
                # Handle both ./ and ../ relative paths
                if output_folder.startswith('./'):
                    # Remove ./ prefix
                    relative_path = output_folder[2:]
                elif output_folder.startswith('../'):
                    # Keep ../ as is for parent directory access
                    relative_path = output_folder
                else:
                    # No prefix, use as is
                    relative_path = output_folder
                
                final_output_path = Path('/app/host_github') / relative_path
                logger.info(f"Relative path mapped to host_github: {output_folder} -> {final_output_path}")
            
            # Create final target directory
            final_output_path.mkdir(parents=True, exist_ok=True)
            
            # Move file from temp to final location
            final_excel_path = final_output_path / "rag_testset.xlsx"
            shutil.move(str(temp_excel_path), str(final_excel_path))
            
            logger.info(f"RAG testset moved to final location: {final_excel_path}")
            
            # Verify file was moved successfully before cleanup
            if final_excel_path.exists():
                logger.info(f"File move verified: {final_excel_path}")
                # Clean up temp files
                shutil.rmtree(temp_output_path)
                logger.info(f"Cleaned up temp directory: {temp_output_path}")
            else:
                logger.error(f"File move failed: {final_excel_path}")
                raise FileNotFoundError(f"Failed to move file to {final_excel_path}")
            
            logger.info(f"Generated {len(all_qa_pairs)} QA pairs from {len(chunks)} chunks")
            
            return {
                "success": True,
                "excel_path": str(final_excel_path),
                "total_qa_pairs": len(all_qa_pairs),
                "total_chunks": len(chunks),
                "total_documents": len(documents),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "qa_per_chunk": qa_per_chunk
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG testset: {e}")
            return {
                "success": False,
                "error": str(e)
            }
