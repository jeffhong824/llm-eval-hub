"""
Agent Testset Generator

Generates Agent test sets from documents by creating task-oriented conversations.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from configs.settings import settings

logger = logging.getLogger(__name__)


class AgentTestsetGenerator:
    """Generate Agent test sets from documents."""
    
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
                # D:\workplace\project_management\my_github\ -> /app/host_github/
                folder_path = folder_path.replace('D:\\workplace\\project_management\\my_github', '/app/host_github')
                folder_path = folder_path.replace('\\', '/')
                logger.info(f"Mapped Windows path to Docker path: {folder_path}")
            elif folder_path.startswith('D:'):
                # Handle D: without \\
                folder_path = folder_path.replace('D:\\workplace\\project_management\\my_github', '/app/host_github')
                folder_path = folder_path.replace('\\', '/')
                logger.info(f"Mapped Windows path to Docker path: {folder_path}")
            
            folder_path = Path(folder_path)
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
    
    def generate_agent_tasks_for_chunk(self, chunk: str, num_tasks: int = 3, language: str = "繁體中文") -> List[Dict[str, Any]]:
        """Generate task-oriented conversations for a single chunk."""
        try:
            # Load system prompt
            from ai.resources.prompt_manager import prompt_manager
            system_prompt = prompt_manager.get_agent_task_generation_prompt()
            
            prompt = f"""{system_prompt}

## Specific Task
Please generate {num_tasks} diverse task scenarios with clear user personas based on the following text fragment:

Text Fragment:
{chunk}

Language Requirement: {language}

Please ensure:
1. Generate exactly {num_tasks} task scenarios
2. Each task has a distinct, well-defined user persona
3. Task types are diverse, covering different user needs
4. User personas are realistic and memorable
5. Tasks focus on clear success criteria for evaluation
6. Use natural, conversational language patterns
7. Each task should be practical and realistic

Return Format:
```json
[
  {{
    "task_id": "task_1",
    "task_type": "information_retrieval|problem_resolution|decision_support|learning_education|process_guidance",
    "user_persona": {{
      "role": "Specific user role/background",
      "background": "Detailed user background and context",
      "communication_style": "How this user typically communicates",
      "motivation": "Why the user is seeking help"
    }},
    "task_context": "Detailed description of what the user is trying to accomplish",
    "initial_request": "The user's first message to the Agent",
    "expected_outcome": "What the Agent should provide to successfully help the user",
    "success_criteria": "Specific criteria for evaluating Agent performance",
    "difficulty": "easy|medium|hard",
    "context_used": "Relevant text fragment from the source material"
  }}
]
```

Return only the JSON array, no additional text.
"""
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            logger.info(f"LLM response length: {len(content)}")
            logger.info(f"LLM response preview: {content[:500]}...")
            
            # Clean up the response content - remove markdown code blocks
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]   # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
            
            logger.info(f"Cleaned response preview: {content[:500]}...")
            
            # Try to parse JSON response
            import json
            try:
                tasks = json.loads(content)
                logger.info(f"Successfully parsed JSON, got {len(tasks)} tasks")
                if not isinstance(tasks, list):
                    raise ValueError("Response is not a list")
                
                # Validate structure
                required_fields = ["task_id", "task_type", "user_persona", "task_context", "initial_request", "expected_outcome", "success_criteria", "difficulty", "context_used"]
                for task in tasks:
                    if not all(key in task for key in required_fields):
                        raise ValueError(f"Missing required fields in task: {required_fields}")
                    
                    # Validate task_type
                    valid_types = ["information_retrieval", "problem_resolution", "decision_support", "learning_education", "process_guidance"]
                    if task.get("task_type") not in valid_types:
                        logger.warning(f"Invalid task_type: {task.get('task_type')}, setting to 'information_retrieval'")
                        task["task_type"] = "information_retrieval"
                    
                    # Validate user_persona structure
                    if not isinstance(task.get("user_persona"), dict):
                        raise ValueError("user_persona must be a dictionary")
                    
                    persona_required_fields = ["role", "background", "communication_style", "motivation"]
                    if not all(key in task["user_persona"] for key in persona_required_fields):
                        raise ValueError(f"Missing required fields in user_persona: {persona_required_fields}")
                
                # Validate the number of tasks
                if len(tasks) != num_tasks:
                    logger.warning(f"Expected {num_tasks} tasks, got {len(tasks)}")
                    if len(tasks) < num_tasks:
                        # Generate additional fallback tasks to reach the target number
                        for i in range(len(tasks), num_tasks):
                            tasks.append({
                                "task_id": f"fallback_task_{i+1}",
                                "task_type": "information_retrieval",
                                "user_persona": {
                                    "role": "一般用戶",
                                    "background": "對文本內容感興趣的普通用戶",
                                    "communication_style": "直接、友善的詢問方式",
                                    "motivation": "希望理解文本的主要內容和重點"
                                },
                                "task_context": f"用戶需要理解第{i+1}段文本內容的主要重點和關鍵信息",
                                "initial_request": "你能幫我理解這段信息嗎？這段文本的主要重點是什麼？",
                                "expected_outcome": "Agent 能夠清楚總結文本的主要重點和關鍵信息",
                                "success_criteria": "Agent 提供準確、完整的文本摘要，幫助用戶理解核心內容",
                                "difficulty": "medium",
                                "context_used": chunk[:200] + "..." if len(chunk) > 200 else chunk
                            })
                        logger.info(f"Added fallback tasks to reach target count: {len(tasks)}")
                    elif len(tasks) > num_tasks:
                        # Truncate to the requested number
                        tasks = tasks[:num_tasks]
                        logger.info(f"Truncated to target count: {len(tasks)}")
                
                logger.info(f"Generated {len(tasks)} agent tasks for chunk")
                return tasks
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {content}")
                # Return fallback tasks with the requested number
                fallback_tasks = []
                for i in range(num_tasks):
                    fallback_tasks.append({
                        "task_id": f"fallback_task_{i+1}",
                        "task_type": "information_retrieval",
                        "user_persona": {
                            "role": "一般用戶",
                            "background": "對文本內容感興趣的普通用戶",
                            "communication_style": "直接、友善的詢問方式",
                            "motivation": "希望理解文本的主要內容和重點"
                        },
                        "task_context": f"用戶需要理解第{i+1}段文本內容的主要重點和關鍵信息",
                        "initial_request": "你能幫我理解這段信息嗎？這段文本的主要重點是什麼？",
                        "expected_outcome": "Agent 能夠清楚總結文本的主要重點和關鍵信息",
                        "success_criteria": "Agent 提供準確、完整的文本摘要，幫助用戶理解核心內容",
                        "difficulty": "medium",
                        "context_used": chunk[:200] + "..." if len(chunk) > 200 else chunk
                    })
                logger.warning(f"Using fallback tasks due to JSON decode error: {len(fallback_tasks)} tasks")
                return fallback_tasks
            except ValueError as e:
                logger.error(f"Validation error: {e}")
                logger.error(f"Response content: {content}")
                # Return fallback tasks with the requested number
                fallback_tasks = []
                for i in range(num_tasks):
                    fallback_tasks.append({
                        "task_id": f"validation_fallback_task_{i+1}",
                        "task_type": "information_retrieval",
                        "user_persona": {
                            "role": "一般用戶",
                            "background": "對文本內容感興趣的普通用戶",
                            "communication_style": "直接、友善的詢問方式",
                            "motivation": "希望理解文本的主要內容和重點"
                        },
                        "task_context": f"用戶需要理解第{i+1}段文本內容的主要重點和關鍵信息",
                        "initial_request": "你能幫我理解這段信息嗎？這段文本的主要重點是什麼？",
                        "expected_outcome": "Agent 能夠清楚總結文本的主要重點和關鍵信息",
                        "success_criteria": "Agent 提供準確、完整的文本摘要，幫助用戶理解核心內容",
                        "difficulty": "medium",
                        "context_used": chunk[:200] + "..." if len(chunk) > 200 else chunk
                    })
                logger.warning(f"Using fallback tasks due to validation error: {len(fallback_tasks)} tasks")
                return fallback_tasks
                
        except Exception as e:
            logger.error(f"Error generating agent tasks for chunk: {e}")
            # Return fallback tasks with the requested number
            fallback_tasks = []
            for i in range(num_tasks):
                fallback_tasks.append({
                    "task_id": f"error_task_{i+1}",
                    "task_type": "information_retrieval",
                    "user_persona": {
                        "role": "一般用戶",
                        "background": "對文本內容感興趣的普通用戶",
                        "communication_style": "直接、友善的詢問方式",
                        "motivation": "希望理解文本的主要內容和重點"
                    },
                    "task_context": f"用戶需要理解第{i+1}段文本內容的主要重點和關鍵信息",
                    "initial_request": "你能幫我理解這段信息嗎？這段文本的主要重點是什麼？",
                    "expected_outcome": "Agent 能夠清楚總結文本的主要重點和關鍵信息",
                    "success_criteria": "Agent 提供準確、完整的文本摘要，幫助用戶理解核心內容",
                    "difficulty": "medium",
                    "context_used": chunk[:200] + "..." if len(chunk) > 200 else chunk
                })
            logger.warning(f"Using fallback tasks due to error: {len(fallback_tasks)} tasks")
            return fallback_tasks
    
    def generate_testset(self, documents_folder: str, output_folder: str, 
                        chunk_size: int = 5000, chunk_overlap: int = 200, 
                        tasks_per_chunk: int = 3, language: str = "繁體中文") -> Dict[str, Any]:
        """Generate complete Agent testset."""
        try:
            logger.info(f"Starting Agent testset generation from {documents_folder}")
            
            # Load documents
            documents = self.load_documents_from_folder(documents_folder)
            if not documents:
                raise ValueError("No documents found in the specified folder")
            
            # Chunk documents
            chunks = self.chunk_documents(documents, chunk_size, chunk_overlap)
            if not chunks:
                raise ValueError("No chunks created from documents")
            
            # Generate tasks for each chunk
            all_tasks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                tasks = self.generate_agent_tasks_for_chunk(chunk, tasks_per_chunk, language)
                
                # Add chunk metadata
                for task in tasks:
                    task["chunk_id"] = i + 1
                    task["chunk_size"] = len(chunk)
                    task["source_folder"] = documents_folder
                
                all_tasks.extend(tasks)
            
            # Create temporary output directory in /app/outputs
            temp_output_path = Path("/app/outputs/agent_testset_temp")
            temp_output_path.mkdir(parents=True, exist_ok=True)
            
            # Save to Excel in temp location
            temp_excel_path = temp_output_path / "agent_testset.xlsx"
            
            # Flatten user_persona data for Excel
            flattened_data = []
            for task in all_tasks:
                base_task = {k: v for k, v in task.items() if k != "user_persona"}
                # Flatten user_persona into separate columns
                persona = task.get("user_persona", {})
                base_task["user_role"] = persona.get("role", "")
                base_task["user_background"] = persona.get("background", "")
                base_task["communication_style"] = persona.get("communication_style", "")
                base_task["user_motivation"] = persona.get("motivation", "")
                flattened_data.append(base_task)
            
            df = pd.DataFrame(flattened_data)
            df.to_excel(temp_excel_path, index=False)
            
            logger.info(f"Agent testset saved to temp location: {temp_excel_path}")
            
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
            final_excel_path = final_output_path / "agent_testset.xlsx"
            shutil.move(str(temp_excel_path), str(final_excel_path))
            
            logger.info(f"Agent testset moved to final location: {final_excel_path}")
            
            # Verify file was moved successfully before cleanup
            if final_excel_path.exists():
                logger.info(f"File move verified: {final_excel_path}")
                # Clean up temp files
                shutil.rmtree(temp_output_path)
                logger.info(f"Cleaned up temp directory: {temp_output_path}")
            else:
                logger.error(f"File move failed: {final_excel_path}")
                raise FileNotFoundError(f"Failed to move file to {final_excel_path}")
            
            logger.info(f"Generated {len(all_tasks)} tasks from {len(chunks)} chunks")
            
            return {
                "success": True,
                "excel_path": str(final_excel_path),
                "total_tasks": len(all_tasks),
                "total_chunks": len(chunks),
                "total_documents": len(documents),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "tasks_per_chunk": tasks_per_chunk
            }
            
        except Exception as e:
            logger.error(f"Error generating Agent testset: {e}")
            return {
                "success": False,
                "error": str(e)
            }
