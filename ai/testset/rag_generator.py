"""
RAG Testset Generator

Generates RAG test sets from documents using RAGAS framework.
Loads documents, chunks them, and generates QA pairs using RAGAS.
"""

# Import pydantic patch FIRST, before any langchain imports
import ai.core.pydantic_patch  # noqa: F401

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from configs.settings import settings

# Try to import RAGAS components
try:
    from ragas.testset import TestsetGenerator
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.synthesizers import (
        SingleHopSpecificQuerySynthesizer,
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("RAGAS not available, falling back to LLM-based generation")

logger = logging.getLogger(__name__)


class RAGTestsetGenerator:
    """Generate RAG test sets from documents using RAGAS framework."""
    
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-3.5-turbo"):
        self.model_provider = model_provider
        self.model_name = model_name
        self.llm = self._create_llm()
        self.embeddings = self._create_embeddings()
        self.use_ragas = RAGAS_AVAILABLE
    
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
    
    def _create_embeddings(self):
        """Create embeddings instance for RAGAS."""
        if self.model_provider == "openai":
            return OpenAIEmbeddings(api_key=settings.openai_api_key)
        elif self.model_provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=settings.gemini_api_key
            )
        else:
            # Fallback to OpenAI embeddings
            return OpenAIEmbeddings(api_key=settings.openai_api_key)
    
    def load_documents_from_folder(self, folder_path: str, selected_files: Optional[List[str]] = None) -> List[Document]:
        """
        Load documents from a folder and return as LangChain Documents.
        
        Args:
            folder_path: Path to the documents folder
            selected_files: Optional list of file names to load. If None, load all .txt files.
        
        Returns:
            List of Document objects
        """
        try:
            from ai.testset.output_manager import OutputManager
            
            # Map path for Docker compatibility
            folder_path = OutputManager.map_path_for_docker(folder_path)
            
            # Handle Docker paths - map to local if not in Docker
            import os
            is_docker = os.path.exists('/app') and os.path.isdir('/app')
            if folder_path.startswith('/app/') and not is_docker:
                # Map to local path
                relative_path = folder_path.replace('/app/', '')
                current_dir = Path.cwd()
                project_root = current_dir
                if (current_dir / "outputs").exists():
                    project_root = current_dir
                else:
                    for parent in current_dir.parents:
                        if (parent / "outputs").exists():
                            project_root = parent
                            break
                folder_path = str(project_root / relative_path)
            
            folder_path = Path(folder_path)
            logger.info(f"Loading documents from: {folder_path}")
            
            if not folder_path.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            # If selected_files is provided, only load those specific files
            if selected_files:
                documents = []
                for filename in selected_files:
                    file_path = folder_path / filename
                    if file_path.exists():
                        try:
                            # Handle different file types
                            if filename.endswith('.txt'):
                                loader = TextLoader(str(file_path), encoding="utf-8")
                                docs = loader.load()
                                documents.extend(docs)
                                logger.info(f"Loaded .txt file: {filename}")
                            elif filename.endswith('.md'):
                                loader = TextLoader(str(file_path), encoding="utf-8")
                                docs = loader.load()
                                documents.extend(docs)
                                logger.info(f"Loaded .md file: {filename}")
                            elif filename.endswith('.pdf'):
                                # Try to load PDF files using PyPDFLoader if available
                                try:
                                    from langchain_community.document_loaders import PyPDFLoader
                                    loader = PyPDFLoader(str(file_path))
                                    docs = loader.load()
                                    documents.extend(docs)
                                    logger.info(f"Loaded .pdf file: {filename}")
                                except ImportError:
                                    logger.warning(f"PyPDFLoader not available, skipping PDF file: {filename}")
                                except Exception as e:
                                    logger.warning(f"Failed to load PDF file {filename}: {e}")
                            elif filename.endswith('.json'):
                                # Load JSON files (document metadata files)
                                try:
                                    import json
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        json_data = json.load(f)
                                    
                                    # Handle both single document dict and list of documents
                                    if isinstance(json_data, list):
                                        doc_list = json_data
                                    elif isinstance(json_data, dict):
                                        # If it's a single document or metadata file
                                        if 'documents' in json_data:
                                            doc_list = json_data['documents']
                                        else:
                                            doc_list = [json_data]
                                    else:
                                        logger.warning(f"Unexpected JSON structure in {filename}")
                                        continue
                                    
                                    # Convert JSON documents to Document objects
                                    for doc_dict in doc_list:
                                        # Extract content from document
                                        content = doc_dict.get('content', '')
                                        if not content:
                                            # Try to construct content from other fields
                                            title = doc_dict.get('title', 'Untitled')
                                            category = doc_dict.get('category', 'N/A')
                                            content = f"# {title}\n\n分類: {category}\n\n{doc_dict.get('description', '')}"
                                        
                                        # Create Document object
                                        from langchain_core.documents import Document
                                        doc = Document(
                                            page_content=content,
                                            metadata={
                                                "source": str(file_path),
                                                "document_id": doc_dict.get('document_id', 'unknown'),
                                                "title": doc_dict.get('title', 'Untitled'),
                                                "category": doc_dict.get('category', 'N/A'),
                                                "target_audience": doc_dict.get('target_audience', 'N/A'),
                                                "complexity_level": doc_dict.get('complexity_level', 'N/A')
                                            }
                                        )
                                        documents.append(doc)
                                    
                                    logger.info(f"Loaded .json file: {filename} ({len(doc_list)} documents)")
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Invalid JSON in {filename}: {e}")
                                except Exception as e:
                                    logger.warning(f"Failed to load JSON file {filename}: {e}")
                            else:
                                logger.warning(f"Unsupported file type: {filename}")
                        except Exception as e:
                            logger.warning(f"Failed to load {filename}: {e}")
                    else:
                        logger.warning(f"File not found: {filename}")
                
                logger.info(f"Loaded {len(documents)} documents from {len(selected_files)} selected files")
            else:
                # Load all supported files (.txt, .md, .pdf, .json) from the specified folder (non-recursive)
                documents = []
                
                # Load .txt and .md files
                for ext in ['*.txt', '*.md']:
                    loader = DirectoryLoader(
                        str(folder_path),
                        glob=ext,
                        loader_cls=TextLoader,
                        loader_kwargs={"encoding": "utf-8"},
                        silent_errors=True
                    )
                    docs = loader.load()
                    documents.extend(docs)
                
                # Load .pdf files
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    for pdf_file in folder_path.glob("*.pdf"):
                        try:
                            loader = PyPDFLoader(str(pdf_file))
                            docs = loader.load()
                            documents.extend(docs)
                        except Exception as e:
                            logger.warning(f"Failed to load PDF {pdf_file.name}: {e}")
                except ImportError:
                    logger.warning("PyPDFLoader not available, skipping PDF files")
                
                # Load .json files (document metadata files)
                for json_file in folder_path.glob("*.json"):
                    # Skip metadata files that contain lists of documents
                    if json_file.name.endswith('_documents_metadata.json'):
                        try:
                            import json
                            with open(json_file, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            
                            # Handle both single document dict and list of documents
                            if isinstance(json_data, list):
                                doc_list = json_data
                            elif isinstance(json_data, dict):
                                if 'documents' in json_data:
                                    doc_list = json_data['documents']
                                else:
                                    doc_list = [json_data]
                            else:
                                continue
                            
                            # Convert JSON documents to Document objects
                            for doc_dict in doc_list:
                                content = doc_dict.get('content', '')
                                if not content:
                                    title = doc_dict.get('title', 'Untitled')
                                    category = doc_dict.get('category', 'N/A')
                                    content = f"# {title}\n\n分類: {category}\n\n{doc_dict.get('description', '')}"
                                
                                from langchain_core.documents import Document
                                doc = Document(
                                    page_content=content,
                                    metadata={
                                        "source": str(json_file),
                                        "document_id": doc_dict.get('document_id', 'unknown'),
                                        "title": doc_dict.get('title', 'Untitled'),
                                        "category": doc_dict.get('category', 'N/A'),
                                        "target_audience": doc_dict.get('target_audience', 'N/A'),
                                        "complexity_level": doc_dict.get('complexity_level', 'N/A')
                                    }
                                )
                                documents.append(doc)
                            
                            logger.info(f"Loaded JSON metadata file: {json_file.name} ({len(doc_list)} documents)")
                        except Exception as e:
                            logger.warning(f"Failed to load JSON file {json_file.name}: {e}")
                
                logger.info(f"Loaded {len(documents)} documents from {folder_path}")
            
            # Log which files were loaded for debugging
            if documents:
                loaded_files = [doc.metadata.get("source", "unknown") for doc in documents]
                logger.info(f"Loaded files: {loaded_files[:10]}{'...' if len(loaded_files) > 10 else ''}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from {folder_path}: {e}")
            raise
    
    def chunk_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Split documents into chunks using RecursiveCharacterTextSplitter."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunked_docs = []
            for doc in documents:
                chunks = text_splitter.split_documents([doc])
                chunked_docs.extend(chunks)
            
            logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise
    
    async def generate_qa_pairs_for_chunk(
        self, 
        chunk: str, 
        num_pairs: int = 3, 
        language: str = "繁體中文",
        personas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, str]]:
        """Generate QA pairs for a single chunk, optionally using specific personas (async version)."""
        try:
            # Load system prompt
            from ai.resources.prompt_manager import prompt_manager
            system_prompt = prompt_manager.get_qa_generation_prompt()
            
            # Build persona context if provided
            persona_context = ""
            if personas and len(personas) > 0:
                persona_context = "\n## 用戶角色參考\n\n請根據以下用戶角色來生成問題，確保問題能夠反映這些真實用戶的需求、背景和問問題的方式：\n\n"
                for i, persona in enumerate(personas[:num_pairs], 1):
                    # Support both needs_and_goals and needs_and_pain_points for backward compatibility
                    needs_data = persona.get('needs_and_goals', {}) or persona.get('needs_and_pain_points', {})
                    
                    persona_context += f"""
### 角色 {i}: {persona.get('persona_name', 'Unknown')}
- **職業背景**: {persona.get('basic_info', {}).get('occupation', 'N/A')}
- **年齡**: {persona.get('basic_info', {}).get('age', 'N/A')}
- **核心需求**: {', '.join(needs_data.get('core_needs', [])[:2])}
- **痛點**: {', '.join(needs_data.get('pain_points', [])[:2])}
- **表達清晰度**: {persona.get('communication_style', {}).get('clarity_level', 'N/A')}

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
5. Use language patterns that match the persona's clarity level
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
    "question_type": "practical|problem_solving|decision_making|learning|use_case|research"{trailing_comma}
    {persona_name_field}
    {persona_id_field}
  }}
]
```

Return only the JSON array.
"""
            
            # Use async invoke to avoid blocking the event loop
            response = await self.llm.ainvoke(prompt)
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
                    
                    # Validate question_type (support more types including research)
                    valid_types = ["practical", "problem_solving", "decision_making", "learning", "use_case", "research"]
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
    
    async def _generate_testset_with_ragas(
        self,
        chunked_documents: List[Document],
        testset_size: int = 30,
        personas: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """Generate testset using RAGAS framework."""
        try:
            if not RAGAS_AVAILABLE:
                raise ImportError("RAGAS is not available")
            
            logger.info("Using RAGAS framework to generate testset")
            
            # Wrap LLM and embeddings for RAGAS
            ragas_llm = LangchainLLMWrapper(self.llm)
            ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
            
            # Create knowledge graph from chunked documents
            kg = KnowledgeGraph()
            for doc in chunked_documents:
                kg.nodes.append(
                    Node(
                        type=NodeType.DOCUMENT,
                        properties={
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        }
                    )
                )
            
            logger.info(f"Created knowledge graph with {len(kg.nodes)} nodes")
            
            # Apply transforms to enrich the knowledge graph
            # This creates additional node types (entities, relations) that RAGAS needs
            try:
                from ragas.testset.transforms import default_transforms, apply_transforms
                transforms = default_transforms(
                    documents=chunked_documents,
                    llm=ragas_llm,
                    embedding_model=ragas_embeddings
                )
                # apply_transforms might return a coroutine in some versions, handle both cases
                result = apply_transforms(kg, transforms)
                if hasattr(result, '__await__'):
                    # It's a coroutine, await it
                    await result
                logger.info(f"Applied transforms, knowledge graph now has {len(kg.nodes)} nodes")
            except Exception as transform_error:
                logger.warning(f"Failed to apply transforms: {transform_error}, continuing with basic graph")
            
            # Verify we have enough nodes
            if len(kg.nodes) == 0:
                raise ValueError("Knowledge graph has no nodes after creation")
            
            # Create testset generator
            generator = TestsetGenerator(
                llm=ragas_llm,
                embedding_model=ragas_embeddings,
                knowledge_graph=kg
            )
            
            # Define query distribution (focus on single-hop for simplicity)
            # Use only SingleHopSpecificQuerySynthesizer to avoid filter issues
            query_distribution = [
                (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 1.0),
            ]
            
            # Generate testset with error handling
            logger.info(f"Generating {testset_size} QA pairs using RAGAS...")
            try:
                dataset = generator.generate(
                    testset_size=testset_size,
                    query_distribution=query_distribution
                )
                
                # Convert to pandas DataFrame
                df = dataset.to_pandas()
                logger.info(f"Generated {len(df)} QA pairs using RAGAS")
                
                # Add persona information if available
                if personas and len(personas) > 0:
                    # Distribute personas across QA pairs
                    for idx, row in df.iterrows():
                        persona_idx = idx % len(personas)
                        persona = personas[persona_idx]
                        df.at[idx, 'persona_id'] = persona.get('persona_id', f"persona_{persona_idx}")
                        df.at[idx, 'persona_name'] = persona.get('persona_name', 'Unknown')
                
                return df
            except Exception as gen_error:
                error_msg = str(gen_error)
                if "No nodes that satisfied" in error_msg or "filter" in error_msg.lower():
                    logger.warning(f"RAGAS filter error: {error_msg}, falling back to LLM generation")
                    raise ValueError("RAGAS filter error, use LLM fallback")
                else:
                    raise
            
        except (ImportError, ValueError) as e:
            # Re-raise these as they indicate we should use LLM fallback
            logger.warning(f"RAGAS generation failed: {e}, will use LLM fallback")
            raise
        except Exception as e:
            logger.error(f"Error generating testset with RAGAS: {e}")
            raise
    
    async def generate_testset_async(
        self, 
        documents_folder: str, 
        output_folder: str, 
        chunk_size: int = 5000, 
        chunk_overlap: int = 200, 
        qa_per_chunk: int = 3, 
        language: str = "繁體中文",
        personas: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[callable] = None,
        selected_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate complete RAG testset using RAGAS or fallback to LLM (async version)."""
        try:
            logger.info(f"Starting RAG testset generation from {documents_folder}")
            if selected_files:
                logger.info(f"Using selected files: {selected_files}")
            
            # Load documents (with optional file selection)
            documents = self.load_documents_from_folder(documents_folder, selected_files=selected_files)
            if not documents:
                raise ValueError("No documents found in the specified folder")
            
            # Chunk documents
            chunked_docs = self.chunk_documents(documents, chunk_size, chunk_overlap)
            if not chunked_docs:
                raise ValueError("No chunks created from documents")
            
            logger.info(f"Loaded {len(documents)} documents, created {len(chunked_docs)} chunks")
            
            # Send initial info via progress callback
            if progress_callback:
                # Get list of loaded file names and their status
                loaded_file_names = []
                failed_files = []
                
                # Track which selected files were successfully loaded
                if selected_files:
                    for filename in selected_files:
                        # Check if this file was loaded by searching documents
                        file_loaded = False
                        for doc in documents:
                            if isinstance(doc, Document) and doc.metadata:
                                source = doc.metadata.get("source", "unknown")
                                import os
                                source_filename = os.path.basename(source)
                                if source_filename == filename:
                                    file_loaded = True
                                    if filename not in loaded_file_names:
                                        loaded_file_names.append(filename)
                                    break
                        if not file_loaded:
                            failed_files.append(filename)
                else:
                    # No file selection, get all loaded files
                    for doc in documents:
                        if isinstance(doc, Document) and doc.metadata:
                            source = doc.metadata.get("source", "unknown")
                            import os
                            filename = os.path.basename(source)
                            if filename not in loaded_file_names:
                                loaded_file_names.append(filename)
                
                await progress_callback({
                    "type": "generation_info",
                    "message": "文件載入完成，開始生成 QA Pairs",
                    "loaded_files": loaded_file_names,
                    "failed_files": failed_files,
                    "total_files": len(loaded_file_names),
                    "total_documents": len(documents),
                    "total_chunks": len(chunked_docs),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "qa_per_chunk": qa_per_chunk,
                    "selected_files": selected_files if selected_files else None,
                    "expected_qa_pairs": len(chunked_docs) * qa_per_chunk
                })
            
            # Try to use RAGAS if available
            all_qa_pairs = []
            chunks = []
            ragas_success = False
            
            if self.use_ragas and RAGAS_AVAILABLE:
                # Calculate testset size based on chunks and qa_per_chunk
                testset_size = min(len(chunked_docs) * qa_per_chunk, 100)  # Cap at 100
                
                # Generate using RAGAS (await since we're in async context)
                try:
                    df = await self._generate_testset_with_ragas(
                        chunked_docs,
                        testset_size=testset_size,
                        personas=personas
                    )
                    
                    all_qa_pairs = df.to_dict('records')
                    chunks = chunked_docs  # For return value
                    ragas_success = True
                    logger.info(f"Successfully generated {len(all_qa_pairs)} QA pairs using RAGAS")
                except (ValueError, ImportError) as ragas_error:
                    # RAGAS failed with filter error or import error, fall back to LLM
                    logger.warning(f"RAGAS generation failed: {ragas_error}, falling back to LLM generation")
                    ragas_success = False
                except Exception as ragas_error:
                    # Other RAGAS errors, log and fall back
                    logger.error(f"RAGAS error: {ragas_error}, falling back to LLM generation")
                    ragas_success = False
            
            # Prepare output folder early for streaming writes
            from ai.testset.output_manager import OutputManager
            output_path = Path(OutputManager.map_path_for_docker(output_folder))
            
            # Handle Docker paths
            import os
            is_docker = os.path.exists('/app') and os.path.isdir('/app')
            if str(output_path).startswith('/app/') and not is_docker:
                relative_path = str(output_path).replace('/app/', '')
                current_dir = Path.cwd()
                project_root = current_dir
                if (current_dir / "outputs").exists():
                    project_root = current_dir
                else:
                    for parent in current_dir.parents:
                        if (parent / "outputs").exists():
                            project_root = parent
                            break
                output_path = project_root / relative_path
            
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            excel_path = output_path / "rag_testset.xlsx"
            
            # Use LLM generation if RAGAS not available or failed
            if not ragas_success:
                # Fallback to LLM-based generation with streaming writes
                logger.info("Using LLM-based generation (RAGAS not available)")
                all_qa_pairs = []
                chunks = []
                
                import asyncio
                from openpyxl import Workbook
                from openpyxl.writer.excel import save_workbook
                
                # Initialize Excel file for streaming writes
                wb = Workbook()
                ws = wb.active
                ws.title = "QA Pairs"
                
                # Write headers
                headers = ["question", "answer", "context", "difficulty", "question_type", 
                          "persona_name", "persona_id", "chunk_id", "chunk_size", "source_file"]
                ws.append(headers)
                
                # Save initial Excel file
                temp_excel = excel_path.with_suffix('.xlsx.tmp')
                wb.save(temp_excel)
                
                total_chunks = len(chunked_docs)
                processed_chunks = 0
                total_qa_pairs_generated = 0  # Track total QA pairs generated
                
                async def process_single_chunk(i: int, chunk_doc: Document):
                    """Process a single chunk asynchronously with streaming write."""
                    nonlocal processed_chunks, total_qa_pairs_generated
                    logger.info(f"Processing chunk {i+1}/{total_chunks}")
                    chunk_text = chunk_doc.page_content if isinstance(chunk_doc, Document) else chunk_doc
                    
                    # Select different personas for each chunk if available
                    chunk_personas = None
                    if personas and len(personas) > 0:
                        start_idx = (i * qa_per_chunk) % len(personas)
                        chunk_personas = []
                        for j in range(qa_per_chunk):
                            persona_idx = (start_idx + j) % len(personas)
                            chunk_personas.append(personas[persona_idx])
                    
                    # Use async version
                    qa_pairs = await self.generate_qa_pairs_for_chunk(chunk_text, qa_per_chunk, language, chunk_personas)
                    
                    # Add chunk metadata and write to Excel immediately
                    for pair in qa_pairs:
                        pair["chunk_id"] = i + 1
                        pair["chunk_size"] = len(chunk_text)
                        pair["source_folder"] = documents_folder
                        if isinstance(chunk_doc, Document) and chunk_doc.metadata:
                            pair["source_file"] = chunk_doc.metadata.get("source", "unknown")
                        
                        # Write row to Excel immediately
                        row = [
                            pair.get("question", ""),
                            pair.get("answer", ""),
                            pair.get("context", ""),
                            pair.get("difficulty", ""),
                            pair.get("question_type", ""),
                            pair.get("persona_name", ""),
                            pair.get("persona_id", ""),
                            pair.get("chunk_id", ""),
                            pair.get("chunk_size", ""),
                            pair.get("source_file", "")
                        ]
                        ws.append(row)
                        
                        # Save Excel file after each row (for real-time updates)
                        wb.save(temp_excel)
                        
                        # Call progress callback if provided
                        if progress_callback:
                            try:
                                total_qa_pairs_generated += 1
                                await progress_callback({
                                    "type": "qa_pair_generated",
                                    "qa_pair": {
                                        "question": pair.get("question", "")[:100] + "..." if len(pair.get("question", "")) > 100 else pair.get("question", ""),
                                        "persona_name": pair.get("persona_name", ""),
                                        "persona_id": pair.get("persona_id", "")
                                    },
                                    "total_generated": total_qa_pairs_generated,
                                    "chunk_progress": f"{i+1}/{total_chunks}",
                                    "progress_percent": int((i + 1) / total_chunks * 90) if total_chunks > 0 else 0
                                })
                            except Exception as e:
                                logger.warning(f"Progress callback error: {e}")
                    
                    processed_chunks += 1
                    chunks.append(chunk_text)
                    all_qa_pairs.extend(qa_pairs)
                    
                    return chunk_text, qa_pairs
                
                # Process chunks concurrently (limit concurrency to avoid overwhelming the API)
                semaphore = asyncio.Semaphore(5)  # Process up to 5 chunks concurrently
                
                async def process_with_semaphore(i: int, chunk_doc: Document):
                    async with semaphore:
                        return await process_single_chunk(i, chunk_doc)
                
                # Create tasks for all chunks
                tasks = [process_with_semaphore(i, chunk_doc) for i, chunk_doc in enumerate(chunked_docs)]
                
                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results (already processed in process_single_chunk, but check for errors)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error processing chunk: {result}")
                        continue
                
                # Send final progress update (95% - saving file)
                if progress_callback:
                    try:
                        await progress_callback({
                            "type": "saving_file",
                            "message": "正在保存 Excel 檔案...",
                            "progress_percent": 95
                        })
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
                
                # Final save: atomic rename
                try:
                    if excel_path.exists():
                        excel_path.unlink()
                    temp_excel.replace(excel_path)
                    
                    # Send completion progress (100%)
                    if progress_callback:
                        try:
                            await progress_callback({
                                "type": "file_saved",
                                "message": "Excel 檔案已保存",
                                "progress_percent": 100,
                                "excel_path": str(excel_path)
                            })
                        except Exception as e:
                            logger.warning(f"Progress callback error: {e}")
                except Exception as write_error:
                    logger.error(f"Error finalizing Excel file: {write_error}")
                    raise
            
            # Prepare output folder
            from ai.testset.output_manager import OutputManager
            output_path = Path(OutputManager.map_path_for_docker(output_folder))
            
            # Handle Docker paths
            import os
            is_docker = os.path.exists('/app') and os.path.isdir('/app')
            if str(output_path).startswith('/app/') and not is_docker:
                relative_path = str(output_path).replace('/app/', '')
                current_dir = Path.cwd()
                project_root = current_dir
                if (current_dir / "outputs").exists():
                    project_root = current_dir
                else:
                    for parent in current_dir.parents:
                        if (parent / "outputs").exists():
                            project_root = parent
                            break
                output_path = project_root / relative_path
            
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save to Excel using atomic write
            excel_path = output_path / "rag_testset.xlsx"
            temp_excel = excel_path.with_suffix('.xlsx.tmp')
            df = pd.DataFrame(all_qa_pairs)
            try:
                # Write to temp file first
                df.to_excel(temp_excel, index=False)
                # Atomic rename
                if excel_path.exists():
                    excel_path.unlink()
                temp_excel.replace(excel_path)
            except Exception as write_error:
                # Clean up temp file on error
                if temp_excel.exists():
                    try:
                        temp_excel.unlink()
                    except:
                        pass
                raise write_error
            
            logger.info(f"RAG testset saved to: {excel_path}")
            logger.info(f"Generated {len(all_qa_pairs)} QA pairs from {len(chunks)} chunks")
            
            return {
                "success": True,
                "excel_path": str(excel_path),
                "total_qa_pairs": len(all_qa_pairs),
                "total_chunks": len(chunks),
                "total_documents": len(documents),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "qa_per_chunk": qa_per_chunk,
                "method": "ragas" if (self.use_ragas and RAGAS_AVAILABLE) else "llm"
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG testset: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
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
        """Generate complete RAG testset using RAGAS or fallback to LLM (sync version for backward compatibility)."""
        import asyncio
        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we can't use asyncio.run()
                    # Fall back to sync LLM generation
                    logger.warning("Event loop is running, falling back to sync LLM generation")
                    return self._generate_testset_sync(
                        documents_folder, output_folder, chunk_size, chunk_overlap,
                        qa_per_chunk, language, personas
                    )
                else:
                    # Loop exists but not running, we can use it
                    return loop.run_until_complete(
                        self.generate_testset_async(
                            documents_folder, output_folder, chunk_size, chunk_overlap,
                            qa_per_chunk, language, personas
                        )
                    )
            except RuntimeError:
                # No event loop, create new one
                return asyncio.run(
                    self.generate_testset_async(
                        documents_folder, output_folder, chunk_size, chunk_overlap,
                        qa_per_chunk, language, personas
                    )
                )
        except Exception as e:
            logger.error(f"Error in generate_testset: {e}")
            # Fallback to sync generation
            return self._generate_testset_sync(
                documents_folder, output_folder, chunk_size, chunk_overlap,
                qa_per_chunk, language, personas
            )
    
    def _generate_testset_sync(
        self,
        documents_folder: str,
        output_folder: str,
        chunk_size: int = 5000,
        chunk_overlap: int = 200,
        qa_per_chunk: int = 3,
        language: str = "繁體中文",
        personas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Synchronous fallback for testset generation (LLM only, no RAGAS)."""
        try:
            logger.info(f"Starting sync RAG testset generation from {documents_folder}")
            
            # Load documents
            documents = self.load_documents_from_folder(documents_folder)
            if not documents:
                raise ValueError("No documents found in the specified folder")
            
            # Chunk documents
            chunked_docs = self.chunk_documents(documents, chunk_size, chunk_overlap)
            if not chunked_docs:
                raise ValueError("No chunks created from documents")
            
            logger.info(f"Loaded {len(documents)} documents, created {len(chunked_docs)} chunks")
            logger.info("Using LLM-based generation (sync mode)")
            
            # Generate QA pairs using LLM
            all_qa_pairs = []
            chunks = []
            for i, chunk_doc in enumerate(chunked_docs):
                logger.info(f"Processing chunk {i+1}/{len(chunked_docs)}")
                chunk_text = chunk_doc.page_content if isinstance(chunk_doc, Document) else chunk_doc
                chunks.append(chunk_text)
                
                # Select different personas for each chunk if available
                chunk_personas = None
                if personas and len(personas) > 0:
                    start_idx = (i * qa_per_chunk) % len(personas)
                    chunk_personas = []
                    for j in range(qa_per_chunk):
                        persona_idx = (start_idx + j) % len(personas)
                        chunk_personas.append(personas[persona_idx])
                
                qa_pairs = self.generate_qa_pairs_for_chunk(chunk_text, qa_per_chunk, language, chunk_personas)
                
                # Add chunk metadata
                for pair in qa_pairs:
                    pair["chunk_id"] = i + 1
                    pair["chunk_size"] = len(chunk_text)
                    pair["source_folder"] = documents_folder
                    if isinstance(chunk_doc, Document) and chunk_doc.metadata:
                        pair["source_file"] = chunk_doc.metadata.get("source", "unknown")
                
                all_qa_pairs.extend(qa_pairs)
            
            # Prepare output folder
            from ai.testset.output_manager import OutputManager
            output_path = Path(OutputManager.map_path_for_docker(output_folder))
            
            # Handle Docker paths
            import os
            is_docker = os.path.exists('/app') and os.path.isdir('/app')
            if str(output_path).startswith('/app/') and not is_docker:
                relative_path = str(output_path).replace('/app/', '')
                current_dir = Path.cwd()
                project_root = current_dir
                if (current_dir / "outputs").exists():
                    project_root = current_dir
                else:
                    for parent in current_dir.parents:
                        if (parent / "outputs").exists():
                            project_root = parent
                            break
                output_path = project_root / relative_path
            
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save to Excel using atomic write
            excel_path = output_path / "rag_testset.xlsx"
            temp_excel = excel_path.with_suffix('.xlsx.tmp')
            df = pd.DataFrame(all_qa_pairs)
            try:
                # Write to temp file first
                df.to_excel(temp_excel, index=False)
                # Atomic rename
                if excel_path.exists():
                    excel_path.unlink()
                temp_excel.replace(excel_path)
            except Exception as write_error:
                # Clean up temp file on error
                if temp_excel.exists():
                    try:
                        temp_excel.unlink()
                    except:
                        pass
                raise write_error
            
            logger.info(f"RAG testset saved to: {excel_path}")
            logger.info(f"Generated {len(all_qa_pairs)} QA pairs from {len(chunks)} chunks")
            
            return {
                "success": True,
                "excel_path": str(excel_path),
                "total_qa_pairs": len(all_qa_pairs),
                "total_chunks": len(chunks),
                "total_documents": len(documents),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "qa_per_chunk": qa_per_chunk,
                "method": "ragas" if (self.use_ragas and RAGAS_AVAILABLE) else "llm"
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG testset: {e}")
            return {
                "success": False,
                "error": str(e)
            }
