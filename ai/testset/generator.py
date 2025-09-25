"""Testset generation module for RAG and Agent systems."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json

import pandas as pd
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ai.document_loaders.loader import DocumentProcessor
from configs.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TestsetConfig:
    """Configuration for testset generation."""
    
    testset_size: int = 10
    single_hop_ratio: float = 0.5
    multi_hop_abstract_ratio: float = 0.25
    multi_hop_specific_ratio: float = 0.25
    include_personas: bool = True
    custom_personas: Optional[List[str]] = None
    save_knowledge_graph: bool = True
    enrich_with_transforms: bool = True


@dataclass
class GeneratedTestset:
    """Generated testset data structure."""
    
    testset_id: str
    timestamp: datetime
    testset_size: int
    documents_count: int
    generation_time: float
    testset_data: pd.DataFrame
    knowledge_graph_path: Optional[str] = None
    config: Optional[TestsetConfig] = None
    status: str = "success"
    error_message: Optional[str] = None


class TestsetGeneratorService:
    """Service for generating testsets for RAG and Agent systems."""
    
    def __init__(self, openai_api_key: str):
        """Initialize the testset generator service."""
        self.openai_api_key = openai_api_key
        
        # Initialize LLM and embeddings
        self.llm = LangchainLLMWrapper(
            ChatOpenAI(
                api_key=openai_api_key,
                model=settings.openai_model,
                temperature=0.1
            )
        )
        
        self.embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                api_key=openai_api_key,
                model="text-embedding-3-small"
            )
        )
        
        # Default personas for testset generation
        self.default_personas = [
            "A curious student learning about the topic",
            "A professional researcher seeking detailed information",
            "A general user with basic knowledge",
            "An expert looking for specific technical details",
            "A beginner asking fundamental questions"
        ]
    
    async def generate_rag_testset_from_folder(
        self,
        folder_path: str,
        config: Optional[TestsetConfig] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> GeneratedTestset:
        """
        Generate testset for RAG system evaluation from folder.
        
        Args:
            folder_path: Path to folder containing documents
            config: Testset generation configuration
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            GeneratedTestset with generated QA pairs
        """
        start_time = datetime.now()
        testset_id = f"rag_testset_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        if config is None:
            config = TestsetConfig()
        
        try:
            logger.info(f"Starting RAG testset generation from folder: {folder_path}")
            
            # Load and process documents
            doc_processor = DocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            documents = doc_processor.load_documents_from_folder(folder_path)
            
            if not documents:
                raise ValueError(f"No documents found in folder: {folder_path}")
            
            # Chunk documents
            chunked_documents = doc_processor.chunk_documents(documents)
            
            logger.info(f"Loaded {len(documents)} documents, created {len(chunked_documents)} chunks")
            
            # Create knowledge graph
            kg = await self._create_knowledge_graph(chunked_documents, config)
            
            # Generate testset
            testset_data = await self._generate_testset_data(kg, config)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = GeneratedTestset(
                testset_id=testset_id,
                timestamp=start_time,
                testset_size=len(testset_data),
                documents_count=len(documents),
                generation_time=generation_time,
                testset_data=testset_data,
                knowledge_graph_path=kg.save_path if hasattr(kg, 'save_path') else None,
                config=config,
                status="success"
            )
            
            logger.info(f"RAG testset generation completed in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating RAG testset: {str(e)}")
            return GeneratedTestset(
                testset_id=testset_id,
                timestamp=start_time,
                testset_size=0,
                documents_count=0,
                generation_time=(datetime.now() - start_time).total_seconds(),
                testset_data=pd.DataFrame(),
                config=config,
                status="error",
                error_message=str(e)
            )
    
    async def generate_agent_testset_with_excel(
        self,
        scenarios: List[Dict[str, Any]],
        config: Optional[TestsetConfig] = None,
        output_path: Optional[str] = None
    ) -> GeneratedTestset:
        """
        Generate testset for Agent system evaluation with Excel output.
        
        Args:
            scenarios: List of agent scenarios/tasks
            config: Testset generation configuration
            output_path: Path to save Excel file
            
        Returns:
            GeneratedTestset with generated agent test cases
        """
        start_time = datetime.now()
        testset_id = f"agent_testset_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        if config is None:
            config = TestsetConfig()
        
        try:
            logger.info(f"Starting agent testset generation with {len(scenarios)} scenarios")
            
            # Generate agent-specific testset data
            testset_data = await self._generate_agent_testset_data_from_scenarios(scenarios)
            
            # Save to Excel if output path provided
            excel_path = None
            if output_path:
                excel_path = await self._save_agent_testset_to_excel(testset_data, output_path, testset_id)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = GeneratedTestset(
                testset_id=testset_id,
                timestamp=start_time,
                testset_size=len(testset_data),
                documents_count=len(scenarios),
                generation_time=generation_time,
                testset_data=testset_data,
                knowledge_graph_path=excel_path,
                config=config,
                status="success"
            )
            
            logger.info(f"Agent testset generation completed in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating agent testset: {str(e)}")
            return GeneratedTestset(
                testset_id=testset_id,
                timestamp=start_time,
                testset_size=0,
                documents_count=len(scenarios),
                generation_time=(datetime.now() - start_time).total_seconds(),
                testset_data=pd.DataFrame(),
                config=config,
                status="error",
                error_message=str(e)
            )
    
    async def generate_from_user_scenarios(
        self,
        user_scenarios: List[str],
        config: Optional[TestsetConfig] = None
    ) -> GeneratedTestset:
        """
        Generate testset from user-provided scenarios.
        
        Args:
            user_scenarios: List of user-provided scenarios
            config: Testset generation configuration
            
        Returns:
            GeneratedTestset with generated test cases
        """
        start_time = datetime.now()
        testset_id = f"scenario_testset_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        if config is None:
            config = TestsetConfig()
        
        try:
            logger.info(f"Starting scenario-based testset generation with {len(user_scenarios)} scenarios")
            
            # Generate documents from scenarios
            documents = await self._generate_documents_from_scenarios(user_scenarios)
            
            # Create knowledge graph
            kg = await self._create_knowledge_graph(documents, config)
            
            # Generate testset
            testset_data = await self._generate_testset_data(kg, config)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = GeneratedTestset(
                testset_id=testset_id,
                timestamp=start_time,
                testset_size=len(testset_data),
                documents_count=len(documents),
                generation_time=generation_time,
                testset_data=testset_data,
                knowledge_graph_path=kg.save_path if hasattr(kg, 'save_path') else None,
                config=config,
                status="success"
            )
            
            logger.info(f"Scenario-based testset generation completed in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating scenario-based testset: {str(e)}")
            return GeneratedTestset(
                testset_id=testset_id,
                timestamp=start_time,
                testset_size=0,
                documents_count=len(user_scenarios),
                generation_time=(datetime.now() - start_time).total_seconds(),
                testset_data=pd.DataFrame(),
                config=config,
                status="error",
                error_message=str(e)
            )
    
    async def _create_knowledge_graph(
        self, 
        documents: List[Document], 
        config: TestsetConfig
    ) -> KnowledgeGraph:
        """Create and enrich knowledge graph from documents."""
        
        # Create knowledge graph
        kg = KnowledgeGraph()
        
        # Add documents to knowledge graph
        for doc in documents:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata
                    }
                )
            )
        
        logger.info(f"Created knowledge graph with {len(kg.nodes)} nodes")
        
        # Apply transformations if enabled
        if config.enrich_with_transforms:
            transforms = default_transforms(
                documents=documents,
                llm=self.llm,
                embedding_model=self.embeddings
            )
            
            apply_transforms(kg, transforms)
            logger.info(f"Applied transforms, knowledge graph now has {len(kg.nodes)} nodes")
        
        # Save knowledge graph if requested
        if config.save_knowledge_graph:
            kg_path = f"{settings.artifacts_dir}/knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            kg.save(kg_path)
            kg.save_path = kg_path
            logger.info(f"Saved knowledge graph to {kg_path}")
        
        return kg
    
    async def _generate_testset_data(
        self, 
        kg: KnowledgeGraph, 
        config: TestsetConfig
    ) -> pd.DataFrame:
        """Generate testset data from knowledge graph."""
        
        # Create testset generator
        generator = TestsetGenerator(
            llm=self.llm,
            embedding_model=self.embeddings,
            knowledge_graph=kg
        )
        
        # Define query distribution
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=self.llm), config.single_hop_ratio),
            (MultiHopAbstractQuerySynthesizer(llm=self.llm), config.multi_hop_abstract_ratio),
            (MultiHopSpecificQuerySynthesizer(llm=self.llm), config.multi_hop_specific_ratio),
        ]
        
        # Generate testset
        dataset = generator.generate(
            testset_size=config.testset_size,
            query_distribution=query_distribution
        )
        
        return dataset.to_pandas()
    
    async def _generate_agent_testset_data(
        self, 
        kg: KnowledgeGraph, 
        config: TestsetConfig,
        scenarios: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Generate agent-specific testset data."""
        
        # For agent testsets, we create scenarios that test different agent capabilities
        agent_test_cases = []
        
        for scenario in scenarios:
            # Generate test cases for each scenario
            test_case = {
                "question": scenario.get("question", ""),
                "context": scenario.get("context", ""),
                "expected_answer": scenario.get("expected_answer", ""),
                "scenario_type": scenario.get("type", "general"),
                "difficulty": scenario.get("difficulty", "medium"),
                "tools_required": scenario.get("tools", []),
                "expected_turns": scenario.get("expected_turns", 3)
            }
            agent_test_cases.append(test_case)
        
        return pd.DataFrame(agent_test_cases)
    
    def _scenarios_to_documents(self, scenarios: List[Dict[str, Any]]) -> List[Document]:
        """Convert agent scenarios to documents."""
        documents = []
        
        for scenario in scenarios:
            content = f"""
            Scenario: {scenario.get('description', '')}
            Type: {scenario.get('type', 'general')}
            Difficulty: {scenario.get('difficulty', 'medium')}
            Tools Required: {', '.join(scenario.get('tools', []))}
            Expected Outcome: {scenario.get('expected_outcome', '')}
            """
            
            doc = Document(
                page_content=content.strip(),
                metadata={
                    "scenario_id": scenario.get("id", ""),
                    "type": scenario.get("type", "general"),
                    "difficulty": scenario.get("difficulty", "medium")
                }
            )
            documents.append(doc)
        
        return documents
    
    async def _generate_documents_from_scenarios(self, scenarios: List[str]) -> List[Document]:
        """Generate documents from user-provided scenarios using LLM."""
        documents = []
        
        for i, scenario in enumerate(scenarios):
            # Use LLM to expand scenario into document
            prompt = f"""
            Expand the following scenario into a comprehensive document that could be used for generating test questions:
            
            Scenario: {scenario}
            
            Please provide:
            1. A detailed description of the scenario
            2. Key concepts and information
            3. Potential questions that could be asked
            4. Expected answers or outcomes
            
            Format as a structured document.
            """
            
            try:
                response = await self._call_llm(prompt)
                
                doc = Document(
                    page_content=response,
                    metadata={
                        "scenario_id": f"scenario_{i}",
                        "source": "user_provided",
                        "original_scenario": scenario
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Error expanding scenario {i}: {str(e)}")
                # Fallback to original scenario
                doc = Document(
                    page_content=scenario,
                    metadata={
                        "scenario_id": f"scenario_{i}",
                        "source": "user_provided",
                        "expanded": False
                    }
                )
                documents.append(doc)
        
        return documents
    
    async def _generate_agent_testset_data_from_scenarios(
        self, 
        scenarios: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Generate agent testset data from scenarios."""
        agent_test_cases = []
        
        for i, scenario in enumerate(scenarios):
            # Generate test cases for each scenario
            test_case = {
                "scenario_id": f"scenario_{i+1}",
                "scenario_name": scenario.get("name", f"Scenario {i+1}"),
                "description": scenario.get("description", ""),
                "goal": scenario.get("goal", ""),
                "expected_outcome": scenario.get("expected_outcome", ""),
                "success_criteria": scenario.get("success_criteria", ""),
                "difficulty": scenario.get("difficulty", "medium"),
                "category": scenario.get("category", "general"),
                "tools_required": ", ".join(scenario.get("tools", [])),
                "expected_turns": scenario.get("expected_turns", 3),
                "time_limit": scenario.get("time_limit", 300),
                "context": scenario.get("context", ""),
                "initial_message": scenario.get("initial_message", ""),
                "evaluation_notes": scenario.get("evaluation_notes", "")
            }
            agent_test_cases.append(test_case)
        
        return pd.DataFrame(agent_test_cases)
    
    async def _save_agent_testset_to_excel(
        self, 
        testset_data: pd.DataFrame, 
        output_path: str, 
        testset_id: str
    ) -> str:
        """Save agent testset to Excel file."""
        import os
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate Excel filename
        excel_filename = f"agent_testset_{testset_id}.xlsx"
        excel_path = output_dir / excel_filename
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main testset data
            testset_data.to_excel(writer, sheet_name='Test Cases', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Scenarios', 'Average Expected Turns', 'Difficulty Distribution'],
                'Value': [
                    len(testset_data),
                    testset_data['expected_turns'].mean(),
                    testset_data['difficulty'].value_counts().to_dict()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Categories sheet
            if 'category' in testset_data.columns:
                category_stats = testset_data['category'].value_counts().reset_index()
                category_stats.columns = ['Category', 'Count']
                category_stats.to_excel(writer, sheet_name='Categories', index=False)
        
        logger.info(f"Saved agent testset to Excel: {excel_path}")
        return str(excel_path)
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate content."""
        # This would be implemented with actual LLM call
        # For now, return a placeholder
        return f"Generated content for: {prompt[:100]}..."
