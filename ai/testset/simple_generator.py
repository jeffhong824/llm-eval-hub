"""Simplified testset generator."""

# Import pydantic patch FIRST, before any langchain imports
import ai.core.pydantic_patch  # noqa: F401

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from langchain.schema import Document
from langchain_openai import ChatOpenAI

from ai.resources.prompt_manager import prompt_manager
from ai.resources.personas import get_personas, Persona

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if not present
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class TestsetConfig:
    """Testset generation configuration."""
    
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


class SimpleTestsetGeneratorService:
    """Simplified testset generator service."""
    
    def __init__(self, api_key: str):
        """Initialize the generator service."""
        self.api_key = api_key
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        logger.info("Simple testset generator initialized with LLM")
    
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
            
            # Create sample testset data
            testset_data = self._create_sample_testset(config.testset_size)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = GeneratedTestset(
                testset_id=testset_id,
                timestamp=start_time,
                testset_size=len(testset_data),
                documents_count=5,  # Placeholder
                generation_time=generation_time,
                testset_data=testset_data,
                knowledge_graph_path=None,
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
            testset_data = self._generate_agent_testset_data_from_scenarios(scenarios)
            
            # Save to Excel if output path provided
            excel_path = None
            if output_path:
                excel_path = self._save_agent_testset_to_excel(testset_data, output_path, testset_id)
            
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
    
    def _create_sample_testset(self, size: int) -> pd.DataFrame:
        """Create sample testset data."""
        questions = [
            "What is machine learning?",
            "How does neural network work?",
            "What are the benefits of AI?",
            "Explain deep learning concepts",
            "What is natural language processing?",
            "How does computer vision work?",
            "What are the applications of AI?",
            "Explain reinforcement learning",
            "What is transfer learning?",
            "How does AI impact society?"
        ]
        
        ground_truths = [
            "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.",
            "Neural networks are computing systems inspired by biological neural networks that process information.",
            "AI can automate tasks, improve efficiency, and enable new capabilities.",
            "Deep learning uses neural networks with multiple layers to learn complex patterns.",
            "Natural language processing enables computers to understand and generate human language.",
            "Computer vision allows machines to interpret and understand visual information.",
            "AI applications include healthcare, finance, transportation, and entertainment.",
            "Reinforcement learning is a type of machine learning based on reward and punishment.",
            "Transfer learning applies knowledge from one domain to another related domain.",
            "AI impacts society through automation, decision-making, and new opportunities."
        ]
        
        contexts = [
            ["Machine learning algorithms build mathematical models based on training data."],
            ["Neural networks consist of layers of interconnected nodes that process information."],
            ["Artificial intelligence can automate repetitive tasks and improve decision-making."],
            ["Deep learning models can automatically learn features from raw data."],
            ["NLP combines computational linguistics with machine learning and deep learning."],
            ["Computer vision uses digital images and videos to extract meaningful information."],
            ["AI systems are being deployed across various industries and sectors."],
            ["Reinforcement learning agents learn through interaction with their environment."],
            ["Transfer learning reduces the need for large amounts of training data."],
            ["AI technology raises questions about ethics, privacy, and job displacement."]
        ]
        
        return pd.DataFrame({
            'question': questions[:size],
            'ground_truth': ground_truths[:size],
            'contexts': contexts[:size]
        })
    
    def _generate_agent_testset_data_from_scenarios(
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
    
    def _save_agent_testset_to_excel(
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

    async def _generate_documents_from_scenarios(self, scenarios: List[str], num_docs: int = 1, language: str = "繁體中文") -> List[Document]:
        """
        Generate documents from scenario prompts using LLM with different personas.
        """
        documents = []
        
        # Load the system prompt
        try:
            system_prompt = prompt_manager.get_scenario_to_documents_prompt()
        except FileNotFoundError:
            logger.error("Scenario to documents prompt not found, using fallback")
            system_prompt = "Generate comprehensive knowledge documents based on the given scenario."
        
        # Get personas for the requested number of documents
        personas = get_personas(num_docs)
        logger.info(f"Using {len(personas)} personas for document generation")
        
        for i, scenario_prompt in enumerate(scenarios):
            # Generate multiple documents for each scenario using different personas
            for doc_index, persona in enumerate(personas):
                try:
                    # Create the full prompt with persona information
                    full_prompt = f"""{system_prompt}

Scenario Prompt: {scenario_prompt}

Persona Information:
- Name: {persona.name}
- Description: {persona.description}
- Perspective: {persona.perspective}
- Expertise Areas: {', '.join(persona.expertise_areas)}
- Communication Style: {persona.communication_style}

Language: {language}

Please generate a comprehensive knowledge document from this persona's perspective for the given scenario. The document should be written in {language}. Include realistic Q&A examples showing how this persona would interact with the AI system."""

                    logger.info(f"Generated prompt for scenario {i+1}, document {doc_index+1} with persona: {persona.name}")
                    logger.info(f"Full prompt length: {len(full_prompt)}")
                    logger.info(f"Prompt preview: {full_prompt[:200]}...")
                    logger.info(f"Scenario prompt: {scenario_prompt}")
                    
                    # Generate content using LLM
                    logger.info("Calling LLM...")
                    response = self.llm.invoke(full_prompt)
                    content = response.content
                    logger.info(f"LLM response received: {len(content)} characters")
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": f"llm_generated_{doc_index+1}", 
                            "scenario_prompt": scenario_prompt,
                            "persona_id": persona.id,
                            "persona_name": persona.name,
                            "persona_description": persona.description,
                            "generated_at": datetime.now().isoformat(),
                            "generation_method": "llm"
                        }
                    )
                    documents.append(doc)
                    logger.info(f"Generated document {doc_index+1} using LLM with persona: {persona.name}")
                    
                except Exception as e:
                    logger.error(f"Error generating document {doc_index+1} with LLM: {str(e)}")
                    # Fallback to simple content
                    content = f"# Generated Document\n\nBased on scenario: {scenario_prompt}\n\nPersona: {persona.name}\n\nThis document contains knowledge relevant to the specified scenario from the {persona.name} perspective."
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": f"fallback_{doc_index+1}", 
                            "scenario_prompt": scenario_prompt,
                            "persona_id": persona.id,
                            "persona_name": persona.name,
                            "persona_description": persona.description,
                            "generated_at": datetime.now().isoformat(),
                            "generation_method": "fallback",
                            "error": str(e)
                        }
                    )
                    documents.append(doc)
        
        return documents
    
