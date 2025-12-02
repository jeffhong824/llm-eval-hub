"""Testset generation API endpoints."""

# Import pydantic patch FIRST, before any langchain imports
import ai.core.pydantic_patch  # noqa: F401

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import json
import asyncio

from ai.testset.simple_generator import SimpleTestsetGeneratorService, TestsetConfig, GeneratedTestset
from ai.testset.rag_generator import RAGTestsetGenerator
from ai.testset.agent_generator import AgentTestsetGenerator
from ai.document_loaders.loader import DocumentProcessor
from ai.core.model_checker import model_checker
from ai.resources.personas import get_persona_names
from configs.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class TestsetGenerationRequest(BaseModel):
    """Request model for testset generation."""
    
    testset_size: int = Field(default=10, ge=1, le=100)
    single_hop_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    multi_hop_abstract_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    multi_hop_specific_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    include_personas: bool = Field(default=True)
    custom_personas: Optional[List[str]] = None
    save_knowledge_graph: bool = Field(default=True)
    enrich_with_transforms: bool = Field(default=True)


class RAGTestsetFromFolderRequest(BaseModel):
    """Request model for RAG testset generation from folder."""
    
    folder_path: str
    testset_size: int = Field(default=10, ge=1, le=100)
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    single_hop_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    multi_hop_abstract_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    multi_hop_specific_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    include_personas: bool = Field(default=True)
    custom_personas: Optional[List[str]] = None
    save_knowledge_graph: bool = Field(default=True)
    enrich_with_transforms: bool = Field(default=True)


class AgentTestsetRequest(BaseModel):
    """Request model for agent testset generation."""
    
    scenarios: List[Dict[str, Any]] = Field(min_items=1, max_items=50)
    output_path: Optional[str] = None
    config: Optional[TestsetGenerationRequest] = None


class ScenarioToDocsRequest(BaseModel):
    """Request model for converting scenarios to documents."""
    
    prompt: str
    output_folder: str
    num_docs: int = Field(default=5, ge=1, le=20)
    language: str = Field(default="繁體中文", description="Language for document generation")


class ScenarioBasedRequest(BaseModel):
    """Request model for scenario-based testset generation."""
    
    scenarios: List[str] = Field(min_items=1, max_items=50)
    config: Optional[TestsetGenerationRequest] = None


class AgentTestsetRequest(BaseModel):
    """Request model for agent testset generation."""
    
    scenarios: List[Dict[str, Any]] = Field(min_items=1, max_items=50)
    config: Optional[TestsetGenerationRequest] = None


class TestsetGenerationResponse(BaseModel):
    """Response model for testset generation."""
    
    testset_id: str
    status: str
    timestamp: datetime
    testset_size: int
    documents_count: int
    generation_time: float
    knowledge_graph_path: Optional[str] = None
    error_message: Optional[str] = None


# Dependency injection
def get_testset_generator() -> SimpleTestsetGeneratorService:
    """Get testset generator service instance."""
    return SimpleTestsetGeneratorService(settings.openai_api_key)


@router.post("/generate/rag/from-folder", response_model=TestsetGenerationResponse)
async def generate_rag_testset_from_folder(
    request: RAGTestsetFromFolderRequest,
    background_tasks: BackgroundTasks,
    generator: SimpleTestsetGeneratorService = Depends(get_testset_generator)
):
    """
    Generate testset for RAG system evaluation from folder.
    
    Args:
        request: RAG testset generation request from folder
        background_tasks: Background tasks for async processing
        generator: Testset generator service
        
    Returns:
        TestsetGenerationResponse with generation results
    """
    try:
        logger.info(f"Starting RAG testset generation from folder: {request.folder_path}")
        
        # Validate ratios
        total_ratio = request.single_hop_ratio + request.multi_hop_abstract_ratio + request.multi_hop_specific_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail="Query type ratios must sum to 1.0"
            )
        
        # Create config
        config = TestsetConfig(
            testset_size=request.testset_size,
            single_hop_ratio=request.single_hop_ratio,
            multi_hop_abstract_ratio=request.multi_hop_abstract_ratio,
            multi_hop_specific_ratio=request.multi_hop_specific_ratio,
            include_personas=request.include_personas,
            custom_personas=request.custom_personas,
            save_knowledge_graph=request.save_knowledge_graph,
            enrich_with_transforms=request.enrich_with_transforms
        )
        
        # Generate testset from folder
        result = await generator.generate_rag_testset_from_folder(
            folder_path=request.folder_path,
            config=config,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        # Save testset to background task
        background_tasks.add_task(save_testset, result)
        
        return TestsetGenerationResponse(
            testset_id=result.testset_id,
            status=result.status,
            timestamp=result.timestamp,
            testset_size=result.testset_size,
            documents_count=result.documents_count,
            generation_time=result.generation_time,
            knowledge_graph_path=result.knowledge_graph_path,
            error_message=result.error_message
        )
        
    except Exception as e:
        logger.error(f"Error generating RAG testset from folder: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/scenario", response_model=TestsetGenerationResponse)
async def generate_scenario_testset(
    request: ScenarioBasedRequest,
    background_tasks: BackgroundTasks,
    generator: SimpleTestsetGeneratorService = Depends(get_testset_generator)
):
    """
    Generate testset from user-provided scenarios.
    
    Args:
        request: Scenario-based generation request
        background_tasks: Background tasks for async processing
        generator: Testset generator service
        
    Returns:
        TestsetGenerationResponse with generation results
    """
    try:
        logger.info(f"Starting scenario-based testset generation with {len(request.scenarios)} scenarios")
        
        # Create config
        config = None
        if request.config:
            config = TestsetConfig(
                testset_size=request.config.testset_size,
                single_hop_ratio=request.config.single_hop_ratio,
                multi_hop_abstract_ratio=request.config.multi_hop_abstract_ratio,
                multi_hop_specific_ratio=request.config.multi_hop_specific_ratio,
                include_personas=request.config.include_personas,
                custom_personas=request.config.custom_personas,
                save_knowledge_graph=request.config.save_knowledge_graph,
                enrich_with_transforms=request.config.enrich_with_transforms
            )
        
        # Generate testset from scenarios
        result = await generator.generate_from_user_scenarios(request.scenarios, config)
        
        # Save testset to background task
        background_tasks.add_task(save_testset, result)
        
        return TestsetGenerationResponse(
            testset_id=result.testset_id,
            status=result.status,
            timestamp=result.timestamp,
            testset_size=result.testset_size,
            documents_count=result.documents_count,
            generation_time=result.generation_time,
            knowledge_graph_path=result.knowledge_graph_path,
            error_message=result.error_message
        )
        
    except Exception as e:
        logger.error(f"Error generating scenario testset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/agent/excel", response_model=TestsetGenerationResponse)
async def generate_agent_testset_with_excel(
    request: AgentTestsetRequest,
    background_tasks: BackgroundTasks,
    generator: SimpleTestsetGeneratorService = Depends(get_testset_generator)
):
    """
    Generate testset for agent system evaluation.
    
    Args:
        request: Agent testset generation request
        background_tasks: Background tasks for async processing
        generator: Testset generator service
        
    Returns:
        TestsetGenerationResponse with generation results
    """
    try:
        logger.info(f"Starting agent testset generation with {len(request.scenarios)} scenarios")
        
        # Create config
        config = None
        if request.config:
            config = TestsetConfig(
                testset_size=request.config.testset_size,
                single_hop_ratio=request.config.single_hop_ratio,
                multi_hop_abstract_ratio=request.config.multi_hop_abstract_ratio,
                multi_hop_specific_ratio=request.config.multi_hop_specific_ratio,
                include_personas=request.config.include_personas,
                custom_personas=request.config.custom_personas,
                save_knowledge_graph=request.config.save_knowledge_graph,
                enrich_with_transforms=request.config.enrich_with_transforms
            )
        
        # Generate agent testset with Excel output
        result = await generator.generate_agent_testset_with_excel(
            scenarios=request.scenarios,
            config=config,
            output_path=request.output_path
        )
        
        # Save testset to background task
        background_tasks.add_task(save_testset, result)
        
        return TestsetGenerationResponse(
            testset_id=result.testset_id,
            status=result.status,
            timestamp=result.timestamp,
            testset_size=result.testset_size,
            documents_count=result.documents_count,
            generation_time=result.generation_time,
            knowledge_graph_path=result.knowledge_graph_path,
            error_message=result.error_message
        )
        
    except Exception as e:
        logger.error(f"Error generating agent testset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload documents for testset generation.
    
    Args:
        files: List of uploaded files
        background_tasks: Background tasks for async processing
        
    Returns:
        Upload result with document IDs
    """
    try:
        logger.info(f"Uploading {len(files)} documents")
        
        document_ids = []
        
        for file in files:
            # Validate file type
            if not file.filename.endswith(('.txt', '.pdf', '.md', '.json')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}"
                )
            
            # Save file
            file_path = f"{settings.upload_dir}/{file.filename}"
            content = await file.read()
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            document_ids.append(file.filename)
            
            # Process document in background
            background_tasks.add_task(process_document, file_path)
        
        return {
            "message": f"Successfully uploaded {len(files)} documents",
            "document_ids": document_ids
        }
        
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/testsets")
async def list_testsets(
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None
):
    """List generated testsets with pagination."""
    try:
        # This would query testsets from storage
        # For now, return placeholder
        return {
            "testsets": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error listing testsets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/testsets/{testset_id}")
async def get_testset(testset_id: str):
    """Get testset by ID."""
    try:
        # This would load testset from storage
        # For now, return placeholder
        return {
            "testset_id": testset_id,
            "status": "available",
            "message": "Testset would be loaded from storage"
        }
    except Exception as e:
        logger.error(f"Error loading testset: {str(e)}")
        raise HTTPException(status_code=404, detail="Testset not found")


@router.get("/testsets/{testset_id}/download")
async def download_testset(testset_id: str):
    """Download testset as CSV."""
    try:
        # This would load and return testset data
        # For now, return placeholder
        return {
            "testset_id": testset_id,
            "message": "Testset download would be implemented"
        }
    except Exception as e:
        logger.error(f"Error downloading testset: {str(e)}")
        raise HTTPException(status_code=404, detail="Testset not found")


@router.delete("/testsets/{testset_id}")
async def delete_testset(testset_id: str):
    """Delete testset by ID."""
    try:
        # This would delete testset from storage
        logger.info(f"Deleting testset: {testset_id}")
        return {"message": "Testset deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting testset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def save_testset(testset: GeneratedTestset):
    """Save generated testset to storage."""
    try:
        # This would save to database or file system
        logger.info(f"Saving testset: {testset.testset_id}")
    except Exception as e:
        logger.error(f"Error saving testset: {str(e)}")


@router.post("/scenario-to-docs")
async def convert_scenario_to_docs(
    request: ScenarioToDocsRequest,
    background_tasks: BackgroundTasks,
    generator: SimpleTestsetGeneratorService = Depends(get_testset_generator)
):
    """
    Convert scenario prompt to documents.
    
    Args:
        request: Scenario to documents conversion request
        background_tasks: Background tasks for async processing
        generator: Testset generator service
        
    Returns:
        Conversion result with document content for download
    """
    try:
        logger.info(f"Converting scenario to documents: {request.prompt[:100]}...")
        logger.info(f"Full prompt: {request.prompt}")
        logger.info(f"Output folder: {request.output_folder}")
        logger.info(f"Number of docs: {request.num_docs}")
        
        # Generate documents from scenario prompt with specified number of documents
        documents = await generator._generate_documents_from_scenarios([request.prompt], request.num_docs, request.language)
        logger.info(f"Generated {len(documents)} documents")
        
        # Save documents to specified folder
        doc_processor = DocumentProcessor()
        save_result = doc_processor.save_documents_to_folder(
            documents=documents,
            output_folder=request.output_folder,
            format="txt",
            cleanup_temp=True
        )
        
        # Prepare document content for download
        document_contents = []
        for i, doc in enumerate(documents):
            document_contents.append({
                "filename": f"document_{i+1}.txt",
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Only return success if files were actually saved
        if save_result["saved_files"] > 0:
            return {
                "message": f"Successfully generated and saved {len(documents)} documents to {request.output_folder}",
                "output_folder": request.output_folder,
                "documents_count": len(documents),
                "files": save_result["files"],
                "documents": document_contents,
                "download_ready": True,
                "status": "success"
            }
        else:
            return {
                "message": f"Generated {len(documents)} documents but failed to save to {request.output_folder}",
                "output_folder": request.output_folder,
                "documents_count": len(documents),
                "files": [],
                "documents": document_contents,
                "download_ready": True,
                "status": "warning"
            }
        
    except Exception as e:
        logger.error(f"Error converting scenario to docs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/available")
async def get_available_models():
    """Get available LLM models."""
    try:
        logger.info("Checking available models...")
        models_info = await model_checker.check_all_models()
        summary = model_checker.get_available_models_summary(models_info)
        
        return {
            "models": models_info,
            "summary": summary,
            "total_available": sum(len(models) for models in summary.values())
        }
        
    except Exception as e:
        logger.error(f"Error checking available models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/personas")
async def get_personas(count: int = 20):
    """Get available personas."""
    try:
        persona_names = get_persona_names(count)
        return {
            "personas": persona_names,
            "count": len(persona_names),
            "total_available": 20
        }
        
    except Exception as e:
        logger.error(f"Error getting personas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class RAGTestsetFromDocumentsRequest(BaseModel):
    """Request model for RAG testset generation from documents folder."""
    
    documents_folder: str = Field(description="Path to folder containing documents")
    output_folder: str = Field(description="Path to save the generated testset")
    model_provider: str = Field(default="openai", description="Model provider (openai, gemini, ollama)")
    model_name: str = Field(description="Model name to use for generation")
    language: str = Field(default="繁體中文", description="Language for generation")
    chunk_size: int = Field(default=5000, ge=100, le=20000, description="Size of each text chunk")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")
    qa_per_chunk: int = Field(default=3, ge=1, le=10, description="Number of QA pairs per chunk")


class AgentTestsetFromDocumentsRequest(BaseModel):
    """Request model for Agent testset generation from documents folder."""
    
    documents_folder: str = Field(description="Path to folder containing documents")
    output_folder: str = Field(description="Path to save the generated testset")
    model_provider: str = Field(default="openai", description="Model provider (openai, gemini, ollama)")
    model_name: str = Field(description="Model name to use for generation")
    language: str = Field(default="繁體中文", description="Language for generation")
    chunk_size: int = Field(default=5000, ge=100, le=20000, description="Size of each text chunk")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")
    tasks_per_chunk: int = Field(default=3, ge=1, le=10, description="Number of tasks per chunk")


@router.post("/rag-testset-from-documents")
async def generate_rag_testset_from_documents(request: RAGTestsetFromDocumentsRequest):
    """Generate RAG testset from documents folder."""
    try:
        logger.info(f"Generating RAG testset from {request.documents_folder}")
        
        generator = RAGTestsetGenerator(
            model_provider=request.model_provider,
            model_name=request.model_name
        )
        result = generator.generate_testset(
            documents_folder=request.documents_folder,
            output_folder=request.output_folder,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            qa_per_chunk=request.qa_per_chunk,
            language=request.language
        )
        
        if result["success"]:
            return {
                "message": "RAG testset generated successfully!",
                "excel_path": result["excel_path"],
                "total_qa_pairs": result["total_qa_pairs"],
                "total_chunks": result["total_chunks"],
                "total_documents": result["total_documents"],
                "parameters": {
                    "chunk_size": result["chunk_size"],
                    "chunk_overlap": result["chunk_overlap"],
                    "qa_per_chunk": result["qa_per_chunk"]
                }
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Error generating RAG testset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent-testset-from-documents")
async def generate_agent_testset_from_documents(request: AgentTestsetFromDocumentsRequest):
    """Generate Agent testset from documents folder."""
    try:
        logger.info(f"Generating Agent testset from {request.documents_folder}")
        
        generator = AgentTestsetGenerator(
            model_provider=request.model_provider,
            model_name=request.model_name
        )
        result = generator.generate_testset(
            documents_folder=request.documents_folder,
            output_folder=request.output_folder,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            tasks_per_chunk=request.tasks_per_chunk,
            language=request.language
        )
        
        if result["success"]:
            return {
                "message": "Agent testset generated successfully!",
                "excel_path": result["excel_path"],
                "total_tasks": result["total_tasks"],
                "total_chunks": result["total_chunks"],
                "total_documents": result["total_documents"],
                "parameters": {
                    "chunk_size": result["chunk_size"],
                    "chunk_overlap": result["chunk_overlap"],
                    "tasks_per_chunk": result["tasks_per_chunk"]
                }
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Error generating Agent testset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_document(file_path: str):
    """Process uploaded document."""
    try:
        # This would process the document (extract text, create embeddings, etc.)
        logger.info(f"Processing document: {file_path}")
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")


# ==================== New Workflow APIs ====================

class GeneratePersonasRequest(BaseModel):
    """Request for generating personas from scenario."""
    scenario_description: str = Field(description="Description of the use case/scenario")
    num_personas: int = Field(default=10, ge=1, le=50, description="Number of personas to generate")
    output_folder: str = Field(description="Base output folder for all workflow stages")
    scenario_name: Optional[str] = Field(default=None, description="Name for the scenario (optional)")
    model_provider: str = Field(default="openai", description="LLM provider for persona generation")
    model_name: str = Field(default="gpt-4", description="LLM model for persona generation")
    language: str = Field(default="繁體中文", description="Language for persona descriptions")


class GenerateDocumentsRequest(BaseModel):
    """Request for generating documents from scenario."""
    scenario_description: str = Field(description="Description of the use case/scenario")
    num_documents: int = Field(default=10, ge=1, le=50, description="Number of documents to generate")
    output_folder: str = Field(description="Base output folder (same as personas)")
    scenario_name: Optional[str] = Field(default=None, description="Name for the scenario (optional)")
    model_provider: str = Field(default="openai", description="LLM provider for document generation")
    model_name: str = Field(default="gpt-4", description="LLM model for document generation")
    language: str = Field(default="繁體中文", description="Language for document content")


class GenerateRAGTestsetWithPersonasRequest(BaseModel):
    """Request for generating RAG testset with personas."""
    documents_folder: str = Field(description="Path to documents folder (from stage 2)")
    personas_json_path: str = Field(description="Path to personas JSON file (from stage 1)")
    output_folder: str = Field(description="Base output folder (same as previous stages)")
    scenario_name: Optional[str] = Field(default=None, description="Name for the scenario")
    model_provider: str = Field(default="openai", description="LLM provider")
    model_name: str = Field(default="gpt-3.5-turbo", description="LLM model")
    language: str = Field(default="繁體中文", description="Language")
    chunk_size: int = Field(default=5000, ge=100, le=20000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    qa_per_chunk: int = Field(default=3, ge=1, le=10)
    selected_files: Optional[List[str]] = Field(default=None, description="List of selected file names to use (if None, use all files)")


class GenerateAgentTestsetWithPersonasRequest(BaseModel):
    """Request for generating Agent testset with personas."""
    documents_folder: str = Field(description="Path to documents folder (from stage 2)")
    personas_json_path: str = Field(description="Path to personas JSON file (from stage 1)")
    output_folder: str = Field(description="Base output folder (same as previous stages)")
    scenario_name: Optional[str] = Field(default=None, description="Name for the scenario")
    model_provider: str = Field(default="openai", description="LLM provider")
    model_name: str = Field(default="gpt-3.5-turbo", description="LLM model")
    language: str = Field(default="繁體中文", description="Language")
    chunk_size: int = Field(default=5000, ge=100, le=20000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    tasks_per_chunk: int = Field(default=3, ge=1, le=10)


@router.post("/workflow/generate-personas")
async def generate_personas_from_scenario(request: GeneratePersonasRequest):
    """
    Stage 1: Generate detailed personas from scenario description.
    
    This is the first shared stage for both RAG and Agent workflows.
    """
    try:
        logger.info(f"Starting persona generation for scenario: {request.scenario_description[:100]}...")
        
        from ai.testset.persona_generator import PersonaGenerator
        from ai.testset.output_manager import OutputManager
        
        # Initialize output manager
        output_manager = OutputManager(request.output_folder, request.scenario_name)
        
        # Generate personas
        persona_generator = PersonaGenerator(
            model_provider=request.model_provider,
            model_name=request.model_name
        )
        
        logger.info(f"Generating {request.num_personas} personas...")
        personas = await persona_generator.generate_personas(
            scenario_description=request.scenario_description,
            num_personas=request.num_personas,
            language=request.language
        )
        
        logger.info(f"Saving {len(personas)} personas to files...")
        # Save personas to stage folder
        save_result = persona_generator.save_personas_to_files(
            personas=personas,
            output_folder=str(output_manager.scenario_folder),
            scenario_name=output_manager.scenario_name
        )
        
        # Mark stage complete
        output_manager.mark_stage_complete("personas", save_result)
        
        logger.info(f"Persona generation completed successfully")
        
        return {
            "success": True,
            "message": f"Successfully generated {len(personas)} personas",
            "scenario_name": output_manager.scenario_name,
            "scenario_folder": str(output_manager.scenario_folder),
            "personas_folder": save_result["personas_folder"],
            "json_file": save_result["json_file"],
            "total_personas": len(personas),
            "personas": personas,  # Return all personas for editing
            "personas_preview": personas,  # Show all personas
            "progress": 100
        }
        
    except Exception as e:
        logger.error(f"Error generating personas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/generate-documents")
async def generate_documents_from_scenario(request: GenerateDocumentsRequest):
    """
    Stage 2: Generate documents from scenario description.
    
    This is the second shared stage for both RAG and Agent workflows.
    """
    try:
        logger.info(f"Starting document generation for scenario: {request.scenario_description[:100]}...")
        
        from ai.testset.scenario_generator import ScenarioDocumentGenerator
        from ai.testset.output_manager import OutputManager
        
        # Initialize output manager
        output_manager = OutputManager(request.output_folder, request.scenario_name)
        
        # Generate documents
        doc_generator = ScenarioDocumentGenerator(
            model_provider=request.model_provider,
            model_name=request.model_name
        )
        
        documents = await doc_generator.generate_documents(
            scenario_description=request.scenario_description,
            num_documents=request.num_documents,
            language=request.language
        )
        
        # Save documents to stage folder
        save_result = doc_generator.save_documents(
            documents=documents,
            output_folder=str(output_manager.scenario_folder),
            scenario_name=output_manager.scenario_name
        )
        
        # Mark stage complete
        output_manager.mark_stage_complete("documents", save_result)
        
        return {
            "success": True,
            "message": f"Successfully generated {len(documents)} documents",
            "scenario_name": output_manager.scenario_name,
            "scenario_folder": str(output_manager.scenario_folder),
            "documents_folder": save_result["documents_folder"],
            "metadata_file": save_result["metadata_file"],
            "total_documents": len(documents),
            "documents": documents,  # Return all documents for editing
            "documents_preview": documents,  # Show all documents
            "documents_preview_old": [
                {"title": doc["title"], "category": doc["category"]}
                for doc in documents[:5]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/generate-rag-testset")
async def generate_rag_testset_with_personas(request: GenerateRAGTestsetWithPersonasRequest):
    """
    Stage 3 (RAG): Generate RAG testset using documents and personas with streaming progress.
    """
    async def event_generator():
        """Generator function for Server-Sent Events."""
        try:
            import json
            from pathlib import Path
            from ai.testset.rag_generator import RAGTestsetGenerator
            from ai.testset.output_manager import OutputManager
            
            # Map paths for Docker compatibility
            personas_json_path = OutputManager.map_path_for_docker(request.personas_json_path)
            documents_folder = OutputManager.map_path_for_docker(request.documents_folder)
            
            # Load personas
            logger.info(f"Loading personas from: {personas_json_path}")
            try:
                with open(personas_json_path, 'r', encoding='utf-8') as f:
                    personas = json.load(f)
                logger.info(f"Loaded {len(personas)} personas")
            except FileNotFoundError:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Personas file not found: {personas_json_path}'})}\n\n"
                return
            except json.JSONDecodeError as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Invalid JSON format in personas file: {e}'})}\n\n"
                return
            
            # Initialize output manager
            output_manager = OutputManager(request.output_folder, request.scenario_name)
            output_manager.set_workflow_type("rag")
            
            # Generate RAG testset
            generator = RAGTestsetGenerator(
                model_provider=request.model_provider,
                model_name=request.model_name
            )
            
            # Get output folder for this stage
            stage_folder = output_manager.get_stage_folder("rag_testset")
            
            # Progress tracking queue for streaming updates
            progress_queue = asyncio.Queue()
            
            # Progress callback for streaming updates
            async def progress_callback(progress_data):
                """Callback to send progress updates via queue."""
                await progress_queue.put(progress_data)
            
            # Start generation task
            async def generate_task():
                """Background task to run generation."""
                try:
                    result = await generator.generate_testset_async(
                        documents_folder=documents_folder,
                        output_folder=str(stage_folder),
                        chunk_size=request.chunk_size,
                        chunk_overlap=request.chunk_overlap,
                        qa_per_chunk=request.qa_per_chunk,
                        language=request.language,
                        personas=personas,
                        progress_callback=progress_callback,
                        selected_files=request.selected_files  # Pass selected files list
                    )
                    await progress_queue.put({"type": "generation_complete", "result": result})
                except Exception as e:
                    await progress_queue.put({"type": "generation_error", "error": str(e)})
            
            # Generate RAG testset with streaming
            logger.info(f"Generating testset from documents folder: {documents_folder}")
            if request.selected_files:
                logger.info(f"Selected files: {request.selected_files}")
            try:
                # Send initial progress with file info
                initial_info = {
                    'type': 'start',
                    'message': '開始生成 QA Pairs...',
                    'documents_folder': documents_folder,
                    'selected_files': request.selected_files if request.selected_files else None,
                    'chunk_size': request.chunk_size,
                    'chunk_overlap': request.chunk_overlap,
                    'qa_per_chunk': request.qa_per_chunk
                }
                yield f"data: {json.dumps(initial_info)}\n\n"
                
                # Start generation in background
                generation_task = asyncio.create_task(generate_task())
                
                # Initialize result variable
                result = None
                generation_complete = False
                
                # Stream progress updates
                while True:
                    try:
                        # Wait for progress update with timeout
                        progress_data = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                        
                        if progress_data.get("type") == "generation_complete":
                            result = progress_data.get("result")
                            generation_complete = True
                            
                            if not result or not result.get("success", False):
                                error_msg = result.get("error", "Unknown error occurred") if result else "Generation failed without error message"
                                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                                return
                            
                            # Mark stage complete
                            try:
                                output_manager.mark_stage_complete("rag_testset", result)
                            except Exception as e:
                                logger.warning(f"Failed to mark stage complete: {e}, but testset was generated successfully")
                            
                            # Send final result
                            yield f"data: {json.dumps({
                                'type': 'complete',
                                'success': True,
                                'message': 'RAG testset generated successfully with personas!',
                                'scenario_name': output_manager.scenario_name,
                                'scenario_folder': str(output_manager.scenario_folder),
                                'excel_path': result['excel_path'],
                                'total_qa_pairs': result['total_qa_pairs'],
                                'total_chunks': result['total_chunks'],
                                'total_documents': result['total_documents'],
                                'personas_used': len(personas),
                                'method': result.get('method', 'llm')
                            })}\n\n"
                            return
                            
                        elif progress_data.get("type") == "generation_error":
                            error_msg = progress_data.get("error", "Unknown error occurred")
                            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                            return
                            
                        else:
                            # Send progress update
                            yield f"data: {json.dumps(progress_data)}\n\n"
                            
                    except asyncio.TimeoutError:
                        # Check if generation task is done
                        if generation_task.done():
                            # Task completed, try to get any remaining messages
                            try:
                                # Check if task had an exception
                                try:
                                    generation_task.result()  # This will raise if there was an exception
                                except Exception as task_exception:
                                    yield f"data: {json.dumps({'type': 'error', 'message': f'Generation task failed: {str(task_exception)}'})}\n\n"
                                    return
                                
                                # Task completed successfully, try to get the result message
                                # Wait a bit more for the message to arrive (it might be in queue)
                                try:
                                    remaining_data = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                                    if remaining_data.get("type") == "generation_complete":
                                        result = remaining_data.get("result")
                                        generation_complete = True
                                        
                                        if not result or not result.get("success", False):
                                            error_msg = result.get("error", "Unknown error occurred") if result else "Generation failed without error message"
                                            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                                            return
                                        
                                        # Mark stage complete
                                        try:
                                            output_manager.mark_stage_complete("rag_testset", result)
                                        except Exception as e:
                                            logger.warning(f"Failed to mark stage complete: {e}, but testset was generated successfully")
                                        
                                        # Send final result
                                        yield f"data: {json.dumps({
                                            'type': 'complete',
                                            'success': True,
                                            'message': 'RAG testset generated successfully with personas!',
                                            'scenario_name': output_manager.scenario_name,
                                            'scenario_folder': str(output_manager.scenario_folder),
                                            'excel_path': result['excel_path'],
                                            'total_qa_pairs': result['total_qa_pairs'],
                                            'total_chunks': result['total_chunks'],
                                            'total_documents': result['total_documents'],
                                            'personas_used': len(personas),
                                            'method': result.get('method', 'llm')
                                        })}\n\n"
                                        return
                                    elif remaining_data.get("type") == "generation_error":
                                        error_msg = remaining_data.get("error", "Unknown error occurred")
                                        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                                        return
                                except asyncio.TimeoutError:
                                    # No message in queue, but task is done - this shouldn't happen
                                    logger.error("Generation task completed but no result message found in queue")
                                    yield f"data: {json.dumps({'type': 'error', 'message': 'Generation completed but no result message received. Please check server logs.'})}\n\n"
                                    return
                            except Exception as e:
                                logger.error(f"Error checking generation task: {e}", exc_info=True)
                                yield f"data: {json.dumps({'type': 'error', 'message': f'Error checking generation status: {str(e)}'})}\n\n"
                                return
                        # Send heartbeat to keep connection alive
                        yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                        continue
                
                # Mark stage complete
                try:
                    output_manager.mark_stage_complete("rag_testset", result)
                except Exception as e:
                    logger.warning(f"Failed to mark stage complete: {e}, but testset was generated successfully")
                
                # Send final result
                yield f"data: {json.dumps({
                    'type': 'complete',
                    'success': True,
                    'message': 'RAG testset generated successfully with personas!',
                    'scenario_name': output_manager.scenario_name,
                    'scenario_folder': str(output_manager.scenario_folder),
                    'excel_path': result['excel_path'],
                    'total_qa_pairs': result['total_qa_pairs'],
                    'total_chunks': result['total_chunks'],
                    'total_documents': result['total_documents'],
                    'personas_used': len(personas),
                    'method': result.get('method', 'llm')
                })}\n\n"
                
            except Exception as e:
                logger.error(f"Error generating testset: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to generate testset: {str(e)}'})}\n\n"
                
        except Exception as e:
            logger.error(f"Error in event generator: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': f'Internal error: {str(e)}'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/workflow/generate-agent-testset")
async def generate_agent_testset_with_personas(request: GenerateAgentTestsetWithPersonasRequest):
    """
    Stage 3 (Agent): Generate Agent testset using documents and personas.
    """
    try:
        logger.info(f"Starting Agent testset generation with personas")
        
        import json
        from pathlib import Path
        from ai.testset.agent_generator import AgentTestsetGenerator
        from ai.testset.output_manager import OutputManager
        
        # Load personas
        with open(request.personas_json_path, 'r', encoding='utf-8') as f:
            personas = json.load(f)
        
        logger.info(f"Loaded {len(personas)} personas")
        
        # Initialize output manager
        output_manager = OutputManager(request.output_folder, request.scenario_name)
        output_manager.set_workflow_type("agent")
        
        # Generate Agent testset
        generator = AgentTestsetGenerator(
            model_provider=request.model_provider,
            model_name=request.model_name
        )
        
        # Get output folder for this stage
        stage_folder = output_manager.get_stage_folder("agent_testset")
        
        result = generator.generate_testset(
            documents_folder=request.documents_folder,
            output_folder=str(stage_folder),
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            tasks_per_chunk=request.tasks_per_chunk,
            language=request.language,
            personas=personas
        )
        
        if result["success"]:
            # Mark stage complete
            output_manager.mark_stage_complete("agent_testset", result)
            
            return {
                "success": True,
                "message": "Agent testset generated successfully with personas!",
                "scenario_name": output_manager.scenario_name,
                "scenario_folder": str(output_manager.scenario_folder),
                "excel_path": result["excel_path"],
                "total_tasks": result["total_tasks"],
                "total_chunks": result["total_chunks"],
                "total_documents": result["total_documents"],
                "personas_used": len(personas)
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Error generating Agent testset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/status/{scenario_name}")
async def get_workflow_status(scenario_name: str, base_folder: str):
    """Get the current status of a workflow."""
    try:
        from ai.testset.output_manager import OutputManager
        
        output_manager = OutputManager(base_folder, scenario_name)
        summary = output_manager.get_summary()
        
        return {
            "success": True,
            "workflow_status": summary
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class CheckExistingDataRequest(BaseModel):
    """Request model for checking existing data."""
    output_folder: str
    scenario_name: Optional[str] = None  # Optional scenario name to check specific project


@router.post("/workflow/check-existing-data")
async def check_existing_data(request: CheckExistingDataRequest):
    """Check if there's existing persona and document data in the output folder."""
    try:
        import os
        import json
        import glob
        from ai.testset.output_manager import OutputManager
        
        output_folder = request.output_folder
        scenario_name = request.scenario_name  # Get scenario_name from request
        
        # Map path for Docker compatibility
        output_folder = OutputManager.map_path_for_docker(output_folder)
        
        # Handle Docker paths (/app/outputs/...) - map to local paths if not in Docker
        import os
        from pathlib import Path
        
        # Check if we're actually in Docker (check if /app exists and is accessible)
        is_docker = os.path.exists('/app') and os.path.isdir('/app')
        
        # If path starts with /app/outputs but we're not in Docker, map to local
        if output_folder.startswith('/app/outputs') and not is_docker:
            # Extract the relative path after /app/outputs/
            relative_path = output_folder.replace('/app/outputs', '').strip('/')
            # Find project root
            current_dir = Path.cwd()
            project_root = current_dir
            if (current_dir / "outputs").exists():
                project_root = current_dir
            else:
                for parent in current_dir.parents:
                    if (parent / "outputs").exists():
                        project_root = parent
                        break
            # Build the mapped path
            if relative_path:
                output_folder = str(project_root / "outputs" / relative_path)
            else:
                output_folder = str(project_root / "outputs")
            logger.info(f"Mapped Docker path to local: {request.output_folder} -> {output_folder}")
        # Convert to absolute path if relative
        elif not os.path.isabs(output_folder):
            # Get the project root (assuming we're in the project directory)
            current_dir = Path.cwd()
            project_root = current_dir
            # If outputs folder exists in current dir, use it
            if (current_dir / "outputs").exists():
                project_root = current_dir
            else:
                # Try parent directories
                for parent in current_dir.parents:
                    if (parent / "outputs").exists():
                        project_root = parent
                        break
            
            if output_folder.startswith('./'):
                output_folder = str(project_root / output_folder[2:])
            elif not os.path.isabs(output_folder):
                output_folder = str(project_root / output_folder)
        
        logger.info(f"Checking existing data in: {output_folder} (absolute: {os.path.abspath(output_folder)})")
        
        # If scenario_name is provided, check that specific project folder
        if scenario_name:
            # Check specific scenario folder
            scenario_folder = os.path.join(output_folder, scenario_name)
            direct_personas_folder = os.path.join(scenario_folder, "01_personas")
            logger.info(f"Checking specific scenario: {scenario_name} in {scenario_folder}")
            logger.info(f"Checking direct personas folder: {direct_personas_folder} (exists: {os.path.exists(direct_personas_folder)})")
            
            if not os.path.exists(direct_personas_folder):
                logger.warning(f"Scenario folder {scenario_folder} does not contain 01_personas")
                return {
                    "has_personas": False,
                    "has_documents": False
                }
            
            personas_folder = direct_personas_folder
            logger.info(f"Found scenario folder: {scenario_folder}, name: {scenario_name}")
        else:
            # Check if the output_folder itself is a scenario folder (contains 01_personas)
            direct_personas_folder = os.path.join(output_folder, "01_personas")
            logger.info(f"Checking direct personas folder: {direct_personas_folder} (exists: {os.path.exists(direct_personas_folder)})")
            
            if os.path.exists(direct_personas_folder):
                # User provided a scenario folder directly
                personas_folder = direct_personas_folder
                scenario_folder = output_folder
                scenario_name = os.path.basename(scenario_folder)
                logger.info(f"Found scenario folder directly: {scenario_folder}, name: {scenario_name}")
            else:
                # Look for personas folder in subdirectories
                search_pattern = os.path.join(output_folder, "*/01_personas")
                logger.info(f"Searching for personas folders with pattern: {search_pattern}")
                personas_folders = glob.glob(search_pattern)
                logger.info(f"Found {len(personas_folders)} personas folders: {personas_folders}")
                
                if not personas_folders:
                    logger.warning(f"No personas folders found in {output_folder}")
                    return {
                        "has_personas": False,
                        "has_documents": False
                    }
                
                # Use the most recent personas folder
                personas_folder = max(personas_folders, key=os.path.getctime)
                scenario_folder = os.path.dirname(personas_folder)
                scenario_name = os.path.basename(scenario_folder)
                logger.info(f"Using most recent scenario folder: {scenario_folder}, name: {scenario_name}")
        
        # Find JSON file
        json_pattern = os.path.join(personas_folder, "*_personas_full.json")
        logger.info(f"Searching for JSON files with pattern: {json_pattern}")
        json_files = glob.glob(json_pattern)
        logger.info(f"Found {len(json_files)} JSON files: {json_files}")
        
        if not json_files:
            logger.warning(f"No personas JSON files found in {personas_folder}")
            return {
                "has_personas": False,
                "has_documents": False
            }
        
        personas_json_path = json_files[0]
        
        # Count personas
        with open(personas_json_path, 'r', encoding='utf-8') as f:
            personas = json.load(f)
            total_personas = len(personas) if isinstance(personas, list) else 0
        
        # Check for documents folder and metadata JSON
        documents_folder = os.path.join(scenario_folder, "02_documents")
        has_documents_folder = os.path.exists(documents_folder) and os.path.isdir(documents_folder)
        
        # Check for documents metadata JSON file
        has_documents_json = False
        documents_metadata_path = None
        if has_documents_folder:
            # Look for metadata JSON file
            metadata_files = glob.glob(os.path.join(documents_folder, "*_documents_metadata.json"))
            if metadata_files:
                documents_metadata_path = metadata_files[0]
                has_documents_json = True
                # Verify JSON file is valid and has content
                try:
                    with open(documents_metadata_path, 'r', encoding='utf-8') as f:
                        documents_metadata = json.load(f)
                        if documents_metadata and len(documents_metadata.get('documents', [])) > 0:
                            has_documents_json = True
                        else:
                            has_documents_json = False
                except Exception:
                    has_documents_json = False
        
        result = {
            "has_personas": True,
            "personas_json_path": personas_json_path,
            "scenario_name": scenario_name,
            "scenario_folder": str(scenario_folder),
            "total_personas": total_personas,
            "has_documents": has_documents_json,  # Only true if JSON exists and has content
            "has_documents_folder": has_documents_folder
        }
        
        if has_documents_json and documents_metadata_path:
            result["documents_folder"] = documents_folder
            result["documents_metadata_path"] = documents_metadata_path
        
        return result
        
    except Exception as e:
        logger.error(f"Error checking existing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class ValidateDataRequest(BaseModel):
    """Request model for validating data integrity."""
    output_folder: str
    scenario_name: str


@router.post("/workflow/validate-data")
async def validate_data_integrity(request: ValidateDataRequest):
    """Validate that all required files exist for a scenario."""
    try:
        import os
        import glob as glob_module
        
        logger.info(f"Validating data for: {request.output_folder}/{request.scenario_name}")
        
        # Check if output_folder is actually the scenario folder itself
        if os.path.exists(os.path.join(request.output_folder, "01_personas")):
            # output_folder is the scenario folder
            scenario_folder = request.output_folder
            logger.info(f"Using output_folder as scenario_folder: {scenario_folder}")
        else:
            # Traditional case: output_folder contains scenario_name subfolder
            scenario_folder = os.path.join(request.output_folder, request.scenario_name)
            logger.info(f"Using combined path as scenario_folder: {scenario_folder}")
        
        if not os.path.exists(scenario_folder):
            logger.error(f"Scenario folder does not exist: {scenario_folder}")
            return {"is_valid": False, "missing": ["scenario_folder"]}
        
        missing_items = []
        
        # Check for personas folder and files
        personas_folder = os.path.join(scenario_folder, "01_personas")
        if not os.path.exists(personas_folder):
            missing_items.append("01_personas")
        else:
            # Check for essential files
            json_files = glob_module.glob(os.path.join(personas_folder, "*_personas_full.json"))
            if not json_files:
                missing_items.append("personas_json")
        
        # Check for documents folder and files
        documents_folder = os.path.join(scenario_folder, "02_documents")
        if not os.path.exists(documents_folder):
            missing_items.append("02_documents")
        else:
            # Check if there are any document files
            doc_files = glob_module.glob(os.path.join(documents_folder, "*.txt"))
            if not doc_files:
                missing_items.append("document_files")
        
        is_valid = len(missing_items) == 0
        
        return {
            "is_valid": is_valid,
            "missing": missing_items if not is_valid else []
        }
        
    except Exception as e:
        logger.error(f"Error validating data integrity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class CheckExistingTestsetRequest(BaseModel):
    """Request model for checking existing testset."""
    output_folder: str
    scenario_name: str
    testset_type: str  # 'rag' or 'agent'


@router.get("/workflow/list-output-folders")
async def list_output_folders(base_path: str = "/app/outputs"):
    """List all folders in the output directory."""
    try:
        import os
        from pathlib import Path
        
        output_path = Path(base_path)
        
        if not output_path.exists():
            return {"folders": []}
        
        # Get all directories
        folders = []
        for item in output_path.iterdir():
            if item.is_dir():
                folders.append({
                    "name": item.name,
                    "path": str(item),
                    "created": os.path.getctime(item) if os.path.exists(item) else 0
                })
        
        # Sort by creation time (newest first)
        folders.sort(key=lambda x: x["created"], reverse=True)
        
        return {
            "base_path": base_path,
            "folders": [f["name"] for f in folders],
            "full_paths": [f["path"] for f in folders]
        }
        
    except Exception as e:
        logger.error(f"Error listing output folders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/load-personas")
async def load_personas(json_path: str):
    """Load personas from JSON file."""
    try:
        import json
        
        with open(json_path, 'r', encoding='utf-8') as f:
            personas = json.load(f)
        
        return {
            "success": True,
            "personas": personas if isinstance(personas, list) else [],
            "total": len(personas) if isinstance(personas, list) else 0
        }
        
    except Exception as e:
        logger.error(f"Error loading personas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/load-documents")
async def load_documents(metadata_path: str):
    """Load documents from metadata JSON file."""
    try:
        import json
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        documents = metadata.get('documents', []) if isinstance(metadata, dict) else []
        
        return {
            "success": True,
            "documents": documents,
            "total": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class UpdatePersonasRequest(BaseModel):
    """Request model for updating personas."""
    json_path: str
    personas: List[Dict[str, Any]]


class UpdateDocumentsRequest(BaseModel):
    """Request model for updating documents."""
    metadata_file: str
    documents: List[Dict[str, Any]]


@router.post("/workflow/update-personas")
async def update_personas(request: UpdatePersonasRequest):
    """Update personas in JSON file after editing."""
    try:
        import json
        from pathlib import Path
        import uuid
        
        json_path = Path(request.json_path)
        
        # Verify the file exists
        if not json_path.exists():
            raise HTTPException(status_code=404, detail=f"Personas file not found: {json_path}")
        
        # Save updated personas to JSON file using atomic write
        temp_json = json_path.with_suffix('.json.tmp')
        try:
            with open(temp_json, 'w', encoding='utf-8') as f:
                json.dump(request.personas, f, ensure_ascii=False, indent=2)
            # Atomic rename
            if json_path.exists():
                json_path.unlink()
            temp_json.replace(json_path)
        except Exception as write_error:
            if temp_json.exists():
                try:
                    temp_json.unlink()
                except:
                    pass
            raise write_error
        
        logger.info(f"Updated {len(request.personas)} personas in {json_path}")
        
        return {
            "success": True,
            "message": f"Successfully updated {len(request.personas)} personas",
            "json_file": str(json_path),
            "total_personas": len(request.personas)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating personas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class UpdateDocumentsRequest(BaseModel):
    """Request model for updating documents."""
    metadata_file: str
    documents: List[Dict[str, Any]]


@router.post("/workflow/update-documents")
async def update_documents(request: UpdateDocumentsRequest):
    """Update documents metadata and content files after editing."""
    try:
        import json
        from pathlib import Path
        from datetime import datetime
        from ai.testset.output_manager import OutputManager
        
        metadata_file = Path(OutputManager.map_path_for_docker(request.metadata_file))
        
        # Verify the file exists
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail=f"Documents metadata file not found: {metadata_file}")
        
        # Get documents folder
        docs_folder = metadata_file.parent
        
        # Update metadata JSON using atomic write
        temp_metadata = metadata_file.with_suffix('.json.tmp')
        try:
            with open(temp_metadata, 'w', encoding='utf-8') as f:
                json.dump(request.documents, f, ensure_ascii=False, indent=2)
            # Atomic rename
            if metadata_file.exists():
                metadata_file.unlink()
            temp_metadata.replace(metadata_file)
        except Exception as write_error:
            if temp_metadata.exists():
                try:
                    temp_metadata.unlink()
                except:
                    pass
            raise write_error
        
        # Update individual document text files
        for doc in request.documents:
            doc_id = doc.get("document_id", "unknown")
            title = doc.get("title", "Untitled").replace("/", "_").replace("\\", "_")
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
            
            # Use atomic write for document files
            temp_file = txt_file.with_suffix('.txt.tmp')
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                # Atomic rename
                if txt_file.exists():
                    txt_file.unlink()
                temp_file.replace(txt_file)
            except Exception as write_error:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
                logger.warning(f"Failed to update document file {txt_file}: {write_error}")
        
        logger.info(f"Updated {len(request.documents)} documents in {metadata_file}")
        
        return {
            "success": True,
            "message": f"Successfully updated {len(request.documents)} documents",
            "metadata_file": str(metadata_file),
            "total_documents": len(request.documents)
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update documents: {str(e)}")


class ListDocumentsRequest(BaseModel):
    """Request model for listing documents in project folder."""
    output_folder: str
    scenario_name: str


@router.post("/workflow/upload-documents")
async def upload_documents_to_project(
    files: List[UploadFile] = File(...),
    output_folder: str = Form(...),
    scenario_name: str = Form(...)
):
    """Upload documents to project's 02_documents folder."""
    try:
        from pathlib import Path
        from ai.testset.output_manager import OutputManager
        import uuid
        
        # Map path for Docker compatibility
        output_folder = OutputManager.map_path_for_docker(output_folder)
        
        # Construct 02_documents folder path
        output_manager = OutputManager(output_folder, scenario_name)
        docs_folder = output_manager.scenario_folder / "02_documents"
        
        # Create folder if it doesn't exist
        docs_folder.mkdir(parents=True, exist_ok=True)
        
        uploaded_count = 0
        uploaded_files = []
        
        for file in files:
            # Validate file type
            if not file.filename.endswith(('.txt', '.md', '.pdf')):
                logger.warning(f"Skipping unsupported file type: {file.filename}")
                continue
            
            # Read file content
            content = await file.read()
            
            # Save file using atomic write
            filename = file.filename.replace('/', '_').replace('\\', '_')
            temp_file = docs_folder / f"{filename}_{uuid.uuid4().hex[:8]}.tmp"
            target_file = docs_folder / filename
            
            try:
                with open(temp_file, 'wb') as f:
                    f.write(content)
                
                # Atomic rename
                if target_file.exists():
                    target_file.unlink()
                temp_file.replace(target_file)
                
                uploaded_count += 1
                uploaded_files.append(filename)
                logger.info(f"Uploaded file: {target_file}")
            except Exception as write_error:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
                logger.error(f"Failed to upload {filename}: {write_error}")
        
        return {
            "success": True,
            "message": f"Successfully uploaded {uploaded_count} files",
            "uploaded_count": uploaded_count,
            "uploaded_files": uploaded_files,
            "documents_folder": str(docs_folder)
        }
        
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload documents: {str(e)}")


class GetScenarioFolderRequest(BaseModel):
    """Request model for getting scenario folder path."""
    output_folder: str
    scenario_name: str


@router.post("/workflow/get-scenario-folder")
async def get_scenario_folder(request: GetScenarioFolderRequest):
    """Get the correct scenario folder path using OutputManager."""
    try:
        from ai.testset.output_manager import OutputManager
        
        # Map path for Docker compatibility
        output_folder = OutputManager.map_path_for_docker(request.output_folder)
        
        # Use OutputManager to get the correct scenario folder
        output_manager = OutputManager(output_folder, request.scenario_name)
        
        return {
            "success": True,
            "scenario_folder": str(output_manager.scenario_folder),
            "base_output_folder": str(output_manager.base_output_folder),
            "scenario_name": output_manager.scenario_name
        }
        
    except Exception as e:
        logger.error(f"Error getting scenario folder: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get scenario folder: {str(e)}")


@router.post("/workflow/list-documents")
async def list_documents_in_project(request: ListDocumentsRequest):
    """List all documents in project's 02_documents folder."""
    try:
        from pathlib import Path
        from ai.testset.output_manager import OutputManager
        
        # Map path for Docker compatibility
        output_folder = OutputManager.map_path_for_docker(request.output_folder)
        
        # Construct 02_documents folder path using OutputManager
        output_manager = OutputManager(output_folder, request.scenario_name)
        docs_folder = output_manager.scenario_folder / "02_documents"
        
        if not docs_folder.exists():
            return {
                "success": True,
                "files": [],
                "message": "02_documents folder does not exist"
            }
        
        # List all supported document files (.txt, .md, .pdf)
        files = []
        supported_extensions = ['*.txt', '*.md', '*.pdf']
        for ext in supported_extensions:
            for file_path in docs_folder.glob(ext):
                try:
                    file_size = file_path.stat().st_size
                    files.append({
                        "name": file_path.name,
                        "size": file_size,
                        "path": str(file_path),
                        "extension": file_path.suffix.lower()
                    })
                except Exception as e:
                    logger.warning(f"Error reading file info for {file_path}: {e}")
        
        # Sort by name
        files.sort(key=lambda x: x["name"])
        
        return {
            "success": True,
            "files": files,
            "total": len(files),
            "documents_folder": str(docs_folder)
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.post("/workflow/check-existing-testset")
async def check_existing_testset(request: CheckExistingTestsetRequest):
    """Check if there's an existing testset for the scenario."""
    try:
        import os
        import glob as glob_module
        
        logger.info(f"Checking existing testset: {request.output_folder}/{request.scenario_name} ({request.testset_type})")
        
        # Check if output_folder is actually the scenario folder itself
        if os.path.exists(os.path.join(request.output_folder, "01_personas")):
            # output_folder is the scenario folder
            scenario_folder = request.output_folder
            logger.info(f"Using output_folder as scenario_folder: {scenario_folder}")
        else:
            # Traditional case: output_folder contains scenario_name subfolder
            scenario_folder = os.path.join(request.output_folder, request.scenario_name)
            logger.info(f"Using combined path as scenario_folder: {scenario_folder}")
        
        # Look for testset folder based on type
        if request.testset_type == 'rag':
            testset_folder = os.path.join(scenario_folder, "03_rag_testset")
        else:
            testset_folder = os.path.join(scenario_folder, "03_agent_testset")
        
        logger.info(f"Looking for testset in: {testset_folder}")
        
        if not os.path.exists(testset_folder):
            logger.info(f"Testset folder does not exist: {testset_folder}")
            return {"has_testset": False}
        
        # Look for Excel file
        excel_files = glob_module.glob(os.path.join(testset_folder, "*.xlsx"))
        
        if not excel_files:
            return {"has_testset": False}
        
        testset_path = excel_files[0]
        
        # Try to read QA count from Excel
        try:
            import pandas as pd
            df = pd.read_excel(testset_path)
            total_qa_pairs = len(df)
        except:
            total_qa_pairs = None
        
        return {
            "has_testset": True,
            "testset_path": testset_path,
            "total_qa_pairs": total_qa_pairs
        }
        
    except Exception as e:
        logger.error(f"Error checking existing testset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
