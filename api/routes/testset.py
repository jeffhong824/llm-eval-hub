"""Testset generation API endpoints."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field
import pandas as pd

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
