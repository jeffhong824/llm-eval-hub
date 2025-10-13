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
        
        personas = persona_generator.generate_personas(
            scenario_description=request.scenario_description,
            num_personas=request.num_personas,
            language=request.language
        )
        
        # Save personas to stage folder
        save_result = persona_generator.save_personas_to_files(
            personas=personas,
            output_folder=str(output_manager.scenario_folder),
            scenario_name=output_manager.scenario_name
        )
        
        # Mark stage complete
        output_manager.mark_stage_complete("personas", save_result)
        
        return {
            "success": True,
            "message": f"Successfully generated {len(personas)} personas",
            "scenario_name": output_manager.scenario_name,
            "scenario_folder": str(output_manager.scenario_folder),
            "personas_folder": save_result["personas_folder"],
            "excel_file": save_result["excel_file"],
            "json_file": save_result["json_file"],
            "total_personas": len(personas),
            "personas_preview": personas[:3]  # Show first 3 personas
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
        
        documents = doc_generator.generate_documents(
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
            "documents_preview": [
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
    Stage 3 (RAG): Generate RAG testset using documents and personas.
    """
    try:
        logger.info(f"Starting RAG testset generation with personas")
        
        import json
        from pathlib import Path
        from ai.testset.rag_generator import RAGTestsetGenerator
        from ai.testset.output_manager import OutputManager
        
        # Load personas
        with open(request.personas_json_path, 'r', encoding='utf-8') as f:
            personas = json.load(f)
        
        logger.info(f"Loaded {len(personas)} personas")
        
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
        
        result = generator.generate_testset(
            documents_folder=request.documents_folder,
            output_folder=str(stage_folder),
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            qa_per_chunk=request.qa_per_chunk,
            language=request.language,
            personas=personas
        )
        
        if result["success"]:
            # Mark stage complete
            output_manager.mark_stage_complete("rag_testset", result)
            
            return {
                "success": True,
                "message": "RAG testset generated successfully with personas!",
                "scenario_name": output_manager.scenario_name,
                "scenario_folder": str(output_manager.scenario_folder),
                "excel_path": result["excel_path"],
                "total_qa_pairs": result["total_qa_pairs"],
                "total_chunks": result["total_chunks"],
                "total_documents": result["total_documents"],
                "personas_used": len(personas)
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Error generating RAG testset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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


@router.post("/workflow/check-existing-data")
async def check_existing_data(request: CheckExistingDataRequest):
    """Check if there's existing persona and document data in the output folder."""
    try:
        import os
        import json
        import glob
        
        output_folder = request.output_folder
        
        # Check if the output_folder itself is a scenario folder (contains 01_personas)
        direct_personas_folder = os.path.join(output_folder, "01_personas")
        
        if os.path.exists(direct_personas_folder):
            # User provided a scenario folder directly
            personas_folder = direct_personas_folder
            scenario_folder = output_folder
            scenario_name = os.path.basename(scenario_folder)
        else:
            # Look for personas folder in subdirectories
            personas_folders = glob.glob(os.path.join(output_folder, "*/01_personas"))
            
            if not personas_folders:
                return {
                    "has_personas": False,
                    "has_documents": False
                }
            
            # Use the most recent personas folder
            personas_folder = max(personas_folders, key=os.path.getctime)
            scenario_folder = os.path.dirname(personas_folder)
            scenario_name = os.path.basename(scenario_folder)
        
        # Find JSON file
        json_files = glob.glob(os.path.join(personas_folder, "*_personas_full.json"))
        
        if not json_files:
            return {
                "has_personas": False,
                "has_documents": False
            }
        
        personas_json_path = json_files[0]
        
        # Count personas
        with open(personas_json_path, 'r', encoding='utf-8') as f:
            personas = json.load(f)
            total_personas = len(personas) if isinstance(personas, list) else 0
        
        # Check for documents
        documents_folder = os.path.join(scenario_folder, "02_documents")
        has_documents = os.path.exists(documents_folder) and os.path.isdir(documents_folder)
        
        result = {
            "has_personas": True,
            "personas_json_path": personas_json_path,
            "scenario_name": scenario_name,
            "total_personas": total_personas,
            "has_documents": has_documents
        }
        
        if has_documents:
            result["documents_folder"] = documents_folder
        
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
