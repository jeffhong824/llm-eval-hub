"""
Output Manager Module

Manages staged output directories and files for the entire evaluation workflow.
Ensures organized, hierarchical output structure with clear stage progression.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages staged output for the evaluation workflow."""
    
    # Stage folder names (fixed as per requirements)
    STAGE_FOLDERS = {
        "personas": "01_personas",
        "documents": "02_documents",
        "rag_testset": "03_rag_testset",
        "agent_testset": "03_agent_testset",
        "rag_evaluation": "04_rag_evaluation",
        "agent_evaluation": "04_agent_evaluation"
    }
    
    def __init__(self, base_output_folder: str, scenario_name: Optional[str] = None):
        """
        Initialize output manager.
        
        Args:
            base_output_folder: Base directory for all outputs
            scenario_name: Optional name for the scenario (will be used in folder naming)
        """
        self.base_output_folder = Path(base_output_folder)
        self.scenario_name = scenario_name or f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create main scenario folder
        self.scenario_folder = self.base_output_folder / self.scenario_name
        self.scenario_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self.metadata = {
            "scenario_name": self.scenario_name,
            "created_at": datetime.now().isoformat(),
            "base_folder": str(self.base_output_folder),
            "scenario_folder": str(self.scenario_folder),
            "stages_completed": [],
            "workflow_type": None,  # "rag" or "agent"
            "stage_outputs": {}
        }
        
        self._save_metadata()
        logger.info(f"Initialized output manager for scenario: {self.scenario_name}")
    
    def get_stage_folder(self, stage: str) -> Path:
        """
        Get the path for a specific stage folder.
        
        Args:
            stage: Stage name (personas, documents, rag_testset, agent_testset, etc.)
            
        Returns:
            Path to the stage folder
        """
        if stage not in self.STAGE_FOLDERS:
            raise ValueError(f"Unknown stage: {stage}. Valid stages: {list(self.STAGE_FOLDERS.keys())}")
        
        folder_name = self.STAGE_FOLDERS[stage]
        stage_folder = self.scenario_folder / folder_name
        stage_folder.mkdir(parents=True, exist_ok=True)
        
        return stage_folder
    
    def mark_stage_complete(self, stage: str, stage_output: Dict[str, Any]):
        """
        Mark a stage as complete and save its output metadata.
        
        Args:
            stage: Stage name
            stage_output: Dictionary with stage output information
        """
        if stage not in self.metadata["stages_completed"]:
            self.metadata["stages_completed"].append(stage)
        
        self.metadata["stage_outputs"][stage] = {
            "completed_at": datetime.now().isoformat(),
            "output": stage_output
        }
        
        self._save_metadata()
        logger.info(f"Marked stage '{stage}' as complete")
    
    def set_workflow_type(self, workflow_type: str):
        """
        Set the workflow type (RAG or Agent).
        
        Args:
            workflow_type: Either "rag" or "agent"
        """
        if workflow_type not in ["rag", "agent"]:
            raise ValueError("workflow_type must be 'rag' or 'agent'")
        
        self.metadata["workflow_type"] = workflow_type
        self._save_metadata()
        logger.info(f"Set workflow type to: {workflow_type}")
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        metadata_file = self.scenario_folder / "workflow_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current workflow state."""
        return {
            "scenario_name": self.scenario_name,
            "scenario_folder": str(self.scenario_folder),
            "workflow_type": self.metadata.get("workflow_type"),
            "stages_completed": self.metadata.get("stages_completed", []),
            "total_stages": len(self.metadata.get("stages_completed", [])),
            "created_at": self.metadata.get("created_at"),
            "last_updated": datetime.now().isoformat()
        }
    
    @staticmethod
    def map_path_for_docker(path: str) -> str:
        """
        Map Windows/Mac paths to Docker paths.
        
        Args:
            path: Original path
            
        Returns:
            Mapped path for Docker environment
        """
        path_obj = Path(path)
        
        # Handle Windows absolute paths
        if path.startswith('D:\\') or path.startswith('D:/'):
            # D:\workplace\project_management\my_github\ -> /app/host_github/
            mapped = path.replace('D:\\workplace\\project_management\\my_github', '/app/host_github')
            mapped = mapped.replace('D:/workplace/project_management/my_github', '/app/host_github')
            mapped = mapped.replace('\\', '/')
            logger.info(f"Mapped Windows path to Docker: {path} -> {mapped}")
            return mapped
        
        # Handle Mac/Linux paths - check if it's under user's home or common locations
        elif path.startswith('/Users/') or path.startswith('/home/'):
            # For Mac/Linux, assume mounted at /app/host_files
            # This will need to be configured in docker-compose.yml
            if '/Users/' in path:
                user_dir = path.split('/Users/')[1].split('/')[0]
                rest_path = '/'.join(path.split('/')[3:])
                mapped = f'/app/host_files/{rest_path}'
            else:
                mapped = path.replace('/home/', '/app/host_files/')
            logger.info(f"Mapped Unix path to Docker: {path} -> {mapped}")
            return mapped
        
        # If already looks like a Docker path or relative path
        elif path.startswith('/app/'):
            return path
        else:
            # Relative path - keep as is or map to a default location
            logger.info(f"Keeping path as is: {path}")
            return path


class StageOutputHandler:
    """Helper class to handle outputs for specific stages."""
    
    @staticmethod
    def save_stage_summary(
        stage_folder: Path,
        stage_name: str,
        summary_data: Dict[str, Any]
    ):
        """Save a summary file for a stage."""
        summary_file = stage_folder / f"{stage_name}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved stage summary: {summary_file}")
    
    @staticmethod
    def create_readme(stage_folder: Path, stage_name: str, description: str):
        """Create a README file for a stage folder."""
        readme_file = stage_folder / "README.md"
        content = f"""# {stage_name.replace('_', ' ').title()}

{description}

## Generated Files

This folder contains files generated during the {stage_name} stage.

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
*LLM Evaluation Hub - Automated Workflow*
"""
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Created README: {readme_file}")

