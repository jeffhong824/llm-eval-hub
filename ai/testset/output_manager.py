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
import uuid
import os
import tempfile
import shutil

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
        import os
        
        # Check if we're running in Docker (check if /app exists and is read-only)
        is_docker = os.path.exists('/app') and not os.access('/app', os.W_OK)
        
        # Map path for Docker if needed
        mapped_folder = self.map_path_for_docker(base_output_folder)
        
        # If path starts with /app and we're in Docker, try to use a writable location
        if mapped_folder.startswith('/app/') and is_docker:
            # Extract relative path from /app/outputs or /app
            if mapped_folder.startswith('/app/outputs'):
                relative_path = mapped_folder.replace('/app/outputs', '').lstrip('/')
            elif mapped_folder == '/app':
                relative_path = ''
            else:
                relative_path = mapped_folder.replace('/app/', '').lstrip('/')
            
            # Try to use /app/host_github/outputs (mounted from project root)
            if relative_path:
                mapped_folder = f'/app/host_github/outputs/{relative_path}'
            else:
                mapped_folder = '/app/host_github/outputs'
            logger.info(f"Path {base_output_folder} mapped to {mapped_folder} (Docker environment)")
        elif not is_docker and mapped_folder.startswith('/app/'):
            # Running locally but path looks like Docker path, convert to relative
            if mapped_folder.startswith('/app/outputs'):
                relative_path = mapped_folder.replace('/app/outputs', '').lstrip('/')
                if relative_path:
                    mapped_folder = f'./outputs/{relative_path}'
                else:
                    mapped_folder = './outputs'
            elif mapped_folder.startswith('/app/host_github'):
                relative_path = mapped_folder.replace('/app/host_github', '').lstrip('/')
                mapped_folder = f'./{relative_path}' if relative_path else '.'
            logger.info(f"Path {base_output_folder} mapped to {mapped_folder} (local environment)")
        
        # Final check: if path is still /app or /app/outputs and we can't write, use current directory
        if mapped_folder.startswith('/app') and is_docker:
            try:
                test_path = Path(mapped_folder)
                # Use unique temp file name to avoid conflicts in concurrent access
                test_file = test_path / f'.write_test_{uuid.uuid4().hex[:8]}'
                test_file.write_text('test')
                test_file.unlink()
                logger.info(f"Path {mapped_folder} is writable")
            except (PermissionError, OSError) as e:
                # Use current working directory as fallback
                cwd = os.getcwd()
                if mapped_folder.startswith('/app/outputs'):
                    relative_path = mapped_folder.replace('/app/outputs', '').lstrip('/')
                    mapped_folder = str(Path(cwd) / 'outputs' / relative_path) if relative_path else str(Path(cwd) / 'outputs')
                else:
                    mapped_folder = str(Path(cwd) / 'outputs')
                logger.warning(f"Path {base_output_folder} mapped to {mapped_folder} (fallback due to read-only: {e})")
        
        self.base_output_folder = Path(mapped_folder)
        
        # Generate unique scenario name with UUID to avoid conflicts in concurrent usage
        if scenario_name:
            self.scenario_name = scenario_name
        else:
            # Add UUID suffix to ensure uniqueness even if multiple users create scenarios at the same time
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
            self.scenario_name = f"scenario_{timestamp}_{unique_id}"
        
        # Check if base_output_folder already contains scenario_name to avoid double nesting
        if self.scenario_name and self.base_output_folder.name == self.scenario_name:
            # base_output_folder is already the scenario folder
            self.scenario_folder = self.base_output_folder
            logger.info(f"Using existing scenario folder: {self.scenario_folder}")
        else:
            # Create main scenario folder with retry logic for concurrent access
            self.scenario_folder = self.base_output_folder / self.scenario_name
            max_retries = 3
            retry_count = 0
            created = False
            
            while retry_count < max_retries and not created:
                try:
                    # Use exist_ok=True to handle race conditions where folder might be created by another process
                    self.scenario_folder.mkdir(parents=True, exist_ok=True)
                    # Verify the folder was actually created (handle race condition)
                    if self.scenario_folder.exists():
                        created = True
                        logger.info(f"Created new scenario folder: {self.scenario_folder}")
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            import time
                            time.sleep(0.1 * retry_count)  # Exponential backoff
                except (PermissionError, OSError) as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        # If we can't write to the path, try /app/host_github/outputs as fallback
                        logger.error(f"Cannot write to {self.scenario_folder}: {e}")
                        fallback_folder = Path('/app/host_github/outputs') / self.scenario_name
                        try:
                            fallback_folder.mkdir(parents=True, exist_ok=True)
                            self.scenario_folder = fallback_folder
                            self.base_output_folder = Path('/app/host_github/outputs')
                            logger.warning(f"Using fallback folder: {self.scenario_folder}")
                            created = True
                        except Exception as fallback_error:
                            # Last resort: use current working directory
                            cwd_outputs = Path(os.getcwd()) / 'outputs' / self.scenario_name
                            cwd_outputs.mkdir(parents=True, exist_ok=True)
                            self.scenario_folder = cwd_outputs
                            self.base_output_folder = Path(os.getcwd()) / 'outputs'
                            logger.error(f"All fallbacks failed, using current directory: {self.scenario_folder} (error: {fallback_error})")
                            created = True
                    else:
                        import time
                        time.sleep(0.1 * retry_count)
        
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
        """Save metadata to JSON file using atomic write to prevent corruption in concurrent access."""
        try:
            metadata_file = self.scenario_folder / "workflow_metadata.json"
            # Ensure the directory exists
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use atomic write: write to temp file first, then rename
            # This prevents file corruption if multiple processes write simultaneously
            temp_file = metadata_file.with_suffix('.json.tmp')
            try:
                # Write to temporary file first
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
                
                # Atomic rename (works on most filesystems)
                # On Windows, if target exists, replace it
                if metadata_file.exists():
                    metadata_file.unlink()
                temp_file.replace(metadata_file)
                
                logger.info(f"Saved workflow metadata to: {metadata_file}")
            except Exception as write_error:
                # Clean up temp file on error
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
                raise write_error
                
        except (PermissionError, OSError, FileNotFoundError) as e:
            logger.error(f"Failed to save metadata to {self.scenario_folder / 'workflow_metadata.json'}: {e}")
            # Try to save to a fallback location
            try:
                fallback_path = Path(os.getcwd()) / 'outputs' / self.scenario_name / "workflow_metadata.json"
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Use atomic write for fallback too
                temp_fallback = fallback_path.with_suffix('.json.tmp')
                try:
                    with open(temp_fallback, 'w', encoding='utf-8') as f:
                        json.dump(self.metadata, f, ensure_ascii=False, indent=2)
                    if fallback_path.exists():
                        fallback_path.unlink()
                    temp_fallback.replace(fallback_path)
                    logger.warning(f"Saved metadata to fallback location: {fallback_path}")
                except Exception as fallback_write_error:
                    if temp_fallback.exists():
                        try:
                            temp_fallback.unlink()
                        except:
                            pass
                    raise fallback_write_error
            except Exception as fallback_error:
                logger.error(f"Failed to save metadata to fallback location: {fallback_error}")
    
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
        Only maps if actually running in Docker, otherwise returns local path.
        
        Args:
            path: Original path
            
        Returns:
            Mapped path for Docker environment, or original path if not in Docker
        """
        import os
        
        # Check if we're actually running in Docker
        is_docker = os.path.exists('/app') and os.path.isdir('/app')
        
        # Handle Windows absolute paths
        if path.startswith('D:\\') or path.startswith('D:/'):
            if is_docker:
                # D:\workplace\project_management\my_github\ -> /app/host_github/
                mapped = path.replace('D:\\workplace\\project_management\\my_github', '/app/host_github')
                mapped = mapped.replace('D:/workplace/project_management/my_github', '/app/host_github')
                mapped = mapped.replace('\\', '/')
                logger.info(f"Mapped Windows path to Docker: {path} -> {mapped}")
                return mapped
            else:
                # Not in Docker, keep original path
                return path
        
        # Handle Mac/Linux absolute paths
        elif path.startswith('/Users/') or path.startswith('/home/'):
            if is_docker:
                # Only map to /app/host_files if actually in Docker
                if '/Users/' in path:
                    # Extract path after /Users/username/
                    parts = path.split('/')
                    if len(parts) > 3:
                        rest_path = '/'.join(parts[3:])
                        mapped = f'/app/host_files/{rest_path}'
                    else:
                        mapped = path
                else:
                    mapped = path.replace('/home/', '/app/host_files/')
                logger.info(f"Mapped Unix path to Docker: {path} -> {mapped}")
                return mapped
            else:
                # Not in Docker, keep original path
                logger.info(f"Not in Docker, keeping path as is: {path}")
                return path
        
        # If already looks like a Docker path
        elif path.startswith('/app/'):
            if not is_docker:
                # Not in Docker but path looks like Docker path, convert to local
                if path.startswith('/app/outputs'):
                    relative_path = path.replace('/app/outputs', '').lstrip('/')
                    current_dir = Path.cwd()
                    # Find project root (where outputs folder should be)
                    project_root = current_dir
                    if (current_dir / "outputs").exists():
                        project_root = current_dir
                    else:
                        for parent in current_dir.parents:
                            if (parent / "outputs").exists():
                                project_root = parent
                                break
                    if relative_path:
                        mapped = str(project_root / "outputs" / relative_path)
                    else:
                        mapped = str(project_root / "outputs")
                    logger.info(f"Converted Docker path to local: {path} -> {mapped}")
                    return mapped
            return path
        else:
            # Relative path - keep as is
            logger.info(f"Keeping relative path as is: {path}")
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

