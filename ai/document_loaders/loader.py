"""Document loaders for various file types using LangChain."""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd

from langchain.schema import Document
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    WebBaseLoader,
    DirectoryLoader,
    UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders.base import BaseLoader

from configs.settings import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Document processor for loading and chunking various file types."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """Initialize document processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators for text splitting
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        
        # Supported file extensions and their loaders
        self.loader_mapping = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.json': JSONLoader,
            '.md': UnstructuredMarkdownLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.ppt': UnstructuredPowerPointLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
            '.html': WebBaseLoader,
            '.htm': WebBaseLoader,
        }
    
    def load_documents_from_folder(
        self,
        folder_path: str,
        glob_pattern: str = "**/*",
        recursive: bool = True
    ) -> List[Document]:
        """
        Load documents from a folder.
        
        Args:
            folder_path: Path to the folder containing documents
            glob_pattern: Glob pattern for file matching
            recursive: Whether to search recursively
            
        Returns:
            List of loaded documents
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise ValueError(f"Folder path does not exist: {folder_path}")
        
        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        documents = []
        
        # Get all files matching the pattern
        if recursive:
            files = list(folder_path.rglob(glob_pattern))
        else:
            files = list(folder_path.glob(glob_pattern))
        
        # Filter for supported file types
        supported_files = [
            f for f in files 
            if f.is_file() and f.suffix.lower() in self.loader_mapping
        ]
        
        logger.info(f"Found {len(supported_files)} supported files in {folder_path}")
        
        for file_path in supported_files:
            try:
                # Load document
                loader_class = self.loader_mapping[file_path.suffix.lower()]
                
                if file_path.suffix.lower() in ['.html', '.htm']:
                    # WebBaseLoader needs URL
                    loader = loader_class(str(file_path))
                else:
                    loader = loader_class(str(file_path))
                
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        'source_file': str(file_path),
                        'file_name': file_path.name,
                        'file_type': file_path.suffix.lower(),
                        'file_size': file_path.stat().st_size
                    })
                
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                continue
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def load_documents_from_files(
        self,
        file_paths: List[str]
    ) -> List[Document]:
        """
        Load documents from specific files.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        for file_path in file_paths:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                continue
            
            if not file_path.is_file():
                logger.warning(f"Path is not a file: {file_path}")
                continue
            
            if file_path.suffix.lower() not in self.loader_mapping:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                continue
            
            try:
                # Load document
                loader_class = self.loader_mapping[file_path.suffix.lower()]
                
                if file_path.suffix.lower() in ['.html', '.htm']:
                    loader = loader_class(str(file_path))
                else:
                    loader = loader_class(str(file_path))
                
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        'source_file': str(file_path),
                        'file_name': file_path.name,
                        'file_type': file_path.suffix.lower(),
                        'file_size': file_path.stat().st_size
                    })
                
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                continue
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunked documents
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        
        # Update text splitter if parameters changed
        if chunk_size is not None or chunk_overlap is not None:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        
        chunked_documents = []
        
        for doc in documents:
            try:
                chunks = self.text_splitter.split_documents([doc])
                
                # Add chunk metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk.page_content)
                    })
                
                chunked_documents.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error chunking document: {str(e)}")
                continue
        
        logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
        return chunked_documents
    
    def save_documents_to_folder(
        self,
        documents: List[Document],
        output_folder: str,
        format: str = "txt",
        cleanup_temp: bool = True
    ) -> Dict[str, Any]:
        """
        Save documents to a folder.
        
        Args:
            documents: List of documents to save
            output_folder: Output folder path
            format: Output format (txt, json, csv)
            cleanup_temp: Whether to cleanup temporary files after saving
            
        Returns:
            Dictionary with save results
        """
        # Handle both absolute and relative paths
        # Check for Windows absolute paths first (even if os.path.isabs returns False in container)
        if output_folder.startswith('D:\\'):
            # Windows absolute path - map to Docker volume mount
            # Convert Windows path to container path
            if output_folder.startswith('D:\\workplace\\project_management\\my_github'):
                # Map to host_github volume mount
                relative_path = output_folder.replace('D:\\workplace\\project_management\\my_github', '').replace('\\', '/')
                output_path = Path('/app/host_github') / relative_path.lstrip('/')
                logger.info(f"Windows path mapped: {output_folder} -> {output_path}")
            else:
                # For other Windows paths, try to use as is
                output_path = Path(output_folder)
                logger.info(f"Windows path used as is: {output_path}")
        elif os.path.isabs(output_folder):
            # Unix absolute path - use as is
            output_path = Path(output_folder)
            logger.info(f"Unix path used as is: {output_path}")
        else:
            # Relative path - map to host_github volume mount for Docker compatibility
            # This ensures relative paths are accessible from the host machine
            output_path = Path('/app/host_github') / output_folder.lstrip('./')
            logger.info(f"Relative path mapped to host_github: {output_folder} -> {output_path}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        temp_files = []  # Track temporary files for cleanup
        
        for i, doc in enumerate(documents):
            try:
                if format == "txt":
                    filename = f"document_{i+1}.txt"
                    file_path = output_path / filename
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(doc.page_content)
                    
                elif format == "json":
                    filename = f"document_{i+1}.json"
                    file_path = output_path / filename
                    
                    doc_data = {
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    }
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(doc_data, f, ensure_ascii=False, indent=2)
                
                elif format == "csv":
                    filename = f"document_{i+1}.csv"
                    file_path = output_path / filename
                    
                    df = pd.DataFrame([{
                        'content': doc.page_content,
                        'metadata': str(doc.metadata)
                    }])
                    
                    df.to_csv(file_path, index=False, encoding='utf-8')
                
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                saved_files.append(str(file_path))
                logger.info(f"Saved document {i} to {filename}")
                
            except Exception as e:
                logger.error(f"Error saving document {i}: {str(e)}")
                continue
        
        # Cleanup temporary files if requested
        if cleanup_temp:
            try:
                # Find and remove temporary files in data directory
                data_dir = Path('./data')
                if data_dir.exists():
                    for temp_dir in data_dir.glob('generated_docs_*'):
                        if temp_dir.is_dir():
                            import shutil
                            shutil.rmtree(temp_dir)
                            logger.info(f"Cleaned up temporary directory: {temp_dir}")
                            temp_files.append(str(temp_dir))
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {str(e)}")
        
        return {
            'total_documents': len(documents),
            'saved_files': len(saved_files),
            'output_folder': str(output_path),
            'format': format,
            'files': saved_files,
            'cleaned_up_temp_files': temp_files
        }
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about documents."""
        if not documents:
            return {}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        file_types = {}
        file_sizes = []
        
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            file_size = doc.metadata.get('file_size', 0)
            if file_size > 0:
                file_sizes.append(file_size)
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'average_chars_per_doc': total_chars / len(documents),
            'average_words_per_doc': total_words / len(documents),
            'file_types': file_types,
            'total_file_size': sum(file_sizes),
            'average_file_size': sum(file_sizes) / len(file_sizes) if file_sizes else 0
        }
