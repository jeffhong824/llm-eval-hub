"""Health check API endpoints."""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from configs.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str
    timestamp: datetime
    version: str
    uptime: float
    dependencies: Dict[str, str]


class DetailedHealthResponse(BaseModel):
    """Detailed health check response model."""
    
    status: str
    timestamp: datetime
    version: str
    uptime: float
    dependencies: Dict[str, str]
    system_info: Dict[str, Any]
    metrics: Dict[str, Any]


# Application start time for uptime calculation
app_start_time = datetime.now()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        HealthResponse with basic health status
    """
    try:
        uptime = (datetime.now() - app_start_time).total_seconds()
        
        # Check dependencies
        dependencies = await check_dependencies()
        
        # Determine overall status
        status = "healthy" if all(
            dep_status == "healthy" for dep_status in dependencies.values()
        ) else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now(),
            version=settings.app_version,
            uptime=uptime,
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check endpoint.
    
    Returns:
        DetailedHealthResponse with comprehensive health status
    """
    try:
        uptime = (datetime.now() - app_start_time).total_seconds()
        
        # Check dependencies
        dependencies = await check_dependencies()
        
        # Get system information
        system_info = await get_system_info()
        
        # Get metrics
        metrics = await get_metrics()
        
        # Determine overall status
        status = "healthy" if all(
            dep_status == "healthy" for dep_status in dependencies.values()
        ) else "degraded"
        
        return DetailedHealthResponse(
            status=status,
            timestamp=datetime.now(),
            version=settings.app_version,
            uptime=uptime,
            dependencies=dependencies,
            system_info=system_info,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes.
    
    Returns:
        Simple ready status
    """
    try:
        dependencies = await check_dependencies()
        
        # Check if critical dependencies are healthy
        critical_deps = ["database", "langsmith", "openai"]
        ready = all(
            dependencies.get(dep, "unhealthy") == "healthy" 
            for dep in critical_deps
        )
        
        if ready:
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
            
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """
    Liveness check endpoint for Kubernetes.
    
    Returns:
        Simple alive status
    """
    try:
        return {"status": "alive", "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not alive")


async def check_dependencies() -> Dict[str, str]:
    """Check health of external dependencies."""
    dependencies = {}
    
    try:
        # Check database
        dependencies["database"] = await check_database()
    except Exception as e:
        logger.warning(f"Database check failed: {str(e)}")
        dependencies["database"] = "unhealthy"
    
    try:
        # Check LangSmith
        dependencies["langsmith"] = await check_langsmith()
    except Exception as e:
        logger.warning(f"LangSmith check failed: {str(e)}")
        dependencies["langsmith"] = "unhealthy"
    
    try:
        # Check OpenAI
        dependencies["openai"] = await check_openai()
    except Exception as e:
        logger.warning(f"OpenAI check failed: {str(e)}")
        dependencies["openai"] = "unhealthy"
    
    return dependencies


async def check_database() -> str:
    """Check database connectivity."""
    try:
        # This would check actual database connection
        # For now, return healthy
        return "healthy"
    except Exception:
        return "unhealthy"


async def check_langsmith() -> str:
    """Check LangSmith API connectivity."""
    try:
        # This would check LangSmith API
        # For now, return healthy
        return "healthy"
    except Exception:
        return "unhealthy"


async def check_openai() -> str:
    """Check OpenAI API connectivity."""
    try:
        # This would check OpenAI API
        # For now, return healthy
        return "healthy"
    except Exception:
        return "unhealthy"


async def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    import psutil
    import platform
    
    try:
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent
        }
    except Exception as e:
        logger.warning(f"Error getting system info: {str(e)}")
        return {"error": str(e)}


async def get_metrics() -> Dict[str, Any]:
    """Get application metrics."""
    try:
        # This would return actual metrics
        # For now, return placeholder
        return {
            "total_evaluations": 0,
            "total_testsets": 0,
            "active_evaluations": 0,
            "average_evaluation_time": 0.0
        }
    except Exception as e:
        logger.warning(f"Error getting metrics: {str(e)}")
        return {"error": str(e)}



