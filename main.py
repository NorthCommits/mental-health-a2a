"""
Mental Health A2A Agent Ecosystem - Main Application

FastAPI application that orchestrates the mental health agent ecosystem
using the Agent-to-Agent (A2A) protocol.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import json

# Import A2A protocol components
from shared.a2a_protocol.communication_layer import A2ACommunicator, A2AMessage, MessageType, ContentType
from shared.a2a_protocol.agent_discovery import AgentDiscovery, AgentCard, CapabilityType
from shared.a2a_protocol.task_management import TaskManager, Task, TaskPriority
from shared.a2a_protocol.security import A2ASecurity, AgentRole, AccessLevel, AuthenticationManager

# Import agents
from agents.primary_screening_agent.screening_agent import PrimaryScreeningAgent
from agents.crisis_detection_agent.crisis_agent import CrisisDetectionAgent

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health A2A Agent Ecosystem",
    description="A comprehensive mental health support system using Agent-to-Agent protocol",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
a2a_security = A2ASecurity(secret_key="your-secret-key-here")
auth_manager = AuthenticationManager(a2a_security)

# Initialize A2A components
a2a_communicator = A2ACommunicator("main-orchestrator", "http://localhost:8000")
agent_discovery = AgentDiscovery()
task_manager = TaskManager()

# Initialize agents
primary_screening_agent = PrimaryScreeningAgent(
    agent_id="primary-screening-001",
    openai_api_key="your-openai-api-key",  # Load from environment
    a2a_communicator=a2a_communicator,
    agent_discovery=agent_discovery,
    task_manager=task_manager,
    security=a2a_security
)

crisis_detection_agent = CrisisDetectionAgent(
    agent_id="crisis-detection-001",
    a2a_communicator=a2a_communicator,
    agent_discovery=agent_discovery,
    task_manager=task_manager,
    security=a2a_security
)

# Global state
active_sessions: Dict[str, Dict[str, Any]] = {}
agent_registry: Dict[str, Any] = {
    "primary-screening-001": primary_screening_agent,
    "crisis-detection-001": crisis_detection_agent
}


# Pydantic models for API
class ScreeningRequest(BaseModel):
    user_id: str
    input_data: Dict[str, Any]
    session_context: Optional[Dict[str, Any]] = None


class ScreeningResponse(BaseModel):
    session_id: str
    status: str
    message: str
    assessment_results: Optional[List[Dict[str, Any]]] = None


class CrisisAnalysisRequest(BaseModel):
    user_id: str
    session_id: str
    interaction_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class CrisisAnalysisResponse(BaseModel):
    risk_assessment: Optional[Dict[str, Any]] = None
    crisis_detected: bool
    alert_level: Optional[str] = None
    recommended_actions: List[str] = []


class AgentStatusResponse(BaseModel):
    agent_id: str
    status: str
    capabilities: List[Dict[str, Any]]
    last_activity: datetime


# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    # In production, implement proper JWT validation
    return {"user_id": "demo-user", "role": "patient"}


# API Routes

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Mental Health A2A Agent Ecosystem",
        "version": "1.0.0",
        "status": "operational",
        "agents": list(agent_registry.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "agents_online": len(agent_registry),
        "active_sessions": len(active_sessions)
    }


@app.post("/screening/start", response_model=ScreeningResponse)
async def start_screening(
    request: ScreeningRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Start a new mental health screening session
    
    This endpoint initiates a comprehensive mental health assessment
    using the Primary Screening Agent.
    """
    try:
        # Start screening session
        session_id = await primary_screening_agent.start_screening_session(
            user_id=request.user_id,
            input_data=request.input_data,
            session_context=request.session_context
        )
        
        # Store session
        active_sessions[session_id] = {
            "user_id": request.user_id,
            "agent_id": "primary-screening-001",
            "started_at": datetime.utcnow(),
            "status": "processing"
        }
        
        # Start background analysis
        background_tasks.add_task(
            analyze_interaction_for_crisis,
            request.user_id,
            session_id,
            request.input_data,
            request.session_context
        )
        
        return ScreeningResponse(
            session_id=session_id,
            status="started",
            message="Screening session started successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/screening/{session_id}/status")
async def get_screening_status(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get the status of a screening session"""
    try:
        # Get session status from agent
        status = await primary_screening_agent.get_session_status(session_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "status": status["status"],
            "current_step": status.get("current_step"),
            "progress": status.get("progress", 0),
            "message": status.get("message", "")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/screening/{session_id}/results")
async def get_screening_results(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get assessment results for a completed screening session"""
    try:
        # Get results from agent
        results = await primary_screening_agent.get_assessment_results(session_id)
        
        if results is None:
            raise HTTPException(status_code=404, detail="Results not available")
        
        return {
            "session_id": session_id,
            "results": [result.model_dump() for result in results],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/crisis/analyze", response_model=CrisisAnalysisResponse)
async def analyze_crisis(
    request: CrisisAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze interaction for crisis indicators
    
    This endpoint uses the Crisis Detection Agent to analyze
    user interactions for potential crisis situations.
    """
    try:
        # Analyze interaction
        risk_assessment = await crisis_detection_agent.analyze_interaction(
            user_id=request.user_id,
            session_id=request.session_id,
            interaction_data=request.interaction_data,
            context=request.context
        )
        
        if risk_assessment:
            return CrisisAnalysisResponse(
                risk_assessment=risk_assessment.model_dump(),
                crisis_detected=True,
                alert_level=risk_assessment.crisis_level.value,
                recommended_actions=risk_assessment.get_required_interventions()
            )
        else:
            return CrisisAnalysisResponse(
                risk_assessment=None,
                crisis_detected=False,
                alert_level=None,
                recommended_actions=[]
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/crisis/alerts")
async def get_crisis_alerts(
    current_user: dict = Depends(get_current_user)
):
    """Get active crisis alerts"""
    try:
        alerts = await crisis_detection_agent.get_active_alerts()
        
        return {
            "alerts": [alert.model_dump() for alert in alerts],
            "count": len(alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/status")
async def get_agents_status(
    current_user: dict = Depends(get_current_user)
):
    """Get status of all agents"""
    try:
        agent_statuses = []
        
        for agent_id, agent in agent_registry.items():
            # Get agent card
            agent_card = await agent_discovery.get_agent_card(agent_id)
            
            if agent_card:
                agent_statuses.append(AgentStatusResponse(
                    agent_id=agent_id,
                    status=agent_card.availability_status,
                    capabilities=[cap.model_dump() for cap in agent_card.capabilities],
                    last_activity=agent_card.last_updated
                ))
        
        return {
            "agents": agent_statuses,
            "total_agents": len(agent_statuses),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_id}/capabilities")
async def get_agent_capabilities(
    agent_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get capabilities of a specific agent"""
    try:
        agent_card = await agent_discovery.get_agent_card(agent_id)
        
        if not agent_card:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "agent_id": agent_id,
            "capabilities": [cap.model_dump() for cap in agent_card.capabilities],
            "supported_languages": agent_card.supported_languages,
            "compliance_certifications": agent_card.compliance_certifications,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/{agent_id}/tasks")
async def create_agent_task(
    agent_id: str,
    task_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Create a task for a specific agent"""
    try:
        if agent_id not in agent_registry:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Create task
        task = await task_manager.create_task(
            title=task_data.get("title", "New Task"),
            description=task_data.get("description", ""),
            task_type=task_data.get("task_type", "general"),
            created_by=current_user["user_id"],
            assigned_to=agent_id,
            priority=TaskPriority(task_data.get("priority", "normal")),
            input_data=task_data.get("input_data", {})
        )
        
        return {
            "task_id": task.task_id,
            "status": "created",
            "message": "Task created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def get_tasks(
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get tasks with optional filtering"""
    try:
        tasks = []
        
        if agent_id:
            agent_tasks = await task_manager.get_agent_tasks(agent_id)
            tasks = [task.model_dump() for task in agent_tasks]
        else:
            # Get all tasks (simplified for demo)
            tasks = []
        
        # Filter by status if provided
        if status:
            tasks = [task for task in tasks if task.get("status") == status]
        
        return {
            "tasks": tasks,
            "count": len(tasks),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks
async def analyze_interaction_for_crisis(
    user_id: str,
    session_id: str,
    interaction_data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
):
    """Background task to analyze interaction for crisis indicators"""
    try:
        await crisis_detection_agent.analyze_interaction(
            user_id=user_id,
            session_id=session_id,
            interaction_data=interaction_data,
            context=context
        )
    except Exception as e:
        print(f"Error in background crisis analysis: {e}")


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    try:
        # Initialize agents
        await primary_screening_agent.initialize()
        await crisis_detection_agent.initialize()
        
        # Register agents with discovery service
        for agent_id, agent in agent_registry.items():
            await agent_discovery.register_agent(
                AgentCard(
                    agent_id=agent_id,
                    name=f"Agent {agent_id}",
                    description=f"Mental health agent {agent_id}",
                    version="1.0.0",
                    capabilities=[],
                    contact_endpoint=f"http://localhost:8000/agents/{agent_id}"
                )
            )
        
        print("Mental Health A2A Agent Ecosystem started successfully")
        
    except Exception as e:
        print(f"Error during startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        # Close A2A communicator
        await a2a_communicator.close()
        
        print("Mental Health A2A Agent Ecosystem shutdown complete")
        
    except Exception as e:
        print(f"Error during shutdown: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
