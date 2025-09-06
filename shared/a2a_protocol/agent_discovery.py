"""
Agent Discovery and Capability Management

Implements agent discovery, capability advertisement, and service negotiation
as specified in the A2A protocol using Agent Cards.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pydantic import BaseModel, Field


class CapabilityType(str, Enum):
    """Types of capabilities that agents can provide"""
    SCREENING = "screening"
    CRISIS_DETECTION = "crisis_detection"
    THERAPEUTIC_INTERVENTION = "therapeutic_intervention"
    CARE_COORDINATION = "care_coordination"
    PROGRESS_ANALYTICS = "progress_analytics"
    DATA_PROCESSING = "data_processing"
    EMERGENCY_RESPONSE = "emergency_response"
    ASSESSMENT = "assessment"


class InputModality(str, Enum):
    """Supported input modalities"""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    DOCUMENT = "document"
    SENSOR_DATA = "sensor_data"
    VIDEO = "video"


class OutputFormat(str, Enum):
    """Supported output formats"""
    TEXT_RESPONSE = "text_response"
    STRUCTURED_DATA = "structured_data"
    ASSESSMENT_SCORE = "assessment_score"
    CRISIS_ALERT = "crisis_alert"
    TREATMENT_PLAN = "treatment_plan"
    APPOINTMENT_SCHEDULE = "appointment_schedule"


class Capability(BaseModel):
    """Individual capability that an agent can provide"""
    capability_type: CapabilityType
    description: str
    input_modalities: List[InputModality]
    output_formats: List[OutputFormat]
    parameters: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None


class AgentCard(BaseModel):
    """
    Agent Card as specified in A2A protocol
    Contains metadata about an agent's capabilities and requirements
    """
    agent_id: str
    name: str
    description: str
    version: str
    capabilities: List[Capability]
    supported_languages: List[str] = Field(default=["en"])
    max_concurrent_tasks: int = Field(default=10)
    response_time_sla: float = Field(default=5.0)  # seconds
    availability_status: str = Field(default="available")  # available, busy, offline
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    contact_endpoint: str
    authentication_required: bool = Field(default=True)
    supported_protocols: List[str] = Field(default=["http", "websocket"])
    privacy_level: str = Field(default="high")  # low, medium, high, maximum
    compliance_certifications: List[str] = Field(default=["hipaa"])


class AgentDiscovery:
    """
    Manages agent discovery and capability negotiation
    Implements the A2A protocol for agent registration and discovery
    """
    
    def __init__(self, registry_endpoint: str = "http://localhost:8000/registry"):
        self.registry_endpoint = registry_endpoint
        self.registered_agents: Dict[str, AgentCard] = {}
        self.capability_index: Dict[CapabilityType, Set[str]] = {}
    
    async def register_agent(self, agent_card: AgentCard) -> bool:
        """
        Register an agent with the discovery service
        
        Args:
            agent_card: Agent metadata and capabilities
            
        Returns:
            bool: True if registration was successful
        """
        try:
            # Store locally
            self.registered_agents[agent_card.agent_id] = agent_card
            
            # Update capability index
            for capability in agent_card.capabilities:
                if capability.capability_type not in self.capability_index:
                    self.capability_index[capability.capability_type] = set()
                self.capability_index[capability.capability_type].add(agent_card.agent_id)
            
            # Register with central registry (in production, this would be a real service)
            # For now, we'll just store locally
            print(f"Agent {agent_card.agent_id} registered successfully")
            return True
            
        except Exception as e:
            print(f"Failed to register agent: {e}")
            return False
    
    async def discover_agents_by_capability(
        self, 
        capability_type: CapabilityType,
        required_input_modalities: Optional[List[InputModality]] = None,
        required_output_formats: Optional[List[OutputFormat]] = None
    ) -> List[AgentCard]:
        """
        Discover agents that can provide specific capabilities
        
        Args:
            capability_type: Type of capability needed
            required_input_modalities: Required input types
            required_output_formats: Required output types
            
        Returns:
            List of agents that match the requirements
        """
        matching_agents = []
        
        # Get agents with the required capability
        agent_ids = self.capability_index.get(capability_type, set())
        
        for agent_id in agent_ids:
            agent_card = self.registered_agents.get(agent_id)
            if not agent_card or agent_card.availability_status != "available":
                continue
            
            # Check if agent supports required input modalities
            if required_input_modalities:
                agent_input_modalities = set()
                for capability in agent_card.capabilities:
                    if capability.capability_type == capability_type:
                        agent_input_modalities.update(capability.input_modalities)
                
                if not set(required_input_modalities).issubset(agent_input_modalities):
                    continue
            
            # Check if agent supports required output formats
            if required_output_formats:
                agent_output_formats = set()
                for capability in agent_card.capabilities:
                    if capability.capability_type == capability_type:
                        agent_output_formats.update(capability.output_formats)
                
                if not set(required_output_formats).issubset(agent_output_formats):
                    continue
            
            matching_agents.append(agent_card)
        
        return matching_agents
    
    async def get_agent_card(self, agent_id: str) -> Optional[AgentCard]:
        """
        Get the agent card for a specific agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            AgentCard if found, None otherwise
        """
        return self.registered_agents.get(agent_id)
    
    async def update_agent_status(self, agent_id: str, status: str) -> bool:
        """
        Update the availability status of an agent
        
        Args:
            agent_id: ID of the agent
            status: New status (available, busy, offline)
            
        Returns:
            bool: True if update was successful
        """
        if agent_id in self.registered_agents:
            self.registered_agents[agent_id].availability_status = status
            self.registered_agents[agent_id].last_updated = datetime.utcnow()
            return True
        return False
    
    async def negotiate_capability(
        self,
        requesting_agent_id: str,
        capability_type: CapabilityType,
        requirements: Dict[str, Any]
    ) -> Optional[AgentCard]:
        """
        Negotiate with agents to find the best match for specific requirements
        
        Args:
            requesting_agent_id: ID of the agent making the request
            capability_type: Type of capability needed
            requirements: Specific requirements and constraints
            
        Returns:
            Best matching AgentCard, or None if no suitable agent found
        """
        # Find agents with the required capability
        candidate_agents = await self.discover_agents_by_capability(capability_type)
        
        if not candidate_agents:
            return None
        
        # Score agents based on requirements
        best_agent = None
        best_score = -1
        
        for agent in candidate_agents:
            score = await self._calculate_agent_score(agent, requirements)
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    async def _calculate_agent_score(
        self, 
        agent: AgentCard, 
        requirements: Dict[str, Any]
    ) -> float:
        """
        Calculate a compatibility score for an agent based on requirements
        
        Args:
            agent: Agent to score
            requirements: Requirements to match against
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        score = 0.0
        total_weight = 0.0
        
        # Performance metrics weight
        if "response_time" in requirements:
            weight = 0.3
            total_weight += weight
            required_time = requirements["response_time"]
            if agent.response_time_sla <= required_time:
                score += weight
            else:
                # Penalty for slower response times
                penalty = min(0.2, (agent.response_time_sla - required_time) / required_time * 0.2)
                score += weight - penalty
        
        # Privacy level weight
        if "privacy_level" in requirements:
            weight = 0.2
            total_weight += weight
            required_privacy = requirements["privacy_level"]
            privacy_levels = {"low": 1, "medium": 2, "high": 3, "maximum": 4}
            if privacy_levels.get(agent.privacy_level, 0) >= privacy_levels.get(required_privacy, 0):
                score += weight
        
        # Compliance certifications weight
        if "compliance_requirements" in requirements:
            weight = 0.2
            total_weight += weight
            required_certifications = set(requirements["compliance_requirements"])
            agent_certifications = set(agent.compliance_certifications)
            if required_certifications.issubset(agent_certifications):
                score += weight
        
        # Availability weight
        weight = 0.1
        total_weight += weight
        if agent.availability_status == "available":
            score += weight
        
        # Capability parameters weight
        if "capability_parameters" in requirements:
            weight = 0.2
            total_weight += weight
            # This would involve more complex matching logic
            # For now, we'll give a base score if the agent has the capability
            score += weight * 0.5
        
        return score / total_weight if total_weight > 0 else 0.0
