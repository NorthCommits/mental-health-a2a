"""
Resource Recommendation Agent
Provides personalized resource recommendations based on conversation analysis
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from shared.a2a_protocol.communication_layer import A2AMessage, MessageType, ContentType, A2ACommunicator
from shared.a2a_protocol.agent_discovery import AgentDiscovery, AgentCard, CapabilityType, InputModality, OutputFormat
from shared.a2a_protocol.task_management import TaskManager
from shared.a2a_protocol.security import A2ASecurity
from utils.file_storage import FileStorageManager
from utils.logger import logger

class ResourceType(str, Enum):
    """Types of resources that can be recommended"""
    CRISIS_HOTLINE = "crisis_hotline"
    THERAPIST = "therapist"
    PSYCHIATRIST = "psychiatrist"
    SUPPORT_GROUP = "support_group"
    SELF_HELP = "self_help"
    MOBILE_APP = "mobile_app"
    WEBSITE = "website"
    BOOK = "book"
    WORKSHOP = "workshop"
    EMERGENCY_SERVICE = "emergency_service"

class UrgencyLevel(str, Enum):
    """Urgency levels for resource recommendations"""
    IMMEDIATE = "immediate"  # Crisis situations
    URGENT = "urgent"        # Within 24 hours
    IMPORTANT = "important"  # Within a week
    ROUTINE = "routine"      # General recommendations

@dataclass
class Resource:
    """Represents a recommended resource"""
    resource_id: str
    name: str
    resource_type: ResourceType
    description: str
    contact_info: str
    availability: str
    cost: str
    urgency_level: UrgencyLevel
    target_conditions: List[str]
    accessibility_features: List[str]
    location: Optional[str] = None
    website: Optional[str] = None
    rating: Optional[float] = None

@dataclass
class ResourceRecommendation:
    """Complete resource recommendation with context"""
    user_id: str
    conversation_id: str
    recommended_resources: List[Resource]
    reasoning: str
    urgency_level: UrgencyLevel
    follow_up_plan: List[str]
    generated_at: datetime

class ResourceRecommendationAgent:
    """Provides personalized resource recommendations based on conversation analysis"""
    
    def __init__(
        self,
        agent_id: str = "resource-recommendation-001",
        a2a_communicator: A2ACommunicator = None,
        agent_discovery: AgentDiscovery = None,
        task_manager: TaskManager = None,
        security: A2ASecurity = None,
        storage_manager: FileStorageManager = None
    ):
        self.agent_id = agent_id
        self.a2a_communicator = a2a_communicator
        self.agent_discovery = agent_discovery
        self.task_manager = task_manager
        self.security = security
        self.storage_manager = storage_manager
        
        # Agent state
        self.resource_database: Dict[str, Resource] = {}
        self.recommendation_history: Dict[str, List[ResourceRecommendation]] = {}
        
        # Flags for async initialization
        self._message_handlers_registered = False
        self._capabilities_registered = False
        
        # Initialize resource database
        self._initialize_resource_database()
    
    async def initialize(self):
        """Initialize the agent asynchronously"""
        if not self._message_handlers_registered:
            await self._register_message_handlers()
            self._message_handlers_registered = True
        
        if not self._capabilities_registered:
            await self._register_agent_capabilities()
            self._capabilities_registered = True
        
        logger.log_system_event("resource_recommendation_agent_initialized", {
            "agent_id": self.agent_id,
            "capabilities_registered": self._capabilities_registered
        })
    
    def _initialize_resource_database(self):
        """Initialize the resource database with common mental health resources"""
        
        # Crisis Resources
        self.resource_database["crisis_national"] = Resource(
            resource_id="crisis_national",
            name="National Suicide Prevention Lifeline",
            resource_type=ResourceType.CRISIS_HOTLINE,
            description="24/7 crisis support and suicide prevention",
            contact_info="988",
            availability="24/7",
            cost="Free",
            urgency_level=UrgencyLevel.IMMEDIATE,
            target_conditions=["suicidal_ideation", "crisis", "emergency"],
            accessibility_features=["phone", "text", "chat"],
            website="https://suicidepreventionlifeline.org"
        )
        
        self.resource_database["crisis_text"] = Resource(
            resource_id="crisis_text",
            name="Crisis Text Line",
            resource_type=ResourceType.CRISIS_HOTLINE,
            description="24/7 crisis support via text message",
            contact_info="Text HOME to 741741",
            availability="24/7",
            cost="Free",
            urgency_level=UrgencyLevel.IMMEDIATE,
            target_conditions=["crisis", "text_preference", "anxiety"],
            accessibility_features=["text", "deaf_friendly"],
            website="https://www.crisistextline.org"
        )
        
        # Mental Health Organizations
        self.resource_database["nami"] = Resource(
            resource_id="nami",
            name="National Alliance on Mental Illness (NAMI)",
            resource_type=ResourceType.WEBSITE,
            description="Comprehensive mental health information and support",
            contact_info="1-800-950-NAMI (6264)",
            availability="Monday-Friday 10am-6pm ET",
            cost="Free",
            urgency_level=UrgencyLevel.ROUTINE,
            target_conditions=["general_mental_health", "family_support", "education"],
            accessibility_features=["website", "phone", "support_groups"],
            website="https://www.nami.org"
        )
        
        self.resource_database["mha"] = Resource(
            resource_id="mha",
            name="Mental Health America",
            resource_type=ResourceType.WEBSITE,
            description="Mental health screening tools and resources",
            contact_info="1-800-969-6642",
            availability="24/7 online",
            cost="Free",
            urgency_level=UrgencyLevel.ROUTINE,
            target_conditions=["screening", "self_assessment", "general_mental_health"],
            accessibility_features=["website", "screening_tools", "mobile_app"],
            website="https://www.mhanational.org"
        )
        
        # ADHD Resources
        self.resource_database["chadd"] = Resource(
            resource_id="chadd",
            name="CHADD (Children and Adults with ADHD)",
            resource_type=ResourceType.WEBSITE,
            description="ADHD information, support groups, and resources",
            contact_info="1-800-233-4050",
            availability="Monday-Friday 9am-5pm ET",
            cost="Free",
            urgency_level=UrgencyLevel.ROUTINE,
            target_conditions=["adhd", "attention_deficit", "neurodivergent"],
            accessibility_features=["website", "support_groups", "webinars"],
            website="https://chadd.org"
        )
        
        # Autism Resources
        self.resource_database["autism_society"] = Resource(
            resource_id="autism_society",
            name="Autism Society of America",
            resource_type=ResourceType.WEBSITE,
            description="Autism support, advocacy, and resources",
            contact_info="1-800-3-AUTISM",
            availability="Monday-Friday 9am-5pm ET",
            cost="Free",
            urgency_level=UrgencyLevel.ROUTINE,
            target_conditions=["autism", "asd", "neurodivergent"],
            accessibility_features=["website", "local_chapters", "support_groups"],
            website="https://www.autism-society.org"
        )
        
        # Self-Help Resources
        self.resource_database["headspace"] = Resource(
            resource_id="headspace",
            name="Headspace",
            resource_type=ResourceType.MOBILE_APP,
            description="Meditation and mindfulness app",
            contact_info="Mobile app",
            availability="24/7",
            cost="Free with premium options",
            urgency_level=UrgencyLevel.ROUTINE,
            target_conditions=["anxiety", "stress", "mindfulness", "meditation"],
            accessibility_features=["mobile_app", "audio", "guided_meditation"],
            website="https://www.headspace.com"
        )
        
        self.resource_database["calm"] = Resource(
            resource_id="calm",
            name="Calm",
            resource_type=ResourceType.MOBILE_APP,
            description="Sleep, meditation, and relaxation app",
            contact_info="Mobile app",
            availability="24/7",
            cost="Free with premium options",
            urgency_level=UrgencyLevel.ROUTINE,
            target_conditions=["sleep", "anxiety", "stress", "relaxation"],
            accessibility_features=["mobile_app", "audio", "sleep_stories"],
            website="https://www.calm.com"
        )
        
        # Emergency Services
        self.resource_database["emergency_911"] = Resource(
            resource_id="emergency_911",
            name="Emergency Services",
            resource_type=ResourceType.EMERGENCY_SERVICE,
            description="Immediate emergency response",
            contact_info="911",
            availability="24/7",
            cost="Free",
            urgency_level=UrgencyLevel.IMMEDIATE,
            target_conditions=["emergency", "immediate_danger", "medical_emergency"],
            accessibility_features=["phone", "emergency_response"],
            location="Local"
        )
    
    async def _register_message_handlers(self):
        """Register message handlers for resource recommendation requests"""
        if self.a2a_communicator:
            await self.a2a_communicator.register_message_handler(
                "recommend_resources",
                self._handle_recommend_resources
            )
            await self.a2a_communicator.register_message_handler(
                "crisis_resources",
                self._handle_crisis_resources
            )
            await self.a2a_communicator.register_message_handler(
                "follow_up_resources",
                self._handle_follow_up_resources
            )
    
    async def _register_agent_capabilities(self):
        """Register agent capabilities with the discovery service"""
        agent_card = AgentCard(
            agent_id=self.agent_id,
            name="Resource Recommendation Agent",
            description="Provides personalized resource recommendations based on conversation analysis",
            version="1.0.0",
            capabilities=[
                {
                    "capability_type": CapabilityType.CARE_COORDINATION,
                    "description": "Resource recommendation and care coordination",
                    "input_modalities": [
                        InputModality.TEXT,
                        InputModality.STRUCTURED_DATA
                    ],
                    "output_formats": [
                        OutputFormat.STRUCTURED_DATA,
                        OutputFormat.TEXT_RESPONSE
                    ],
                    "parameters": {
                        "resource_types": ["crisis_hotline", "therapist", "support_group", "self_help", "mobile_app"],
                        "urgency_levels": ["immediate", "urgent", "important", "routine"],
                        "accessibility_features": ["phone", "text", "website", "mobile_app", "deaf_friendly"]
                    }
                }
            ],
            supported_languages=["en"],
            availability_status="available",
            contact_endpoint="http://localhost:8000/resource-recommendation",
            compliance_certifications=["hipaa"]
        )
        
        await self.agent_discovery.register_agent(agent_card)
        logger.log_system_event("resource_recommendation_agent_registered", {
            "agent_id": self.agent_id,
            "capabilities": len(agent_card.capabilities)
        })
    
    async def _handle_recommend_resources(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle general resource recommendation requests"""
        try:
            content = message.content
            user_id = content.get("user_id", "anonymous")
            conversation_id = content.get("conversation_id", "unknown")
            conversation_analysis = content.get("conversation_analysis", {})
            identified_conditions = content.get("identified_conditions", [])
            urgency_level = content.get("urgency_level", "routine")
            
            # Generate resource recommendations
            recommendation = await self.recommend_resources(
                user_id=user_id,
                conversation_id=conversation_id,
                conversation_analysis=conversation_analysis,
                identified_conditions=identified_conditions,
                urgency_level=urgency_level
            )
            
            return {
                "status": "success",
                "recommendation": self._recommendation_to_dict(recommendation),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.log_system_event("resource_recommendation_error", {"error": str(e)})
            return {"status": "error", "error": str(e)}
    
    async def _handle_crisis_resources(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle crisis resource recommendation requests"""
        try:
            content = message.content
            user_id = content.get("user_id", "anonymous")
            conversation_id = content.get("conversation_id", "unknown")
            crisis_level = content.get("crisis_level", "high")
            
            # Generate crisis resource recommendations
            recommendation = await self.recommend_crisis_resources(
                user_id=user_id,
                conversation_id=conversation_id,
                crisis_level=crisis_level
            )
            
            return {
                "status": "success",
                "recommendation": self._recommendation_to_dict(recommendation),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.log_system_event("crisis_resource_error", {"error": str(e)})
            return {"status": "error", "error": str(e)}
    
    async def _handle_follow_up_resources(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle follow-up resource recommendation requests"""
        try:
            content = message.content
            user_id = content.get("user_id", "anonymous")
            conversation_id = content.get("conversation_id", "unknown")
            previous_recommendations = content.get("previous_recommendations", [])
            
            # Generate follow-up resource recommendations
            recommendation = await self.recommend_follow_up_resources(
                user_id=user_id,
                conversation_id=conversation_id,
                previous_recommendations=previous_recommendations
            )
            
            return {
                "status": "success",
                "recommendation": self._recommendation_to_dict(recommendation),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.log_system_event("follow_up_resource_error", {"error": str(e)})
            return {"status": "error", "error": str(e)}
    
    async def recommend_resources(
        self,
        user_id: str,
        conversation_id: str,
        conversation_analysis: Dict[str, Any],
        identified_conditions: List[str],
        urgency_level: str = "routine"
    ) -> ResourceRecommendation:
        """Generate personalized resource recommendations"""
        
        # Determine urgency level
        urgency = UrgencyLevel(urgency_level)
        
        # Filter resources based on identified conditions
        relevant_resources = self._filter_resources_by_conditions(identified_conditions)
        
        # Filter by urgency level
        relevant_resources = [r for r in relevant_resources if r.urgency_level == urgency or urgency == UrgencyLevel.ROUTINE]
        
        # Sort by relevance and rating
        relevant_resources.sort(key=lambda x: (x.rating or 0), reverse=True)
        
        # Limit to top 5 recommendations
        recommended_resources = relevant_resources[:5]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(identified_conditions, recommended_resources)
        
        # Generate follow-up plan
        follow_up_plan = self._generate_follow_up_plan(recommended_resources, urgency)
        
        # Create recommendation
        recommendation = ResourceRecommendation(
            user_id=user_id,
            conversation_id=conversation_id,
            recommended_resources=recommended_resources,
            reasoning=reasoning,
            urgency_level=urgency,
            follow_up_plan=follow_up_plan,
            generated_at=datetime.utcnow()
        )
        
        # Store recommendation
        if user_id not in self.recommendation_history:
            self.recommendation_history[user_id] = []
        self.recommendation_history[user_id].append(recommendation)
        
        # Log recommendation
        logger.log_system_event("resource_recommendation_generated", {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "resource_count": len(recommended_resources),
            "urgency_level": urgency.value
        })
        
        return recommendation
    
    async def recommend_crisis_resources(
        self,
        user_id: str,
        conversation_id: str,
        crisis_level: str = "high"
    ) -> ResourceRecommendation:
        """Generate crisis-specific resource recommendations"""
        
        # Get crisis resources
        crisis_resources = [
            r for r in self.resource_database.values()
            if r.urgency_level == UrgencyLevel.IMMEDIATE
        ]
        
        # Add emergency services if crisis level is critical
        if crisis_level == "critical":
            crisis_resources.append(self.resource_database["emergency_911"])
        
        # Generate reasoning
        reasoning = f"Based on the crisis level ({crisis_level}), I'm recommending immediate support resources."
        
        # Generate follow-up plan
        follow_up_plan = [
            "Contact crisis resources immediately",
            "Follow up with mental health professional within 24 hours",
            "Create safety plan with trusted person",
            "Monitor symptoms and seek additional help if needed"
        ]
        
        # Create recommendation
        recommendation = ResourceRecommendation(
            user_id=user_id,
            conversation_id=conversation_id,
            recommended_resources=crisis_resources,
            reasoning=reasoning,
            urgency_level=UrgencyLevel.IMMEDIATE,
            follow_up_plan=follow_up_plan,
            generated_at=datetime.utcnow()
        )
        
        # Store recommendation
        if user_id not in self.recommendation_history:
            self.recommendation_history[user_id] = []
        self.recommendation_history[user_id].append(recommendation)
        
        return recommendation
    
    async def recommend_follow_up_resources(
        self,
        user_id: str,
        conversation_id: str,
        previous_recommendations: List[Dict[str, Any]]
    ) -> ResourceRecommendation:
        """Generate follow-up resource recommendations"""
        
        # Get previously recommended resource IDs
        previous_resource_ids = [r.get("resource_id") for r in previous_recommendations]
        
        # Find complementary resources
        complementary_resources = []
        for resource in self.resource_database.values():
            if resource.resource_id not in previous_resource_ids:
                complementary_resources.append(resource)
        
        # Sort by relevance
        complementary_resources.sort(key=lambda x: (x.rating or 0), reverse=True)
        
        # Take top 3 complementary resources
        recommended_resources = complementary_resources[:3]
        
        # Generate reasoning
        reasoning = "These are additional resources that complement your previous recommendations."
        
        # Generate follow-up plan
        follow_up_plan = [
            "Review previous recommendations",
            "Try new recommended resources",
            "Schedule follow-up assessment in 2 weeks",
            "Track progress and adjust as needed"
        ]
        
        # Create recommendation
        recommendation = ResourceRecommendation(
            user_id=user_id,
            conversation_id=conversation_id,
            recommended_resources=recommended_resources,
            reasoning=reasoning,
            urgency_level=UrgencyLevel.ROUTINE,
            follow_up_plan=follow_up_plan,
            generated_at=datetime.utcnow()
        )
        
        return recommendation
    
    def _filter_resources_by_conditions(self, conditions: List[str]) -> List[Resource]:
        """Filter resources based on identified conditions"""
        
        if not conditions:
            return list(self.resource_database.values())
        
        relevant_resources = []
        
        for resource in self.resource_database.values():
            # Check if resource targets any of the identified conditions
            if any(condition in resource.target_conditions for condition in conditions):
                relevant_resources.append(resource)
        
        # If no specific matches, return general mental health resources
        if not relevant_resources:
            relevant_resources = [
                r for r in self.resource_database.values()
                if "general_mental_health" in r.target_conditions
            ]
        
        return relevant_resources
    
    def _generate_reasoning(self, conditions: List[str], resources: List[Resource]) -> str:
        """Generate reasoning for resource recommendations"""
        
        if not conditions:
            return "Based on our conversation, I'm recommending these general mental health resources."
        
        condition_text = ", ".join(conditions)
        resource_types = [r.resource_type.value for r in resources]
        
        return f"Based on the conditions we discussed ({condition_text}), I'm recommending these {', '.join(resource_types)} resources that are specifically designed to help with these concerns."
    
    def _generate_follow_up_plan(self, resources: List[Resource], urgency: UrgencyLevel) -> List[str]:
        """Generate follow-up plan based on resources and urgency"""
        
        plan = []
        
        if urgency == UrgencyLevel.IMMEDIATE:
            plan.extend([
                "Contact crisis resources immediately",
                "Follow up with mental health professional within 24 hours",
                "Create safety plan with trusted person"
            ])
        elif urgency == UrgencyLevel.URGENT:
            plan.extend([
                "Contact recommended resources within 24 hours",
                "Schedule appointment with mental health professional within a week",
                "Begin using self-help resources immediately"
            ])
        else:
            plan.extend([
                "Review recommended resources",
                "Choose 2-3 resources to try first",
                "Schedule follow-up assessment in 2 weeks",
                "Track progress and adjust as needed"
            ])
        
        return plan
    
    def _recommendation_to_dict(self, recommendation: ResourceRecommendation) -> Dict[str, Any]:
        """Convert ResourceRecommendation to dictionary"""
        return {
            "user_id": recommendation.user_id,
            "conversation_id": recommendation.conversation_id,
            "recommended_resources": [self._resource_to_dict(r) for r in recommendation.recommended_resources],
            "reasoning": recommendation.reasoning,
            "urgency_level": recommendation.urgency_level.value,
            "follow_up_plan": recommendation.follow_up_plan,
            "generated_at": recommendation.generated_at.isoformat()
        }
    
    def _resource_to_dict(self, resource: Resource) -> Dict[str, Any]:
        """Convert Resource to dictionary"""
        return {
            "resource_id": resource.resource_id,
            "name": resource.name,
            "resource_type": resource.resource_type.value,
            "description": resource.description,
            "contact_info": resource.contact_info,
            "availability": resource.availability,
            "cost": resource.cost,
            "urgency_level": resource.urgency_level.value,
            "target_conditions": resource.target_conditions,
            "accessibility_features": resource.accessibility_features,
            "location": resource.location,
            "website": resource.website,
            "rating": resource.rating
        }
