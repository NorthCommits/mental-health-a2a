"""
Therapeutic Intervention Agent

Provides ongoing therapy sessions and interventions using evidence-based techniques
including CBT, mindfulness, and coping strategies. Follows A2A protocol for
seamless collaboration with other mental health agents.
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field

from shared.a2a_protocol.communication_layer import A2ACommunicator, A2AMessage, MessageType, ContentType
from shared.a2a_protocol.agent_discovery import AgentDiscovery, AgentCard, CapabilityType, InputModality, OutputFormat
from shared.a2a_protocol.task_management import TaskManager, Task, TaskPriority, TaskStatus
from shared.a2a_protocol.security import A2ASecurity, AgentRole, AccessLevel

from utils.therapeutic_modules import TherapeuticModules
from utils.file_storage import FileStorageManager


class TherapySessionType(str, Enum):
    """Types of therapy sessions"""
    INDIVIDUAL = "individual"
    GROUP = "group"
    CRISIS_INTERVENTION = "crisis_intervention"
    FOLLOW_UP = "follow_up"
    ASSESSMENT = "assessment"


class InterventionType(str, Enum):
    """Types of therapeutic interventions"""
    CBT_TECHNIQUE = "cbt_technique"
    MINDFULNESS = "mindfulness"
    COPING_STRATEGY = "coping_strategy"
    BREATHING_EXERCISE = "breathing_exercise"
    PROGRESSIVE_RELAXATION = "progressive_relaxation"
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"
    BEHAVIORAL_ACTIVATION = "behavioral_activation"
    EXPOSURE_THERAPY = "exposure_therapy"


class TherapySession(BaseModel):
    """Therapy session data structure"""
    session_id: str
    user_id: str
    session_type: TherapySessionType
    intervention_type: InterventionType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    therapist_notes: str = ""
    client_feedback: str = ""
    mood_before: Optional[int] = None  # 1-10 scale
    mood_after: Optional[int] = None   # 1-10 scale
    intervention_content: Dict[str, Any] = Field(default_factory=dict)
    homework_assigned: List[str] = Field(default_factory=list)
    next_session_scheduled: Optional[datetime] = None
    effectiveness_rating: Optional[float] = None  # 1-5 scale
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TherapeuticInterventionAgent:
    """
    Therapeutic Intervention Agent for ongoing therapy sessions
    
    This agent provides evidence-based therapeutic interventions including
    CBT techniques, mindfulness exercises, and coping strategies.
    """
    
    def __init__(
        self,
        agent_id: str,
        a2a_communicator: A2ACommunicator,
        agent_discovery: AgentDiscovery,
        task_manager: TaskManager,
        security: A2ASecurity,
        storage_manager: FileStorageManager
    ):
        self.agent_id = agent_id
        self.a2a_communicator = a2a_communicator
        self.agent_discovery = agent_discovery
        self.task_manager = task_manager
        self.security = security
        self.storage_manager = storage_manager
        
        # Initialize therapeutic modules
        self.therapeutic_modules = TherapeuticModules()
        
        # Agent state
        self.active_sessions: Dict[str, TherapySession] = {}
        self.session_history: Dict[str, List[TherapySession]] = {}
        
        # Register message handlers and capabilities
        # Note: These will be called when the agent is started
        self._message_handlers_registered = False
        self._capabilities_registered = False
    
    async def initialize(self):
        """Initialize the agent by registering handlers and capabilities"""
        if not self._message_handlers_registered:
            await self._register_message_handlers()
            self._message_handlers_registered = True
        
        if not self._capabilities_registered:
            await self._register_agent_capabilities()
            self._capabilities_registered = True
    
    async def _register_message_handlers(self):
        """Register message handlers for A2A communication"""
        await self.a2a_communicator.register_message_handler(
            MessageType.TASK_REQUEST,
            self._handle_task_request
        )
        await self.a2a_communicator.register_message_handler(
            MessageType.COLLABORATION,
            self._handle_collaboration
        )
        await self.a2a_communicator.register_message_handler(
            MessageType.CRISIS_ALERT,
            self._handle_crisis_alert
        )
    
    async def _register_agent_capabilities(self):
        """Register agent capabilities with the discovery service"""
        agent_card = AgentCard(
            agent_id=self.agent_id,
            name="Therapeutic Intervention Agent",
            description="Provides ongoing therapy sessions and evidence-based interventions",
            version="1.0.0",
            capabilities=[
                {
                    "capability_type": CapabilityType.THERAPEUTIC_INTERVENTION,
                    "description": "Evidence-based therapeutic interventions and therapy sessions",
                    "input_modalities": [
                        InputModality.TEXT,
                        InputModality.AUDIO,
                        InputModality.IMAGE
                    ],
                    "output_formats": [
                        OutputFormat.TEXT_RESPONSE,
                        OutputFormat.STRUCTURED_DATA,
                        OutputFormat.TREATMENT_PLAN
                    ],
                    "parameters": {
                        "supported_interventions": [
                            "cbt_techniques", "mindfulness", "coping_strategies",
                            "breathing_exercises", "cognitive_restructuring"
                        ],
                        "response_time_sla": 3.0,
                        "privacy_level": "maximum",
                        "therapy_modalities": ["individual", "group", "crisis"]
                    }
                }
            ],
            contact_endpoint=f"http://localhost:8000/agents/{self.agent_id}",
            compliance_certifications=["hipaa", "therapy_standards", "evidence_based_practice"]
        )
        
        await self.agent_discovery.register_agent(agent_card)
    
    async def start_therapy_session(
        self,
        user_id: str,
        session_type: TherapySessionType,
        intervention_type: InterventionType,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new therapy session
        
        Args:
            user_id: ID of the user
            session_type: Type of therapy session
            intervention_type: Type of intervention to use
            context: Additional context for the session
            
        Returns:
            Session ID for tracking the therapy session
        """
        session_id = str(uuid.uuid4())
        
        # Create therapy session
        session = TherapySession(
            session_id=session_id,
            user_id=user_id,
            session_type=session_type,
            intervention_type=intervention_type,
            start_time=datetime.utcnow(),
            intervention_content=await self._prepare_intervention_content(intervention_type, context)
        )
        
        # Store session
        self.active_sessions[session_id] = session
        
        # Add to user's session history
        if user_id not in self.session_history:
            self.session_history[user_id] = []
        self.session_history[user_id].append(session)
        
        # Save to file storage
        await self.storage_manager.save_therapy_session(session_id, session.model_dump())
        
        return session_id
    
    async def _prepare_intervention_content(
        self,
        intervention_type: InterventionType,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare intervention content based on type"""
        try:
            if intervention_type == InterventionType.CBT_TECHNIQUE:
                return await self._get_cbt_technique(context)
            elif intervention_type == InterventionType.MINDFULNESS:
                return await self._get_mindfulness_exercise(context)
            elif intervention_type == InterventionType.COPING_STRATEGY:
                return await self._get_coping_strategy(context)
            elif intervention_type == InterventionType.BREATHING_EXERCISE:
                return await self._get_breathing_exercise(context)
            elif intervention_type == InterventionType.PROGRESSIVE_RELAXATION:
                return await self._get_progressive_relaxation(context)
            elif intervention_type == InterventionType.COGNITIVE_RESTRUCTURING:
                return await self._get_cognitive_restructuring(context)
            elif intervention_type == InterventionType.BEHAVIORAL_ACTIVATION:
                return await self._get_behavioral_activation(context)
            elif intervention_type == InterventionType.EXPOSURE_THERAPY:
                return await self._get_exposure_therapy(context)
            else:
                return {"error": "Unknown intervention type"}
        except Exception as e:
            print(f"Error preparing intervention content: {e}")
            return {"error": str(e)}
    
    async def _get_cbt_technique(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get CBT technique based on context"""
        # Get CBT module
        cbt_module = self.therapeutic_modules.get_module("cbt_techniques")
        if not cbt_module:
            return {"error": "CBT module not found"}
        
        # Select appropriate technique based on context
        technique_id = "thought_challenging"  # default
        if context and "anxiety_level" in context:
            if context["anxiety_level"] > 7:
                technique_id = "thought_challenging"
            elif context["anxiety_level"] > 4:
                technique_id = "behavioral_experiments"
            else:
                technique_id = "cognitive_restructuring"
        
        technique = self.therapeutic_modules.get_technique("cbt_techniques", technique_id)
        if not technique:
            # Fallback to random technique
            technique = self.therapeutic_modules.get_random_technique("cbt_techniques")
        
        if not technique:
            return {"error": "No CBT techniques available"}
        
        return {
            "technique_name": technique.get("name", "CBT Technique"),
            "description": technique.get("description", ""),
            "steps": technique.get("steps", []),
            "homework": technique.get("homework", []),
            "duration_minutes": technique.get("duration_minutes", 30)
        }
    
    async def _get_mindfulness_exercise(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get mindfulness exercise based on context"""
        # Get mindfulness module
        mindfulness_module = self.therapeutic_modules.get_module("mindfulness")
        if not mindfulness_module:
            return {"error": "Mindfulness module not found"}
        
        # Select appropriate exercise
        exercise_id = "breathing_meditation"  # default
        if context and "stress_level" in context:
            if context["stress_level"] > 7:
                exercise_id = "body_scan"
            elif context["stress_level"] > 4:
                exercise_id = "breathing_meditation"
            else:
                exercise_id = "loving_kindness"
        
        exercise = self.therapeutic_modules.get_technique("mindfulness", exercise_id)
        if not exercise:
            # Fallback to random exercise
            exercise = self.therapeutic_modules.get_random_technique("mindfulness")
        
        if not exercise:
            return {"error": "No mindfulness exercises available"}
        
        return {
            "exercise_name": exercise.get("name", "Mindfulness Exercise"),
            "description": exercise.get("description", ""),
            "instructions": exercise.get("instructions", []),
            "duration_minutes": exercise.get("duration_minutes", 10),
            "audio_guide": exercise.get("audio_guide")
        }
    
    async def _get_coping_strategy(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get coping strategy based on context"""
        # Get coping strategies module
        coping_module = self.therapeutic_modules.get_module("coping_strategies")
        if not coping_module:
            return {"error": "Coping strategies module not found"}
        
        # Select appropriate strategy
        strategy_id = "general_coping"  # default
        if context and "emotion" in context:
            emotion = context["emotion"].lower()
            if emotion in ["anxiety", "panic"]:
                strategy_id = "grounding_techniques"
            elif emotion in ["sadness", "depression"]:
                strategy_id = "behavioral_activation"
            elif emotion in ["anger", "frustration"]:
                strategy_id = "anger_management"
        
        strategy = self.therapeutic_modules.get_technique("coping_strategies", strategy_id)
        if not strategy:
            # Fallback to random strategy
            strategy = self.therapeutic_modules.get_random_technique("coping_strategies")
        
        if not strategy:
            return {"error": "No coping strategies available"}
        
        return {
            "strategy_name": strategy.get("name", "Coping Strategy"),
            "description": strategy.get("description", ""),
            "techniques": strategy.get("techniques", []),
            "when_to_use": strategy.get("when_to_use", ""),
            "effectiveness_tips": strategy.get("effectiveness_tips", [])
        }
    
    async def _get_breathing_exercise(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get breathing exercise"""
        # Get breathing exercises module
        breathing_module = self.therapeutic_modules.get_module("breathing_exercises")
        if not breathing_module:
            return {"error": "Breathing exercises module not found"}
        
        # Select appropriate exercise
        exercise_id = "deep_breathing"  # default
        if context and "anxiety_level" in context:
            if context["anxiety_level"] > 7:
                exercise_id = "4_7_8_breathing"
            elif context["anxiety_level"] > 4:
                exercise_id = "box_breathing"
        
        exercise = self.therapeutic_modules.get_technique("breathing_exercises", exercise_id)
        if not exercise:
            # Fallback to random exercise
            exercise = self.therapeutic_modules.get_random_technique("breathing_exercises")
        
        if not exercise:
            return {"error": "No breathing exercises available"}
        
        return {
            "exercise_name": exercise.get("name", "Breathing Exercise"),
            "description": exercise.get("description", ""),
            "instructions": exercise.get("instructions", []),
            "duration_minutes": exercise.get("duration_minutes", 5),
            "breathing_pattern": exercise.get("breathing_pattern", "")
        }
    
    async def _get_progressive_relaxation(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get progressive relaxation exercise"""
        # Get progressive relaxation module
        relaxation_module = self.therapeutic_modules.get_module("progressive_relaxation")
        if not relaxation_module:
            return {"error": "Progressive relaxation module not found"}
        
        exercise = self.therapeutic_modules.get_random_technique("progressive_relaxation")
        if not exercise:
            return {"error": "No progressive relaxation exercises available"}
        
        return {
            "exercise_name": exercise.get("name", "Progressive Relaxation"),
            "description": exercise.get("description", ""),
            "instructions": exercise.get("instructions", []),
            "duration_minutes": exercise.get("duration_minutes", 15),
            "body_parts": exercise.get("body_parts", [])
        }
    
    async def _get_cognitive_restructuring(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get cognitive restructuring exercise"""
        # Get cognitive restructuring module
        cr_module = self.therapeutic_modules.get_module("cognitive_restructuring")
        if not cr_module:
            return {"error": "Cognitive restructuring module not found"}
        
        technique = self.therapeutic_modules.get_random_technique("cognitive_restructuring")
        if not technique:
            return {"error": "No cognitive restructuring techniques available"}
        
        return {
            "technique_name": technique.get("name", "Cognitive Restructuring"),
            "description": technique.get("description", ""),
            "steps": technique.get("steps", []),
            "thought_record_template": technique.get("thought_record_template", {}),
            "duration_minutes": technique.get("duration_minutes", 20)
        }
    
    async def _get_behavioral_activation(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get behavioral activation exercise"""
        # Get behavioral activation module
        ba_module = self.therapeutic_modules.get_module("behavioral_activation")
        if not ba_module:
            return {"error": "Behavioral activation module not found"}
        
        technique = self.therapeutic_modules.get_random_technique("behavioral_activation")
        if not technique:
            return {"error": "No behavioral activation techniques available"}
        
        return {
            "technique_name": technique.get("name", "Behavioral Activation"),
            "description": technique.get("description", ""),
            "activity_planning": technique.get("activity_planning", {}),
            "pleasure_activities": technique.get("pleasure_activities", []),
            "mastery_activities": technique.get("mastery_activities", [])
        }
    
    async def _get_exposure_therapy(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get exposure therapy exercise"""
        # Get exposure therapy module
        et_module = self.therapeutic_modules.get_module("exposure_therapy")
        if not et_module:
            return {"error": "Exposure therapy module not found"}
        
        technique = self.therapeutic_modules.get_random_technique("exposure_therapy")
        if not technique:
            return {"error": "No exposure therapy techniques available"}
        
        return {
            "technique_name": technique.get("name", "Exposure Therapy"),
            "description": technique.get("description", ""),
            "exposure_hierarchy": technique.get("exposure_hierarchy", []),
            "safety_behaviors": technique.get("safety_behaviors", []),
            "duration_minutes": technique.get("duration_minutes", 30)
        }
    
    async def conduct_intervention(
        self,
        session_id: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Conduct the therapeutic intervention
        
        Args:
            session_id: ID of the therapy session
            user_input: User's input during the session
            context: Additional context
            
        Returns:
            Intervention response and guidance
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return {"error": "Session not found"}
            
            # Process user input and provide therapeutic response
            response = await self._process_therapeutic_input(
                session, user_input, context
            )
            
            # Update session with user input
            session.therapist_notes += f"\nUser input: {user_input}\nTherapist response: {response['therapist_response']}\n"
            
            # Save updated session
            await self.storage_manager.save_therapy_session(session_id, session.model_dump())
            
            return response
            
        except Exception as e:
            print(f"Error conducting intervention: {e}")
            return {"error": str(e)}
    
    async def _process_therapeutic_input(
        self,
        session: TherapySession,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process user input and generate therapeutic response"""
        
        # Analyze user input for emotional content
        emotional_analysis = await self._analyze_emotional_content(user_input)
        
        # Generate therapeutic response based on intervention type
        if session.intervention_type == InterventionType.CBT_TECHNIQUE:
            response = await self._generate_cbt_response(session, user_input, emotional_analysis)
        elif session.intervention_type == InterventionType.MINDFULNESS:
            response = await self._generate_mindfulness_response(session, user_input, emotional_analysis)
        elif session.intervention_type == InterventionType.COPING_STRATEGY:
            response = await self._generate_coping_response(session, user_input, emotional_analysis)
        else:
            response = await self._generate_general_therapeutic_response(session, user_input, emotional_analysis)
        
        return response
    
    async def _analyze_emotional_content(self, user_input: str) -> Dict[str, Any]:
        """Analyze emotional content of user input"""
        # This would use NLP models to analyze emotional content
        # For now, return basic analysis
        return {
            "emotion": "neutral",
            "intensity": 5,
            "sentiment": "neutral",
            "key_themes": ["general"]
        }
    
    async def _generate_cbt_response(
        self,
        session: TherapySession,
        user_input: str,
        emotional_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate CBT-based therapeutic response"""
        technique = session.intervention_content
        
        return {
            "therapist_response": f"Let's work with the {technique['technique_name']} technique. I notice you mentioned '{user_input}'. Let's examine this thought together using the steps we discussed.",
            "intervention_guidance": technique["steps"],
            "homework_assignment": technique["homework"],
            "next_steps": "Continue practicing this technique and we'll review your progress next session.",
            "technique_name": technique["technique_name"]
        }
    
    async def _generate_mindfulness_response(
        self,
        session: TherapySession,
        user_input: str,
        emotional_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate mindfulness-based therapeutic response"""
        exercise = session.intervention_content
        
        return {
            "therapist_response": f"Let's practice the {exercise['exercise_name']} together. I hear that you're experiencing some challenges. Let's use this mindfulness exercise to help you stay present and grounded.",
            "exercise_instructions": exercise["instructions"],
            "duration_minutes": exercise["duration_minutes"],
            "next_steps": "Practice this exercise daily and notice how it affects your mood and stress levels.",
            "exercise_name": exercise["exercise_name"]
        }
    
    async def _generate_coping_response(
        self,
        session: TherapySession,
        user_input: str,
        emotional_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate coping strategy response"""
        strategy = session.intervention_content
        
        return {
            "therapist_response": f"Let's use the {strategy['strategy_name']} to help you cope with what you're experiencing. I can see this is challenging for you, and these techniques can help.",
            "coping_techniques": strategy["techniques"],
            "when_to_use": strategy["when_to_use"],
            "effectiveness_tips": strategy["effectiveness_tips"],
            "next_steps": "Practice these techniques when you feel overwhelmed and track which ones work best for you.",
            "strategy_name": strategy["strategy_name"]
        }
    
    async def _generate_general_therapeutic_response(
        self,
        session: TherapySession,
        user_input: str,
        emotional_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate general therapeutic response"""
        return {
            "therapist_response": f"I hear what you're saying about '{user_input}'. That sounds really challenging. Let's work through this together using the techniques we've been practicing.",
            "intervention_guidance": session.intervention_content.get("description", ""),
            "next_steps": "Continue practicing the techniques we've discussed and let me know how you're feeling.",
            "technique_name": session.intervention_type.value
        }
    
    async def end_therapy_session(
        self,
        session_id: str,
        mood_after: int,
        client_feedback: str,
        effectiveness_rating: float
    ) -> bool:
        """
        End a therapy session
        
        Args:
            session_id: ID of the therapy session
            mood_after: User's mood after session (1-10 scale)
            client_feedback: Client's feedback about the session
            effectiveness_rating: Effectiveness rating (1-5 scale)
            
        Returns:
            bool: True if session ended successfully
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return False
            
            # Update session with end data
            session.end_time = datetime.utcnow()
            session.duration_minutes = int((session.end_time - session.start_time).total_seconds() / 60)
            session.mood_after = mood_after
            session.client_feedback = client_feedback
            session.effectiveness_rating = effectiveness_rating
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            # Save final session data
            await self.storage_manager.save_therapy_session(session_id, session.model_dump())
            
            # Schedule next session if needed
            if session.next_session_scheduled:
                await self._schedule_next_session(session)
            
            return True
            
        except Exception as e:
            print(f"Error ending therapy session: {e}")
            return False
    
    async def _schedule_next_session(self, session: TherapySession):
        """Schedule next therapy session"""
        # This would integrate with Care Coordination Agent
        # For now, just log the scheduling
        print(f"Next session scheduled for {session.next_session_scheduled}")
    
    async def get_session_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get therapy session history for a user"""
        if user_id not in self.session_history:
            return []
        
        return [session.model_dump() for session in self.session_history[user_id]]
    
    async def get_active_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get active therapy session"""
        session = self.active_sessions.get(session_id)
        return session.model_dump() if session else None
    
    async def _handle_task_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle incoming task requests"""
        try:
            task_data = message.parts[0].content if message.parts else ""
            
            # Create task
            task = await self.task_manager.create_task(
                title=f"Therapy task from {message.sender_agent_id}",
                description=task_data,
                task_type="therapeutic_intervention",
                created_by=message.sender_agent_id,
                assigned_to=self.agent_id,
                priority=TaskPriority.NORMAL,
                input_data={"message": message.model_dump()}
            )
            
            return {
                "task_id": task.task_id,
                "status": "accepted",
                "message": "Therapy task accepted and queued"
            }
            
        except Exception as e:
            print(f"Error handling task request: {e}")
            return {"error": str(e)}
    
    async def _handle_collaboration(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle collaboration messages"""
        try:
            collaboration_data = json.loads(message.parts[0].content) if message.parts else {}
            
            # Handle collaboration request
            return {
                "status": "collaboration_acknowledged",
                "message": "Therapy collaboration request received"
            }
            
        except Exception as e:
            print(f"Error handling collaboration: {e}")
            return {"error": str(e)}
    
    async def _handle_crisis_alert(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle crisis alerts"""
        try:
            crisis_data = json.loads(message.parts[0].content) if message.parts else {}
            
            # Handle crisis situation
            return {
                "status": "crisis_acknowledged",
                "message": "Crisis alert received, providing therapeutic support"
            }
            
        except Exception as e:
            print(f"Error handling crisis alert: {e}")
            return {"error": str(e)}
