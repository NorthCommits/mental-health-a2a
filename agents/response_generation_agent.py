"""
Response Generation Agent
Generates contextually appropriate responses for mental health conversations
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

class ResponseType(str, Enum):
    """Types of responses the agent can generate"""
    EMPATHETIC = "empathetic"
    THERAPEUTIC = "therapeutic"
    CRISIS_SUPPORT = "crisis_support"
    NEURODIVERGENT_FRIENDLY = "neurodivergent_friendly"
    CLARIFYING = "clarifying"
    SUPPORTIVE = "supportive"

class ResponseTone(str, Enum):
    """Tone of responses"""
    WARM = "warm"
    PROFESSIONAL = "professional"
    GENTLE = "gentle"
    ENCOURAGING = "encouraging"
    CALM = "calm"

@dataclass
class ResponseContext:
    """Context for response generation"""
    user_message: str
    conversation_history: List[Dict[str, Any]]
    user_emotional_state: str
    crisis_level: str
    neurodivergent_considerations: bool
    conversation_stage: str
    topics_discussed: List[str]

@dataclass
class GeneratedResponse:
    """Generated response with metadata"""
    response_text: str
    response_type: ResponseType
    tone: ResponseTone
    confidence: float
    therapeutic_techniques: List[str]
    follow_up_suggestions: List[str]
    timestamp: datetime

class ResponseGenerationAgent:
    """Generates contextually appropriate responses for mental health conversations"""
    
    def __init__(
        self,
        agent_id: str = "response-generation-001",
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
        self.response_templates: Dict[str, List[str]] = {}
        self.therapeutic_techniques: Dict[str, List[str]] = {}
        
        # Flags for async initialization
        self._message_handlers_registered = False
        self._capabilities_registered = False
        
        # Initialize response templates and techniques
        self._initialize_response_templates()
        self._initialize_therapeutic_techniques()
    
    async def initialize(self):
        """Initialize the agent asynchronously"""
        if not self._message_handlers_registered:
            await self._register_message_handlers()
            self._message_handlers_registered = True
        
        if not self._capabilities_registered:
            await self._register_agent_capabilities()
            self._capabilities_registered = True
        
        logger.log_system_event("response_generation_agent_initialized", {
            "agent_id": self.agent_id,
            "capabilities_registered": self._capabilities_registered
        })
    
    def _initialize_response_templates(self):
        """Initialize response templates for different situations"""
        self.response_templates = {
            "empathetic": [
                "I can hear that {emotion} is really present for you right now.",
                "It sounds like you're going through a lot with {topic}.",
                "I appreciate you sharing that with me. That must be {difficulty_level}.",
                "I can sense the {emotion} in what you're describing.",
                "Thank you for trusting me with this. I can see how {situation} affects you."
            ],
            "therapeutic": [
                "Let's explore that together. When you say {key_phrase}, what comes up for you?",
                "I'm curious about your experience with {topic}. Can you tell me more?",
                "It sounds like {situation} is important to you. What makes it feel that way?",
                "I notice you mentioned {key_phrase}. How does that show up in your daily life?",
                "What would it be like if {situation} were different?"
            ],
            "crisis_support": [
                "I'm here with you right now. You're not alone in this.",
                "It sounds like you're in a lot of pain. I want to help you through this.",
                "I can hear how difficult this is for you. Let's take this one step at a time.",
                "You've been so brave to share this with me. Let's work together on this.",
                "I'm concerned about you. Can you tell me more about what's happening?"
            ],
            "neurodivergent_friendly": [
                "I understand that {situation} might feel overwhelming. That's completely valid.",
                "It sounds like your brain processes {topic} in a unique way. That's okay.",
                "I can see how {situation} might be challenging for you. What helps you feel more comfortable?",
                "Your experience with {topic} is important. How does it feel in your body?",
                "I appreciate you explaining how {situation} works for you. That helps me understand better."
            ],
            "clarifying": [
                "I want to make sure I understand correctly. Are you saying that {key_phrase}?",
                "Can you help me understand more about {topic}?",
                "When you mention {key_phrase}, what does that look like for you?",
                "I'm not sure I'm following. Could you help me understand {situation} better?",
                "Let me make sure I have this right. You're saying that {key_phrase}?"
            ],
            "supportive": [
                "You're doing really well in sharing this with me.",
                "It takes courage to talk about {topic}. I'm proud of you.",
                "You're not alone in this. Many people experience {situation}.",
                "It's okay to feel {emotion}. That's a normal response to {situation}.",
                "You're taking important steps by talking about this."
            ]
        }
    
    def _initialize_therapeutic_techniques(self):
        """Initialize therapeutic techniques for different situations"""
        self.therapeutic_techniques = {
            "active_listening": [
                "Reflect back what you hear",
                "Ask open-ended questions",
                "Validate emotions",
                "Show genuine interest"
            ],
            "cognitive_behavioral": [
                "Identify thought patterns",
                "Challenge negative thoughts",
                "Explore evidence for/against thoughts",
                "Develop alternative perspectives"
            ],
            "mindfulness": [
                "Grounding techniques",
                "Breathing exercises",
                "Present moment awareness",
                "Body scan techniques"
            ],
            "trauma_informed": [
                "Maintain safety",
                "Avoid retraumatization",
                "Empower the person",
                "Use gentle language"
            ],
            "crisis_intervention": [
                "Assess immediate safety",
                "Provide emotional support",
                "Identify coping strategies",
                "Connect to resources"
            ]
        }
    
    async def _register_message_handlers(self):
        """Register message handlers for response generation requests"""
        if self.a2a_communicator:
            await self.a2a_communicator.register_message_handler(
                "generate_response",
                self._handle_generate_response
            )
            await self.a2a_communicator.register_message_handler(
                "generate_therapeutic_response",
                self._handle_generate_therapeutic_response
            )
            await self.a2a_communicator.register_message_handler(
                "generate_crisis_response",
                self._handle_generate_crisis_response
            )
    
    async def _register_agent_capabilities(self):
        """Register agent capabilities with the discovery service"""
        agent_card = AgentCard(
            agent_id=self.agent_id,
            name="Response Generation Agent",
            description="Generates contextually appropriate responses for mental health conversations",
            version="1.0.0",
            capabilities=[
                {
                    "capability_type": CapabilityType.THERAPEUTIC_INTERVENTION,
                    "description": "Generate empathetic and therapeutic responses",
                    "input_modalities": [
                        InputModality.TEXT,
                        InputModality.AUDIO
                    ],
                    "output_formats": [
                        OutputFormat.TEXT_RESPONSE,
                        OutputFormat.STRUCTURED_DATA
                    ],
                    "parameters": {
                        "response_types": ["empathetic", "therapeutic", "crisis_support", "neurodivergent_friendly"],
                        "therapeutic_techniques": ["active_listening", "cognitive_behavioral", "mindfulness"],
                        "tone_options": ["warm", "professional", "gentle", "encouraging", "calm"]
                    }
                }
            ],
            supported_languages=["en"],
            availability_status="available",
            contact_endpoint="http://localhost:8000/response-generation",
            compliance_certifications=["hipaa"]
        )
        
        await self.agent_discovery.register_agent(agent_card)
        logger.log_system_event("response_generation_agent_registered", {
            "agent_id": self.agent_id,
            "capabilities": len(agent_card.capabilities)
        })
    
    async def _handle_generate_response(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle general response generation requests"""
        try:
            content = message.content
            context = ResponseContext(
                user_message=content.get("user_message", ""),
                conversation_history=content.get("conversation_history", []),
                user_emotional_state=content.get("emotional_state", "neutral"),
                crisis_level=content.get("crisis_level", "low"),
                neurodivergent_considerations=content.get("neurodivergent_considerations", False),
                conversation_stage=content.get("conversation_stage", "early"),
                topics_discussed=content.get("topics_discussed", [])
            )
            
            # Generate response
            response = await self.generate_response(context)
            
            return {
                "status": "success",
                "response": self._response_to_dict(response),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.log_system_event("response_generation_error", {"error": str(e)})
            return {"status": "error", "error": str(e)}
    
    async def _handle_generate_therapeutic_response(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle therapeutic response generation requests"""
        try:
            content = message.content
            context = ResponseContext(
                user_message=content.get("user_message", ""),
                conversation_history=content.get("conversation_history", []),
                user_emotional_state=content.get("emotional_state", "neutral"),
                crisis_level=content.get("crisis_level", "low"),
                neurodivergent_considerations=content.get("neurodivergent_considerations", False),
                conversation_stage=content.get("conversation_stage", "early"),
                topics_discussed=content.get("topics_discussed", [])
            )
            
            # Generate therapeutic response
            response = await self.generate_therapeutic_response(context)
            
            return {
                "status": "success",
                "response": self._response_to_dict(response),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.log_system_event("therapeutic_response_error", {"error": str(e)})
            return {"status": "error", "error": str(e)}
    
    async def _handle_generate_crisis_response(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle crisis response generation requests"""
        try:
            content = message.content
            context = ResponseContext(
                user_message=content.get("user_message", ""),
                conversation_history=content.get("conversation_history", []),
                user_emotional_state=content.get("emotional_state", "distressed"),
                crisis_level=content.get("crisis_level", "high"),
                neurodivergent_considerations=content.get("neurodivergent_considerations", False),
                conversation_stage=content.get("conversation_stage", "crisis"),
                topics_discussed=content.get("topics_discussed", [])
            )
            
            # Generate crisis response
            response = await self.generate_crisis_response(context)
            
            return {
                "status": "success",
                "response": self._response_to_dict(response),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.log_system_event("crisis_response_error", {"error": str(e)})
            return {"status": "error", "error": str(e)}
    
    async def generate_response(self, context: ResponseContext) -> GeneratedResponse:
        """Generate a contextually appropriate response"""
        
        # Determine response type based on context
        response_type = self._determine_response_type(context)
        
        # Determine tone based on context
        tone = self._determine_tone(context)
        
        # Generate response text
        response_text = await self._generate_response_text(context, response_type, tone)
        
        # Determine therapeutic techniques
        therapeutic_techniques = self._select_therapeutic_techniques(context, response_type)
        
        # Generate follow-up suggestions
        follow_up_suggestions = self._generate_follow_up_suggestions(context, response_type)
        
        # Calculate confidence
        confidence = self._calculate_confidence(context, response_type)
        
        return GeneratedResponse(
            response_text=response_text,
            response_type=response_type,
            tone=tone,
            confidence=confidence,
            therapeutic_techniques=therapeutic_techniques,
            follow_up_suggestions=follow_up_suggestions,
            timestamp=datetime.utcnow()
        )
    
    async def generate_therapeutic_response(self, context: ResponseContext) -> GeneratedResponse:
        """Generate a specifically therapeutic response"""
        context.response_type = ResponseType.THERAPEUTIC
        return await self.generate_response(context)
    
    async def generate_crisis_response(self, context: ResponseContext) -> GeneratedResponse:
        """Generate a crisis-appropriate response"""
        context.response_type = ResponseType.CRISIS_SUPPORT
        return await self.generate_response(context)
    
    def _determine_response_type(self, context: ResponseContext) -> ResponseType:
        """Determine the most appropriate response type"""
        
        # Crisis takes priority
        if context.crisis_level in ["high", "critical"]:
            return ResponseType.CRISIS_SUPPORT
        
        # Neurodivergent considerations
        if context.neurodivergent_considerations:
            return ResponseType.NEURODIVERGENT_FRIENDLY
        
        # Emotional state considerations
        if context.user_emotional_state in ["distressed", "anxious", "sad"]:
            return ResponseType.EMPATHETIC
        
        # Conversation stage considerations
        if context.conversation_stage in ["early", "introduction"]:
            return ResponseType.CLARIFYING
        
        # Default to therapeutic
        return ResponseType.THERAPEUTIC
    
    def _determine_tone(self, context: ResponseContext) -> ResponseTone:
        """Determine the most appropriate tone"""
        
        # Crisis situations need calm tone
        if context.crisis_level in ["high", "critical"]:
            return ResponseTone.CALM
        
        # Distressed users need gentle tone
        if context.user_emotional_state in ["distressed", "anxious", "sad"]:
            return ResponseTone.GENTLE
        
        # Early conversation needs warm tone
        if context.conversation_stage in ["early", "introduction"]:
            return ResponseTone.WARM
        
        # Default to encouraging
        return ResponseTone.ENCOURAGING
    
    async def _generate_response_text(self, context: ResponseContext, response_type: ResponseType, tone: ResponseTone) -> str:
        """Generate the actual response text"""
        
        # Get appropriate templates
        templates = self.response_templates.get(response_type.value, [])
        
        if not templates:
            return "I'm here to listen and support you. Can you tell me more about what's on your mind?"
        
        # Select template based on context
        template = self._select_template(templates, context)
        
        # Fill in template variables
        response_text = self._fill_template(template, context)
        
        # Adjust tone if needed
        response_text = self._adjust_tone(response_text, tone)
        
        return response_text
    
    def _select_template(self, templates: List[str], context: ResponseContext) -> str:
        """Select the most appropriate template"""
        
        # Simple selection based on conversation stage
        if context.conversation_stage == "early":
            return templates[0] if templates else "I'm here to listen and support you."
        elif context.conversation_stage == "exploration":
            return templates[1] if len(templates) > 1 else templates[0]
        else:
            return templates[-1] if templates else "Thank you for sharing that with me."
    
    def _fill_template(self, template: str, context: ResponseContext) -> str:
        """Fill in template variables with context data"""
        
        # Extract key information from user message
        user_message = context.user_message.lower()
        
        # Simple keyword extraction
        emotions = ["anxious", "sad", "angry", "frustrated", "worried", "scared", "happy", "excited"]
        topics = ["work", "family", "relationships", "health", "mental health", "stress", "anxiety", "depression"]
        
        detected_emotion = next((emotion for emotion in emotions if emotion in user_message), "concerned")
        detected_topic = next((topic for topic in topics if topic in user_message), "what you're experiencing")
        
        # Fill template
        filled_template = template.format(
            emotion=detected_emotion,
            topic=detected_topic,
            difficulty_level="challenging",
            situation="this situation",
            key_phrase=context.user_message[:50] + "..." if len(context.user_message) > 50 else context.user_message
        )
        
        return filled_template
    
    def _adjust_tone(self, response_text: str, tone: ResponseTone) -> str:
        """Adjust response tone"""
        
        if tone == ResponseTone.WARM:
            return response_text.replace("I", "I'm here to help you, and I")
        elif tone == ResponseTone.GENTLE:
            return response_text.replace(".", " gently.")
        elif tone == ResponseTone.ENCOURAGING:
            return response_text + " You're doing great by sharing this with me."
        elif tone == ResponseTone.CALM:
            return response_text.replace("!", ".").replace("?", ".")
        
        return response_text
    
    def _select_therapeutic_techniques(self, context: ResponseContext, response_type: ResponseType) -> List[str]:
        """Select appropriate therapeutic techniques"""
        
        techniques = []
        
        # Always include active listening
        techniques.extend(self.therapeutic_techniques["active_listening"][:2])
        
        # Add techniques based on response type
        if response_type == ResponseType.CRISIS_SUPPORT:
            techniques.extend(self.therapeutic_techniques["crisis_intervention"][:2])
        elif response_type == ResponseType.THERAPEUTIC:
            techniques.extend(self.therapeutic_techniques["cognitive_behavioral"][:2])
        elif context.user_emotional_state in ["anxious", "stressed"]:
            techniques.extend(self.therapeutic_techniques["mindfulness"][:2])
        
        return techniques[:4]  # Limit to 4 techniques
    
    def _generate_follow_up_suggestions(self, context: ResponseContext, response_type: ResponseType) -> List[str]:
        """Generate follow-up suggestions"""
        
        suggestions = []
        
        if response_type == ResponseType.CRISIS_SUPPORT:
            suggestions = [
                "Would you like to talk about what's making you feel this way?",
                "Can you tell me more about what's happening right now?",
                "What would help you feel safer right now?"
            ]
        elif response_type == ResponseType.THERAPEUTIC:
            suggestions = [
                "Can you tell me more about that experience?",
                "What was that like for you?",
                "How did that make you feel?"
            ]
        elif response_type == ResponseType.EMPATHETIC:
            suggestions = [
                "I'm here to listen. What else would you like to share?",
                "That sounds really important to you. Can you tell me more?",
                "What's it like for you when that happens?"
            ]
        else:
            suggestions = [
                "Can you help me understand that better?",
                "What would you like to explore next?",
                "Is there anything else you'd like to discuss?"
            ]
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _calculate_confidence(self, context: ResponseContext, response_type: ResponseType) -> float:
        """Calculate confidence in the generated response"""
        
        base_confidence = 0.7
        
        # Increase confidence based on context clarity
        if len(context.user_message) > 20:
            base_confidence += 0.1
        
        if len(context.conversation_history) > 2:
            base_confidence += 0.1
        
        # Increase confidence for crisis responses (more standardized)
        if response_type == ResponseType.CRISIS_SUPPORT:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _response_to_dict(self, response: GeneratedResponse) -> Dict[str, Any]:
        """Convert GeneratedResponse to dictionary"""
        return {
            "response_text": response.response_text,
            "response_type": response.response_type.value,
            "tone": response.tone.value,
            "confidence": response.confidence,
            "therapeutic_techniques": response.therapeutic_techniques,
            "follow_up_suggestions": response.follow_up_suggestions,
            "timestamp": response.timestamp.isoformat()
        }
