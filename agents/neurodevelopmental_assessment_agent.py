"""
Neurodevelopmental Assessment Agent

Specialized agent for conducting conversational assessments of ADHD and ASD traits
through natural, engaging chat flows that feel supportive rather than clinical.
Integrates with the existing A2A protocol and mental health ecosystem.
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

from utils.file_storage import FileStorageManager


class AssessmentType(str, Enum):
    """Types of neurodevelopmental assessments"""
    ADHD = "adhd"
    ASD = "asd"
    COMBINED = "combined"


class ConversationStage(str, Enum):
    """Stages of the conversational assessment"""
    INTRODUCTION = "introduction"
    ADHD_ATTENTION = "adhd_attention"
    ADHD_HYPERACTIVITY = "adhd_hyperactivity"
    ADHD_IMPULSIVITY = "adhd_impulsivity"
    ASD_SOCIAL_COMMUNICATION = "asd_social_communication"
    ASD_SENSORY = "asd_sensory"
    ASD_REPETITIVE_BEHAVIORS = "asd_repetitive_behaviors"
    ASD_INTERESTS = "asd_interests"
    WRAP_UP = "wrap_up"
    RESULTS = "results"


class ResponsePattern(BaseModel):
    """Pattern identified in user responses"""
    category: str
    description: str
    strength_level: int = Field(ge=1, le=5)  # 1-5 scale
    examples: List[str] = []
    implications: str = ""


class AssessmentInsight(BaseModel):
    """Insight derived from assessment responses"""
    area: str
    pattern_description: str
    potential_traits: List[str] = []
    strengths: List[str] = []
    challenges: List[str] = []
    recommendations: List[str] = []
    resources: List[str] = []


class NeurodevelopmentalAssessmentResult(BaseModel):
    """Results of neurodevelopmental assessment"""
    assessment_id: str
    user_id: str
    assessment_type: AssessmentType
    conversation_stages_completed: List[ConversationStage]
    adhd_insights: List[AssessmentInsight] = []
    asd_insights: List[AssessmentInsight] = []
    overall_patterns: List[ResponsePattern] = []
    recommendations: List[str] = []
    resources: List[str] = []
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    requires_professional_evaluation: bool = False
    next_steps: List[str] = []


class ConversationFlow(BaseModel):
    """Defines a conversation flow for assessment"""
    stage: ConversationStage
    prompt: str
    follow_up_questions: List[str] = []
    keywords_to_explore: List[str] = []
    expected_response_types: List[str] = []


class NeurodevelopmentalAssessmentAgent:
    """Agent for conducting neurodevelopmental assessments through natural conversation"""
    
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
        
        # Agent state
        self.active_assessments: Dict[str, Dict[str, Any]] = {}
        self.conversation_flows: Dict[AssessmentType, List[ConversationFlow]] = {}
        
        # Flags for async initialization
        self._message_handlers_registered = False
        self._capabilities_registered = False
        
        # Initialize conversation flows
        self._initialize_conversation_flows()
    
    async def initialize(self):
        """Initialize the agent asynchronously"""
        if not self._message_handlers_registered:
            await self._register_message_handlers()
            self._message_handlers_registered = True
        
        if not self._capabilities_registered:
            await self._register_agent_capabilities()
            self._capabilities_registered = True
    
    def _initialize_conversation_flows(self):
        """Initialize conversation flows for different assessment types"""
        
        # ADHD Assessment Flows
        self.conversation_flows[AssessmentType.ADHD] = [
            ConversationFlow(
                stage=ConversationStage.INTRODUCTION,
                prompt="I'd like to learn about how your mind works and how you experience the world. This isn't a test - we're just having a conversation about your experiences. What's something you're really passionate about or interested in?",
                follow_up_questions=[
                    "How do you usually approach learning about new things?",
                    "What helps you stay focused when you're really interested in something?"
                ],
                keywords_to_explore=["focus", "interest", "passion", "learning"],
                expected_response_types=["personal_interest", "learning_style"]
            ),
            ConversationFlow(
                stage=ConversationStage.ADHD_ATTENTION,
                prompt="I'm curious about how you handle focus and attention in your daily life. Can you tell me about a time when you were trying to focus on something important but your mind kept wandering?",
                follow_up_questions=[
                    "How do you typically feel when you have to sit through long meetings or lectures?",
                    "What strategies do you use when you need to concentrate on something that doesn't interest you?",
                    "Do you find that you work better under pressure or with plenty of time?"
                ],
                keywords_to_explore=["wandering", "meetings", "concentrate", "pressure", "boredom"],
                expected_response_types=["attention_challenges", "focus_strategies", "work_style"]
            ),
            ConversationFlow(
                stage=ConversationStage.ADHD_HYPERACTIVITY,
                prompt="I'd love to hear about your energy levels and how you like to move through your day. How do you typically feel when you have to sit still for long periods?",
                follow_up_questions=[
                    "What do you do when you feel restless or have excess energy?",
                    "How do you prefer to work - do you like to move around or stay in one place?",
                    "Do you find that physical activity helps you think better?"
                ],
                keywords_to_explore=["restless", "energy", "movement", "sitting", "physical"],
                expected_response_types=["energy_patterns", "movement_preferences", "restlessness"]
            ),
            ConversationFlow(
                stage=ConversationStage.ADHD_IMPULSIVITY,
                prompt="Let's talk about decision-making and how you handle situations that require quick thinking. Can you describe a time when you made a decision quickly and how that turned out?",
                follow_up_questions=[
                    "How do you feel about interrupting others when you have an idea?",
                    "What happens when you get excited about something - do you tend to jump right in?",
                    "How do you handle waiting for things you're excited about?"
                ],
                keywords_to_explore=["quick_decision", "interrupting", "excitement", "waiting", "impulse"],
                expected_response_types=["decision_making", "social_interaction", "patience"]
            )
        ]
        
        # ASD Assessment Flows
        self.conversation_flows[AssessmentType.ASD] = [
            ConversationFlow(
                stage=ConversationStage.ASD_SOCIAL_COMMUNICATION,
                prompt="I'm interested in learning about your social experiences. Can you tell me about how you feel at parties or large social gatherings?",
                follow_up_questions=[
                    "How do you prefer to communicate with others - do you like face-to-face conversations, texting, or other ways?",
                    "What's something you wish people understood better about how you communicate?",
                    "How do you handle small talk or casual conversations?"
                ],
                keywords_to_explore=["parties", "social", "communication", "small_talk", "understanding"],
                expected_response_types=["social_preferences", "communication_style", "social_challenges"]
            ),
            ConversationFlow(
                stage=ConversationStage.ASD_SENSORY,
                prompt="I'd love to hear about your sensory experiences. Are there certain sounds, textures, lights, or smells that really bother you or that you really enjoy?",
                follow_up_questions=[
                    "How do you feel in crowded or noisy environments?",
                    "Are there certain fabrics or materials you prefer or avoid?",
                    "What kind of lighting helps you feel most comfortable?"
                ],
                keywords_to_explore=["sounds", "textures", "lights", "smells", "crowded", "noisy"],
                expected_response_types=["sensory_sensitivities", "environmental_preferences", "sensory_seeking"]
            ),
            ConversationFlow(
                stage=ConversationStage.ASD_REPETITIVE_BEHAVIORS,
                prompt="I'm curious about your daily routines and habits. Do you have certain rituals or ways of doing things that feel important to you?",
                follow_up_questions=[
                    "How do you feel when your routine gets disrupted?",
                    "Are there certain movements or actions that help you feel calm or focused?",
                    "What helps you feel most comfortable and secure in your daily life?"
                ],
                keywords_to_explore=["routine", "rituals", "disrupted", "movements", "calm", "secure"],
                expected_response_types=["routine_preferences", "coping_behaviors", "comfort_strategies"]
            ),
            ConversationFlow(
                stage=ConversationStage.ASD_INTERESTS,
                prompt="Tell me about your favorite hobbies or interests and how deeply you engage with them. What draws you to these particular activities?",
                follow_up_questions=[
                    "How do you feel when you're deeply engaged in something you love?",
                    "Do you find that you notice details that others might miss?",
                    "What's something you could talk about for hours without getting tired of it?"
                ],
                keywords_to_explore=["hobbies", "interests", "deeply", "details", "hours", "passion"],
                expected_response_types=["special_interests", "attention_to_detail", "intense_focus"]
            )
        ]
    
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
            name="Neurodevelopmental Assessment Agent",
            description="Conducts conversational assessments for ADHD and ASD traits through natural, supportive dialogue",
            version="1.0.0",
            capabilities=[
                {
                    "capability_type": CapabilityType.ASSESSMENT,
                    "description": "Neurodevelopmental trait assessment through conversation",
                    "input_modalities": [
                        InputModality.TEXT,
                        InputModality.AUDIO
                    ],
                    "output_formats": [
                        OutputFormat.STRUCTURED_DATA,
                        OutputFormat.ASSESSMENT_SCORE,
                        OutputFormat.TEXT_RESPONSE
                    ],
                    "parameters": {
                        "supported_assessments": ["adhd", "asd", "combined"],
                        "conversation_based": True,
                        "neurodiversity_affirming": True
                    }
                }
            ],
            supported_languages=["en"],
            availability_status="available",
            contact_endpoint="http://localhost:8000/neurodevelopmental-assessment"
        )
        
        await self.agent_discovery.register_agent(agent_card)
    
    async def start_assessment(self, user_id: str, assessment_type: AssessmentType, context: Optional[Dict[str, Any]] = None) -> str:
        """Start a new neurodevelopmental assessment"""
        assessment_id = str(uuid.uuid4())
        
        # Initialize assessment session
        self.active_assessments[assessment_id] = {
            "user_id": user_id,
            "assessment_type": assessment_type,
            "current_stage": ConversationStage.INTRODUCTION,
            "conversation_history": [],
            "responses": {},
            "insights": [],
            "started_at": datetime.utcnow(),
            "context": context or {}
        }
        
        # Get initial conversation flow
        flows = self.conversation_flows.get(assessment_type, [])
        if flows:
            initial_flow = flows[0]
            return initial_flow.prompt
        
        return "I'd like to learn about how your mind works and how you experience the world. This isn't a test - we're just having a conversation about your experiences. What's something you're really passionate about or interested in?"
    
    async def process_response(self, assessment_id: str, user_response: str) -> Dict[str, Any]:
        """Process user response and generate next conversation step"""
        if assessment_id not in self.active_assessments:
            return {"error": "Assessment not found"}
        
        assessment = self.active_assessments[assessment_id]
        
        # Add response to conversation history
        assessment["conversation_history"].append({
            "role": "user",
            "content": user_response,
            "timestamp": datetime.utcnow()
        })
        
        # Analyze response for patterns
        insights = await self._analyze_response(assessment, user_response)
        assessment["insights"].extend(insights)
        
        # Determine next conversation step
        next_step = await self._determine_next_step(assessment)
        
        # Add agent response to conversation history
        assessment["conversation_history"].append({
            "role": "assistant",
            "content": next_step["prompt"],
            "timestamp": datetime.utcnow()
        })
        
        return {
            "assessment_id": assessment_id,
            "next_prompt": next_step["prompt"],
            "stage": next_step["stage"],
            "insights_so_far": len(assessment["insights"]),
            "is_complete": next_step.get("is_complete", False)
        }
    
    async def _analyze_response(self, assessment: Dict[str, Any], response: str) -> List[AssessmentInsight]:
        """Analyze user response for neurodevelopmental patterns"""
        insights = []
        response_lower = response.lower()
        
        # ADHD Pattern Analysis
        if assessment["assessment_type"] in [AssessmentType.ADHD, AssessmentType.COMBINED]:
            adhd_insights = await self._analyze_adhd_patterns(response_lower, assessment["current_stage"])
            insights.extend(adhd_insights)
        
        # ASD Pattern Analysis
        if assessment["assessment_type"] in [AssessmentType.ASD, AssessmentType.COMBINED]:
            asd_insights = await self._analyze_asd_patterns(response_lower, assessment["current_stage"])
            insights.extend(asd_insights)
        
        return insights
    
    async def _analyze_adhd_patterns(self, response: str, stage: ConversationStage) -> List[AssessmentInsight]:
        """Analyze response for ADHD-related patterns"""
        insights = []
        
        # Attention patterns
        if any(word in response for word in ["wandering", "distracted", "mind", "focus", "concentrate"]):
            insights.append(AssessmentInsight(
                area="Attention",
                pattern_description="Mentions of attention or focus challenges",
                potential_traits=["attention_difficulties", "mind_wandering"],
                strengths=["self_awareness"],
                challenges=["sustained_attention"],
                recommendations=["Consider attention management strategies", "Explore focus techniques"],
                resources=["ADHD focus apps", "Pomodoro technique", "Environmental modifications"]
            ))
        
        # Hyperactivity patterns
        if any(word in response for word in ["restless", "energy", "move", "sitting", "fidget"]):
            insights.append(AssessmentInsight(
                area="Activity Level",
                pattern_description="Mentions of high energy or restlessness",
                potential_traits=["hyperactivity", "high_energy"],
                strengths=["active_lifestyle", "energy_utilization"],
                challenges=["sitting_still", "calm_activities"],
                recommendations=["Regular physical activity", "Movement breaks", "Fidget tools"],
                resources=["Exercise routines", "Standing desk", "Movement apps"]
            ))
        
        # Impulsivity patterns
        if any(word in response for word in ["quick", "jump", "excited", "interrupt", "waiting"]):
            insights.append(AssessmentInsight(
                area="Impulse Control",
                pattern_description="Mentions of quick decisions or excitement",
                potential_traits=["impulsivity", "enthusiasm"],
                strengths=["decisiveness", "enthusiasm"],
                challenges=["impulse_control", "patience"],
                recommendations=["Pause techniques", "Decision-making frameworks"],
                resources=["Mindfulness practices", "Decision journals"]
            ))
        
        return insights
    
    async def _analyze_asd_patterns(self, response: str, stage: ConversationStage) -> List[AssessmentInsight]:
        """Analyze response for ASD-related patterns"""
        insights = []
        
        # Social communication patterns
        if any(word in response for word in ["parties", "social", "small talk", "communication", "understanding"]):
            insights.append(AssessmentInsight(
                area="Social Communication",
                pattern_description="Mentions of social interaction preferences or challenges",
                potential_traits=["social_preferences", "communication_style"],
                strengths=["honest_communication", "deep_connections"],
                challenges=["small_talk", "social_nuances"],
                recommendations=["Social skills groups", "Communication strategies"],
                resources=["Social skills apps", "Communication guides", "Support groups"]
            ))
        
        # Sensory patterns
        if any(word in response for word in ["sounds", "textures", "lights", "smells", "crowded", "noisy"]):
            insights.append(AssessmentInsight(
                area="Sensory Processing",
                pattern_description="Mentions of sensory preferences or sensitivities",
                potential_traits=["sensory_sensitivity", "sensory_seeking"],
                strengths=["sensory_awareness", "environmental_adaptation"],
                challenges=["sensory_overload", "environmental_control"],
                recommendations=["Sensory accommodations", "Environmental modifications"],
                resources=["Sensory tools", "Noise-canceling headphones", "Sensory-friendly spaces"]
            ))
        
        # Repetitive behaviors and routines
        if any(word in response for word in ["routine", "rituals", "disrupted", "movements", "calm", "secure"]):
            insights.append(AssessmentInsight(
                area="Repetitive Behaviors",
                pattern_description="Mentions of routine preferences or repetitive behaviors",
                potential_traits=["routine_preference", "repetitive_behaviors"],
                strengths=["consistency", "self_regulation"],
                challenges=["flexibility", "change_adaptation"],
                recommendations=["Gradual routine changes", "Flexibility building"],
                resources=["Routine apps", "Change preparation strategies"]
            ))
        
        # Special interests
        if any(word in response for word in ["hobbies", "interests", "deeply", "details", "hours", "passion"]):
            insights.append(AssessmentInsight(
                area="Special Interests",
                pattern_description="Mentions of intense interests or attention to detail",
                potential_traits=["special_interests", "attention_to_detail"],
                strengths=["expertise", "focus", "passion"],
                challenges=["interest_balance", "social_sharing"],
                recommendations=["Interest integration", "Social sharing strategies"],
                resources=["Interest groups", "Skill development", "Mentorship programs"]
            ))
        
        return insights
    
    async def _create_personalized_prompt(self, next_flow: ConversationFlow, user_response: str, assessment_type: AssessmentType) -> str:
        """Create a personalized prompt based on user response and next conversation stage"""
        response_lower = user_response.lower()
        
        # Base prompt from the flow
        base_prompt = next_flow.prompt
        
        # Add personalized elements based on user response
        if "tech" in response_lower or "technology" in response_lower or "programming" in response_lower or "coding" in response_lower:
            if next_flow.stage == ConversationStage.ADHD_ATTENTION:
                return f"That's wonderful that you're passionate about tech! I can see how that field might appeal to you. {base_prompt} For example, when you're working on a coding project, do you find yourself getting completely absorbed in it, or do you sometimes struggle to maintain focus?"
            elif next_flow.stage == ConversationStage.ADHD_ACTIVITY:
                return f"Tech work can be really engaging! {base_prompt} When you're deep in a programming session, do you find yourself needing to move around or take breaks, or can you sit still for long periods?"
            elif next_flow.stage == ConversationStage.ASD_INTERESTS:
                return f"Your interest in tech is really interesting! {base_prompt} What specifically draws you to technology? Is it the problem-solving aspect, the creativity, or something else?"
            elif next_flow.stage == ConversationStage.ASD_SOCIAL:
                return f"Tech can be a great field for collaboration. {base_prompt} How do you feel about working with others on technical projects? Do you prefer pair programming, team meetings, or working independently?"
        
        # General personalized responses
        if "work" in response_lower or "job" in response_lower:
            if next_flow.stage == ConversationStage.ADHD_ATTENTION:
                return f"I appreciate you sharing about your work. {base_prompt} In your work environment, do you find it easy to focus on tasks, or do you sometimes get distracted by other things happening around you?"
            elif next_flow.stage == ConversationStage.ASD_SOCIAL:
                return f"Work can involve a lot of social interaction. {base_prompt} How do you feel about team meetings or collaborating with colleagues on projects?"
        
        if "learning" in response_lower or "study" in response_lower:
            if next_flow.stage == ConversationStage.ADHD_ATTENTION:
                return f"Learning new things is great! {base_prompt} When you're studying or learning something new, do you find it easy to concentrate, or do your thoughts sometimes wander to other topics?"
            elif next_flow.stage == ConversationStage.ASD_INTERESTS:
                return f"Learning is such an important part of growth. {base_prompt} What's your favorite way to learn new things? Do you prefer hands-on experience, reading, or something else?"
        
        # If no specific pattern matches, use the base prompt with a gentle acknowledgment
        return f"Thank you for sharing that with me. {base_prompt}"
    
    async def _determine_next_step(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the next conversation step based on current progress"""
        flows = self.conversation_flows.get(assessment["assessment_type"], [])
        current_stage = assessment["current_stage"]
        conversation_history = assessment["conversation_history"]
        
        # Get the last user response
        last_user_response = ""
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg["role"] == "user":
                    last_user_response = msg["content"]
                    break
        
        # Find current flow index
        current_index = next((i for i, flow in enumerate(flows) if flow.stage == current_stage), 0)
        
        # Generate a more engaging response based on user input
        if last_user_response and current_index < len(flows) - 1:
            # Create a personalized follow-up based on their response
            next_flow = flows[current_index + 1]
            personalized_prompt = await self._create_personalized_prompt(
                next_flow, last_user_response, assessment["assessment_type"]
            )
            assessment["current_stage"] = next_flow.stage
            return {
                "stage": next_flow.stage,
                "prompt": personalized_prompt,
                "is_complete": False
            }
        elif current_index < len(flows) - 1:
            # Move to next stage with original prompt
            next_flow = flows[current_index + 1]
            assessment["current_stage"] = next_flow.stage
            return {
                "stage": next_flow.stage,
                "prompt": next_flow.prompt,
                "is_complete": False
            }
        else:
            # Assessment complete
            return await self._complete_assessment(assessment)
    
    async def _complete_assessment(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Complete the assessment and generate results"""
        # Generate comprehensive results
        result = await self._generate_assessment_results(assessment)
        
        # Save results
        await self._save_assessment_results(result)
        
        # Mark as complete
        assessment["completed_at"] = datetime.utcnow()
        assessment["status"] = "completed"
        
        return {
            "stage": ConversationStage.RESULTS,
            "prompt": "Thank you for sharing your experiences with me. I've compiled some insights based on our conversation that might be helpful for you to reflect on.",
            "is_complete": True,
            "results": result
        }
    
    async def _generate_assessment_results(self, assessment: Dict[str, Any]) -> NeurodevelopmentalAssessmentResult:
        """Generate comprehensive assessment results"""
        # Organize insights by area
        adhd_insights = [insight for insight in assessment["insights"] if insight.area in ["Attention", "Activity Level", "Impulse Control"]]
        asd_insights = [insight for insight in assessment["insights"] if insight.area in ["Social Communication", "Sensory Processing", "Repetitive Behaviors", "Special Interests"]]
        
        # Generate recommendations
        recommendations = []
        resources = []
        
        for insight in assessment["insights"]:
            recommendations.extend(insight.recommendations)
            resources.extend(insight.resources)
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        resources = list(set(resources))
        
        # Determine if professional evaluation is recommended
        requires_evaluation = len(assessment["insights"]) > 3 or any(
            insight.area in ["Attention", "Social Communication", "Sensory Processing"] 
            for insight in assessment["insights"]
        )
        
        return NeurodevelopmentalAssessmentResult(
            assessment_id=str(uuid.uuid4()),
            user_id=assessment["user_id"],
            assessment_type=assessment["assessment_type"],
            conversation_stages_completed=[ConversationStage.INTRODUCTION],  # Simplified for now
            adhd_insights=adhd_insights,
            asd_insights=asd_insights,
            overall_patterns=[],  # Would be populated with pattern analysis
            recommendations=recommendations,
            resources=resources,
            requires_professional_evaluation=requires_evaluation,
            next_steps=[
                "Review the insights and recommendations provided",
                "Consider discussing findings with a healthcare provider",
                "Explore the suggested resources and strategies",
                "Continue self-reflection and self-advocacy"
            ]
        )
    
    async def _save_assessment_results(self, result: NeurodevelopmentalAssessmentResult):
        """Save assessment results to storage"""
        # Save to file storage
        self.storage_manager.save_neurodevelopmental_assessment(result.dict())
    
    async def _handle_task_request(self, message: A2AMessage):
        """Handle task requests from other agents"""
        # Implementation for handling task requests
        pass
    
    async def _handle_collaboration(self, message: A2AMessage):
        """Handle collaboration requests from other agents"""
        # Implementation for handling collaboration
        pass
    
    async def _handle_crisis_alert(self, message: A2AMessage):
        """Handle crisis alerts with neurodivergent considerations"""
        # Implementation for handling crisis alerts with special considerations
        pass
