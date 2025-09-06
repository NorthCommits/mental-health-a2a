"""
Crisis Detection Agent

Continuously monitors all interactions for high-risk content and provides
immediate crisis intervention when needed.
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from shared.a2a_protocol.communication_layer import A2ACommunicator, A2AMessage, MessageType, ContentType
from shared.a2a_protocol.agent_discovery import AgentDiscovery, AgentCard, CapabilityType, InputModality, OutputFormat
from shared.a2a_protocol.task_management import TaskManager, Task, TaskPriority, TaskStatus
from shared.a2a_protocol.security import A2ASecurity, AgentRole, AccessLevel

from .risk_models import (
    RiskAssessment, CrisisLevel, RiskFactor, CrisisAlert, 
    SafetyPlan, CrisisIntervention, CrisisStatistics, CrisisPattern
)
from .nlp_models import CrisisNLPModel, SentimentAnalyzer, IntentClassifier


class CrisisAgentStatus(str, Enum):
    """Status of the crisis detection agent"""
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    CRISIS_RESPONSE = "crisis_response"
    INTERVENTION = "intervention"
    MAINTENANCE = "maintenance"


class NeurodivergentCrisisType(str, Enum):
    """Types of neurodivergent-specific crisis situations"""
    SENSORY_OVERLOAD = "sensory_overload"
    MELTDOWN = "meltdown"
    SHUTDOWN = "shutdown"
    EXECUTIVE_FUNCTION_FAILURE = "executive_function_failure"
    SOCIAL_OVERWHELM = "social_overwhelm"
    ROUTINE_DISRUPTION = "routine_disruption"
    MASKING_EXHAUSTION = "masking_exhaustion"
    BURNOUT = "burnout"


class CrisisDetectionAgent:
    """
    Crisis Detection Agent for mental health monitoring
    
    This agent continuously monitors all interactions for high-risk content
    and provides immediate crisis intervention when needed.
    """
    
    def __init__(
        self,
        agent_id: str,
        a2a_communicator: A2ACommunicator,
        agent_discovery: AgentDiscovery,
        task_manager: TaskManager,
        security: A2ASecurity
    ):
        self.agent_id = agent_id
        self.a2a_communicator = a2a_communicator
        self.agent_discovery = agent_discovery
        self.task_manager = task_manager
        self.security = security
        
        # Initialize NLP models
        self.crisis_nlp = CrisisNLPModel()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.intent_classifier = IntentClassifier()
        
        # Agent state
        self.status = CrisisAgentStatus.MONITORING
        self.active_alerts: Dict[str, CrisisAlert] = {}
        self.safety_plans: Dict[str, SafetyPlan] = {}
        self.intervention_history: List[CrisisIntervention] = []
        self.statistics = CrisisStatistics(time_period="24h")
        
        # Crisis detection thresholds
        self.crisis_thresholds = {
            CrisisLevel.LOW: 0.2,
            CrisisLevel.MEDIUM: 0.4,
            CrisisLevel.HIGH: 0.6,
            CrisisLevel.CRITICAL: 0.8,
            CrisisLevel.EMERGENCY: 0.9
        }
        
        # Register message handlers and capabilities
        # Note: These will be called when the agent is started
        self._message_handlers_registered = False
        self._capabilities_registered = False
        self._monitoring_started = False
    
    async def initialize(self):
        """Initialize the agent by registering handlers and capabilities"""
        if not self._message_handlers_registered:
            await self._register_message_handlers()
            self._message_handlers_registered = True
        
        if not self._capabilities_registered:
            await self._register_agent_capabilities()
            self._capabilities_registered = True
        
        if not self._monitoring_started:
            asyncio.create_task(self._monitoring_loop())
            self._monitoring_started = True
    
    async def _register_message_handlers(self):
        """Register message handlers for A2A communication"""
        await self.a2a_communicator.register_message_handler(
            MessageType.CRISIS_ALERT,
            self._handle_crisis_alert
        )
        await self.a2a_communicator.register_message_handler(
            MessageType.TASK_REQUEST,
            self._handle_task_request
        )
        await self.a2a_communicator.register_message_handler(
            MessageType.COLLABORATION,
            self._handle_collaboration
        )
    
    async def _register_agent_capabilities(self):
        """Register agent capabilities with the discovery service"""
        agent_card = AgentCard(
            agent_id=self.agent_id,
            name="Crisis Detection Agent",
            description="Monitors interactions for crisis indicators and provides immediate intervention",
            version="1.0.0",
            capabilities=[
                {
                    "capability_type": CapabilityType.CRISIS_DETECTION,
                    "description": "Real-time crisis detection and intervention",
                    "input_modalities": [
                        InputModality.TEXT,
                        InputModality.AUDIO,
                        InputModality.IMAGE
                    ],
                    "output_formats": [
                        OutputFormat.CRISIS_ALERT,
                        OutputFormat.STRUCTURED_DATA
                    ],
                    "parameters": {
                        "response_time_sla": 1.0,  # 1 second for crisis detection
                        "accuracy_threshold": 0.85,
                        "privacy_level": "maximum"
                    }
                }
            ],
            contact_endpoint=f"http://localhost:8000/agents/{self.agent_id}",
            compliance_certifications=["hipaa", "crisis_intervention_standards"]
        )
        
        await self.agent_discovery.register_agent(agent_card)
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop for crisis detection"""
        while True:
            try:
                # Check for new data to analyze
                await self._process_pending_analysis()
                
                # Update statistics
                await self._update_statistics()
                
                # Check for overdue interventions
                await self._check_overdue_interventions()
                
                # Sleep for monitoring interval
                await asyncio.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying
    
    async def analyze_interaction(
        self,
        user_id: str,
        session_id: str,
        interaction_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[RiskAssessment]:
        """
        Analyze an interaction for crisis indicators
        
        Args:
            user_id: ID of the user
            session_id: ID of the session
            interaction_data: Data from the interaction
            context: Additional context
            
        Returns:
            RiskAssessment if crisis indicators found, None otherwise
        """
        try:
            self.status = CrisisAgentStatus.ANALYZING
            
            # Extract text for analysis
            text_content = self._extract_text_content(interaction_data)
            
            if not text_content:
                return None
            
            # Analyze text for crisis indicators
            analysis_result = await self.crisis_nlp.analyze_text(text_content, context)
            
            # Check if crisis indicators are present
            risk_score = analysis_result.get("risk_score", 0.0)
            confidence = analysis_result.get("confidence", 0.0)
            
            if risk_score < self.crisis_thresholds[CrisisLevel.LOW]:
                return None
            
            # Create risk assessment
            risk_assessment = await self._create_risk_assessment(
                user_id, session_id, analysis_result, context
            )
            
            # Check if immediate action is required
            if risk_assessment.crisis_level in [CrisisLevel.CRITICAL, CrisisLevel.EMERGENCY]:
                await self._handle_crisis_situation(risk_assessment)
            
            return risk_assessment
            
        except Exception as e:
            print(f"Error analyzing interaction: {e}")
            return None
        finally:
            self.status = CrisisAgentStatus.MONITORING
    
    def _extract_text_content(self, interaction_data: Dict[str, Any]) -> str:
        """Extract text content from interaction data"""
        # Try different sources of text content
        if "text" in interaction_data:
            return interaction_data["text"]
        elif "message" in interaction_data:
            return interaction_data["message"]
        elif "transcript" in interaction_data:
            return interaction_data["transcript"]
        elif "content" in interaction_data:
            return str(interaction_data["content"])
        else:
            return ""
    
    async def _create_risk_assessment(
        self,
        user_id: str,
        session_id: str,
        analysis_result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """Create a risk assessment from analysis results"""
        
        # Extract risk factors
        risk_factors = []
        crisis_indicators = analysis_result.get("crisis_indicators", {})
        
        for risk_type in RiskFactor:
            if risk_type.value in crisis_indicators.get("risk_factors", []):
                risk_factors.append(risk_type)
        
        # Create crisis patterns
        detected_patterns = []
        patterns = crisis_indicators.get("detected_patterns", {})
        
        for pattern_type, pattern_data in patterns.items():
            pattern = CrisisPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type=RiskFactor(pattern_type),
                confidence_score=pattern_data.get("score", 0.0),
                severity_weight=pattern_data.get("severity", 0.0),
                context=pattern_data.get("context", {})
            )
            detected_patterns.append(pattern)
        
        # Calculate scores
        risk_score = crisis_indicators.get("overall_risk_score", 0.0)
        confidence_score = crisis_indicators.get("confidence", 0.0)
        
        # Determine crisis level
        crisis_level = self._determine_crisis_level(risk_score)
        
        # Calculate urgency and safety scores
        urgency_score = min(1.0, risk_score * 1.2)  # Urgency increases with risk
        safety_score = max(0.0, 1.0 - risk_score)  # Safety decreases with risk
        
        # Create risk assessment
        assessment = RiskAssessment(
            assessment_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            crisis_level=crisis_level,
            risk_factors=risk_factors,
            detected_patterns=detected_patterns,
            confidence_score=confidence_score,
            urgency_score=urgency_score,
            safety_score=safety_score,
            requires_immediate_action=crisis_level in [CrisisLevel.CRITICAL, CrisisLevel.EMERGENCY],
            context=context or {},
            metadata=analysis_result
        )
        
        return assessment
    
    def _determine_crisis_level(self, risk_score: float) -> CrisisLevel:
        """Determine crisis level based on risk score"""
        for level in [CrisisLevel.EMERGENCY, CrisisLevel.CRITICAL, CrisisLevel.HIGH, CrisisLevel.MEDIUM, CrisisLevel.LOW]:
            if risk_score >= self.crisis_thresholds[level]:
                return level
        return CrisisLevel.NONE
    
    async def _handle_crisis_situation(self, risk_assessment: RiskAssessment):
        """Handle a crisis situation requiring immediate action"""
        try:
            self.status = CrisisAgentStatus.CRISIS_RESPONSE
            
            # Create crisis alert
            alert = CrisisAlert(
                alert_id=str(uuid.uuid4()),
                user_id=risk_assessment.user_id,
                session_id=risk_assessment.session_id,
                crisis_level=risk_assessment.crisis_level,
                risk_assessment=risk_assessment,
                alert_message=self._generate_alert_message(risk_assessment),
                required_actions=risk_assessment.get_required_interventions(),
                contact_emergency_services=risk_assessment.crisis_level == CrisisLevel.EMERGENCY,
                notify_family=risk_assessment.crisis_level in [CrisisLevel.CRITICAL, CrisisLevel.EMERGENCY]
            )
            
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            
            # Send alerts to relevant agents
            await self._send_crisis_alerts(alert)
            
            # Initiate immediate interventions
            await self._initiate_immediate_interventions(alert)
            
            # Update statistics
            self.statistics.crisis_detected += 1
            
        except Exception as e:
            print(f"Error handling crisis situation: {e}")
        finally:
            self.status = CrisisAgentStatus.MONITORING
    
    def _generate_alert_message(self, risk_assessment: RiskAssessment) -> str:
        """Generate alert message for crisis situation"""
        level = risk_assessment.crisis_level.value.upper()
        factors = ", ".join([factor.value for factor in risk_assessment.risk_factors])
        
        return f"CRISIS ALERT - {level} level detected. Risk factors: {factors}. " \
               f"Confidence: {risk_assessment.confidence_score:.2f}. " \
               f"Safety score: {risk_assessment.safety_score:.2f}"
    
    async def _send_crisis_alerts(self, alert: CrisisAlert):
        """Send crisis alerts to relevant agents"""
        try:
            # Send to Care Coordination Agent
            care_agents = await self.agent_discovery.discover_agents_by_capability(
                CapabilityType.CARE_COORDINATION
            )
            
            for care_agent in care_agents:
                crisis_message = await self.a2a_communicator.create_crisis_alert(
                    care_agent.agent_id,
                    {
                        "alert_id": alert.alert_id,
                        "user_id": alert.user_id,
                        "crisis_level": alert.crisis_level.value,
                        "risk_assessment": alert.risk_assessment.model_dump(),
                        "required_actions": alert.required_actions,
                        "contact_emergency_services": alert.contact_emergency_services,
                        "notify_family": alert.notify_family,
                        "timestamp": alert.created_at.isoformat()
                    }
                )
                
                await self.a2a_communicator.send_message(crisis_message)
            
            # Send to Therapeutic Intervention Agent if available
            therapy_agents = await self.agent_discovery.discover_agents_by_capability(
                CapabilityType.THERAPEUTIC_INTERVENTION
            )
            
            for therapy_agent in therapy_agents:
                crisis_message = await self.a2a_communicator.create_crisis_alert(
                    therapy_agent.agent_id,
                    {
                        "alert_id": alert.alert_id,
                        "user_id": alert.user_id,
                        "crisis_level": alert.crisis_level.value,
                        "intervention_needed": True,
                        "timestamp": alert.created_at.isoformat()
                    }
                )
                
                await self.a2a_communicator.send_message(crisis_message)
            
        except Exception as e:
            print(f"Error sending crisis alerts: {e}")
    
    async def _initiate_immediate_interventions(self, alert: CrisisAlert):
        """Initiate immediate crisis interventions"""
        try:
            self.status = CrisisAgentStatus.INTERVENTION
            
            # Create intervention tasks
            for intervention_type in alert.required_actions:
                task = await self.task_manager.create_task(
                    title=f"Crisis intervention: {intervention_type}",
                    description=f"Immediate crisis intervention for user {alert.user_id}",
                    task_type="crisis_intervention",
                    created_by=self.agent_id,
                    assigned_to=self.agent_id,  # Handle internally first
                    priority=TaskPriority.EMERGENCY,
                    input_data={
                        "alert_id": alert.alert_id,
                        "intervention_type": intervention_type,
                        "user_id": alert.user_id,
                        "crisis_level": alert.crisis_level.value
                    }
                )
                
                # Record intervention
                intervention = CrisisIntervention(
                    intervention_id=str(uuid.uuid4()),
                    user_id=alert.user_id,
                    crisis_alert_id=alert.alert_id,
                    intervention_type=intervention_type,
                    action_taken=f"Initiated {intervention_type} intervention",
                    outcome="in_progress",
                    follow_up_required=True,
                    performed_by=self.agent_id
                )
                
                self.intervention_history.append(intervention)
            
        except Exception as e:
            print(f"Error initiating interventions: {e}")
        finally:
            self.status = CrisisAgentStatus.MONITORING
    
    async def _process_pending_analysis(self):
        """Process any pending analysis requests"""
        # This would check for new data to analyze
        # For now, it's a placeholder
        pass
    
    async def _update_statistics(self):
        """Update crisis detection statistics"""
        # Update statistics based on recent activity
        current_time = datetime.utcnow()
        
        # Count recent assessments
        recent_assessments = len([
            alert for alert in self.active_alerts.values()
            if (current_time - alert.created_at).total_seconds() < 3600  # Last hour
        ])
        
        self.statistics.total_assessments = recent_assessments
    
    async def _check_overdue_interventions(self):
        """Check for overdue crisis interventions"""
        current_time = datetime.utcnow()
        
        for intervention in self.intervention_history:
            if (not intervention.follow_up_required or 
                intervention.follow_up_date is None or
                current_time < intervention.follow_up_date):
                continue
            
            # Follow up is overdue
            await self._handle_overdue_intervention(intervention)
    
    async def _handle_overdue_intervention(self, intervention: CrisisIntervention):
        """Handle overdue crisis intervention"""
        try:
            # Create follow-up task
            task = await self.task_manager.create_task(
                title=f"Follow-up for intervention {intervention.intervention_id}",
                description=f"Follow-up required for crisis intervention",
                task_type="crisis_follow_up",
                created_by=self.agent_id,
                assigned_to=self.agent_id,
                priority=TaskPriority.HIGH,
                input_data={
                    "intervention_id": intervention.intervention_id,
                    "user_id": intervention.user_id,
                    "overdue": True
                }
            )
            
        except Exception as e:
            print(f"Error handling overdue intervention: {e}")
    
    async def _handle_crisis_alert(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle incoming crisis alerts"""
        try:
            # Process crisis alert from other agents
            alert_data = json.loads(message.parts[0].content) if message.parts else {}
            
            # Acknowledge the alert
            return {
                "status": "acknowledged",
                "message": "Crisis alert received and being processed",
                "alert_id": alert_data.get("alert_id")
            }
            
        except Exception as e:
            print(f"Error handling crisis alert: {e}")
            return {"error": str(e)}
    
    async def _handle_task_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle incoming task requests"""
        try:
            # Process task request
            task_data = message.parts[0].content if message.parts else ""
            
            # Create task
            task = await self.task_manager.create_task(
                title=f"Crisis detection task from {message.sender_agent_id}",
                description=task_data,
                task_type="crisis_detection_request",
                created_by=message.sender_agent_id,
                assigned_to=self.agent_id,
                priority=TaskPriority.HIGH,
                input_data={"message": message.model_dump()}
            )
            
            return {
                "task_id": task.task_id,
                "status": "accepted",
                "message": "Crisis detection task accepted"
            }
            
        except Exception as e:
            print(f"Error handling task request: {e}")
            return {"error": str(e)}
    
    async def _handle_collaboration(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle collaboration messages"""
        try:
            # Process collaboration request
            collaboration_data = json.loads(message.parts[0].content) if message.parts else {}
            
            # Handle collaboration
            return {
                "status": "collaboration_acknowledged",
                "message": "Collaboration request received"
            }
            
        except Exception as e:
            print(f"Error handling collaboration: {e}")
            return {"error": str(e)}
    
    async def get_active_alerts(self) -> List[CrisisAlert]:
        """Get all active crisis alerts"""
        return list(self.active_alerts.values())
    
    async def get_statistics(self) -> CrisisStatistics:
        """Get crisis detection statistics"""
        return self.statistics
    
    async def create_safety_plan(self, user_id: str, plan_data: Dict[str, Any]) -> SafetyPlan:
        """Create a safety plan for a user"""
        safety_plan = SafetyPlan(
            plan_id=str(uuid.uuid4()),
            user_id=user_id,
            warning_signs=plan_data.get("warning_signs", []),
            coping_strategies=plan_data.get("coping_strategies", []),
            support_contacts=plan_data.get("support_contacts", []),
            professional_contacts=plan_data.get("professional_contacts", []),
            emergency_contacts=plan_data.get("emergency_contacts", [])
        )
        
        self.safety_plans[user_id] = safety_plan
        return safety_plan
    
    async def get_safety_plan(self, user_id: str) -> Optional[SafetyPlan]:
        """Get safety plan for a user"""
        return self.safety_plans.get(user_id)
    
    async def analyze_neurodivergent_crisis(
        self,
        user_id: str,
        session_id: str,
        interaction_data: Dict[str, Any],
        neurodivergent_profile: Optional[Dict[str, Any]] = None
    ) -> Optional[RiskAssessment]:
        """
        Analyze interaction for neurodivergent-specific crisis indicators
        
        Args:
            user_id: ID of the user
            session_id: ID of the session
            interaction_data: Data from the interaction
            neurodivergent_profile: User's neurodivergent traits and needs
            
        Returns:
            RiskAssessment if neurodivergent crisis indicators found, None otherwise
        """
        try:
            text_content = self._extract_text_content(interaction_data)
            if not text_content:
                return None
            
            # Check for neurodivergent crisis patterns
            crisis_type = await self._detect_neurodivergent_crisis_type(text_content, neurodivergent_profile)
            
            if not crisis_type:
                return None
            
            # Create specialized risk assessment
            risk_assessment = await self._create_neurodivergent_risk_assessment(
                user_id, session_id, crisis_type, text_content, neurodivergent_profile
            )
            
            # Handle neurodivergent crisis with appropriate interventions
            if risk_assessment.crisis_level in [CrisisLevel.CRITICAL, CrisisLevel.EMERGENCY]:
                await self._handle_neurodivergent_crisis(risk_assessment, crisis_type)
            
            return risk_assessment
            
        except Exception as e:
            print(f"Error analyzing neurodivergent crisis: {e}")
            return None
    
    async def _detect_neurodivergent_crisis_type(
        self, 
        text_content: str, 
        neurodivergent_profile: Optional[Dict[str, Any]] = None
    ) -> Optional[NeurodivergentCrisisType]:
        """Detect specific type of neurodivergent crisis from text content"""
        text_lower = text_content.lower()
        
        # Sensory overload indicators
        sensory_keywords = [
            "too loud", "too bright", "overwhelming", "sensory", "noise", "light",
            "texture", "smell", "taste", "touch", "overstimulated", "overwhelmed"
        ]
        if any(keyword in text_lower for keyword in sensory_keywords):
            return NeurodivergentCrisisType.SENSORY_OVERLOAD
        
        # Meltdown indicators
        meltdown_keywords = [
            "meltdown", "breaking down", "can't cope", "losing control", "exploding",
            "outburst", "tantrum", "screaming", "crying", "angry", "frustrated"
        ]
        if any(keyword in text_lower for keyword in meltdown_keywords):
            return NeurodivergentCrisisType.MELTDOWN
        
        # Shutdown indicators
        shutdown_keywords = [
            "shutdown", "shut down", "can't speak", "can't move", "frozen",
            "numb", "disconnected", "dissociated", "gone blank", "empty"
        ]
        if any(keyword in text_lower for keyword in shutdown_keywords):
            return NeurodivergentCrisisType.SHUTDOWN
        
        # Executive function failure indicators
        executive_keywords = [
            "can't think", "brain fog", "executive function", "can't organize",
            "can't plan", "can't decide", "overwhelmed by tasks", "paralysis",
            "can't start", "stuck", "frozen"
        ]
        if any(keyword in text_lower for keyword in executive_keywords):
            return NeurodivergentCrisisType.EXECUTIVE_FUNCTION_FAILURE
        
        # Social overwhelm indicators
        social_keywords = [
            "social anxiety", "people are too much", "socially exhausted", "masking",
            "pretending", "acting normal", "socially drained", "people overwhelming"
        ]
        if any(keyword in text_lower for keyword in social_keywords):
            return NeurodivergentCrisisType.SOCIAL_OVERWHELM
        
        # Routine disruption indicators
        routine_keywords = [
            "routine disrupted", "schedule changed", "unexpected", "surprise",
            "can't adapt", "change is hard", "routine broken", "plan changed"
        ]
        if any(keyword in text_lower for keyword in routine_keywords):
            return NeurodivergentCrisisType.ROUTINE_DISRUPTION
        
        # Masking exhaustion indicators
        masking_keywords = [
            "tired of pretending", "exhausted from acting", "masking is hard",
            "can't keep up the act", "tired of being normal", "pretending is exhausting"
        ]
        if any(keyword in text_lower for keyword in masking_keywords):
            return NeurodivergentCrisisType.MASKING_EXHAUSTION
        
        # Burnout indicators
        burnout_keywords = [
            "burnout", "burned out", "exhausted", "drained", "can't function",
            "overwhelmed", "stressed", "anxious", "depressed", "hopeless"
        ]
        if any(keyword in text_lower for keyword in burnout_keywords):
            return NeurodivergentCrisisType.BURNOUT
        
        return None
    
    async def _create_neurodivergent_risk_assessment(
        self,
        user_id: str,
        session_id: str,
        crisis_type: NeurodivergentCrisisType,
        text_content: str,
        neurodivergent_profile: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """Create risk assessment for neurodivergent crisis"""
        
        # Determine crisis level based on type and intensity
        crisis_level = self._determine_neurodivergent_crisis_level(crisis_type, text_content)
        
        # Create appropriate risk factors
        risk_factors = self._create_neurodivergent_risk_factors(crisis_type, neurodivergent_profile)
        
        # Generate intervention recommendations
        interventions = self._generate_neurodivergent_interventions(crisis_type, crisis_level)
        
        return RiskAssessment(
            assessment_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            crisis_level=crisis_level,
            risk_score=self._calculate_neurodivergent_risk_score(crisis_type, crisis_level),
            confidence=0.8,  # High confidence for neurodivergent patterns
            risk_factors=risk_factors,
            crisis_indicators=[crisis_type.value],
            intervention_recommendations=interventions,
            requires_immediate_action=crisis_level in [CrisisLevel.CRITICAL, CrisisLevel.EMERGENCY],
            assessment_timestamp=datetime.utcnow(),
            context={
                "neurodivergent_crisis_type": crisis_type.value,
                "neurodivergent_profile": neurodivergent_profile or {},
                "text_content": text_content[:500]  # Truncate for storage
            }
        )
    
    def _determine_neurodivergent_crisis_level(
        self, 
        crisis_type: NeurodivergentCrisisType, 
        text_content: str
    ) -> CrisisLevel:
        """Determine crisis level for neurodivergent crisis"""
        text_lower = text_content.lower()
        
        # High-intensity keywords indicate higher crisis levels
        high_intensity_keywords = [
            "can't", "unable", "impossible", "never", "always", "completely",
            "totally", "absolutely", "desperate", "hopeless", "suicidal"
        ]
        
        has_high_intensity = any(keyword in text_lower for keyword in high_intensity_keywords)
        
        # Emergency level for certain crisis types with high intensity
        if crisis_type in [NeurodivergentCrisisType.MELTDOWN, NeurodivergentCrisisType.SHUTDOWN] and has_high_intensity:
            return CrisisLevel.EMERGENCY
        
        # Critical level for most neurodivergent crises with high intensity
        if has_high_intensity:
            return CrisisLevel.CRITICAL
        
        # High level for most neurodivergent crises
        if crisis_type in [NeurodivergentCrisisType.MELTDOWN, NeurodivergentCrisisType.SHUTDOWN, NeurodivergentCrisisType.BURNOUT]:
            return CrisisLevel.HIGH
        
        # Medium level for others
        return CrisisLevel.MEDIUM
    
    def _create_neurodivergent_risk_factors(
        self, 
        crisis_type: NeurodivergentCrisisType,
        neurodivergent_profile: Optional[Dict[str, Any]] = None
    ) -> List[RiskFactor]:
        """Create risk factors specific to neurodivergent crisis"""
        risk_factors = []
        
        # Add crisis-specific risk factors
        if crisis_type == NeurodivergentCrisisType.SENSORY_OVERLOAD:
            risk_factors.append(RiskFactor(
                factor="sensory_overload",
                description="Experiencing sensory overwhelm",
                severity="high",
                category="sensory_processing"
            ))
        
        elif crisis_type == NeurodivergentCrisisType.MELTDOWN:
            risk_factors.append(RiskFactor(
                factor="meltdown",
                description="Experiencing emotional/behavioral meltdown",
                severity="critical",
                category="emotional_regulation"
            ))
        
        elif crisis_type == NeurodivergentCrisisType.SHUTDOWN:
            risk_factors.append(RiskFactor(
                factor="shutdown",
                description="Experiencing shutdown response",
                severity="critical",
                category="emotional_regulation"
            ))
        
        elif crisis_type == NeurodivergentCrisisType.EXECUTIVE_FUNCTION_FAILURE:
            risk_factors.append(RiskFactor(
                factor="executive_dysfunction",
                description="Experiencing executive function difficulties",
                severity="medium",
                category="cognitive_processing"
            ))
        
        # Add profile-based risk factors
        if neurodivergent_profile:
            if neurodivergent_profile.get("sensory_sensitivities"):
                risk_factors.append(RiskFactor(
                    factor="sensory_sensitivities",
                    description="Has known sensory sensitivities",
                    severity="medium",
                    category="sensory_processing"
                ))
            
            if neurodivergent_profile.get("social_difficulties"):
                risk_factors.append(RiskFactor(
                    factor="social_difficulties",
                    description="Has known social communication challenges",
                    severity="medium",
                    category="social_processing"
                ))
        
        return risk_factors
    
    def _generate_neurodivergent_interventions(
        self, 
        crisis_type: NeurodivergentCrisisType, 
        crisis_level: CrisisLevel
    ) -> List[str]:
        """Generate intervention recommendations for neurodivergent crisis"""
        interventions = []
        
        if crisis_type == NeurodivergentCrisisType.SENSORY_OVERLOAD:
            interventions.extend([
                "Find a quiet, dimly lit space immediately",
                "Use noise-canceling headphones or earplugs",
                "Remove or reduce sensory stimuli (bright lights, loud sounds)",
                "Practice deep breathing or grounding techniques",
                "Consider weighted blanket or compression clothing"
            ])
        
        elif crisis_type == NeurodivergentCrisisType.MELTDOWN:
            interventions.extend([
                "Ensure physical safety - remove harmful objects",
                "Provide space and time - don't try to stop the meltdown",
                "Use calming sensory tools if available",
                "Speak in a calm, low voice",
                "Avoid physical contact unless requested"
            ])
        
        elif crisis_type == NeurodivergentCrisisType.SHUTDOWN:
            interventions.extend([
                "Provide a safe, quiet space",
                "Don't force communication or interaction",
                "Offer comfort items or sensory tools",
                "Be patient and wait for them to re-engage",
                "Avoid overwhelming them with questions"
            ])
        
        elif crisis_type == NeurodivergentCrisisType.EXECUTIVE_FUNCTION_FAILURE:
            interventions.extend([
                "Break tasks into smaller, manageable steps",
                "Use visual aids or written lists",
                "Remove distractions and simplify environment",
                "Offer to help with decision-making",
                "Provide structure and routine"
            ])
        
        elif crisis_type == NeurodivergentCrisisType.SOCIAL_OVERWHELM:
            interventions.extend([
                "Take a break from social situations",
                "Use communication alternatives (text, writing)",
                "Practice self-compassion about social challenges",
                "Connect with understanding friends or support groups",
                "Consider social skills or therapy support"
            ])
        
        elif crisis_type == NeurodivergentCrisisType.ROUTINE_DISRUPTION:
            interventions.extend([
                "Acknowledge the difficulty of unexpected changes",
                "Help create a new routine or structure",
                "Provide advance notice for future changes when possible",
                "Use visual schedules or calendars",
                "Practice flexibility-building exercises gradually"
            ])
        
        elif crisis_type == NeurodivergentCrisisType.MASKING_EXHAUSTION:
            interventions.extend([
                "Take time to unmask and be authentic",
                "Connect with neurodivergent community",
                "Practice self-acceptance and self-advocacy",
                "Set boundaries around masking expectations",
                "Seek supportive, accepting environments"
            ])
        
        elif crisis_type == NeurodivergentCrisisType.BURNOUT:
            interventions.extend([
                "Take a complete break from overwhelming activities",
                "Focus on basic self-care and rest",
                "Reduce sensory and social demands",
                "Engage in preferred, low-demand activities",
                "Consider professional support for burnout recovery"
            ])
        
        # Add general neurodivergent support
        interventions.extend([
            "Remember that neurodivergent experiences are valid",
            "Consider accommodations and support needs",
            "Connect with neurodivergent community resources",
            "Advocate for your needs and boundaries"
        ])
        
        return interventions
    
    def _calculate_neurodivergent_risk_score(
        self, 
        crisis_type: NeurodivergentCrisisType, 
        crisis_level: CrisisLevel
    ) -> float:
        """Calculate risk score for neurodivergent crisis"""
        base_scores = {
            NeurodivergentCrisisType.SENSORY_OVERLOAD: 0.6,
            NeurodivergentCrisisType.MELTDOWN: 0.8,
            NeurodivergentCrisisType.SHUTDOWN: 0.8,
            NeurodivergentCrisisType.EXECUTIVE_FUNCTION_FAILURE: 0.5,
            NeurodivergentCrisisType.SOCIAL_OVERWHELM: 0.6,
            NeurodivergentCrisisType.ROUTINE_DISRUPTION: 0.4,
            NeurodivergentCrisisType.MASKING_EXHAUSTION: 0.7,
            NeurodivergentCrisisType.BURNOUT: 0.7
        }
        
        level_multipliers = {
            CrisisLevel.LOW: 0.5,
            CrisisLevel.MEDIUM: 0.7,
            CrisisLevel.HIGH: 0.8,
            CrisisLevel.CRITICAL: 0.9,
            CrisisLevel.EMERGENCY: 1.0
        }
        
        base_score = base_scores.get(crisis_type, 0.5)
        multiplier = level_multipliers.get(crisis_level, 0.7)
        
        return min(base_score * multiplier, 1.0)
    
    async def _handle_neurodivergent_crisis(
        self, 
        risk_assessment: RiskAssessment, 
        crisis_type: NeurodivergentCrisisType
    ):
        """Handle neurodivergent crisis with appropriate interventions"""
        try:
            # Create crisis alert
            crisis_alert = CrisisAlert(
                alert_id=str(uuid.uuid4()),
                user_id=risk_assessment.user_id,
                session_id=risk_assessment.session_id,
                crisis_level=risk_assessment.crisis_level,
                alert_type=f"neurodivergent_{crisis_type.value}",
                message=f"Neurodivergent crisis detected: {crisis_type.value}",
                timestamp=datetime.utcnow(),
                requires_immediate_action=risk_assessment.requires_immediate_action,
                intervention_recommendations=risk_assessment.intervention_recommendations
            )
            
            # Store crisis alert
            self.crisis_alerts.append(crisis_alert)
            
            # Send A2A message to other agents
            await self.a2a_communicator.send_message(
                A2AMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    recipient_id="therapeutic-intervention-001",
                    message_type=MessageType.CRISIS_ALERT,
                    content_type=ContentType.JSON,
                    content=crisis_alert.dict(),
                    timestamp=datetime.utcnow()
                )
            )
            
            # Update statistics
            self.crisis_statistics.total_crises += 1
            self.crisis_statistics.neurodivergent_crises += 1
            
            print(f"Neurodivergent crisis alert created: {crisis_type.value} for user {risk_assessment.user_id}")
            
        except Exception as e:
            print(f"Error handling neurodivergent crisis: {e}")
