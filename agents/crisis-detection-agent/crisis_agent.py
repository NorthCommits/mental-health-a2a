"""
Crisis Detection Agent

Continuously monitors all interactions for high-risk content and provides
immediate crisis intervention when needed.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from shared.a2a_protocol.communication_layer import A2ACommunicator, A2AMessage, MessageType, ContentType
from shared.a2a_protocol.agent_discovery import AgentDiscovery, AgentCard, CapabilityType, InputModality, OutputFormat
from shared.a2a_protocol.task_management import TaskManager, Task, TaskPriority, TaskStatus
from shared.a2a_protocol.security import A2ASecurity, AgentRole, AccessLevel

from .risk_models import (
    RiskAssessment, CrisisLevel, RiskFactor, CrisisAlert, 
    SafetyPlan, CrisisIntervention, CrisisStatistics
)
from .nlp_models import CrisisNLPModel, SentimentAnalyzer, IntentClassifier


class CrisisAgentStatus(str, Enum):
    """Status of the crisis detection agent"""
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    CRISIS_RESPONSE = "crisis_response"
    INTERVENTION = "intervention"
    MAINTENANCE = "maintenance"


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
        
        # Register message handlers
        asyncio.create_task(self._register_message_handlers())
        
        # Register agent capabilities
        asyncio.create_task(self._register_agent_capabilities())
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
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
