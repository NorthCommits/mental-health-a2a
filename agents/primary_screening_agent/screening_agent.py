"""
Primary Screening Agent

The main agent responsible for conducting initial mental health assessments
using validated clinical tools and multi-modal input processing.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from shared.a2a_protocol.communication_layer import A2ACommunicator, A2AMessage, MessageType, ContentType
from shared.a2a_protocol.agent_discovery import AgentDiscovery, AgentCard, CapabilityType, InputModality, OutputFormat
from shared.a2a_protocol.task_management import TaskManager, Task, TaskPriority, TaskStatus
from shared.a2a_protocol.security import A2ASecurity, AgentRole, AccessLevel

from .assessment_models import (
    PHQ9Assessment, GAD7Assessment, AssessmentResult, 
    AssessmentType, SeverityLevel, RiskLevel, AssessmentResponse
)
from .input_processors import TextProcessor, AudioProcessor, ImageProcessor, DocumentProcessor


class ScreeningAgentStatus(str, Enum):
    """Status of the screening agent"""
    IDLE = "idle"
    PROCESSING = "processing"
    ASSESSING = "assessing"
    CRISIS_MODE = "crisis_mode"
    MAINTENANCE = "maintenance"


class PrimaryScreeningAgent:
    """
    Primary Screening Agent for mental health assessment
    
    This agent serves as the entry point for all user interactions and conducts
    initial mental health assessments using validated clinical tools.
    """
    
    def __init__(
        self,
        agent_id: str,
        openai_api_key: str,
        a2a_communicator: A2ACommunicator,
        agent_discovery: AgentDiscovery,
        task_manager: TaskManager,
        security: A2ASecurity
    ):
        self.agent_id = agent_id
        self.openai_api_key = openai_api_key
        self.a2a_communicator = a2a_communicator
        self.agent_discovery = agent_discovery
        self.task_manager = task_manager
        self.security = security
        
        # Initialize input processors
        self.text_processor = TextProcessor(openai_api_key)
        self.audio_processor = AudioProcessor()
        self.image_processor = ImageProcessor()
        self.document_processor = DocumentProcessor()
        
        # Agent state
        self.status = ScreeningAgentStatus.IDLE
        self.current_assessments: Dict[str, Dict[str, Any]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
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
            MessageType.CRISIS_ALERT,
            self._handle_crisis_alert
        )
        await self.a2a_communicator.register_message_handler(
            MessageType.COLLABORATION,
            self._handle_collaboration
        )
    
    async def _register_agent_capabilities(self):
        """Register agent capabilities with the discovery service"""
        agent_card = AgentCard(
            agent_id=self.agent_id,
            name="Primary Screening Agent",
            description="Conducts initial mental health assessments using validated clinical tools",
            version="1.0.0",
            capabilities=[
                {
                    "capability_type": CapabilityType.SCREENING,
                    "description": "Mental health screening and assessment",
                    "input_modalities": [
                        InputModality.TEXT,
                        InputModality.AUDIO,
                        InputModality.IMAGE,
                        InputModality.DOCUMENT
                    ],
                    "output_formats": [
                        OutputFormat.ASSESSMENT_SCORE,
                        OutputFormat.STRUCTURED_DATA,
                        OutputFormat.CRISIS_ALERT
                    ],
                    "parameters": {
                        "supported_assessments": ["PHQ9", "GAD7", "PHQ15", "GADQIV"],
                        "response_time_sla": 5.0,
                        "privacy_level": "high"
                    }
                }
            ],
            contact_endpoint=f"http://localhost:8000/agents/{self.agent_id}",
            compliance_certifications=["hipaa", "fda_guidelines"]
        )
        
        await self.agent_discovery.register_agent(agent_card)
    
    async def start_screening_session(
        self,
        user_id: str,
        input_data: Dict[str, Any],
        session_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new screening session for a user
        
        Args:
            user_id: ID of the user
            input_data: Input data from the user
            session_context: Additional context for the session
            
        Returns:
            Session ID for tracking the screening process
        """
        session_id = str(uuid.uuid4())
        
        # Initialize session
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "started_at": datetime.utcnow(),
            "status": "initializing",
            "input_data": input_data,
            "context": session_context or {},
            "assessments": [],
            "current_step": "input_processing"
        }
        
        # Process input data
        await self._process_user_input(session_id, input_data)
        
        return session_id
    
    async def _process_user_input(self, session_id: str, input_data: Dict[str, Any]):
        """Process multi-modal user input"""
        try:
            session = self.active_sessions[session_id]
            session["status"] = "processing_input"
            
            processed_inputs = []
            
            # Process text input
            if "text" in input_data:
                text_input = await self.text_processor.process_text(
                    input_data["text"],
                    session["context"]
                )
                processed_inputs.append(text_input)
            
            # Process audio input
            if "audio" in input_data:
                audio_data = base64.b64decode(input_data["audio"])
                audio_input = await self.audio_processor.process_audio(audio_data)
                processed_inputs.append(audio_input)
            
            # Process image input
            if "image" in input_data:
                image_data = base64.b64decode(input_data["image"])
                image_input = await self.image_processor.process_image(image_data)
                processed_inputs.append(image_input)
            
            # Process document input
            if "document" in input_data:
                doc_data = base64.b64decode(input_data["document"])
                doc_input = await self.document_processor.process_document(
                    doc_data, 
                    input_data.get("filename", "document")
                )
                processed_inputs.append(doc_input)
            
            # Store processed inputs
            session["processed_inputs"] = processed_inputs
            
            # Determine appropriate assessments
            assessments_to_run = await self._determine_assessments(processed_inputs)
            session["assessments_to_run"] = assessments_to_run
            session["current_step"] = "assessment_selection"
            
            # Start assessments
            await self._run_assessments(session_id, assessments_to_run)
            
        except Exception as e:
            print(f"Error processing user input: {e}")
            session["status"] = "error"
            session["error"] = str(e)
    
    async def _determine_assessments(self, processed_inputs: List) -> List[AssessmentType]:
        """Determine which assessments to run based on processed input"""
        assessments = []
        
        # Analyze processed inputs for mental health indicators
        depression_indicators = 0
        anxiety_indicators = 0
        crisis_indicators = 0
        
        for processed_input in processed_inputs:
            if processed_input.input_type == "text":
                indicators = processed_input.metadata.get("mental_health_indicators", {})
                depression_indicators += indicators.get("depression_indicators", 0)
                anxiety_indicators += indicators.get("anxiety_indicators", 0)
                crisis_indicators += indicators.get("crisis_indicators", 0)
        
        # Always run PHQ-9 and GAD-7 for comprehensive screening
        assessments.append(AssessmentType.PHQ9)
        assessments.append(AssessmentType.GAD7)
        
        # Add additional assessments based on indicators
        if depression_indicators > 2:
            assessments.append(AssessmentType.PHQ15)  # Somatic symptoms
        
        if anxiety_indicators > 2:
            assessments.append(AssessmentType.GADQIV)  # More detailed anxiety assessment
        
        if crisis_indicators > 0:
            # Crisis situation - prioritize immediate assessment
            assessments = [AssessmentType.PHQ9, AssessmentType.GAD7]
        
        return assessments
    
    async def _run_assessments(self, session_id: str, assessments: List[AssessmentType]):
        """Run the determined assessments"""
        try:
            session = self.active_sessions[session_id]
            session["status"] = "running_assessments"
            session["current_step"] = "assessment_execution"
            
            assessment_results = []
            
            for assessment_type in assessments:
                if assessment_type == AssessmentType.PHQ9:
                    result = await self._run_phq9_assessment(session_id)
                elif assessment_type == AssessmentType.GAD7:
                    result = await self._run_gad7_assessment(session_id)
                else:
                    # For other assessments, create a basic result
                    result = await self._run_basic_assessment(session_id, assessment_type)
                
                if result:
                    assessment_results.append(result)
            
            session["assessment_results"] = assessment_results
            session["current_step"] = "result_analysis"
            
            # Analyze results and determine next steps
            await self._analyze_assessment_results(session_id, assessment_results)
            
        except Exception as e:
            print(f"Error running assessments: {e}")
            session["status"] = "error"
            session["error"] = str(e)
    
    async def _run_phq9_assessment(self, session_id: str) -> Optional[AssessmentResult]:
        """Run PHQ-9 depression assessment"""
        try:
            session = self.active_sessions[session_id]
            
            # Extract depression-related information from processed inputs
            depression_data = await self._extract_depression_indicators(session["processed_inputs"])
            
            # Create PHQ-9 responses based on extracted data
            responses = await self._generate_phq9_responses(depression_data)
            
            # Create assessment
            assessment = PHQ9Assessment.create_from_responses(
                assessment_id=str(uuid.uuid4()),
                user_id=session["user_id"],
                responses=responses
            )
            
            # Convert to AssessmentResult
            result = AssessmentResult(
                result_id=str(uuid.uuid4()),
                user_id=session["user_id"],
                assessment_type=AssessmentType.PHQ9,
                total_score=assessment.total_score,
                severity_level=assessment.severity_level,
                risk_level=assessment.risk_level,
                completed_at=assessment.completed_at,
                requires_follow_up=assessment.requires_follow_up,
                recommended_actions=assessment.recommended_actions,
                clinical_notes=f"PHQ-9 assessment completed with {assessment.severity_level.value} severity"
            )
            
            return result
            
        except Exception as e:
            print(f"Error running PHQ-9 assessment: {e}")
            return None
    
    async def _run_gad7_assessment(self, session_id: str) -> Optional[AssessmentResult]:
        """Run GAD-7 anxiety assessment"""
        try:
            session = self.active_sessions[session_id]
            
            # Extract anxiety-related information from processed inputs
            anxiety_data = await self._extract_anxiety_indicators(session["processed_inputs"])
            
            # Create GAD-7 responses based on extracted data
            responses = await self._generate_gad7_responses(anxiety_data)
            
            # Create assessment
            assessment = GAD7Assessment.create_from_responses(
                assessment_id=str(uuid.uuid4()),
                user_id=session["user_id"],
                responses=responses
            )
            
            # Convert to AssessmentResult
            result = AssessmentResult(
                result_id=str(uuid.uuid4()),
                user_id=session["user_id"],
                assessment_type=AssessmentType.GAD7,
                total_score=assessment.total_score,
                severity_level=assessment.severity_level,
                risk_level=assessment.risk_level,
                completed_at=assessment.completed_at,
                requires_follow_up=assessment.requires_follow_up,
                recommended_actions=assessment.recommended_actions,
                clinical_notes=f"GAD-7 assessment completed with {assessment.severity_level.value} severity"
            )
            
            return result
            
        except Exception as e:
            print(f"Error running GAD-7 assessment: {e}")
            return None
    
    async def _run_basic_assessment(self, session_id: str, assessment_type: AssessmentType) -> Optional[AssessmentResult]:
        """Run a basic assessment for unsupported types"""
        # This would be implemented for other assessment types
        return None
    
    async def _extract_depression_indicators(self, processed_inputs: List) -> Dict[str, Any]:
        """Extract depression indicators from processed inputs"""
        depression_indicators = {
            "interest_pleasure": 0,
            "mood": 0,
            "sleep": 0,
            "energy": 0,
            "appetite": 0,
            "self_worth": 0,
            "concentration": 0,
            "psychomotor": 0,
            "suicidal_ideation": 0
        }
        
        for processed_input in processed_inputs:
            if processed_input.input_type == "text":
                assessment_data = processed_input.metadata.get("assessment_data", {})
                # Map assessment data to PHQ-9 indicators
                # This would be more sophisticated in a real implementation
                if "mood_score" in assessment_data:
                    depression_indicators["mood"] = max(0, min(3, int(assessment_data["mood_score"] / 10 * 3)))
                if "energy_level" in assessment_data:
                    depression_indicators["energy"] = max(0, min(3, int(assessment_data["energy_level"] / 10 * 3)))
                # ... map other indicators
        
        return depression_indicators
    
    async def _extract_anxiety_indicators(self, processed_inputs: List) -> Dict[str, Any]:
        """Extract anxiety indicators from processed inputs"""
        anxiety_indicators = {
            "nervousness": 0,
            "worry_control": 0,
            "excessive_worry": 0,
            "trouble_relaxing": 0,
            "restlessness": 0,
            "irritability": 0,
            "fear": 0
        }
        
        for processed_input in processed_inputs:
            if processed_input.input_type == "text":
                assessment_data = processed_input.metadata.get("assessment_data", {})
                # Map assessment data to GAD-7 indicators
                if "anxiety_level" in assessment_data:
                    anxiety_indicators["nervousness"] = max(0, min(3, int(assessment_data["anxiety_level"] / 10 * 3)))
                # ... map other indicators
        
        return anxiety_indicators
    
    async def _generate_phq9_responses(self, depression_data: Dict[str, Any]) -> List:
        """Generate PHQ-9 responses from depression indicators"""
        # This would use AI to generate appropriate responses based on indicators
        # For now, return placeholder responses
        responses = []
        questions = PHQ9Assessment.get_questions()
        
        for question in questions:
            # Map depression indicators to question responses
            if "interest_pleasure" in question.question_id:
                score = depression_data.get("interest_pleasure", 0)
            elif "mood" in question.question_id:
                score = depression_data.get("mood", 0)
            # ... map other questions
            else:
                score = 0
            
            responses.append(AssessmentResponse(
                question_id=question.question_id,
                response=score,
                timestamp=datetime.utcnow()
            ))
        
        return responses
    
    async def _generate_gad7_responses(self, anxiety_data: Dict[str, Any]) -> List:
        """Generate GAD-7 responses from anxiety indicators"""
        # This would use AI to generate appropriate responses based on indicators
        # For now, return placeholder responses
        responses = []
        questions = GAD7Assessment.get_questions()
        
        for question in questions:
            # Map anxiety indicators to question responses
            if "nervousness" in question.question_id:
                score = anxiety_data.get("nervousness", 0)
            elif "worry_control" in question.question_id:
                score = anxiety_data.get("worry_control", 0)
            # ... map other questions
            else:
                score = 0
            
            responses.append(AssessmentResponse(
                question_id=question.question_id,
                response=score,
                timestamp=datetime.utcnow()
            ))
        
        return responses
    
    async def _analyze_assessment_results(self, session_id: str, results: List[AssessmentResult]):
        """Analyze assessment results and determine next steps"""
        try:
            session = self.active_sessions[session_id]
            
            # Check for crisis situations
            crisis_results = [r for r in results if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
            
            if crisis_results:
                await self._handle_crisis_situation(session_id, crisis_results)
            else:
                # Normal processing - determine care level needed
                await self._determine_care_level(session_id, results)
            
            session["status"] = "completed"
            session["current_step"] = "results_ready"
            
        except Exception as e:
            print(f"Error analyzing assessment results: {e}")
            session["status"] = "error"
            session["error"] = str(e)
    
    async def _handle_crisis_situation(self, session_id: str, crisis_results: List[AssessmentResult]):
        """Handle crisis situations requiring immediate attention"""
        try:
            session = self.active_sessions[session_id]
            session["status"] = "crisis_mode"
            
            # Send crisis alert to Crisis Detection Agent
            crisis_agent = await self.agent_discovery.discover_agents_by_capability(
                CapabilityType.CRISIS_DETECTION
            )
            
            if crisis_agent:
                crisis_data = {
                    "session_id": session_id,
                    "user_id": session["user_id"],
                    "crisis_results": [r.to_crisis_alert_data() for r in crisis_results],
                    "timestamp": datetime.utcnow().isoformat(),
                    "priority": "critical"
                }
                
                crisis_message = await self.a2a_communicator.create_crisis_alert(
                    crisis_agent[0].agent_id,
                    crisis_data
                )
                
                await self.a2a_communicator.send_message(crisis_message)
            
            # Store crisis information
            session["crisis_detected"] = True
            session["crisis_results"] = crisis_results
            
        except Exception as e:
            print(f"Error handling crisis situation: {e}")
    
    async def _determine_care_level(self, session_id: str, results: List[AssessmentResult]):
        """Determine appropriate care level based on assessment results"""
        try:
            session = self.active_sessions[session_id]
            
            # Analyze overall risk level
            if results:
                max_risk = max([r.risk_level for r in results], key=lambda x: ["low", "medium", "high", "critical"].index(x.value))
            else:
                max_risk = RiskLevel.LOW  # Default to low risk if no results
            
            if max_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                # High risk - refer to human clinician
                session["recommended_care"] = "immediate_clinical_evaluation"
                session["priority"] = "high"
            elif max_risk == RiskLevel.MEDIUM:
                # Medium risk - consider therapy or counseling
                session["recommended_care"] = "therapy_referral"
                session["priority"] = "medium"
            else:
                # Low risk - self-monitoring or basic support
                session["recommended_care"] = "self_monitoring"
                session["priority"] = "low"
            
            # Create task for Care Coordination Agent
            care_agent = await self.agent_discovery.discover_agents_by_capability(
                CapabilityType.CARE_COORDINATION
            )
            
            if care_agent:
                task = await self.task_manager.create_task(
                    title=f"Care coordination for user {session['user_id']}",
                    description=f"Coordinate care based on screening results: {session['recommended_care']}",
                    task_type="care_coordination",
                    created_by=self.agent_id,
                    assigned_to=care_agent[0].agent_id,
                    priority=TaskPriority.HIGH if session["priority"] == "high" else TaskPriority.NORMAL,
                    input_data={
                        "session_id": session_id,
                        "user_id": session["user_id"],
                        "assessment_results": [r.model_dump() for r in results],
                        "recommended_care": session["recommended_care"],
                        "priority": session["priority"]
                    }
                )
            
        except Exception as e:
            print(f"Error determining care level: {e}")
    
    async def _handle_task_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle incoming task requests"""
        try:
            # Process the task request
            task_data = message.parts[0].content if message.parts else ""
            
            # Create a task in the task manager
            task = await self.task_manager.create_task(
                title=f"Task from {message.sender_agent_id}",
                description=task_data,
                task_type="screening_request",
                created_by=message.sender_agent_id,
                assigned_to=self.agent_id,
                priority=TaskPriority.NORMAL,
                input_data={"message": message.model_dump()}
            )
            
            return {
                "task_id": task.task_id,
                "status": "accepted",
                "message": "Task accepted and queued for processing"
            }
            
        except Exception as e:
            print(f"Error handling task request: {e}")
            return {"error": str(e)}
    
    async def _handle_crisis_alert(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle crisis alerts from other agents"""
        try:
            # Process crisis alert
            crisis_data = json.loads(message.parts[0].content) if message.parts else {}
            
            # Update agent status to crisis mode
            self.status = ScreeningAgentStatus.CRISIS_MODE
            
            # Handle the crisis situation
            # This would involve immediate response protocols
            
            return {
                "status": "crisis_acknowledged",
                "message": "Crisis alert received and being processed"
            }
            
        except Exception as e:
            print(f"Error handling crisis alert: {e}")
            return {"error": str(e)}
    
    async def _handle_collaboration(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle collaboration messages from other agents"""
        try:
            # Process collaboration data
            collaboration_data = json.loads(message.parts[0].content) if message.parts else {}
            
            # Handle the collaboration request
            # This would involve sharing relevant screening data or coordinating care
            
            return {
                "status": "collaboration_acknowledged",
                "message": "Collaboration request received and being processed"
            }
            
        except Exception as e:
            print(f"Error handling collaboration: {e}")
            return {"error": str(e)}
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a screening session"""
        return self.active_sessions.get(session_id)
    
    async def get_assessment_results(self, session_id: str) -> Optional[List[AssessmentResult]]:
        """Get assessment results for a completed session"""
        session = self.active_sessions.get(session_id)
        if session and session["status"] == "completed":
            return session.get("assessment_results", [])
        return None
