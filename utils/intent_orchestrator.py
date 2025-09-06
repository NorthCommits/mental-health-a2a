"""
Intent-Based Agent Orchestrator

Analyzes user intent and intelligently routes requests to appropriate agents
based on capabilities, workload, and context.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from .logger import logger
from shared.a2a_protocol.communication_layer import A2AMessage, MessageType, ContentType, A2ACommunicator
from shared.a2a_protocol.agent_discovery import AgentDiscovery, CapabilityType, AgentCard


class IntentType(str, Enum):
    """Types of user intents"""
    CRISIS = "crisis"
    ASSESSMENT = "assessment"
    THERAPY = "therapy"
    NEURODEVELOPMENTAL = "neurodevelopmental"
    RESOURCE_REQUEST = "resource_request"
    GENERAL_SUPPORT = "general_support"
    PROGRESS_TRACKING = "progress_tracking"
    APPOINTMENT = "appointment"


class IntentOrchestrator:
    """
    Intent-Based Agent Orchestrator
    
    Analyzes user input to determine intent and routes to appropriate agents
    based on capabilities, current workload, and context.
    """
    
    def __init__(self, a2a_communicator: A2ACommunicator, agent_discovery: AgentDiscovery):
        self.a2a_communicator = a2a_communicator
        self.agent_discovery = agent_discovery
        self.agent_workloads: Dict[str, int] = {}
        self.agent_response_times: Dict[str, List[float]] = {}
        self.intent_patterns = self._initialize_intent_patterns()
        
    def _initialize_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize patterns for intent detection"""
        return {
            IntentType.CRISIS: [
                "suicide", "kill myself", "end my life", "hurt myself", "self harm",
                "crisis", "emergency", "urgent", "help me", "can't go on", "want to die"
            ],
            IntentType.ASSESSMENT: [
                "assessment", "evaluation", "screening", "test", "check", "evaluate",
                "how am i", "mental health", "depression", "anxiety", "symptoms"
            ],
            IntentType.THERAPY: [
                "therapy", "counseling", "cbt", "treatment", "coping", "strategies",
                "mindfulness", "meditation", "breathing", "relaxation"
            ],
            IntentType.NEURODEVELOPMENTAL: [
                "adhd", "autism", "asd", "focus", "attention", "hyperactive",
                "sensory", "social", "communication", "neurodivergent", "spectrum"
            ],
            IntentType.RESOURCE_REQUEST: [
                "resources", "help", "support", "information", "where to find",
                "recommendations", "suggestions", "guidance"
            ],
            IntentType.PROGRESS_TRACKING: [
                "progress", "tracking", "improvement", "better", "worse", "changes",
                "monitoring", "follow up", "check in"
            ],
            IntentType.APPOINTMENT: [
                "appointment", "schedule", "meeting", "session", "booking",
                "calendar", "when", "available"
            ]
        }
    
    async def analyze_intent(self, user_input: str, context: Dict[str, Any]) -> Tuple[IntentType, float, Dict[str, Any]]:
        """
        Analyze user input to determine intent
        
        Args:
            user_input: User's input text
            context: Conversation context
            
        Returns:
            Tuple of (intent_type, confidence, analysis_data)
        """
        user_lower = user_input.lower()
        intent_scores = {}
        
        # Score each intent type based on pattern matching
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            matches = []
            
            for pattern in patterns:
                if pattern in user_lower:
                    score += 1
                    matches.append(pattern)
            
            # Normalize score by number of patterns
            normalized_score = score / len(patterns) if patterns else 0
            intent_scores[intent_type] = {
                "score": normalized_score,
                "matches": matches
            }
        
        # Find the highest scoring intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1]["score"])
        intent_type = best_intent[0]
        confidence = best_intent[1]["score"]
        
        # Additional context analysis
        analysis_data = {
            "intent_scores": intent_scores,
            "context_indicators": self._analyze_context_indicators(context),
            "urgency_level": self._determine_urgency(user_input, context),
            "emotional_tone": self._analyze_emotional_tone(user_input)
        }
        
        # Log intent analysis
        logger.log_system_event("intent_analysis", {
            "user_input": user_input,
            "detected_intent": intent_type,
            "confidence": confidence,
            "analysis_data": analysis_data
        })
        
        return intent_type, confidence, analysis_data
    
    def _analyze_context_indicators(self, context: Dict[str, Any]) -> List[str]:
        """Analyze conversation context for additional indicators"""
        indicators = []
        
        if context.get("conversation_stage") == "crisis":
            indicators.append("crisis_context")
        
        if context.get("previous_crisis_indicators"):
            indicators.append("crisis_history")
        
        if context.get("neurodevelopmental_indicators"):
            indicators.append("neurodevelopmental_context")
        
        if context.get("assessment_in_progress"):
            indicators.append("assessment_context")
        
        return indicators
    
    def _determine_urgency(self, user_input: str, context: Dict[str, Any]) -> str:
        """Determine urgency level based on input and context"""
        crisis_keywords = ["suicide", "kill", "die", "hurt", "emergency", "urgent"]
        
        if any(keyword in user_input.lower() for keyword in crisis_keywords):
            return "high"
        
        if context.get("urgency_level") == "high":
            return "high"
        
        if context.get("crisis_indicators"):
            return "high"
        
        return "medium" if context.get("assessment_in_progress") else "low"
    
    def _analyze_emotional_tone(self, user_input: str) -> str:
        """Analyze emotional tone of user input"""
        # Simple keyword-based analysis
        positive_words = ["good", "better", "happy", "great", "improved", "well"]
        negative_words = ["bad", "worse", "sad", "terrible", "awful", "hopeless"]
        
        positive_count = sum(1 for word in positive_words if word in user_input.lower())
        negative_count = sum(1 for word in negative_words if word in user_input.lower())
        
        if negative_count > positive_count:
            return "negative"
        elif positive_count > negative_count:
            return "positive"
        else:
            return "neutral"
    
    async def route_to_agents(self, 
                            intent_type: IntentType, 
                            user_input: str, 
                            context: Dict[str, Any],
                            analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Route request to appropriate agents based on intent
        
        Args:
            intent_type: Detected intent type
            user_input: User's input
            context: Conversation context
            analysis_data: Analysis results
            
        Returns:
            List of agent responses
        """
        # Determine required capabilities based on intent
        required_capabilities = self._get_required_capabilities(intent_type, analysis_data)
        
        # Find suitable agents
        suitable_agents = await self._find_suitable_agents(required_capabilities, analysis_data)
        
        if not suitable_agents:
            logger.log_system_event("no_suitable_agents", {
                "intent_type": intent_type,
                "required_capabilities": required_capabilities
            })
            return []
        
        # Route to agents and collect responses
        agent_responses = []
        
        for agent_card in suitable_agents:
            try:
                response = await self._send_to_agent(agent_card, intent_type, user_input, context)
                if response:
                    agent_responses.append({
                        "agent_id": agent_card.agent_id,
                        "agent_name": agent_card.name,
                        "response": response,
                        "capabilities_used": self._get_used_capabilities(agent_card, intent_type)
                    })
                    
                    # Update workload tracking
                    self.agent_workloads[agent_card.agent_id] = self.agent_workloads.get(agent_card.agent_id, 0) + 1
                    
            except Exception as e:
                logger.log_system_event("agent_communication_error", {
                    "agent_id": agent_card.agent_id,
                    "error": str(e)
                })
        
        # Log orchestration results
        logger.log_system_event("agent_orchestration", {
            "intent_type": intent_type,
            "agents_contacted": len(agent_responses),
            "agent_ids": [r["agent_id"] for r in agent_responses]
        })
        
        return agent_responses
    
    def _get_required_capabilities(self, intent_type: IntentType, analysis_data: Dict[str, Any]) -> List[CapabilityType]:
        """Get required capabilities based on intent and analysis"""
        capabilities = []
        
        if intent_type == IntentType.CRISIS:
            capabilities.extend([CapabilityType.CRISIS_DETECTION, CapabilityType.EMERGENCY_RESPONSE])
        
        if intent_type == IntentType.ASSESSMENT:
            capabilities.append(CapabilityType.ASSESSMENT)
        
        if intent_type == IntentType.THERAPY:
            capabilities.append(CapabilityType.THERAPEUTIC_INTERVENTION)
        
        if intent_type == IntentType.NEURODEVELOPMENTAL:
            capabilities.append(CapabilityType.ASSESSMENT)  # Neurodevelopmental assessment
        
        if intent_type == IntentType.RESOURCE_REQUEST:
            capabilities.append(CapabilityType.CARE_COORDINATION)
        
        if intent_type == IntentType.PROGRESS_TRACKING:
            capabilities.append(CapabilityType.PROGRESS_ANALYTICS)
        
        if intent_type == IntentType.APPOINTMENT:
            capabilities.append(CapabilityType.CARE_COORDINATION)
        
        # Always include general assessment for context
        if CapabilityType.ASSESSMENT not in capabilities:
            capabilities.append(CapabilityType.ASSESSMENT)
        
        return capabilities
    
    async def _find_suitable_agents(self, 
                                  required_capabilities: List[CapabilityType], 
                                  analysis_data: Dict[str, Any]) -> List[AgentCard]:
        """Find agents suitable for the required capabilities"""
        suitable_agents = []
        
        for capability in required_capabilities:
            agents = await self.agent_discovery.discover_agents_by_capability(capability)
            
            for agent in agents:
                if agent not in suitable_agents:
                    # Check if agent is available and not overloaded
                    if (agent.availability_status == "available" and 
                        self.agent_workloads.get(agent.agent_id, 0) < agent.max_concurrent_tasks):
                        suitable_agents.append(agent)
        
        # Sort by response time and workload
        suitable_agents.sort(key=lambda x: (
            x.response_time_sla,
            self.agent_workloads.get(x.agent_id, 0)
        ))
        
        return suitable_agents
    
    async def _send_to_agent(self, 
                           agent_card: AgentCard, 
                           intent_type: IntentType, 
                           user_input: str, 
                           context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send request to a specific agent"""
        try:
            # Create task request message
            message = await self.a2a_communicator.create_task_request(
                recipient_agent_id=agent_card.agent_id,
                task_description=f"Process {intent_type.value} request: {user_input}",
                context={
                    "intent_type": intent_type.value,
                    "user_input": user_input,
                    "conversation_context": context,
                    "required_capabilities": [cap.value for cap in self._get_required_capabilities(intent_type, {})]
                },
                priority=10 if intent_type == IntentType.CRISIS else 5
            )
            
            # Send message
            success = await self.a2a_communicator.send_message(message)
            
            if success:
                # In a real implementation, we'd wait for the response
                # For now, we'll simulate a response
                return {
                    "status": "processed",
                    "agent_response": f"Agent {agent_card.name} processed {intent_type.value} request",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return None
                
        except Exception as e:
            logger.log_system_event("agent_communication_error", {
                "agent_id": agent_card.agent_id,
                "error": str(e)
            })
            return None
    
    def _get_used_capabilities(self, agent_card: AgentCard, intent_type: IntentType) -> List[str]:
        """Get capabilities used by agent for specific intent"""
        used_capabilities = []
        
        for capability in agent_card.capabilities:
            if capability.capability_type.value in [intent_type.value, "assessment", "general"]:
                used_capabilities.append(capability.capability_type.value)
        
        return used_capabilities
    
    async def orchestrate_conversation(self, 
                                    user_input: str, 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main orchestration method for conversation handling
        
        Args:
            user_input: User's input
            context: Conversation context
            
        Returns:
            Orchestration results with agent responses
        """
        # Analyze intent
        intent_type, confidence, analysis_data = await self.analyze_intent(user_input, context)
        
        # Route to appropriate agents
        agent_responses = await self.route_to_agents(intent_type, user_input, context, analysis_data)
        
        # Synthesize responses
        synthesized_response = self._synthesize_responses(agent_responses, intent_type, analysis_data)
        
        return {
            "intent_type": intent_type,
            "confidence": confidence,
            "agent_responses": agent_responses,
            "synthesized_response": synthesized_response,
            "analysis_data": analysis_data,
            "orchestration_timestamp": datetime.utcnow().isoformat()
        }
    
    def _synthesize_responses(self, 
                            agent_responses: List[Dict[str, Any]], 
                            intent_type: IntentType,
                            analysis_data: Dict[str, Any]) -> str:
        """Synthesize responses from multiple agents into coherent response"""
        if not agent_responses:
            return "I understand you're reaching out for support. Let me help you with that."
        
        # For crisis situations, prioritize crisis detection agent
        if intent_type == IntentType.CRISIS:
            crisis_agents = [r for r in agent_responses if "crisis" in r["agent_name"].lower()]
            if crisis_agents:
                return crisis_agents[0]["response"]
        
        # For other intents, combine responses
        if len(agent_responses) == 1:
            return agent_responses[0]["response"]
        
        # Multiple agents - create synthesized response
        synthesized = f"Based on our analysis, here's what I can help you with:\n\n"
        
        for i, response in enumerate(agent_responses, 1):
            synthesized += f"{i}. {response['agent_name']}: {response['response']}\n"
        
        return synthesized
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current status of all agents"""
        return {
            "agent_workloads": self.agent_workloads,
            "agent_response_times": self.agent_response_times,
            "total_agents": len(self.agent_workloads),
            "available_agents": len([w for w in self.agent_workloads.values() if w < 5])
        }
