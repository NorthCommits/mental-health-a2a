"""
Agent Communication Manager for inter-agent communication
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from .logger import logger
from shared.a2a_protocol.communication_layer import A2AMessage, MessageType, ContentType
from shared.a2a_protocol.agent_discovery import CapabilityType

class AgentCommunicationManager:
    """Manages communication between different agents"""
    
    def __init__(self, a2a_communicator, agent_discovery):
        self.a2a_communicator = a2a_communicator
        self.agent_discovery = agent_discovery
        self.active_agents = {}
        
    async def initialize_agent_communication(self):
        """Initialize communication with all available agents"""
        try:
            # Get all registered agents
            agents = self.agent_discovery.get_available_agents()
            
            for agent_id, agent_card in agents.items():
                self.active_agents[agent_id] = {
                    "agent_card": agent_card,
                    "last_communication": None,
                    "communication_count": 0
                }
            
            logger.log_system_event("agent_communication_init", {
                "agents_initialized": len(self.active_agents),
                "agent_ids": list(self.active_agents.keys())
            })
            
        except Exception as e:
            logger.log_system_event("agent_communication_init_error", {"error": str(e)})
    
    async def coordinate_assessment(self, 
                                  user_id: str, 
                                  user_input: str, 
                                  conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate assessment across multiple agents"""
        
        # 1. Primary Screening Agent - Initial assessment
        primary_screening_result = await self._communicate_with_agent(
            from_agent="conversation_manager",
            to_agent="primary-screening-001",
            message_type="assessment_request",
            content={
                "user_id": user_id,
                "user_input": user_input,
                "context": conversation_context,
                "assessment_type": "conversation_based"
            }
        )
        
        # 2. Crisis Detection Agent - Check for crisis indicators
        crisis_result = await self._communicate_with_agent(
            from_agent="conversation_manager",
            to_agent="crisis-detection-001",
            message_type="crisis_analysis",
            content={
                "user_id": user_id,
                "user_input": user_input,
                "context": conversation_context,
                "urgency_level": primary_screening_result.get("urgency_level", "low")
            }
        )
        
        # 3. Neurodevelopmental Assessment Agent - Check for neurodivergent traits
        neurodevelopmental_result = await self._communicate_with_agent(
            from_agent="conversation_manager",
            to_agent="neurodevelopmental-assessment-001",
            message_type="trait_analysis",
            content={
                "user_id": user_id,
                "user_input": user_input,
                "context": conversation_context,
                "assessment_focus": conversation_context.get("assessment_focus", {})
            }
        )
        
        # 4. Progress Analytics Agent - Track conversation patterns
        analytics_result = await self._communicate_with_agent(
            from_agent="conversation_manager",
            to_agent="progress-analytics-001",
            message_type="conversation_analysis",
            content={
                "user_id": user_id,
                "user_input": user_input,
                "context": conversation_context,
                "agent_responses": {
                    "primary_screening": primary_screening_result,
                    "crisis_detection": crisis_result,
                    "neurodevelopmental": neurodevelopmental_result
                }
            }
        )
        
        # Compile results
        coordinated_result = {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "primary_screening": primary_screening_result,
            "crisis_detection": crisis_result,
            "neurodevelopmental_assessment": neurodevelopmental_result,
            "analytics": analytics_result,
            "overall_urgency": self._determine_overall_urgency(crisis_result, primary_screening_result),
            "recommended_actions": self._compile_recommended_actions(
                primary_screening_result, crisis_result, neurodevelopmental_result
            )
        }
        
        # Log coordinated assessment
        logger.log_system_event("coordinated_assessment", {
            "user_id": user_id,
            "agents_involved": len(self.active_agents),
            "overall_urgency": coordinated_result["overall_urgency"]
        })
        
        return coordinated_result
    
    async def _communicate_with_agent(self, 
                                    from_agent: str, 
                                    to_agent: str, 
                                    message_type: str, 
                                    content: Dict[str, Any]) -> Dict[str, Any]:
        """Send message to specific agent and get response"""
        
        try:
            # Create A2A message
            message = A2AMessage(
                message_id=f"msg_{datetime.utcnow().timestamp()}",
                from_agent=from_agent,
                to_agent=to_agent,
                message_type=MessageType.REQUEST,
                content_type=ContentType.JSON,
                content=content,
                timestamp=datetime.utcnow(),
                priority="normal"
            )
            
            # Send message
            response = await self.a2a_communicator.send_message(message)
            
            # Log communication
            logger.log_agent_communication(
                from_agent=from_agent,
                to_agent=to_agent,
                message_type=message_type,
                content=content,
                user_id=content.get("user_id")
            )
            
            # Update agent stats
            if to_agent in self.active_agents:
                self.active_agents[to_agent]["last_communication"] = datetime.utcnow()
                self.active_agents[to_agent]["communication_count"] += 1
            
            return response.content if response else {"error": "No response from agent"}
            
        except Exception as e:
            logger.log_system_event("agent_communication_error", {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "error": str(e)
            })
            return {"error": str(e)}
    
    def _determine_overall_urgency(self, crisis_result: Dict[str, Any], primary_result: Dict[str, Any]) -> str:
        """Determine overall urgency level"""
        crisis_level = crisis_result.get("crisis_level", "none")
        primary_urgency = primary_result.get("urgency_level", "low")
        
        if crisis_level in ["high", "critical"]:
            return "high"
        elif crisis_level == "medium" or primary_urgency == "high":
            return "medium"
        else:
            return "low"
    
    def _compile_recommended_actions(self, 
                                   primary_result: Dict[str, Any], 
                                   crisis_result: Dict[str, Any], 
                                   neuro_result: Dict[str, Any]) -> List[str]:
        """Compile recommended actions from all agents"""
        actions = []
        
        # Add primary screening recommendations
        if "recommendations" in primary_result:
            actions.extend(primary_result["recommendations"])
        
        # Add crisis response actions
        if "immediate_actions" in crisis_result:
            actions.extend(crisis_result["immediate_actions"])
        
        # Add neurodevelopmental recommendations
        if "accommodations" in neuro_result:
            actions.extend(neuro_result["accommodations"])
        
        return list(set(actions))  # Remove duplicates
    
    async def generate_comprehensive_report(self, 
                                          user_id: str, 
                                          conversation_history: List[Dict[str, Any]], 
                                          agent_responses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report using all agent insights"""
        
        # Coordinate with Progress Analytics Agent for report generation
        report_result = await self._communicate_with_agent(
            from_agent="conversation_manager",
            to_agent="progress-analytics-001",
            message_type="generate_report",
            content={
                "user_id": user_id,
                "conversation_history": conversation_history,
                "agent_responses": agent_responses,
                "report_type": "comprehensive_assessment"
            }
        )
        
        # Add agent communication summary
        report_result["agent_communication_summary"] = {
            "agents_consulted": list(self.active_agents.keys()),
            "total_communications": sum(agent["communication_count"] for agent in self.active_agents.values()),
            "last_communication": max(
                agent["last_communication"] or datetime.min 
                for agent in self.active_agents.values()
            ).isoformat() if any(agent["last_communication"] for agent in self.active_agents.values()) else None
        }
        
        return report_result
    
    def get_agent_communication_stats(self) -> Dict[str, Any]:
        """Get statistics about agent communications"""
        return {
            "total_agents": len(self.active_agents),
            "active_agents": [
                {
                    "agent_id": agent_id,
                    "capabilities": agent_info["agent_card"].capabilities,
                    "communication_count": agent_info["communication_count"],
                    "last_communication": agent_info["last_communication"].isoformat() if agent_info["last_communication"] else None
                }
                for agent_id, agent_info in self.active_agents.items()
            ]
        }
