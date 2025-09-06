"""
Professional logging system for Mental Health A2A System
"""
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class MentalHealthLogger:
    """Structured logger for mental health assessment system"""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("mental_health_a2a")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup handlers
        self._setup_handlers()
        
        # Agent communication log
        self.agent_log_file = self.logs_dir / "agent_communications.jsonl"
        
    def _setup_handlers(self):
        """Setup logging handlers with colored output"""
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler for all logs
        file_handler = logging.FileHandler(self.logs_dir / "system.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Colored formatter for console
        class ColoredFormatter(logging.Formatter):
            """Custom formatter with colors"""
            
            # Color codes
            COLORS = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[35m', # Magenta
                'RESET': '\033[0m'      # Reset
            }
            
            def format(self, record):
                # Add color to levelname
                levelname = record.levelname
                color = self.COLORS.get(levelname, self.COLORS['RESET'])
                record.levelname = f"{color}{levelname}{self.COLORS['RESET']}"
                
                # Add status code if available
                if hasattr(record, 'status_code'):
                    record.msg = f"[{record.status_code}] {record.msg}"
                
                return super().format(record)
        
        # Formatters
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def log_agent_communication(self, 
                              from_agent: str, 
                              to_agent: str, 
                              message_type: str, 
                              content: Dict[str, Any],
                              user_id: Optional[str] = None):
        """Log agent-to-agent communication"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": message_type,
            "user_id": user_id,
            "content": content
        }
        
        # Write to JSONL file
        with open(self.agent_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Human-readable message
        agent_names = {
            "primary-screening-001": "Primary Screening Agent",
            "crisis-detection-001": "Crisis Detection Agent", 
            "therapeutic-intervention-001": "Therapeutic Intervention Agent",
            "care-coordination-001": "Care Coordination Agent",
            "progress-analytics-001": "Progress Analytics Agent",
            "neurodevelopmental-assessment-001": "Neurodevelopmental Assessment Agent"
        }
        
        from_name = agent_names.get(from_agent, from_agent)
        to_name = agent_names.get(to_agent, to_agent)
        
        self.logger.info(f"[COMMUNICATION] {from_name} is communicating with {to_name} about: {message_type}")
    
    def log_assessment_progress(self, 
                              user_id: str, 
                              stage: str, 
                              user_input: str, 
                              ai_response: str,
                              insights: Optional[Dict[str, Any]] = None):
        """Log assessment conversation progress"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "stage": stage,
            "user_input": user_input,
            "ai_response": ai_response,
            "insights": insights or {}
        }
        
        assessment_log_file = self.logs_dir / f"assessment_{user_id}.jsonl"
        with open(assessment_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Human-readable stage names
        stage_names = {
            "introduction": "Conversation Introduction",
            "assessment": "Mental Health Assessment", 
            "exploration": "Exploring Concerns",
            "analysis": "Analyzing Responses",
            "completion": "Assessment Complete"
        }
        
        stage_name = stage_names.get(stage, stage.title())
        self.logger.info(f"[ASSESSMENT] Assessment Progress: {stage_name} - User: {user_id}")
    
    def log_crisis_detection(self, 
                           user_id: str, 
                           crisis_level: str, 
                           triggers: list,
                           response_actions: list):
        """Log crisis detection events"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "crisis_level": crisis_level,
            "triggers": triggers,
            "response_actions": response_actions
        }
        
        crisis_log_file = self.logs_dir / "crisis_events.jsonl"
        with open(crisis_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Human-readable crisis levels
        crisis_levels = {
            "low": "Low Risk",
            "medium": "Moderate Risk", 
            "high": "High Risk",
            "critical": "Critical Risk"
        }
        
        level_name = crisis_levels.get(crisis_level.lower(), crisis_level.title())
        self.logger.warning(f"[CRISIS] CRISIS ALERT: {level_name} detected for User {user_id}")
    
    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log system events with human-readable messages and status codes"""
        
        # Status codes for different event types
        status_codes = {
            "a2a_components_initialized": "200",
            "adaptive_systems_initialized": "200", 
            "intent_analysis": "200",
            "intent_orchestration": "200",
            "agents_triggered": "200",
            "crisis_detected": "300",
            "assessment_completed": "200",
            "report_generated": "200",
            "agent_coordination": "200",
            "comprehensive_report_generated": "200",
            "a2a_components_reinitialized": "201",
            "no_suitable_agents": "404",
            "orchestration_error": "500",
            "openai_key_missing": "503",
            "a2a_components_missing": "503",
            "adaptive_conversation_error": "500",
            "agent_communication_error": "500",
            "report_generation_error": "500"
        }
        
        # Human-readable event messages
        event_messages = {
            "a2a_components_initialized": "[SYSTEM] A2A Communication System: All components successfully initialized and ready",
            "adaptive_systems_initialized": "[AI] AI Conversation System: Adaptive conversation manager and intent orchestrator are now active",
            "intent_analysis": "[ANALYSIS] Intent Analysis: Analyzing user input to determine appropriate response",
            "intent_orchestration": "[ORCHESTRATION] Agent Orchestration: Routing user request to appropriate AI agents",
            "no_suitable_agents": "[WARNING] Agent Selection: No suitable agents found for this request type",
            "agents_triggered": "[ACTIVATION] Agent Activation: Multiple agents are now working together on this request",
            "crisis_detected": "[CRISIS] Crisis Detection: Potential crisis indicators identified",
            "assessment_completed": "[SUCCESS] Assessment Complete: Mental health assessment has been completed",
            "report_generated": "[REPORT] Report Generated: Comprehensive mental health report created",
            "agent_coordination": "[COORDINATION] Agent Coordination: Multiple agents are collaborating on this case",
            "orchestration_error": "[ERROR] Orchestration Error: There was an issue with agent coordination",
            "openai_key_missing": "[CONFIG] Configuration Error: OpenAI API key not found",
            "a2a_components_missing": "[WARNING] System Warning: A2A components need to be reinitialized",
            "a2a_components_reinitialized": "[RECOVERY] System Recovery: A2A components have been successfully reinitialized",
            "adaptive_conversation_error": "[ERROR] Conversation Error: Issue with adaptive conversation system",
            "comprehensive_report_generated": "[REPORT] Report Complete: Comprehensive mental health assessment report generated",
            "agent_communication_error": "[ERROR] Agent Communication Error: Failed to communicate with agent",
            "report_generation_error": "[ERROR] Report Generation Error: Failed to generate mental health report"
        }
        
        # Get status code and message
        status_code = status_codes.get(event_type, "200")
        message = event_messages.get(event_type, f"System Event: {event_type}")
        
        # Add context from details if relevant
        if details:
            context_parts = []
            
            # Check for specific details to highlight
            if "user_id" in details:
                context_parts.append(f"User: {details['user_id']}")
            
            if "agents_contacted" in details:
                context_parts.append(f"Agents: {details['agents_contacted']}")
                
            if "confidence" in details:
                confidence = details["confidence"]
                # Ensure confidence is a float for comparison
                try:
                    confidence_float = float(confidence) if isinstance(confidence, str) else confidence
                    if confidence_float > 0.7:
                        context_parts.append(f"Confidence: High ({confidence_float:.1%})")
                    elif confidence_float > 0.4:
                        context_parts.append(f"Confidence: Medium ({confidence_float:.1%})")
                    else:
                        context_parts.append(f"Confidence: Low ({confidence_float:.1%})")
                except (ValueError, TypeError):
                    context_parts.append(f"Confidence: {confidence}")
            
            if "urgency_level" in details:
                urgency = details["urgency_level"]
                urgency_indicator = {"low": "[LOW]", "medium": "[MEDIUM]", "high": "[HIGH]"}.get(urgency, "[UNKNOWN]")
                context_parts.append(f"Urgency: {urgency_indicator} {urgency.title()}")
            
            if "error" in details:
                context_parts.append(f"Error: {details['error']}")
            
            # Add context if we have any
            if context_parts:
                message += f" | {' | '.join(context_parts)}"
        
        # Log with appropriate level based on status code
        if status_code.startswith("5"):
            # Server errors - use ERROR level
            self.logger.error(message, extra={"status_code": status_code})
        elif status_code.startswith("4"):
            # Client errors - use WARNING level
            self.logger.warning(message, extra={"status_code": status_code})
        elif status_code.startswith("3"):
            # Redirection/special cases - use WARNING level
            self.logger.warning(message, extra={"status_code": status_code})
        else:
            # Success - use INFO level
            self.logger.info(message, extra={"status_code": status_code})
    
    def get_agent_communications(self, limit: int = 100) -> list:
        """Get recent agent communications"""
        if not self.agent_log_file.exists():
            return []
        
        communications = []
        with open(self.agent_log_file, "r") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    communications.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        return communications
    
    def log_error(self, error_type: str, message: str, status_code: str = "500", details: Optional[Dict[str, Any]] = None):
        """Log errors with specific status codes and red highlighting"""
        error_messages = {
            "agent_initialization_failed": f"[ERROR] Agent Initialization Failed: {message}",
            "communication_failed": f"[ERROR] Communication Failed: {message}",
            "orchestration_failed": f"[ERROR] Orchestration Failed: {message}",
            "assessment_failed": f"[ERROR] Assessment Failed: {message}",
            "report_generation_failed": f"[ERROR] Report Generation Failed: {message}",
            "crisis_detection_failed": f"[ERROR] Crisis Detection Failed: {message}",
            "configuration_error": f"[ERROR] Configuration Error: {message}",
            "api_error": f"[ERROR] API Error: {message}",
            "database_error": f"[ERROR] Database Error: {message}",
            "file_system_error": f"[ERROR] File System Error: {message}"
        }
        
        error_message = error_messages.get(error_type, f"[ERROR] Error: {message}")
        
        if details:
            context_parts = []
            for key, value in details.items():
                context_parts.append(f"{key}: {value}")
            error_message += f" | {' | '.join(context_parts)}"
        
        self.logger.error(error_message, extra={"status_code": status_code})
    
    def log_success(self, success_type: str, message: str, status_code: str = "200", details: Optional[Dict[str, Any]] = None):
        """Log successful operations with green highlighting"""
        success_messages = {
            "agent_initialized": f"[SUCCESS] Agent Initialized: {message}",
            "communication_successful": f"[SUCCESS] Communication Successful: {message}",
            "orchestration_successful": f"[SUCCESS] Orchestration Successful: {message}",
            "assessment_completed": f"[SUCCESS] Assessment Completed: {message}",
            "report_generated": f"[SUCCESS] Report Generated: {message}",
            "crisis_handled": f"[SUCCESS] Crisis Handled: {message}",
            "configuration_loaded": f"[SUCCESS] Configuration Loaded: {message}",
            "api_success": f"[SUCCESS] API Success: {message}",
            "database_operation": f"[SUCCESS] Database Operation: {message}",
            "file_operation": f"[SUCCESS] File Operation: {message}"
        }
        
        success_message = success_messages.get(success_type, f"[SUCCESS] Success: {message}")
        
        if details:
            context_parts = []
            for key, value in details.items():
                context_parts.append(f"{key}: {value}")
            success_message += f" | {' | '.join(context_parts)}"
        
        self.logger.info(success_message, extra={"status_code": status_code})

# Global logger instance
logger = MentalHealthLogger()
