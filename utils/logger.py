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
        """Setup logging handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler for all logs
        file_handler = logging.FileHandler(self.logs_dir / "system.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
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
        
        self.logger.info(f"Agent Communication: {from_agent} -> {to_agent} ({message_type})")
    
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
        
        self.logger.info(f"Assessment Progress: User {user_id} - Stage: {stage}")
    
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
        
        self.logger.warning(f"CRISIS DETECTED: User {user_id} - Level: {crisis_level}")
    
    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log system events"""
        self.logger.info(f"System Event: {event_type} - {details}")
    
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

# Global logger instance
logger = MentalHealthLogger()
