"""
Configuration settings for the Mental Health A2A Agent Ecosystem
"""

import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Mental Health A2A Agent Ecosystem"
    api_version: str = "1.0.0"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    
    # Database Configuration
    database_url: str = "sqlite:///./mental_health.db"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    
    # A2A Protocol Configuration
    a2a_base_url: str = "http://localhost:8000"
    agent_registry_endpoint: str = "http://localhost:8000/registry"
    
    # Agent Configuration
    primary_screening_agent_id: str = "primary-screening-001"
    crisis_detection_agent_id: str = "crisis-detection-001"
    therapeutic_intervention_agent_id: str = "therapeutic-intervention-001"
    care_coordination_agent_id: str = "care-coordination-001"
    progress_analytics_agent_id: str = "progress-analytics-001"
    
    # Crisis Detection Configuration
    crisis_detection_threshold: float = 0.2
    crisis_response_timeout: int = 5  # seconds
    
    # Assessment Configuration
    phq9_thresholds: dict = {
        "minimal": 0,
        "mild": 5,
        "moderate": 10,
        "moderately_severe": 15,
        "severe": 20
    }
    
    gad7_thresholds: dict = {
        "minimal": 0,
        "mild": 5,
        "moderate": 10,
        "severe": 15
    }
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # HIPAA Compliance
    data_retention_days: int = 2555  # 7 years
    encryption_enabled: bool = True
    audit_logging_enabled: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Load OpenAI API key from environment
if not settings.openai_api_key:
    settings.openai_api_key = os.getenv("OPENAI_API_KEY")
