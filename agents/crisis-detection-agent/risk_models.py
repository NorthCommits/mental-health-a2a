"""
Risk Assessment Models for Crisis Detection Agent

Implements risk assessment models and crisis level classification
for detecting high-risk mental health situations.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class CrisisLevel(str, Enum):
    """Levels of crisis severity"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RiskFactor(str, Enum):
    """Types of risk factors for crisis detection"""
    SUICIDAL_IDEATION = "suicidal_ideation"
    SELF_HARM = "self_harm"
    SUBSTANCE_ABUSE = "substance_abuse"
    PSYCHOTIC_SYMPTOMS = "psychotic_symptoms"
    SEVERE_DEPRESSION = "severe_depression"
    PANIC_ATTACK = "panic_attack"
    HOMICIDAL_IDEATION = "homicidal_ideation"
    ISOLATION = "isolation"
    HOPELESSNESS = "hopelessness"
    TRAUMA_REACTIVATION = "trauma_reactivation"


class InterventionType(str, Enum):
    """Types of crisis interventions"""
    IMMEDIATE_SAFETY_CHECK = "immediate_safety_check"
    CRISIS_COUNSELING = "crisis_counseling"
    EMERGENCY_SERVICES = "emergency_services"
    FAMILY_NOTIFICATION = "family_notification"
    CLINICAL_EVALUATION = "clinical_evaluation"
    HOSPITALIZATION = "hospitalization"
    FOLLOW_UP_CARE = "follow_up_care"


class CrisisPattern(BaseModel):
    """Pattern indicating potential crisis situation"""
    pattern_id: str
    pattern_type: RiskFactor
    confidence_score: float = Field(ge=0.0, le=1.0)
    severity_weight: float = Field(ge=0.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class RiskAssessment(BaseModel):
    """Comprehensive risk assessment for crisis detection"""
    assessment_id: str
    user_id: str
    session_id: str
    crisis_level: CrisisLevel
    risk_factors: List[RiskFactor]
    detected_patterns: List[CrisisPattern]
    confidence_score: float = Field(ge=0.0, le=1.0)
    urgency_score: float = Field(ge=0.0, le=1.0)
    safety_score: float = Field(ge=0.0, le=1.0)  # Higher = safer
    recommended_interventions: List[InterventionType]
    requires_immediate_action: bool = False
    assessed_at: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def calculate_risk_score(self) -> float:
        """Calculate overall risk score based on patterns and factors"""
        if not self.detected_patterns:
            return 0.0
        
        # Weighted average of pattern confidence and severity
        weighted_score = sum(
            pattern.confidence_score * pattern.severity_weight
            for pattern in self.detected_patterns
        ) / len(self.detected_patterns)
        
        # Adjust based on number of risk factors
        factor_multiplier = 1.0 + (len(self.risk_factors) - 1) * 0.1
        
        return min(1.0, weighted_score * factor_multiplier)
    
    def determine_crisis_level(self) -> CrisisLevel:
        """Determine crisis level based on risk assessment"""
        risk_score = self.calculate_risk_score()
        
        if risk_score >= 0.9:
            return CrisisLevel.EMERGENCY
        elif risk_score >= 0.8:
            return CrisisLevel.CRITICAL
        elif risk_score >= 0.6:
            return CrisisLevel.HIGH
        elif risk_score >= 0.4:
            return CrisisLevel.MEDIUM
        elif risk_score >= 0.2:
            return CrisisLevel.LOW
        else:
            return CrisisLevel.NONE
    
    def get_required_interventions(self) -> List[InterventionType]:
        """Get required interventions based on crisis level"""
        interventions = []
        
        if self.crisis_level == CrisisLevel.EMERGENCY:
            interventions.extend([
                InterventionType.IMMEDIATE_SAFETY_CHECK,
                InterventionType.EMERGENCY_SERVICES,
                InterventionType.FAMILY_NOTIFICATION,
                InterventionType.HOSPITALIZATION
            ])
        elif self.crisis_level == CrisisLevel.CRITICAL:
            interventions.extend([
                InterventionType.IMMEDIATE_SAFETY_CHECK,
                InterventionType.CRISIS_COUNSELING,
                InterventionType.CLINICAL_EVALUATION,
                InterventionType.FAMILY_NOTIFICATION
            ])
        elif self.crisis_level == CrisisLevel.HIGH:
            interventions.extend([
                InterventionType.CRISIS_COUNSELING,
                InterventionType.CLINICAL_EVALUATION,
                InterventionType.FOLLOW_UP_CARE
            ])
        elif self.crisis_level == CrisisLevel.MEDIUM:
            interventions.extend([
                InterventionType.CRISIS_COUNSELING,
                InterventionType.FOLLOW_UP_CARE
            ])
        elif self.crisis_level == CrisisLevel.LOW:
            interventions.append(InterventionType.FOLLOW_UP_CARE)
        
        return interventions


class CrisisAlert(BaseModel):
    """Crisis alert for immediate action"""
    alert_id: str
    user_id: str
    session_id: str
    crisis_level: CrisisLevel
    risk_assessment: RiskAssessment
    alert_message: str
    required_actions: List[str]
    contact_emergency_services: bool = False
    notify_family: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: Optional[str] = None


class SafetyPlan(BaseModel):
    """Safety plan for crisis prevention"""
    plan_id: str
    user_id: str
    warning_signs: List[str]
    coping_strategies: List[str]
    support_contacts: List[Dict[str, str]]
    professional_contacts: List[Dict[str, str]]
    emergency_contacts: List[Dict[str, str]]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class CrisisIntervention(BaseModel):
    """Record of crisis intervention actions taken"""
    intervention_id: str
    user_id: str
    crisis_alert_id: str
    intervention_type: InterventionType
    action_taken: str
    outcome: str
    follow_up_required: bool = False
    follow_up_date: Optional[datetime] = None
    performed_by: str  # Agent ID or human identifier
    performed_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None


class CrisisStatistics(BaseModel):
    """Statistics for crisis detection monitoring"""
    total_assessments: int = 0
    crisis_detected: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    average_response_time: float = 0.0  # seconds
    intervention_success_rate: float = 0.0
    time_period: str  # e.g., "24h", "7d", "30d"
    generated_at: datetime = Field(default_factory=datetime.utcnow)
