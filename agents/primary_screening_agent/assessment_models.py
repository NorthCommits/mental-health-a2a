"""
Assessment Models for Primary Screening Agent

Implements validated clinical assessment tools including PHQ-9, GAD-7, and other
screening instruments used in mental health evaluation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class SeverityLevel(str, Enum):
    """Severity levels for assessment results"""
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    MODERATELY_SEVERE = "moderately_severe"
    SEVERE = "severe"


class RiskLevel(str, Enum):
    """Risk levels for assessment results"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AssessmentType(str, Enum):
    """Types of assessments available"""
    PHQ9 = "phq9"  # Patient Health Questionnaire-9 (Depression)
    GAD7 = "gad7"  # Generalized Anxiety Disorder 7-item
    PHQ15 = "phq15"  # Patient Health Questionnaire-15 (Somatic Symptoms)
    GADQIV = "gadqiv"  # Generalized Anxiety Disorder Questionnaire-IV
    PCL5 = "pcl5"  # PTSD Checklist for DSM-5
    AUDIT = "audit"  # Alcohol Use Disorders Identification Test
    CAGE = "cage"  # Cut down, Annoyed, Guilty, Eye-opener questionnaire


class AssessmentQuestion(BaseModel):
    """Individual assessment question"""
    question_id: str
    question_text: str
    response_options: List[str]
    response_type: str  # "single_choice", "multiple_choice", "scale", "text"
    required: bool = True
    scoring_weights: Optional[Dict[str, int]] = None


class AssessmentResponse(BaseModel):
    """Response to an assessment question"""
    question_id: str
    response: Union[str, int, List[str]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence_score: Optional[float] = None  # For AI-generated responses


class PHQ9Assessment(BaseModel):
    """
    Patient Health Questionnaire-9 (PHQ-9) for depression screening
    Validated 9-item questionnaire with 0-3 scale scoring
    """
    assessment_id: str
    user_id: str
    responses: List[AssessmentResponse]
    total_score: int
    severity_level: SeverityLevel
    risk_level: RiskLevel
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    requires_follow_up: bool = False
    recommended_actions: List[str] = Field(default_factory=list)
    
    @classmethod
    def create_from_responses(
        cls,
        assessment_id: str,
        user_id: str,
        responses: List[AssessmentResponse]
    ) -> "PHQ9Assessment":
        """Create PHQ-9 assessment from user responses"""
        # PHQ-9 scoring: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe
        total_score = sum(
            response.response if isinstance(response.response, int) else 0
            for response in responses
        )
        
        if total_score <= 4:
            severity = SeverityLevel.MINIMAL
            risk = RiskLevel.LOW
        elif total_score <= 9:
            severity = SeverityLevel.MILD
            risk = RiskLevel.LOW
        elif total_score <= 14:
            severity = SeverityLevel.MODERATE
            risk = RiskLevel.MEDIUM
        elif total_score <= 19:
            severity = SeverityLevel.MODERATELY_SEVERE
            risk = RiskLevel.HIGH
        else:
            severity = SeverityLevel.SEVERE
            risk = RiskLevel.CRITICAL
        
        requires_follow_up = total_score >= 10
        recommended_actions = cls._get_recommended_actions(total_score, severity)
        
        return cls(
            assessment_id=assessment_id,
            user_id=user_id,
            responses=responses,
            total_score=total_score,
            severity_level=severity,
            risk_level=risk,
            requires_follow_up=requires_follow_up,
            recommended_actions=recommended_actions
        )
    
    @staticmethod
    def _get_recommended_actions(score: int, severity: SeverityLevel) -> List[str]:
        """Get recommended actions based on PHQ-9 score"""
        actions = []
        
        if score >= 20:
            actions.extend([
                "Immediate clinical evaluation recommended",
                "Consider psychiatric consultation",
                "Monitor for suicidal ideation",
                "Consider medication evaluation"
            ])
        elif score >= 15:
            actions.extend([
                "Clinical evaluation recommended within 1 week",
                "Consider therapy referral",
                "Monitor symptoms closely"
            ])
        elif score >= 10:
            actions.extend([
                "Clinical evaluation recommended within 2 weeks",
                "Consider therapy or counseling",
                "Self-monitoring recommended"
            ])
        elif score >= 5:
            actions.extend([
                "Self-monitoring recommended",
                "Consider lifestyle interventions",
                "Follow up in 1 month"
            ])
        else:
            actions.extend([
                "Continue current care",
                "Routine follow-up recommended"
            ])
        
        return actions
    
    @staticmethod
    def get_questions() -> List[AssessmentQuestion]:
        """Get PHQ-9 assessment questions"""
        return [
            AssessmentQuestion(
                question_id="phq9_1",
                question_text="Little interest or pleasure in doing things",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="phq9_2",
                question_text="Feeling down, depressed, or hopeless",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="phq9_3",
                question_text="Trouble falling or staying asleep, or sleeping too much",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="phq9_4",
                question_text="Feeling tired or having little energy",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="phq9_5",
                question_text="Poor appetite or overeating",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="phq9_6",
                question_text="Feeling bad about yourself - or that you are a failure or have let yourself or your family down",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="phq9_7",
                question_text="Trouble concentrating on things, such as reading the newspaper or watching television",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="phq9_8",
                question_text="Moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="phq9_9",
                question_text="Thoughts that you would be better off dead, or of hurting yourself",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            )
        ]


class GAD7Assessment(BaseModel):
    """
    Generalized Anxiety Disorder 7-item scale (GAD-7)
    Validated 7-item questionnaire for anxiety screening
    """
    assessment_id: str
    user_id: str
    responses: List[AssessmentResponse]
    total_score: int
    severity_level: SeverityLevel
    risk_level: RiskLevel
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    requires_follow_up: bool = False
    recommended_actions: List[str] = Field(default_factory=list)
    
    @classmethod
    def create_from_responses(
        cls,
        assessment_id: str,
        user_id: str,
        responses: List[AssessmentResponse]
    ) -> "GAD7Assessment":
        """Create GAD-7 assessment from user responses"""
        # GAD-7 scoring: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-21 severe
        total_score = sum(
            response.response if isinstance(response.response, int) else 0
            for response in responses
        )
        
        if total_score <= 4:
            severity = SeverityLevel.MINIMAL
            risk = RiskLevel.LOW
        elif total_score <= 9:
            severity = SeverityLevel.MILD
            risk = RiskLevel.LOW
        elif total_score <= 14:
            severity = SeverityLevel.MODERATE
            risk = RiskLevel.MEDIUM
        else:
            severity = SeverityLevel.SEVERE
            risk = RiskLevel.HIGH
        
        requires_follow_up = total_score >= 10
        recommended_actions = cls._get_recommended_actions(total_score, severity)
        
        return cls(
            assessment_id=assessment_id,
            user_id=user_id,
            responses=responses,
            total_score=total_score,
            severity_level=severity,
            risk_level=risk,
            requires_follow_up=requires_follow_up,
            recommended_actions=recommended_actions
        )
    
    @staticmethod
    def _get_recommended_actions(score: int, severity: SeverityLevel) -> List[str]:
        """Get recommended actions based on GAD-7 score"""
        actions = []
        
        if score >= 15:
            actions.extend([
                "Immediate clinical evaluation recommended",
                "Consider psychiatric consultation",
                "Consider medication evaluation",
                "Consider therapy referral"
            ])
        elif score >= 10:
            actions.extend([
                "Clinical evaluation recommended within 1 week",
                "Consider therapy or counseling",
                "Consider relaxation techniques",
                "Monitor symptoms closely"
            ])
        elif score >= 5:
            actions.extend([
                "Self-monitoring recommended",
                "Consider stress management techniques",
                "Follow up in 1 month"
            ])
        else:
            actions.extend([
                "Continue current care",
                "Routine follow-up recommended"
            ])
        
        return actions
    
    @staticmethod
    def get_questions() -> List[AssessmentQuestion]:
        """Get GAD-7 assessment questions"""
        return [
            AssessmentQuestion(
                question_id="gad7_1",
                question_text="Feeling nervous, anxious or on edge",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="gad7_2",
                question_text="Not being able to stop or control worrying",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="gad7_3",
                question_text="Worrying too much about different things",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="gad7_4",
                question_text="Trouble relaxing",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="gad7_5",
                question_text="Being so restless that it is hard to sit still",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="gad7_6",
                question_text="Becoming easily annoyed or irritable",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            ),
            AssessmentQuestion(
                question_id="gad7_7",
                question_text="Feeling afraid as if something awful might happen",
                response_options=["Not at all", "Several days", "More than half the days", "Nearly every day"],
                response_type="single_choice",
                scoring_weights={"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            )
        ]


class AssessmentResult(BaseModel):
    """Comprehensive assessment result combining multiple screening tools"""
    result_id: str
    user_id: str
    assessment_type: AssessmentType
    total_score: int
    severity_level: SeverityLevel
    risk_level: RiskLevel
    completed_at: datetime
    requires_follow_up: bool
    recommended_actions: List[str]
    clinical_notes: Optional[str] = None
    next_assessment_due: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_crisis_alert_data(self) -> Dict[str, Any]:
        """Convert assessment result to crisis alert data if needed"""
        if self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return {
                "assessment_type": self.assessment_type.value,
                "total_score": self.total_score,
                "severity_level": self.severity_level.value,
                "risk_level": self.risk_level.value,
                "requires_immediate_attention": self.risk_level == RiskLevel.CRITICAL,
                "recommended_actions": self.recommended_actions,
                "clinical_notes": self.clinical_notes,
                "timestamp": self.completed_at.isoformat()
            }
        return {}
