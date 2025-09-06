"""
Progress Analytics Agent

Tracks therapeutic progress and outcomes, generates reports and trend analysis.
Follows A2A protocol for seamless collaboration with other mental health agents.
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field

from shared.a2a_protocol.communication_layer import A2ACommunicator, A2AMessage, MessageType, ContentType
from shared.a2a_protocol.agent_discovery import AgentDiscovery, AgentCard, CapabilityType, InputModality, OutputFormat
from shared.a2a_protocol.task_management import TaskManager, Task, TaskPriority, TaskStatus
from shared.a2a_protocol.security import A2ASecurity, AgentRole, AccessLevel

from utils.file_storage import FileStorageManager


class ProgressMetric(str, Enum):
    """Types of progress metrics"""
    MOOD_SCORE = "mood_score"
    ANXIETY_LEVEL = "anxiety_level"
    DEPRESSION_SCORE = "depression_score"
    SESSION_ATTENDANCE = "session_attendance"
    HOMEWORK_COMPLETION = "homework_completion"
    MEDICATION_ADHERENCE = "medication_adherence"
    SLEEP_QUALITY = "sleep_quality"
    SOCIAL_FUNCTIONING = "social_functioning"
    CRISIS_EPISODES = "crisis_episodes"


class ReportType(str, Enum):
    """Types of progress reports"""
    WEEKLY_SUMMARY = "weekly_summary"
    MONTHLY_PROGRESS = "monthly_progress"
    TREATMENT_EFFECTIVENESS = "treatment_effectiveness"
    GOAL_ACHIEVEMENT = "goal_achievement"
    CRISIS_ANALYSIS = "crisis_analysis"
    PROVIDER_PERFORMANCE = "provider_performance"
    SYSTEM_UTILIZATION = "system_utilization"


class TrendDirection(str, Enum):
    """Trend direction indicators"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    FLUCTUATING = "fluctuating"


class ProgressDataPoint(BaseModel):
    """Individual progress data point"""
    data_point_id: str
    user_id: str
    metric_type: ProgressMetric
    value: float
    timestamp: datetime
    source: str  # e.g., "assessment", "self_report", "provider_rating"
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProgressTrend(BaseModel):
    """Progress trend analysis"""
    trend_id: str
    user_id: str
    metric_type: ProgressMetric
    direction: TrendDirection
    trend_strength: float  # 0-1 scale
    start_date: datetime
    end_date: datetime
    data_points: List[ProgressDataPoint]
    trend_line: List[float] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    significance: str = "low"  # low, medium, high
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProgressReport(BaseModel):
    """Progress report data structure"""
    report_id: str
    user_id: str
    report_type: ReportType
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    trends: List[ProgressTrend] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    goals_achieved: List[str] = Field(default_factory=list)
    goals_pending: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    provider_notes: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProgressAnalyticsAgent:
    """
    Progress Analytics Agent for tracking therapeutic progress and outcomes
    
    This agent analyzes progress data, generates reports, and provides insights
    into treatment effectiveness and patient outcomes.
    """
    
    def __init__(
        self,
        agent_id: str,
        a2a_communicator: A2ACommunicator,
        agent_discovery: AgentDiscovery,
        task_manager: TaskManager,
        security: A2ASecurity,
        storage_manager: FileStorageManager
    ):
        self.agent_id = agent_id
        self.a2a_communicator = a2a_communicator
        self.agent_discovery = agent_discovery
        self.task_manager = task_manager
        self.security = security
        self.storage_manager = storage_manager
        
        # Agent state
        self.progress_data: Dict[str, List[ProgressDataPoint]] = {}  # user_id -> data points
        self.trends: Dict[str, List[ProgressTrend]] = {}  # user_id -> trends
        self.reports: Dict[str, List[ProgressReport]] = {}  # user_id -> reports
        
        # Flags for async initialization
        self._message_handlers_registered = False
        self._capabilities_registered = False
    
    async def initialize(self):
        """Initialize the agent asynchronously"""
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
            MessageType.COLLABORATION,
            self._handle_collaboration
        )
        await self.a2a_communicator.register_message_handler(
            MessageType.NOTIFICATION,
            self._handle_notification
        )
    
    async def _register_agent_capabilities(self):
        """Register agent capabilities with the discovery service"""
        agent_card = AgentCard(
            agent_id=self.agent_id,
            name="Progress Analytics Agent",
            description="Tracks therapeutic progress and generates analytics reports",
            version="1.0.0",
            capabilities=[
                {
                    "capability_type": CapabilityType.PROGRESS_ANALYTICS,
                    "description": "Progress tracking and analytics reporting",
                    "input_modalities": [
                        InputModality.TEXT,
                        InputModality.DOCUMENT
                    ],
                    "output_formats": [
                        OutputFormat.STRUCTURED_DATA,
                        OutputFormat.TEXT_RESPONSE,
                        OutputFormat.ASSESSMENT_SCORE
                    ],
                    "parameters": {
                        "supported_metrics": [
                            "mood_score", "anxiety_level", "depression_score",
                            "session_attendance", "homework_completion"
                        ],
                        "response_time_sla": 5.0,
                        "privacy_level": "maximum",
                        "analytics_services": ["trend_analysis", "report_generation", "outcome_prediction"]
                    }
                }
            ],
            contact_endpoint=f"http://localhost:8000/agents/{self.agent_id}",
            compliance_certifications=["hipaa", "analytics_standards", "data_protection"]
        )
        
        await self.agent_discovery.register_agent(agent_card)
    
    async def record_progress_data(
        self,
        user_id: str,
        metric_type: ProgressMetric,
        value: float,
        source: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a new progress data point
        
        Args:
            user_id: ID of the user
            metric_type: Type of progress metric
            value: Metric value
            source: Source of the data
            context: Additional context
            
        Returns:
            Data point ID
        """
        try:
            data_point_id = str(uuid.uuid4())
            
            data_point = ProgressDataPoint(
                data_point_id=data_point_id,
                user_id=user_id,
                metric_type=metric_type,
                value=value,
                timestamp=datetime.utcnow(),
                source=source,
                context=context or {}
            )
            
            # Store data point
            if user_id not in self.progress_data:
                self.progress_data[user_id] = []
            self.progress_data[user_id].append(data_point)
            
            # Save to file storage
            await self.storage_manager.save_progress_data(data_point_id, data_point.model_dump())
            
            # Trigger trend analysis if enough data points
            await self._analyze_trends(user_id, metric_type)
            
            return data_point_id
            
        except Exception as e:
            print(f"Error recording progress data: {e}")
            return None
    
    async def _analyze_trends(self, user_id: str, metric_type: ProgressMetric):
        """Analyze trends for a specific metric"""
        try:
            if user_id not in self.progress_data:
                return
            
            # Get data points for this metric
            metric_data = [
                dp for dp in self.progress_data[user_id]
                if dp.metric_type == metric_type
            ]
            
            if len(metric_data) < 3:  # Need at least 3 data points for trend analysis
                return
            
            # Sort by timestamp
            metric_data.sort(key=lambda x: x.timestamp)
            
            # Calculate trend
            trend = await self._calculate_trend(metric_data)
            
            if trend:
                # Store trend
                if user_id not in self.trends:
                    self.trends[user_id] = []
                self.trends[user_id].append(trend)
                
                # Save trend
                await self.storage_manager.save_progress_trend(trend.trend_id, trend.model_dump())
                
        except Exception as e:
            print(f"Error analyzing trends: {e}")
    
    async def _calculate_trend(self, data_points: List[ProgressDataPoint]) -> Optional[ProgressTrend]:
        """Calculate trend from data points"""
        try:
            if len(data_points) < 3:
                return None
            
            # Extract values and timestamps
            values = [dp.value for dp in data_points]
            timestamps = [dp.timestamp for dp in data_points]
            
            # Simple linear regression for trend calculation
            n = len(values)
            x = list(range(n))
            
            # Calculate slope
            x_mean = sum(x) / n
            y_mean = sum(values) / n
            
            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
            
            # Determine trend direction
            if slope > 0.1:
                direction = TrendDirection.IMPROVING
            elif slope < -0.1:
                direction = TrendDirection.DECLINING
            else:
                direction = TrendDirection.STABLE
            
            # Calculate trend strength
            trend_strength = min(1.0, abs(slope) * 10)
            
            # Calculate confidence score
            confidence_score = min(1.0, len(data_points) / 10)
            
            # Determine significance
            if confidence_score > 0.8 and trend_strength > 0.7:
                significance = "high"
            elif confidence_score > 0.5 and trend_strength > 0.4:
                significance = "medium"
            else:
                significance = "low"
            
            # Generate trend line
            trend_line = [y_mean + slope * (x[i] - x_mean) for i in range(n)]
            
            trend = ProgressTrend(
                trend_id=str(uuid.uuid4()),
                user_id=data_points[0].user_id,
                metric_type=data_points[0].metric_type,
                direction=direction,
                trend_strength=trend_strength,
                start_date=timestamps[0],
                end_date=timestamps[-1],
                data_points=data_points,
                trend_line=trend_line,
                confidence_score=confidence_score,
                significance=significance
            )
            
            return trend
            
        except Exception as e:
            print(f"Error calculating trend: {e}")
            return None
    
    async def generate_progress_report(
        self,
        user_id: str,
        report_type: ReportType,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> str:
        """
        Generate a progress report for a user
        
        Args:
            user_id: ID of the user
            report_type: Type of report to generate
            period_start: Start of reporting period
            period_end: End of reporting period
            
        Returns:
            Report ID
        """
        try:
            # Set default period if not provided
            if not period_end:
                period_end = datetime.utcnow()
            if not period_start:
                if report_type == ReportType.WEEKLY_SUMMARY:
                    period_start = period_end - timedelta(days=7)
                elif report_type == ReportType.MONTHLY_PROGRESS:
                    period_start = period_end - timedelta(days=30)
                else:
                    period_start = period_end - timedelta(days=30)
            
            # Generate report content
            report_content = await self._generate_report_content(
                user_id, report_type, period_start, period_end
            )
            
            # Create report
            report_id = str(uuid.uuid4())
            report = ProgressReport(
                report_id=report_id,
                user_id=user_id,
                report_type=report_type,
                generated_at=datetime.utcnow(),
                period_start=period_start,
                period_end=period_end,
                **report_content
            )
            
            # Store report
            if user_id not in self.reports:
                self.reports[user_id] = []
            self.reports[user_id].append(report)
            
            # Save report
            await self.storage_manager.save_progress_report(report_id, report.model_dump())
            
            return report_id
            
        except Exception as e:
            print(f"Error generating progress report: {e}")
            return None
    
    async def _generate_report_content(
        self,
        user_id: str,
        report_type: ReportType,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Generate report content based on type"""
        
        if report_type == ReportType.WEEKLY_SUMMARY:
            return await self._generate_weekly_summary(user_id, period_start, period_end)
        elif report_type == ReportType.MONTHLY_PROGRESS:
            return await self._generate_monthly_progress(user_id, period_start, period_end)
        elif report_type == ReportType.TREATMENT_EFFECTIVENESS:
            return await self._generate_treatment_effectiveness(user_id, period_start, period_end)
        elif report_type == ReportType.GOAL_ACHIEVEMENT:
            return await self._generate_goal_achievement(user_id, period_start, period_end)
        elif report_type == ReportType.CRISIS_ANALYSIS:
            return await self._generate_crisis_analysis(user_id, period_start, period_end)
        else:
            return await self._generate_general_report(user_id, period_start, period_end)
    
    async def _generate_weekly_summary(
        self,
        user_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Generate weekly summary report"""
        
        # Get data for the period
        data_points = await self._get_data_for_period(user_id, period_start, period_end)
        
        # Calculate summary statistics
        summary_stats = await self._calculate_summary_statistics(data_points)
        
        # Get trends
        trends = await self._get_trends_for_period(user_id, period_start, period_end)
        
        # Generate summary
        summary = f"Weekly progress summary for {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}. "
        summary += f"Overall mood trend: {summary_stats.get('mood_trend', 'stable')}. "
        summary += f"Key metrics: {summary_stats.get('key_metrics', 'No significant changes')}."
        
        return {
            "summary": summary,
            "key_findings": summary_stats.get("key_findings", []),
            "trends": trends,
            "recommendations": summary_stats.get("recommendations", []),
            "goals_achieved": summary_stats.get("goals_achieved", []),
            "goals_pending": summary_stats.get("goals_pending", []),
            "next_steps": summary_stats.get("next_steps", [])
        }
    
    async def _generate_monthly_progress(
        self,
        user_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Generate monthly progress report"""
        
        # Get data for the period
        data_points = await self._get_data_for_period(user_id, period_start, period_end)
        
        # Calculate monthly statistics
        monthly_stats = await self._calculate_monthly_statistics(data_points)
        
        # Get trends
        trends = await self._get_trends_for_period(user_id, period_start, period_end)
        
        # Generate summary
        summary = f"Monthly progress report for {period_start.strftime('%B %Y')}. "
        summary += f"Overall progress: {monthly_stats.get('overall_progress', 'stable')}. "
        summary += f"Treatment effectiveness: {monthly_stats.get('effectiveness', 'moderate')}."
        
        return {
            "summary": summary,
            "key_findings": monthly_stats.get("key_findings", []),
            "trends": trends,
            "recommendations": monthly_stats.get("recommendations", []),
            "goals_achieved": monthly_stats.get("goals_achieved", []),
            "goals_pending": monthly_stats.get("goals_pending", []),
            "next_steps": monthly_stats.get("next_steps", [])
        }
    
    async def _generate_treatment_effectiveness(
        self,
        user_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Generate treatment effectiveness report"""
        
        # Get data for the period
        data_points = await self._get_data_for_period(user_id, period_start, period_end)
        
        # Calculate effectiveness metrics
        effectiveness_metrics = await self._calculate_effectiveness_metrics(data_points)
        
        # Generate summary
        summary = f"Treatment effectiveness analysis for {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}. "
        summary += f"Overall effectiveness: {effectiveness_metrics.get('overall_effectiveness', 'moderate')}. "
        summary += f"Key improvements: {effectiveness_metrics.get('key_improvements', 'None significant')}."
        
        return {
            "summary": summary,
            "key_findings": effectiveness_metrics.get("key_findings", []),
            "trends": effectiveness_metrics.get("trends", []),
            "recommendations": effectiveness_metrics.get("recommendations", []),
            "goals_achieved": effectiveness_metrics.get("goals_achieved", []),
            "goals_pending": effectiveness_metrics.get("goals_pending", []),
            "next_steps": effectiveness_metrics.get("next_steps", [])
        }
    
    async def _generate_goal_achievement(
        self,
        user_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Generate goal achievement report"""
        
        # Get data for the period
        data_points = await self._get_data_for_period(user_id, period_start, period_end)
        
        # Calculate goal achievement metrics
        goal_metrics = await self._calculate_goal_achievement_metrics(data_points)
        
        # Generate summary
        summary = f"Goal achievement report for {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}. "
        summary += f"Goals achieved: {goal_metrics.get('goals_achieved_count', 0)}. "
        summary += f"Overall progress: {goal_metrics.get('overall_progress', 'moderate')}."
        
        return {
            "summary": summary,
            "key_findings": goal_metrics.get("key_findings", []),
            "trends": goal_metrics.get("trends", []),
            "recommendations": goal_metrics.get("recommendations", []),
            "goals_achieved": goal_metrics.get("goals_achieved", []),
            "goals_pending": goal_metrics.get("goals_pending", []),
            "next_steps": goal_metrics.get("next_steps", [])
        }
    
    async def _generate_crisis_analysis(
        self,
        user_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Generate crisis analysis report"""
        
        # Get data for the period
        data_points = await self._get_data_for_period(user_id, period_start, period_end)
        
        # Calculate crisis metrics
        crisis_metrics = await self._calculate_crisis_metrics(data_points)
        
        # Generate summary
        summary = f"Crisis analysis report for {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}. "
        summary += f"Crisis episodes: {crisis_metrics.get('crisis_episodes', 0)}. "
        summary += f"Risk level: {crisis_metrics.get('risk_level', 'low')}."
        
        return {
            "summary": summary,
            "key_findings": crisis_metrics.get("key_findings", []),
            "trends": crisis_metrics.get("trends", []),
            "recommendations": crisis_metrics.get("recommendations", []),
            "goals_achieved": crisis_metrics.get("goals_achieved", []),
            "goals_pending": crisis_metrics.get("goals_pending", []),
            "next_steps": crisis_metrics.get("next_steps", [])
        }
    
    async def _generate_general_report(
        self,
        user_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Generate general progress report"""
        
        # Get data for the period
        data_points = await self._get_data_for_period(user_id, period_start, period_end)
        
        # Calculate general metrics
        general_metrics = await self._calculate_general_metrics(data_points)
        
        # Generate summary
        summary = f"Progress report for {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}. "
        summary += f"Overall status: {general_metrics.get('overall_status', 'stable')}."
        
        return {
            "summary": summary,
            "key_findings": general_metrics.get("key_findings", []),
            "trends": general_metrics.get("trends", []),
            "recommendations": general_metrics.get("recommendations", []),
            "goals_achieved": general_metrics.get("goals_achieved", []),
            "goals_pending": general_metrics.get("goals_pending", []),
            "next_steps": general_metrics.get("next_steps", [])
        }
    
    async def _get_data_for_period(
        self,
        user_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> List[ProgressDataPoint]:
        """Get data points for a specific period"""
        if user_id not in self.progress_data:
            return []
        
        return [
            dp for dp in self.progress_data[user_id]
            if period_start <= dp.timestamp <= period_end
        ]
    
    async def _get_trends_for_period(
        self,
        user_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> List[ProgressTrend]:
        """Get trends for a specific period"""
        if user_id not in self.trends:
            return []
        
        return [
            trend for trend in self.trends[user_id]
            if period_start <= trend.start_date <= period_end
        ]
    
    async def _calculate_summary_statistics(self, data_points: List[ProgressDataPoint]) -> Dict[str, Any]:
        """Calculate summary statistics from data points"""
        if not data_points:
            return {"key_findings": [], "recommendations": []}
        
        # Group by metric type
        metrics = {}
        for dp in data_points:
            if dp.metric_type not in metrics:
                metrics[dp.metric_type] = []
            metrics[dp.metric_type].append(dp.value)
        
        # Calculate statistics for each metric
        summary_stats = {}
        for metric_type, values in metrics.items():
            if values:
                summary_stats[metric_type] = {
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return {
            "key_findings": [f"Data collected for {len(metrics)} metrics"],
            "recommendations": ["Continue monitoring progress"],
            "goals_achieved": [],
            "goals_pending": [],
            "next_steps": ["Review progress with care team"]
        }
    
    async def _calculate_monthly_statistics(self, data_points: List[ProgressDataPoint]) -> Dict[str, Any]:
        """Calculate monthly statistics from data points"""
        return await self._calculate_summary_statistics(data_points)
    
    async def _calculate_effectiveness_metrics(self, data_points: List[ProgressDataPoint]) -> Dict[str, Any]:
        """Calculate treatment effectiveness metrics"""
        return await self._calculate_summary_statistics(data_points)
    
    async def _calculate_goal_achievement_metrics(self, data_points: List[ProgressDataPoint]) -> Dict[str, Any]:
        """Calculate goal achievement metrics"""
        return await self._calculate_summary_statistics(data_points)
    
    async def _calculate_crisis_metrics(self, data_points: List[ProgressDataPoint]) -> Dict[str, Any]:
        """Calculate crisis metrics"""
        return await self._calculate_summary_statistics(data_points)
    
    async def _calculate_general_metrics(self, data_points: List[ProgressDataPoint]) -> Dict[str, Any]:
        """Calculate general metrics"""
        return await self._calculate_summary_statistics(data_points)
    
    async def get_progress_data(
        self,
        user_id: str,
        metric_type: Optional[ProgressMetric] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get progress data for a user"""
        if user_id not in self.progress_data:
            return []
        
        data_points = self.progress_data[user_id]
        
        # Filter by metric type
        if metric_type:
            data_points = [dp for dp in data_points if dp.metric_type == metric_type]
        
        # Filter by date range
        if start_date:
            data_points = [dp for dp in data_points if dp.timestamp >= start_date]
        if end_date:
            data_points = [dp for dp in data_points if dp.timestamp <= end_date]
        
        return [dp.model_dump() for dp in data_points]
    
    async def get_trends(
        self,
        user_id: str,
        metric_type: Optional[ProgressMetric] = None
    ) -> List[Dict[str, Any]]:
        """Get trends for a user"""
        if user_id not in self.trends:
            return []
        
        trends = self.trends[user_id]
        
        # Filter by metric type
        if metric_type:
            trends = [trend for trend in trends if trend.metric_type == metric_type]
        
        return [trend.model_dump() for trend in trends]
    
    async def get_reports(
        self,
        user_id: str,
        report_type: Optional[ReportType] = None
    ) -> List[Dict[str, Any]]:
        """Get reports for a user"""
        if user_id not in self.reports:
            return []
        
        reports = self.reports[user_id]
        
        # Filter by report type
        if report_type:
            reports = [report for report in reports if report.report_type == report_type]
        
        return [report.model_dump() for report in reports]
    
    async def _handle_task_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle incoming task requests"""
        try:
            task_data = message.parts[0].content if message.parts else ""
            
            # Create task
            task = await self.task_manager.create_task(
                title=f"Analytics task from {message.sender_agent_id}",
                description=task_data,
                task_type="progress_analytics",
                created_by=message.sender_agent_id,
                assigned_to=self.agent_id,
                priority=TaskPriority.NORMAL,
                input_data={"message": message.model_dump()}
            )
            
            return {
                "task_id": task.task_id,
                "status": "accepted",
                "message": "Analytics task accepted"
            }
            
        except Exception as e:
            print(f"Error handling task request: {e}")
            return {"error": str(e)}
    
    async def _handle_collaboration(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle collaboration messages"""
        try:
            collaboration_data = json.loads(message.parts[0].content) if message.parts else {}
            
            # Handle collaboration request
            return {
                "status": "collaboration_acknowledged",
                "message": "Analytics collaboration received"
            }
            
        except Exception as e:
            print(f"Error handling collaboration: {e}")
            return {"error": str(e)}
    
    async def _handle_notification(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle notification messages"""
        try:
            notification_data = json.loads(message.parts[0].content) if message.parts else {}
            
            # Handle notification
            return {
                "status": "notification_acknowledged",
                "message": "Analytics notification received"
            }
            
        except Exception as e:
            print(f"Error handling notification: {e}")
            return {"error": str(e)}
