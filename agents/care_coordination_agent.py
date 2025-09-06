"""
Care Coordination Agent

Manages appointments, provider communication, and coordinates between different
healthcare providers. Follows A2A protocol for seamless collaboration with
other mental health agents.
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


class AppointmentType(str, Enum):
    """Types of appointments"""
    INITIAL_CONSULTATION = "initial_consultation"
    FOLLOW_UP = "follow_up"
    THERAPY_SESSION = "therapy_session"
    CRISIS_INTERVENTION = "crisis_intervention"
    MEDICATION_REVIEW = "medication_review"
    GROUP_THERAPY = "group_therapy"
    FAMILY_SESSION = "family_session"


class AppointmentStatus(str, Enum):
    """Appointment statuses"""
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"
    RESCHEDULED = "rescheduled"


class ProviderType(str, Enum):
    """Types of healthcare providers"""
    PSYCHIATRIST = "psychiatrist"
    PSYCHOLOGIST = "psychologist"
    THERAPIST = "therapist"
    COUNSELOR = "counselor"
    SOCIAL_WORKER = "social_worker"
    NURSE_PRACTITIONER = "nurse_practitioner"
    CASE_MANAGER = "case_manager"


class Appointment(BaseModel):
    """Appointment data structure"""
    appointment_id: str
    user_id: str
    provider_id: str
    provider_name: str
    provider_type: ProviderType
    appointment_type: AppointmentType
    scheduled_time: datetime
    duration_minutes: int
    status: AppointmentStatus
    location: str
    meeting_link: Optional[str] = None
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    reminder_sent: bool = False
    follow_up_required: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Provider(BaseModel):
    """Healthcare provider data structure"""
    provider_id: str
    name: str
    provider_type: ProviderType
    specializations: List[str] = Field(default_factory=list)
    contact_email: str
    contact_phone: str
    availability: Dict[str, List[str]] = Field(default_factory=dict)  # day -> time slots
    location: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CarePlan(BaseModel):
    """Care plan data structure"""
    plan_id: str
    user_id: str
    primary_provider_id: str
    care_team: List[str] = Field(default_factory=list)  # provider IDs
    goals: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    timeline: Dict[str, datetime] = Field(default_factory=dict)
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CareCoordinationAgent:
    """
    Care Coordination Agent for appointment and provider management
    
    This agent manages appointments, coordinates between healthcare providers,
    and ensures seamless care delivery across the mental health ecosystem.
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
        self.providers: Dict[str, Provider] = {}
        self.appointments: Dict[str, Appointment] = {}
        self.care_plans: Dict[str, CarePlan] = {}
        self.user_appointments: Dict[str, List[str]] = {}  # user_id -> appointment_ids
        
        # Register message handlers
        asyncio.create_task(self._register_message_handlers())
        
        # Register agent capabilities
        asyncio.create_task(self._register_agent_capabilities())
        
        # Initialize with sample providers
        asyncio.create_task(self._initialize_sample_providers())
    
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
            MessageType.CRISIS_ALERT,
            self._handle_crisis_alert
        )
    
    async def _register_agent_capabilities(self):
        """Register agent capabilities with the discovery service"""
        agent_card = AgentCard(
            agent_id=self.agent_id,
            name="Care Coordination Agent",
            description="Manages appointments and coordinates between healthcare providers",
            version="1.0.0",
            capabilities=[
                {
                    "capability_type": CapabilityType.CARE_COORDINATION,
                    "description": "Appointment scheduling and provider coordination",
                    "input_modalities": [
                        InputModality.TEXT,
                        InputModality.JSON
                    ],
                    "output_formats": [
                        OutputFormat.APPOINTMENT_SCHEDULE,
                        OutputFormat.STRUCTURED_DATA,
                        OutputFormat.TREATMENT_PLAN
                    ],
                    "parameters": {
                        "supported_appointment_types": [
                            "initial_consultation", "follow_up", "therapy_session",
                            "crisis_intervention", "medication_review"
                        ],
                        "response_time_sla": 2.0,
                        "privacy_level": "maximum",
                        "coordination_services": ["scheduling", "provider_matching", "care_planning"]
                    }
                }
            ],
            contact_endpoint=f"http://localhost:8000/agents/{self.agent_id}",
            compliance_certifications=["hipaa", "healthcare_coordination", "appointment_management"]
        )
        
        await self.agent_discovery.register_agent(agent_card)
    
    async def _initialize_sample_providers(self):
        """Initialize with sample healthcare providers"""
        sample_providers = [
            Provider(
                provider_id="provider_001",
                name="Dr. Sarah Johnson",
                provider_type=ProviderType.PSYCHIATRIST,
                specializations=["depression", "anxiety", "bipolar_disorder"],
                contact_email="sarah.johnson@mentalhealth.com",
                contact_phone="555-0101",
                availability={
                    "monday": ["09:00", "10:00", "11:00", "14:00", "15:00"],
                    "tuesday": ["09:00", "10:00", "11:00", "14:00", "15:00"],
                    "wednesday": ["09:00", "10:00", "11:00", "14:00", "15:00"],
                    "thursday": ["09:00", "10:00", "11:00", "14:00", "15:00"],
                    "friday": ["09:00", "10:00", "11:00"]
                },
                location="Main Office - Room 101"
            ),
            Provider(
                provider_id="provider_002",
                name="Dr. Michael Chen",
                provider_type=ProviderType.PSYCHOLOGIST,
                specializations=["cbt", "trauma", "ptsd"],
                contact_email="michael.chen@mentalhealth.com",
                contact_phone="555-0102",
                availability={
                    "monday": ["08:00", "09:00", "10:00", "13:00", "14:00"],
                    "tuesday": ["08:00", "09:00", "10:00", "13:00", "14:00"],
                    "wednesday": ["08:00", "09:00", "10:00", "13:00", "14:00"],
                    "thursday": ["08:00", "09:00", "10:00", "13:00", "14:00"],
                    "friday": ["08:00", "09:00", "10:00"]
                },
                location="Main Office - Room 102"
            ),
            Provider(
                provider_id="provider_003",
                name="Lisa Rodriguez, LCSW",
                provider_type=ProviderType.SOCIAL_WORKER,
                specializations=["case_management", "family_therapy", "crisis_intervention"],
                contact_email="lisa.rodriguez@mentalhealth.com",
                contact_phone="555-0103",
                availability={
                    "monday": ["10:00", "11:00", "12:00", "15:00", "16:00"],
                    "tuesday": ["10:00", "11:00", "12:00", "15:00", "16:00"],
                    "wednesday": ["10:00", "11:00", "12:00", "15:00", "16:00"],
                    "thursday": ["10:00", "11:00", "12:00", "15:00", "16:00"],
                    "friday": ["10:00", "11:00", "12:00"]
                },
                location="Main Office - Room 103"
            )
        ]
        
        for provider in sample_providers:
            self.providers[provider.provider_id] = provider
            await self.storage_manager.save_provider(provider.provider_id, provider.model_dump())
    
    async def schedule_appointment(
        self,
        user_id: str,
        appointment_type: AppointmentType,
        preferred_time: Optional[datetime] = None,
        preferred_provider_type: Optional[ProviderType] = None,
        urgency: str = "normal",
        notes: str = ""
    ) -> Optional[str]:
        """
        Schedule a new appointment
        
        Args:
            user_id: ID of the user
            appointment_type: Type of appointment needed
            preferred_time: Preferred appointment time
            preferred_provider_type: Preferred type of provider
            urgency: Urgency level (low, normal, high, urgent)
            notes: Additional notes
            
        Returns:
            Appointment ID if successful, None otherwise
        """
        try:
            # Find suitable provider
            provider = await self._find_suitable_provider(
                appointment_type, preferred_provider_type, preferred_time, urgency
            )
            
            if not provider:
                return None
            
            # Find available time slot
            available_time = await self._find_available_time_slot(
                provider, preferred_time, urgency
            )
            
            if not available_time:
                return None
            
            # Create appointment
            appointment_id = str(uuid.uuid4())
            appointment = Appointment(
                appointment_id=appointment_id,
                user_id=user_id,
                provider_id=provider.provider_id,
                provider_name=provider.name,
                provider_type=provider.provider_type,
                appointment_type=appointment_type,
                scheduled_time=available_time,
                duration_minutes=await self._get_appointment_duration(appointment_type),
                status=AppointmentStatus.SCHEDULED,
                location=provider.location,
                notes=notes
            )
            
            # Store appointment
            self.appointments[appointment_id] = appointment
            
            # Add to user's appointments
            if user_id not in self.user_appointments:
                self.user_appointments[user_id] = []
            self.user_appointments[user_id].append(appointment_id)
            
            # Save to file storage
            await self.storage_manager.save_appointment(appointment_id, appointment.model_dump())
            
            # Send confirmation
            await self._send_appointment_confirmation(appointment)
            
            return appointment_id
            
        except Exception as e:
            print(f"Error scheduling appointment: {e}")
            return None
    
    async def _find_suitable_provider(
        self,
        appointment_type: AppointmentType,
        preferred_provider_type: Optional[ProviderType],
        preferred_time: Optional[datetime],
        urgency: str
    ) -> Optional[Provider]:
        """Find a suitable provider for the appointment"""
        
        # Filter providers by type if specified
        candidate_providers = []
        for provider in self.providers.values():
            if not provider.is_active:
                continue
            
            if preferred_provider_type and provider.provider_type != preferred_provider_type:
                continue
            
            # Check if provider has relevant specializations
            if await self._provider_suitable_for_appointment(provider, appointment_type):
                candidate_providers.append(provider)
        
        if not candidate_providers:
            return None
        
        # For urgent appointments, prioritize availability
        if urgency in ["high", "urgent"]:
            for provider in candidate_providers:
                if await self._provider_available_soon(provider):
                    return provider
        
        # Return first suitable provider
        return candidate_providers[0]
    
    async def _provider_suitable_for_appointment(
        self,
        provider: Provider,
        appointment_type: AppointmentType
    ) -> bool:
        """Check if provider is suitable for appointment type"""
        
        if appointment_type == AppointmentType.INITIAL_CONSULTATION:
            return provider.provider_type in [ProviderType.PSYCHIATRIST, ProviderType.PSYCHOLOGIST]
        elif appointment_type == AppointmentType.THERAPY_SESSION:
            return provider.provider_type in [ProviderType.PSYCHOLOGIST, ProviderType.THERAPIST, ProviderType.COUNSELOR]
        elif appointment_type == AppointmentType.CRISIS_INTERVENTION:
            return provider.provider_type in [ProviderType.PSYCHIATRIST, ProviderType.SOCIAL_WORKER]
        elif appointment_type == AppointmentType.MEDICATION_REVIEW:
            return provider.provider_type in [ProviderType.PSYCHIATRIST, ProviderType.NURSE_PRACTITIONER]
        else:
            return True
    
    async def _provider_available_soon(self, provider: Provider) -> bool:
        """Check if provider has availability soon"""
        # Check if provider has availability within next 24 hours
        now = datetime.utcnow()
        tomorrow = now + timedelta(days=1)
        
        # This is a simplified check - in reality, you'd check actual availability
        return True
    
    async def _find_available_time_slot(
        self,
        provider: Provider,
        preferred_time: Optional[datetime],
        urgency: str
    ) -> Optional[datetime]:
        """Find an available time slot for the provider"""
        
        if preferred_time:
            # Check if preferred time is available
            if await self._time_slot_available(provider, preferred_time):
                return preferred_time
        
        # Find next available slot
        now = datetime.utcnow()
        current_time = now
        
        # Look for availability in the next 30 days
        for days_ahead in range(30):
            check_date = current_time + timedelta(days=days_ahead)
            day_name = check_date.strftime("%A").lower()
            
            if day_name in provider.availability:
                for time_slot in provider.availability[day_name]:
                    slot_time = datetime.combine(check_date.date(), datetime.strptime(time_slot, "%H:%M").time())
                    
                    if slot_time > now and await self._time_slot_available(provider, slot_time):
                        return slot_time
        
        return None
    
    async def _time_slot_available(self, provider: Provider, time_slot: datetime) -> bool:
        """Check if a specific time slot is available for a provider"""
        # Check if any existing appointments conflict with this time slot
        for appointment in self.appointments.values():
            if (appointment.provider_id == provider.provider_id and
                appointment.status not in [AppointmentStatus.CANCELLED, AppointmentStatus.COMPLETED] and
                appointment.scheduled_time <= time_slot < appointment.scheduled_time + timedelta(minutes=appointment.duration_minutes)):
                return False
        
        return True
    
    async def _get_appointment_duration(self, appointment_type: AppointmentType) -> int:
        """Get duration in minutes for appointment type"""
        duration_map = {
            AppointmentType.INITIAL_CONSULTATION: 60,
            AppointmentType.FOLLOW_UP: 30,
            AppointmentType.THERAPY_SESSION: 50,
            AppointmentType.CRISIS_INTERVENTION: 90,
            AppointmentType.MEDICATION_REVIEW: 20,
            AppointmentType.GROUP_THERAPY: 90,
            AppointmentType.FAMILY_SESSION: 60
        }
        return duration_map.get(appointment_type, 30)
    
    async def _send_appointment_confirmation(self, appointment: Appointment):
        """Send appointment confirmation to user"""
        # This would integrate with notification system
        print(f"Appointment confirmed: {appointment.appointment_id} for {appointment.user_id}")
    
    async def reschedule_appointment(
        self,
        appointment_id: str,
        new_time: datetime,
        reason: str = ""
    ) -> bool:
        """
        Reschedule an existing appointment
        
        Args:
            appointment_id: ID of the appointment to reschedule
            new_time: New appointment time
            reason: Reason for rescheduling
            
        Returns:
            bool: True if rescheduled successfully
        """
        try:
            appointment = self.appointments.get(appointment_id)
            if not appointment:
                return False
            
            # Check if new time is available
            provider = self.providers.get(appointment.provider_id)
            if not provider or not await self._time_slot_available(provider, new_time):
                return False
            
            # Update appointment
            appointment.scheduled_time = new_time
            appointment.status = AppointmentStatus.RESCHEDULED
            appointment.updated_at = datetime.utcnow()
            appointment.notes += f"\nRescheduled to {new_time}: {reason}"
            
            # Save updated appointment
            await self.storage_manager.save_appointment(appointment_id, appointment.model_dump())
            
            return True
            
        except Exception as e:
            print(f"Error rescheduling appointment: {e}")
            return False
    
    async def cancel_appointment(
        self,
        appointment_id: str,
        reason: str = ""
    ) -> bool:
        """
        Cancel an appointment
        
        Args:
            appointment_id: ID of the appointment to cancel
            reason: Reason for cancellation
            
        Returns:
            bool: True if cancelled successfully
        """
        try:
            appointment = self.appointments.get(appointment_id)
            if not appointment:
                return False
            
            # Update appointment status
            appointment.status = AppointmentStatus.CANCELLED
            appointment.updated_at = datetime.utcnow()
            appointment.notes += f"\nCancelled: {reason}"
            
            # Save updated appointment
            await self.storage_manager.save_appointment(appointment_id, appointment.model_dump())
            
            return True
            
        except Exception as e:
            print(f"Error cancelling appointment: {e}")
            return False
    
    async def get_user_appointments(
        self,
        user_id: str,
        status_filter: Optional[AppointmentStatus] = None
    ) -> List[Dict[str, Any]]:
        """Get appointments for a user"""
        if user_id not in self.user_appointments:
            return []
        
        appointment_ids = self.user_appointments[user_id]
        appointments = [self.appointments[aid] for aid in appointment_ids if aid in self.appointments]
        
        if status_filter:
            appointments = [apt for apt in appointments if apt.status == status_filter]
        
        return [apt.model_dump() for apt in appointments]
    
    async def create_care_plan(
        self,
        user_id: str,
        primary_provider_id: str,
        goals: List[str],
        interventions: List[str],
        timeline: Dict[str, datetime]
    ) -> str:
        """
        Create a care plan for a user
        
        Args:
            user_id: ID of the user
            primary_provider_id: ID of primary care provider
            goals: List of treatment goals
            interventions: List of planned interventions
            timeline: Timeline for achieving goals
            
        Returns:
            Care plan ID
        """
        try:
            plan_id = str(uuid.uuid4())
            
            care_plan = CarePlan(
                plan_id=plan_id,
                user_id=user_id,
                primary_provider_id=primary_provider_id,
                goals=goals,
                interventions=interventions,
                timeline=timeline
            )
            
            # Store care plan
            self.care_plans[plan_id] = care_plan
            
            # Save to file storage
            await self.storage_manager.save_care_plan(plan_id, care_plan.model_dump())
            
            return plan_id
            
        except Exception as e:
            print(f"Error creating care plan: {e}")
            return None
    
    async def update_care_plan(
        self,
        plan_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a care plan"""
        try:
            care_plan = self.care_plans.get(plan_id)
            if not care_plan:
                return False
            
            # Update fields
            for key, value in updates.items():
                if hasattr(care_plan, key):
                    setattr(care_plan, key, value)
            
            care_plan.updated_at = datetime.utcnow()
            
            # Save updated care plan
            await self.storage_manager.save_care_plan(plan_id, care_plan.model_dump())
            
            return True
            
        except Exception as e:
            print(f"Error updating care plan: {e}")
            return False
    
    async def get_care_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get a care plan by ID"""
        care_plan = self.care_plans.get(plan_id)
        return care_plan.model_dump() if care_plan else None
    
    async def get_user_care_plan(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get care plan for a user"""
        for care_plan in self.care_plans.values():
            if care_plan.user_id == user_id and care_plan.status == "active":
                return care_plan.model_dump()
        return None
    
    async def _handle_task_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle incoming task requests"""
        try:
            task_data = message.parts[0].content if message.parts else ""
            
            # Create task
            task = await self.task_manager.create_task(
                title=f"Care coordination task from {message.sender_agent_id}",
                description=task_data,
                task_type="care_coordination",
                created_by=message.sender_agent_id,
                assigned_to=self.agent_id,
                priority=TaskPriority.NORMAL,
                input_data={"message": message.model_dump()}
            )
            
            return {
                "task_id": task.task_id,
                "status": "accepted",
                "message": "Care coordination task accepted"
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
                "message": "Care coordination collaboration received"
            }
            
        except Exception as e:
            print(f"Error handling collaboration: {e}")
            return {"error": str(e)}
    
    async def _handle_crisis_alert(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Handle crisis alerts"""
        try:
            crisis_data = json.loads(message.parts[0].content) if message.parts else {}
            
            # Handle crisis situation - schedule urgent appointment
            user_id = crisis_data.get("user_id")
            if user_id:
                appointment_id = await self.schedule_appointment(
                    user_id=user_id,
                    appointment_type=AppointmentType.CRISIS_INTERVENTION,
                    urgency="urgent",
                    notes="Crisis intervention - urgent appointment"
                )
                
                return {
                    "status": "crisis_acknowledged",
                    "message": "Crisis alert received, urgent appointment scheduled",
                    "appointment_id": appointment_id
                }
            
            return {
                "status": "crisis_acknowledged",
                "message": "Crisis alert received"
            }
            
        except Exception as e:
            print(f"Error handling crisis alert: {e}")
            return {"error": str(e)}
