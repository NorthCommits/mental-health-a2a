"""
Task Management for A2A Protocol

Implements task lifecycle management, artifact handling, and progress tracking
as specified in the A2A protocol.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task lifecycle statuses"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ArtifactType(str, Enum):
    """Types of artifacts that can be produced by tasks"""
    ASSESSMENT_RESULT = "assessment_result"
    CRISIS_ALERT = "crisis_alert"
    TREATMENT_PLAN = "treatment_plan"
    PROGRESS_REPORT = "progress_report"
    APPOINTMENT_SCHEDULE = "appointment_schedule"
    CLINICAL_NOTE = "clinical_note"
    DATA_ANALYSIS = "data_analysis"
    RECOMMENDATION = "recommendation"


class Artifact(BaseModel):
    """Artifact produced by task completion"""
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    artifact_type: ArtifactType
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    access_level: str = Field(default="restricted")  # public, restricted, confidential


class TaskProgress(BaseModel):
    """Progress update for a task"""
    progress_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    status: TaskStatus
    progress_percentage: float = Field(ge=0.0, le=100.0)
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class Task(BaseModel):
    """A2A Task representation"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    task_type: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_by: str  # Agent ID that created the task
    assigned_to: str  # Agent ID assigned to execute the task
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    artifacts: List[Artifact] = Field(default_factory=list)
    progress_history: List[TaskProgress] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None
    parent_task_id: Optional[str] = None
    child_task_ids: List[str] = Field(default_factory=list)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    error_message: Optional[str] = None


class TaskManager:
    """
    Manages task lifecycle, progress tracking, and artifact handling
    Implements the A2A protocol for task management
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.agent_tasks: Dict[str, List[str]] = {}  # agent_id -> task_ids
        self.task_queue: List[str] = []  # Priority-ordered task queue
    
    async def create_task(
        self,
        title: str,
        description: str,
        task_type: str,
        created_by: str,
        assigned_to: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        due_date: Optional[datetime] = None,
        estimated_duration: Optional[timedelta] = None,
        input_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        parent_task_id: Optional[str] = None
    ) -> Task:
        """
        Create a new task
        
        Args:
            title: Task title
            description: Detailed task description
            task_type: Type of task (e.g., "screening", "crisis_detection")
            created_by: Agent ID that created the task
            assigned_to: Agent ID assigned to execute the task
            priority: Task priority level
            due_date: Optional due date for the task
            estimated_duration: Estimated time to complete
            input_data: Input data for the task
            context: Additional context
            parent_task_id: ID of parent task if this is a subtask
            
        Returns:
            Created Task object
        """
        task = Task(
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            created_by=created_by,
            assigned_to=assigned_to,
            due_date=due_date,
            estimated_duration=estimated_duration,
            input_data=input_data or {},
            context=context,
            parent_task_id=parent_task_id
        )
        
        # Store the task
        self.tasks[task.task_id] = task
        
        # Add to agent's task list
        if assigned_to not in self.agent_tasks:
            self.agent_tasks[assigned_to] = []
        self.agent_tasks[assigned_to].append(task.task_id)
        
        # Add to task queue with priority ordering
        await self._add_to_queue(task.task_id, priority)
        
        # Add to parent task's children if applicable
        if parent_task_id and parent_task_id in self.tasks:
            self.tasks[parent_task_id].child_task_ids.append(task.task_id)
        
        return task
    
    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress_percentage: Optional[float] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update task status and progress
        
        Args:
            task_id: ID of the task to update
            status: New status
            progress_percentage: Progress percentage (0-100)
            message: Status message
            metadata: Additional metadata
            
        Returns:
            bool: True if update was successful
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        old_status = task.status
        task.status = status
        task.updated_at = datetime.utcnow()
        
        # Calculate actual duration if completed
        if status == TaskStatus.COMPLETED and task.actual_duration is None:
            task.actual_duration = datetime.utcnow() - task.created_at
        
        # Create progress update
        progress = TaskProgress(
            task_id=task_id,
            status=status,
            progress_percentage=progress_percentage or 0.0,
            message=message or f"Status changed from {old_status} to {status}",
            metadata=metadata
        )
        task.progress_history.append(progress)
        
        # Remove from queue if completed, failed, or cancelled
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            await self._remove_from_queue(task_id)
        
        return True
    
    async def add_artifact(
        self,
        task_id: str,
        artifact_type: ArtifactType,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        access_level: str = "restricted"
    ) -> Optional[Artifact]:
        """
        Add an artifact to a task
        
        Args:
            task_id: ID of the task
            artifact_type: Type of artifact
            content: Artifact content
            metadata: Additional metadata
            access_level: Access level for the artifact
            
        Returns:
            Created Artifact if successful, None otherwise
        """
        if task_id not in self.tasks:
            return None
        
        artifact = Artifact(
            task_id=task_id,
            artifact_type=artifact_type,
            content=content,
            metadata=metadata,
            access_level=access_level
        )
        
        self.tasks[task_id].artifacts.append(artifact)
        return artifact
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task if found, None otherwise
        """
        return self.tasks.get(task_id)
    
    async def get_agent_tasks(
        self, 
        agent_id: str, 
        status_filter: Optional[TaskStatus] = None
    ) -> List[Task]:
        """
        Get all tasks for a specific agent
        
        Args:
            agent_id: ID of the agent
            status_filter: Optional status filter
            
        Returns:
            List of tasks for the agent
        """
        if agent_id not in self.agent_tasks:
            return []
        
        task_ids = self.agent_tasks[agent_id]
        tasks = [self.tasks[tid] for tid in task_ids if tid in self.tasks]
        
        if status_filter:
            tasks = [task for task in tasks if task.status == status_filter]
        
        return tasks
    
    async def get_next_task(self, agent_id: str) -> Optional[Task]:
        """
        Get the next highest priority task for an agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Next task to execute, or None if no tasks available
        """
        agent_task_ids = self.agent_tasks.get(agent_id, [])
        
        # Find the highest priority pending task for this agent
        for task_id in self.task_queue:
            if task_id in agent_task_ids and task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    return task
        
        return None
    
    async def retry_failed_task(self, task_id: str) -> bool:
        """
        Retry a failed task
        
        Args:
            task_id: ID of the task to retry
            
        Returns:
            bool: True if retry was successful
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status != TaskStatus.FAILED:
            return False
        
        if task.retry_count >= task.max_retries:
            return False
        
        # Reset task for retry
        task.status = TaskStatus.PENDING
        task.retry_count += 1
        task.error_message = None
        task.updated_at = datetime.utcnow()
        
        # Add back to queue
        await self._add_to_queue(task_id, task.priority)
        
        return True
    
    async def cancel_task(self, task_id: str, reason: Optional[str] = None) -> bool:
        """
        Cancel a task
        
        Args:
            task_id: ID of the task to cancel
            reason: Reason for cancellation
            
        Returns:
            bool: True if cancellation was successful
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.updated_at = datetime.utcnow()
        task.error_message = reason or "Task cancelled"
        
        # Remove from queue
        await self._remove_from_queue(task_id)
        
        # Cancel child tasks
        for child_id in task.child_task_ids:
            await self.cancel_task(child_id, "Parent task cancelled")
        
        return True
    
    async def _add_to_queue(self, task_id: str, priority: TaskPriority):
        """Add task to priority queue"""
        if task_id in self.task_queue:
            return
        
        # Priority ordering: emergency > critical > high > normal > low
        priority_order = {
            TaskPriority.EMERGENCY: 0,
            TaskPriority.CRITICAL: 1,
            TaskPriority.HIGH: 2,
            TaskPriority.NORMAL: 3,
            TaskPriority.LOW: 4
        }
        
        task_priority = priority_order.get(priority, 3)
        
        # Insert task in correct position based on priority
        inserted = False
        for i, queued_task_id in enumerate(self.task_queue):
            if queued_task_id in self.tasks:
                queued_priority = priority_order.get(self.tasks[queued_task_id].priority, 3)
                if task_priority < queued_priority:
                    self.task_queue.insert(i, task_id)
                    inserted = True
                    break
        
        if not inserted:
            self.task_queue.append(task_id)
    
    async def _remove_from_queue(self, task_id: str):
        """Remove task from priority queue"""
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)
