"""
A2A Protocol Implementation for Mental Health Agent Ecosystem

This module implements the Agent-to-Agent (A2A) protocol as specified by Google Cloud,
enabling seamless communication between specialized mental health agents.
"""

from .communication_layer import A2ACommunicator, A2AMessage
from .agent_discovery import AgentDiscovery, AgentCard
from .security import A2ASecurity, AuthenticationManager
from .task_management import TaskManager, Task, Artifact

__all__ = [
    "A2ACommunicator",
    "A2AMessage", 
    "AgentDiscovery",
    "AgentCard",
    "A2ASecurity",
    "AuthenticationManager",
    "TaskManager",
    "Task",
    "Artifact"
]
