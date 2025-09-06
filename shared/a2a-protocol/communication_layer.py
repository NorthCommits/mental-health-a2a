"""
A2A Communication Layer

Implements the core communication protocols for agent-to-agent interaction
using HTTP, SSE, and JSON-RPC as specified in the A2A protocol.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import httpx
import websockets
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of A2A messages"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    CAPABILITY_DISCOVERY = "capability_discovery"
    COLLABORATION = "collaboration"
    NOTIFICATION = "notification"
    CRISIS_ALERT = "crisis_alert"


class ContentType(str, Enum):
    """Supported content types for message parts"""
    TEXT = "text/plain"
    JSON = "application/json"
    AUDIO = "audio/wav"
    IMAGE = "image/jpeg"
    DOCUMENT = "application/pdf"
    SENSOR_DATA = "application/sensor+json"
    CLINICAL_DATA = "application/clinical+json"


class MessagePart(BaseModel):
    """Individual part of an A2A message"""
    content_type: ContentType
    content: str
    metadata: Optional[Dict[str, Any]] = None


class A2AMessage(BaseModel):
    """Core A2A message structure"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sender_agent_id: str
    recipient_agent_id: str
    message_type: MessageType
    parts: List[MessagePart]
    context: Optional[Dict[str, Any]] = None
    priority: int = Field(default=1, ge=1, le=10)  # 1=low, 10=critical
    correlation_id: Optional[str] = None


class A2ACommunicator:
    """
    Handles communication between agents using the A2A protocol.
    Supports both HTTP and WebSocket connections for real-time communication.
    """
    
    def __init__(self, agent_id: str, base_url: str = "http://localhost:8000"):
        self.agent_id = agent_id
        self.base_url = base_url
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        
    async def send_message(self, message: A2AMessage) -> bool:
        """
        Send a message to another agent via HTTP POST
        
        Args:
            message: The A2A message to send
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            url = f"{self.base_url}/agents/{message.recipient_agent_id}/messages"
            response = await self.http_client.post(
                url,
                json=message.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False
    
    async def send_websocket_message(self, message: A2AMessage) -> bool:
        """
        Send a message via WebSocket for real-time communication
        
        Args:
            message: The A2A message to send
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            if message.recipient_agent_id in self.websocket_connections:
                ws = self.websocket_connections[message.recipient_agent_id]
                await ws.send(json.dumps(message.model_dump()))
                return True
            else:
                # Fallback to HTTP if WebSocket not available
                return await self.send_message(message)
        except Exception as e:
            print(f"Failed to send WebSocket message: {e}")
            return False
    
    async def register_message_handler(self, message_type: MessageType, handler: Callable):
        """
        Register a handler for specific message types
        
        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
        """
        self.message_handlers[message_type] = handler
    
    async def handle_incoming_message(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process an incoming A2A message
        
        Args:
            message_data: Raw message data from JSON
            
        Returns:
            Response data if applicable
        """
        try:
            message = A2AMessage(**message_data)
            message_type = message.message_type
            
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                response = await handler(message)
                return response
            else:
                print(f"No handler registered for message type: {message_type}")
                return None
                
        except Exception as e:
            print(f"Error handling incoming message: {e}")
            return None
    
    async def create_task_request(
        self,
        recipient_agent_id: str,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 1
    ) -> A2AMessage:
        """
        Create a task request message
        
        Args:
            recipient_agent_id: ID of agent to receive the task
            task_description: Description of the task to perform
            context: Additional context for the task
            priority: Priority level (1-10)
            
        Returns:
            A2AMessage configured as a task request
        """
        parts = [
            MessagePart(
                content_type=ContentType.TEXT,
                content=task_description
            )
        ]
        
        return A2AMessage(
            sender_agent_id=self.agent_id,
            recipient_agent_id=recipient_agent_id,
            message_type=MessageType.TASK_REQUEST,
            parts=parts,
            context=context,
            priority=priority
        )
    
    async def create_crisis_alert(
        self,
        recipient_agent_id: str,
        crisis_data: Dict[str, Any],
        priority: int = 10
    ) -> A2AMessage:
        """
        Create a crisis alert message for emergency situations
        
        Args:
            recipient_agent_id: ID of agent to receive the alert
            crisis_data: Crisis-related data and context
            priority: Always set to 10 for crisis alerts
            
        Returns:
            A2AMessage configured as a crisis alert
        """
        parts = [
            MessagePart(
                content_type=ContentType.CLINICAL_DATA,
                content=json.dumps(crisis_data)
            )
        ]
        
        return A2AMessage(
            sender_agent_id=self.agent_id,
            recipient_agent_id=recipient_agent_id,
            message_type=MessageType.CRISIS_ALERT,
            parts=parts,
            context={"crisis_level": "high", "requires_immediate_attention": True},
            priority=priority
        )
    
    async def create_collaboration_message(
        self,
        recipient_agent_id: str,
        collaboration_data: Dict[str, Any],
        content_type: ContentType = ContentType.JSON
    ) -> A2AMessage:
        """
        Create a collaboration message for agent coordination
        
        Args:
            recipient_agent_id: ID of agent to collaborate with
            collaboration_data: Data to share for collaboration
            content_type: Type of content being shared
            
        Returns:
            A2AMessage configured for collaboration
        """
        parts = [
            MessagePart(
                content_type=content_type,
                content=json.dumps(collaboration_data) if content_type == ContentType.JSON else str(collaboration_data)
            )
        ]
        
        return A2AMessage(
            sender_agent_id=self.agent_id,
            recipient_agent_id=recipient_agent_id,
            message_type=MessageType.COLLABORATION,
            parts=parts,
            context={"collaboration_type": "data_sharing"}
        )
    
    async def close(self):
        """Clean up resources"""
        await self.http_client.aclose()
        for ws in self.websocket_connections.values():
            await ws.close()
