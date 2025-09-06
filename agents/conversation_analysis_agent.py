"""
Conversation Analysis Agent
Real-time analysis of conversation patterns, sentiment, and insights
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from shared.a2a_protocol.communication_layer import A2AMessage, MessageType, ContentType, A2ACommunicator
from shared.a2a_protocol.agent_discovery import AgentDiscovery, AgentCard, CapabilityType, InputModality, OutputFormat
from shared.a2a_protocol.task_management import TaskManager
from shared.a2a_protocol.security import A2ASecurity
from utils.file_storage import FileStorageManager
from utils.logger import logger

@dataclass
class ConversationInsight:
    """Represents an insight from conversation analysis"""
    insight_type: str  # sentiment, topic, emotion, pattern
    value: str
    confidence: float
    timestamp: datetime
    context: Dict[str, Any]

@dataclass
class ConversationMetrics:
    """Conversation metrics and statistics"""
    total_messages: int
    user_messages: int
    ai_messages: int
    average_response_length: float
    sentiment_trend: List[float]
    topics_discussed: List[str]
    emotional_states: List[str]
    conversation_flow_score: float

class ConversationAnalysisAgent:
    """Real-time conversation analysis and pattern detection agent"""
    
    def __init__(
        self,
        agent_id: str = "conversation-analysis-001",
        a2a_communicator: A2ACommunicator = None,
        agent_discovery: AgentDiscovery = None,
        task_manager: TaskManager = None,
        security: A2ASecurity = None,
        storage_manager: FileStorageManager = None
    ):
        self.agent_id = agent_id
        self.a2a_communicator = a2a_communicator
        self.agent_discovery = agent_discovery
        self.task_manager = task_manager
        self.security = security
        self.storage_manager = storage_manager
        
        # Agent state
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.conversation_insights: Dict[str, List[ConversationInsight]] = {}
        
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
        
        logger.log_system_event("conversation_analysis_agent_initialized", {
            "agent_id": self.agent_id,
            "capabilities_registered": self._capabilities_registered
        })
    
    async def _register_message_handlers(self):
        """Register message handlers for different conversation analysis requests"""
        if self.a2a_communicator:
            await self.a2a_communicator.register_message_handler(
                "conversation_analysis",
                self._handle_conversation_analysis
            )
            await self.a2a_communicator.register_message_handler(
                "real_time_analysis",
                self._handle_real_time_analysis
            )
            await self.a2a_communicator.register_message_handler(
                "conversation_insights",
                self._handle_conversation_insights
            )
    
    async def _register_agent_capabilities(self):
        """Register agent capabilities with the discovery service"""
        agent_card = AgentCard(
            agent_id=self.agent_id,
            name="Conversation Analysis Agent",
            description="Real-time analysis of conversation patterns, sentiment, and insights for mental health assessment",
            version="1.0.0",
            capabilities=[
                {
                    "capability_type": CapabilityType.DATA_PROCESSING,
                    "description": "Real-time conversation analysis and pattern detection",
                    "input_modalities": [
                        InputModality.TEXT,
                        InputModality.AUDIO
                    ],
                    "output_formats": [
                        OutputFormat.STRUCTURED_DATA,
                        OutputFormat.TEXT_RESPONSE
                    ],
                    "parameters": {
                        "analysis_types": ["sentiment", "topic", "emotion", "pattern"],
                        "real_time_processing": True,
                        "confidence_threshold": 0.7
                    }
                }
            ],
            supported_languages=["en"],
            availability_status="available",
            contact_endpoint="http://localhost:8000/conversation-analysis",
            compliance_certifications=["hipaa"]
        )
        
        await self.agent_discovery.register_agent(agent_card)
        logger.log_system_event("conversation_analysis_agent_registered", {
            "agent_id": self.agent_id,
            "capabilities": len(agent_card.capabilities)
        })
    
    async def _handle_conversation_analysis(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle conversation analysis requests"""
        try:
            content = message.content
            conversation_id = content.get("conversation_id")
            conversation_data = content.get("conversation_data", [])
            
            # Analyze conversation
            analysis_result = await self.analyze_conversation(conversation_id, conversation_data)
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "analysis": analysis_result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.log_system_event("conversation_analysis_error", {"error": str(e)})
            return {"status": "error", "error": str(e)}
    
    async def _handle_real_time_analysis(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle real-time analysis requests"""
        try:
            content = message.content
            conversation_id = content.get("conversation_id")
            new_message = content.get("message")
            
            # Perform real-time analysis
            insights = await self.analyze_message(conversation_id, new_message)
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "insights": insights,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.log_system_event("real_time_analysis_error", {"error": str(e)})
            return {"status": "error", "error": str(e)}
    
    async def _handle_conversation_insights(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle conversation insights requests"""
        try:
            content = message.content
            conversation_id = content.get("conversation_id")
            
            # Get conversation insights
            insights = self.conversation_insights.get(conversation_id, [])
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "insights": [self._insight_to_dict(insight) for insight in insights],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.log_system_event("conversation_insights_error", {"error": str(e)})
            return {"status": "error", "error": str(e)}
    
    async def analyze_conversation(self, conversation_id: str, conversation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entire conversation for patterns and insights"""
        
        # Initialize conversation tracking
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = {
                "messages": [],
                "start_time": datetime.utcnow(),
                "last_analysis": None
            }
        
        # Update conversation data
        self.active_conversations[conversation_id]["messages"] = conversation_data
        
        # Perform analysis
        analysis = {
            "conversation_id": conversation_id,
            "total_messages": len(conversation_data),
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "sentiment_analysis": await self._analyze_sentiment(conversation_data),
            "topic_analysis": await self._analyze_topics(conversation_data),
            "emotional_analysis": await self._analyze_emotions(conversation_data),
            "conversation_flow": await self._analyze_conversation_flow(conversation_data),
            "insights": await self._generate_insights(conversation_data)
        }
        
        # Store analysis
        self.active_conversations[conversation_id]["last_analysis"] = analysis
        
        # Log analysis
        logger.log_system_event("conversation_analyzed", {
            "conversation_id": conversation_id,
            "message_count": len(conversation_data),
            "insights_generated": len(analysis["insights"])
        })
        
        return analysis
    
    async def analyze_message(self, conversation_id: str, message: Dict[str, Any]) -> List[ConversationInsight]:
        """Analyze a single message for real-time insights"""
        
        # Initialize conversation if needed
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = {
                "messages": [],
                "start_time": datetime.utcnow(),
                "last_analysis": None
            }
        
        # Add message to conversation
        self.active_conversations[conversation_id]["messages"].append(message)
        
        # Generate insights for this message
        insights = []
        
        # Sentiment insight
        sentiment = await self._analyze_message_sentiment(message)
        if sentiment:
            insights.append(ConversationInsight(
                insight_type="sentiment",
                value=sentiment["sentiment"],
                confidence=sentiment["confidence"],
                timestamp=datetime.utcnow(),
                context={"message_id": message.get("id", "unknown")}
            ))
        
        # Emotion insight
        emotion = await self._analyze_message_emotion(message)
        if emotion:
            insights.append(ConversationInsight(
                insight_type="emotion",
                value=emotion["emotion"],
                confidence=emotion["confidence"],
                timestamp=datetime.utcnow(),
                context={"message_id": message.get("id", "unknown")}
            ))
        
        # Topic insight
        topic = await self._analyze_message_topic(message)
        if topic:
            insights.append(ConversationInsight(
                insight_type="topic",
                value=topic["topic"],
                confidence=topic["confidence"],
                timestamp=datetime.utcnow(),
                context={"message_id": message.get("id", "unknown")}
            ))
        
        # Store insights
        if conversation_id not in self.conversation_insights:
            self.conversation_insights[conversation_id] = []
        self.conversation_insights[conversation_id].extend(insights)
        
        return insights
    
    async def _analyze_sentiment(self, conversation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment across conversation"""
        user_messages = [msg for msg in conversation_data if msg.get("role") == "user"]
        
        if not user_messages:
            return {"overall_sentiment": "neutral", "confidence": 0.0}
        
        # Simple sentiment analysis (in production, use more sophisticated models)
        positive_words = ["good", "great", "happy", "excited", "love", "enjoy", "wonderful", "amazing"]
        negative_words = ["bad", "terrible", "sad", "angry", "hate", "awful", "horrible", "depressed"]
        
        positive_count = sum(1 for msg in user_messages 
                           for word in positive_words 
                           if word in msg.get("content", "").lower())
        negative_count = sum(1 for msg in user_messages 
                           for word in negative_words 
                           if word in msg.get("content", "").lower())
        
        total_words = positive_count + negative_count
        if total_words == 0:
            return {"overall_sentiment": "neutral", "confidence": 0.0}
        
        sentiment_score = (positive_count - negative_count) / total_words
        
        if sentiment_score > 0.2:
            sentiment = "positive"
        elif sentiment_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "overall_sentiment": sentiment,
            "confidence": abs(sentiment_score),
            "positive_indicators": positive_count,
            "negative_indicators": negative_count
        }
    
    async def _analyze_topics(self, conversation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze topics discussed in conversation"""
        user_messages = [msg for msg in conversation_data if msg.get("role") == "user"]
        
        # Topic keywords
        topic_keywords = {
            "mental_health": ["anxiety", "depression", "stress", "mental", "therapy", "counseling"],
            "relationships": ["family", "friends", "partner", "relationship", "social", "lonely"],
            "work": ["job", "work", "career", "boss", "colleague", "office"],
            "health": ["health", "medical", "doctor", "medication", "treatment"],
            "daily_life": ["routine", "daily", "morning", "evening", "schedule", "habits"]
        }
        
        topic_counts = {topic: 0 for topic in topic_keywords}
        
        for msg in user_messages:
            content = msg.get("content", "").lower()
            for topic, keywords in topic_keywords.items():
                for keyword in keywords:
                    if keyword in content:
                        topic_counts[topic] += 1
        
        # Get most discussed topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        main_topics = [topic for topic, count in sorted_topics if count > 0]
        
        return {
            "main_topics": main_topics[:3],  # Top 3 topics
            "topic_distribution": topic_counts,
            "conversation_focus": main_topics[0] if main_topics else "general"
        }
    
    async def _analyze_emotions(self, conversation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze emotional states in conversation"""
        user_messages = [msg for msg in conversation_data if msg.get("role") == "user"]
        
        # Emotion keywords
        emotion_keywords = {
            "anxiety": ["worried", "anxious", "nervous", "panic", "fear", "scared"],
            "sadness": ["sad", "depressed", "down", "hopeless", "empty", "crying"],
            "anger": ["angry", "mad", "frustrated", "irritated", "annoyed"],
            "happiness": ["happy", "joyful", "excited", "cheerful", "content"],
            "confusion": ["confused", "lost", "unclear", "unsure", "puzzled"]
        }
        
        emotion_counts = {emotion: 0 for emotion in emotion_keywords}
        
        for msg in user_messages:
            content = msg.get("content", "").lower()
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in content:
                        emotion_counts[emotion] += 1
        
        # Get dominant emotions
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        dominant_emotions = [emotion for emotion, count in sorted_emotions if count > 0]
        
        return {
            "dominant_emotions": dominant_emotions[:3],  # Top 3 emotions
            "emotion_distribution": emotion_counts,
            "emotional_state": dominant_emotions[0] if dominant_emotions else "neutral"
        }
    
    async def _analyze_conversation_flow(self, conversation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation flow and engagement"""
        if len(conversation_data) < 2:
            return {"flow_score": 0.5, "engagement": "low"}
        
        # Calculate response lengths
        user_messages = [msg for msg in conversation_data if msg.get("role") == "user"]
        ai_messages = [msg for msg in conversation_data if msg.get("role") == "assistant"]
        
        avg_user_length = sum(len(msg.get("content", "")) for msg in user_messages) / len(user_messages) if user_messages else 0
        avg_ai_length = sum(len(msg.get("content", "")) for msg in ai_messages) / len(ai_messages) if ai_messages else 0
        
        # Calculate engagement score
        engagement_score = min(1.0, (avg_user_length / 100) * 0.5 + (len(user_messages) / 10) * 0.5)
        
        # Calculate flow score
        flow_score = min(1.0, len(conversation_data) / 20)  # Normalize to 20 messages
        
        return {
            "flow_score": flow_score,
            "engagement": "high" if engagement_score > 0.7 else "medium" if engagement_score > 0.4 else "low",
            "avg_user_message_length": avg_user_length,
            "avg_ai_message_length": avg_ai_length,
            "message_count": len(conversation_data)
        }
    
    async def _generate_insights(self, conversation_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable insights from conversation"""
        insights = []
        
        # Sentiment insights
        sentiment_analysis = await self._analyze_sentiment(conversation_data)
        if sentiment_analysis["overall_sentiment"] == "negative":
            insights.append({
                "type": "sentiment_warning",
                "message": "User appears to be expressing negative emotions",
                "priority": "medium",
                "suggestion": "Consider providing additional emotional support"
            })
        
        # Topic insights
        topic_analysis = await self._analyze_topics(conversation_data)
        if "mental_health" in topic_analysis["main_topics"]:
            insights.append({
                "type": "topic_focus",
                "message": "User is actively discussing mental health topics",
                "priority": "high",
                "suggestion": "Continue exploring mental health concerns"
            })
        
        # Engagement insights
        flow_analysis = await self._analyze_conversation_flow(conversation_data)
        if flow_analysis["engagement"] == "low":
            insights.append({
                "type": "engagement_low",
                "message": "User engagement appears low",
                "priority": "medium",
                "suggestion": "Consider asking more open-ended questions"
            })
        
        return insights
    
    async def _analyze_message_sentiment(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze sentiment of a single message"""
        content = message.get("content", "").lower()
        
        positive_words = ["good", "great", "happy", "excited", "love", "enjoy"]
        negative_words = ["bad", "terrible", "sad", "angry", "hate", "awful"]
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        if positive_count > negative_count:
            return {"sentiment": "positive", "confidence": 0.7}
        elif negative_count > positive_count:
            return {"sentiment": "negative", "confidence": 0.7}
        else:
            return {"sentiment": "neutral", "confidence": 0.5}
    
    async def _analyze_message_emotion(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze emotion of a single message"""
        content = message.get("content", "").lower()
        
        emotion_keywords = {
            "anxiety": ["worried", "anxious", "nervous", "panic"],
            "sadness": ["sad", "depressed", "down", "hopeless"],
            "anger": ["angry", "mad", "frustrated", "irritated"],
            "happiness": ["happy", "joyful", "excited", "cheerful"]
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in content for keyword in keywords):
                return {"emotion": emotion, "confidence": 0.8}
        
        return None
    
    async def _analyze_message_topic(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze topic of a single message"""
        content = message.get("content", "").lower()
        
        topic_keywords = {
            "mental_health": ["anxiety", "depression", "stress", "mental"],
            "relationships": ["family", "friends", "partner", "social"],
            "work": ["job", "work", "career", "boss"],
            "health": ["health", "medical", "doctor", "medication"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content for keyword in keywords):
                return {"topic": topic, "confidence": 0.8}
        
        return None
    
    def _insight_to_dict(self, insight: ConversationInsight) -> Dict[str, Any]:
        """Convert ConversationInsight to dictionary"""
        return {
            "insight_type": insight.insight_type,
            "value": insight.value,
            "confidence": insight.confidence,
            "timestamp": insight.timestamp.isoformat(),
            "context": insight.context
        }
    
    async def get_conversation_metrics(self, conversation_id: str) -> Optional[ConversationMetrics]:
        """Get comprehensive conversation metrics"""
        if conversation_id not in self.active_conversations:
            return None
        
        conversation = self.active_conversations[conversation_id]
        messages = conversation["messages"]
        
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        ai_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        
        avg_response_length = sum(len(msg.get("content", "")) for msg in messages) / len(messages) if messages else 0
        
        # Get sentiment trend
        sentiment_trend = []
        for msg in user_messages:
            sentiment = await self._analyze_message_sentiment(msg)
            if sentiment:
                sentiment_trend.append(1.0 if sentiment["sentiment"] == "positive" else -1.0 if sentiment["sentiment"] == "negative" else 0.0)
        
        # Get topics and emotions
        topic_analysis = await self._analyze_topics(messages)
        emotion_analysis = await self._analyze_emotions(messages)
        flow_analysis = await self._analyze_conversation_flow(messages)
        
        return ConversationMetrics(
            total_messages=len(messages),
            user_messages=len(user_messages),
            ai_messages=len(ai_messages),
            average_response_length=avg_response_length,
            sentiment_trend=sentiment_trend,
            topics_discussed=topic_analysis["main_topics"],
            emotional_states=emotion_analysis["dominant_emotions"],
            conversation_flow_score=flow_analysis["flow_score"]
        )
