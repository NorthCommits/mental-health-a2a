"""
Mental Health A2A Agent Ecosystem - Streamlit Interface
A streamlined mental health support system with file-based storage and professional UI
"""

import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import streamlit as st
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import truststore

# Inject truststore into SSL for better certificate handling
truststore.inject_into_ssl()

# Load environment variables
load_dotenv()

# Import new adaptive conversation system
from utils.adaptive_conversation import AdaptiveConversationManager
from utils.agent_communication_manager import AgentCommunicationManager
from utils.logger import logger
import plotly.graph_objects as go
import asyncio
import threading
import time

# Import our A2A agents
from agents.primary_screening_agent.screening_agent import PrimaryScreeningAgent
from agents.crisis_detection_agent.crisis_agent import CrisisDetectionAgent
from agents.therapeutic_intervention_agent import TherapeuticInterventionAgent
from agents.care_coordination_agent import CareCoordinationAgent
from agents.progress_analytics_agent import ProgressAnalyticsAgent
from agents.neurodevelopmental_assessment_agent import NeurodevelopmentalAssessmentAgent, AssessmentType

# Import A2A protocol components
from shared.a2a_protocol.communication_layer import A2ACommunicator, A2AMessage, MessageType
from shared.a2a_protocol.agent_discovery import AgentDiscovery, AgentCard, CapabilityType
from shared.a2a_protocol.task_management import TaskManager, TaskPriority
from shared.a2a_protocol.security import A2ASecurity, AgentRole, AccessLevel

# File storage utilities
from utils.file_storage import FileStorageManager
from utils.session_manager import SessionManager
from utils.therapeutic_modules import TherapeuticModules

# Page configuration
st.set_page_config(
    page_title="Mental Health A2A Support System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for therapy-friendly design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .therapy-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .crisis-alert {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #f39c12;
    }
    
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    
    .agent-communication {
        background: #e3f2fd;
        border: 1px solid #bbdefb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Accessibility Styles */
    .accessibility-reduced-stimulation {
        filter: grayscale(20%) brightness(0.9);
        animation: none !important;
        transition: none !important;
    }
    
    .accessibility-high-contrast {
        background-color: #000000 !important;
        color: #ffffff !important;
        border-color: #ffffff !important;
    }
    
    .accessibility-high-contrast .therapy-card {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border-color: #ffffff !important;
    }
    
    .accessibility-large-text {
        font-size: 1.2em !important;
    }
    
    .accessibility-large-text h1 {
        font-size: 2.5em !important;
    }
    
    .accessibility-large-text h2 {
        font-size: 2em !important;
    }
    
    .accessibility-large-text h3 {
        font-size: 1.8em !important;
    }
    
    .sensory-friendly-button {
        background-color: #4CAF50;
        border: 2px solid #45a049;
        color: white;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        transition: background-color 0.3s;
    }
    
    .sensory-friendly-button:hover {
        background-color: #45a049;
    }
    
    .calm-colors {
        background: linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%);
    }
    
    .neurodivergent-support {
        background: #f0f8ff;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MentalHealthApp:
    """Main Streamlit application for the Mental Health A2A system"""
    
    def __init__(self):
        self.initialize_app()
        self.setup_file_storage()
        self.setup_agents()
        self.setup_adaptive_systems()
        
        # Ensure conversation_manager is always available
        if not hasattr(self, 'conversation_manager'):
            self.conversation_manager = None
        
        # Ensure agents are always available
        if not hasattr(self, 'primary_screening'):
            self.primary_screening = None
        if not hasattr(self, 'crisis_detection'):
            self.crisis_detection = None
        if not hasattr(self, 'therapeutic_intervention'):
            self.therapeutic_intervention = None
        if not hasattr(self, 'care_coordination'):
            self.care_coordination = None
        if not hasattr(self, 'progress_analytics'):
            self.progress_analytics = None
        if not hasattr(self, 'neurodevelopmental_assessment'):
            self.neurodevelopmental_assessment = None
    
    def setup_adaptive_systems(self):
        """Setup adaptive conversation and agent communication systems"""
        try:
            # Get OpenAI API key
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                logger.log_system_event("openai_key_missing", {"error": "OpenAI API key not found"})
                return
            
            # Check if A2A components are available
            a2a_communicator = getattr(self, 'a2a_communicator', None)
            agent_discovery = getattr(self, 'agent_discovery', None)
            
            # Initialize adaptive conversation manager
            self.conversation_manager = AdaptiveConversationManager(
                openai_api_key, 
                a2a_communicator=a2a_communicator,
                agent_discovery=agent_discovery
            )
            
            # Initialize agent communication manager (simplified for now)
            # We'll create a simple version that doesn't require A2A components
            self.agent_communication_manager = None
            
            logger.log_system_event("adaptive_systems_initialized", {
                "conversation_manager": True,
                "agent_communication_manager": "simplified",
                "a2a_communicator_available": a2a_communicator is not None,
                "agent_discovery_available": agent_discovery is not None,
                "intent_orchestrator_available": self.conversation_manager.intent_orchestrator is not None
            })
            
        except Exception as e:
            logger.log_system_event("adaptive_systems_error", {"error": str(e)})
    
    def initialize_app(self):
        """Initialize the Streamlit app state"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        if 'user_id' not in st.session_state:
            st.session_state.user_id = f"user_{int(time.time())}"
        
        if 'agents_initialized' not in st.session_state:
            st.session_state.agents_initialized = False
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if 'current_mood' not in st.session_state:
            st.session_state.current_mood = 5  # Neutral mood (1-10 scale)
        
        if 'crisis_detected' not in st.session_state:
            st.session_state.crisis_detected = False
    
    def setup_file_storage(self):
        """Initialize file-based storage system"""
        self.storage_manager = FileStorageManager()
        self.session_manager = SessionManager(self.storage_manager)
        self.therapeutic_modules = TherapeuticModules()
    
    def setup_agents(self):
        """Initialize all A2A agents"""
        if not st.session_state.agents_initialized:
            try:
                # Initialize A2A components
                self.a2a_communicator = A2ACommunicator("main-orchestrator", "http://localhost:8000")
                self.agent_discovery = AgentDiscovery()
                self.task_manager = TaskManager()
                self.security = A2ASecurity(secret_key="demo-secret-key")
                
                # Initialize agents
                self.primary_screening = PrimaryScreeningAgent(
                    agent_id="primary-screening-001",
                    openai_api_key="demo-key",  # Using demo key for now
                    a2a_communicator=self.a2a_communicator,
                    agent_discovery=self.agent_discovery,
                    task_manager=self.task_manager,
                    security=self.security
                )
                
                self.crisis_detection = CrisisDetectionAgent(
                    agent_id="crisis-detection-001",
                    a2a_communicator=self.a2a_communicator,
                    agent_discovery=self.agent_discovery,
                    task_manager=self.task_manager,
                    security=self.security
                )
                
                self.therapeutic_intervention = TherapeuticInterventionAgent(
                    agent_id="therapeutic-intervention-001",
                    a2a_communicator=self.a2a_communicator,
                    agent_discovery=self.agent_discovery,
                    task_manager=self.task_manager,
                    security=self.security,
                    storage_manager=self.storage_manager
                )
                
                self.care_coordination = CareCoordinationAgent(
                    agent_id="care-coordination-001",
                    a2a_communicator=self.a2a_communicator,
                    agent_discovery=self.agent_discovery,
                    task_manager=self.task_manager,
                    security=self.security,
                    storage_manager=self.storage_manager
                )
                
                self.progress_analytics = ProgressAnalyticsAgent(
                    agent_id="progress-analytics-001",
                    a2a_communicator=self.a2a_communicator,
                    agent_discovery=self.agent_discovery,
                    task_manager=self.task_manager,
                    security=self.security,
                    storage_manager=self.storage_manager
                )
                
                self.neurodevelopmental_assessment = NeurodevelopmentalAssessmentAgent(
                    agent_id="neurodevelopmental-assessment-001",
                    a2a_communicator=self.a2a_communicator,
                    agent_discovery=self.agent_discovery,
                    task_manager=self.task_manager,
                    security=self.security,
                    storage_manager=self.storage_manager
                )
                
                # Initialize agents asynchronously
                import asyncio
                asyncio.run(self.primary_screening.initialize())
                asyncio.run(self.crisis_detection.initialize())
                asyncio.run(self.therapeutic_intervention.initialize())
                asyncio.run(self.care_coordination.initialize())
                asyncio.run(self.progress_analytics.initialize())
                asyncio.run(self.neurodevelopmental_assessment.initialize())
                
                # Store agents in session state
                st.session_state.primary_screening = self.primary_screening
                st.session_state.crisis_detection = self.crisis_detection
                st.session_state.therapeutic_intervention = self.therapeutic_intervention
                st.session_state.care_coordination = self.care_coordination
                st.session_state.progress_analytics = self.progress_analytics
                st.session_state.neurodevelopmental_assessment = self.neurodevelopmental_assessment
                
                st.session_state.agents_initialized = True
                st.success("🧠 Mental Health A2A Agents initialized successfully!")
                
            except Exception as e:
                st.error(f"Failed to initialize agents: {str(e)}")
        else:
            # Load agents from session state
            self.primary_screening = st.session_state.get('primary_screening')
            self.crisis_detection = st.session_state.get('crisis_detection')
            self.therapeutic_intervention = st.session_state.get('therapeutic_intervention')
            self.care_coordination = st.session_state.get('care_coordination')
            self.progress_analytics = st.session_state.get('progress_analytics')
            self.neurodevelopmental_assessment = st.session_state.get('neurodevelopmental_assessment')
    
    def render_header(self):
        """Render the main header"""
        # Apply accessibility styles
        self._apply_accessibility_styles()
        
        st.markdown("""
        <div class="main-header">
            <h1>Mental Health A2A Support System</h1>
            <p>Comprehensive mental health support powered by collaborative AI agents</p>
            <p><strong>Session ID:</strong> {}</p>
        </div>
        """.format(st.session_state.session_id), unsafe_allow_html=True)
    
    def _apply_accessibility_styles(self):
        """Apply accessibility styles based on user preferences"""
        # Apply reduced stimulation
        if st.session_state.get('reduced_stimulation', False):
            st.markdown("""
            <style>
                .main .block-container {
                    filter: grayscale(20%) brightness(0.9);
                    animation: none !important;
                    transition: none !important;
                }
            </style>
            """, unsafe_allow_html=True)
        
        # Apply high contrast
        if st.session_state.get('high_contrast', False):
            st.markdown("""
            <style>
                .main .block-container {
                    background-color: #000000 !important;
                    color: #ffffff !important;
                }
                .therapy-card {
                    background-color: #1a1a1a !important;
                    color: #ffffff !important;
                    border-color: #ffffff !important;
                }
            </style>
            """, unsafe_allow_html=True)
        
        # Apply large text
        if st.session_state.get('large_text', False):
            st.markdown("""
            <style>
                .main .block-container {
                    font-size: 1.2em !important;
                }
                h1 { font-size: 2.5em !important; }
                h2 { font-size: 2em !important; }
                h3 { font-size: 1.8em !important; }
            </style>
            """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.title("Navigation")
        
        # Session info
        st.sidebar.markdown("### Session Info")
        st.sidebar.info(f"**User ID:** {st.session_state.user_id}")
        st.sidebar.info(f"**Session:** {st.session_state.session_id[:8]}...")
        
        # Accessibility settings
        st.sidebar.markdown("### Accessibility Settings")
        
        # Sensory accommodations
        if st.sidebar.checkbox("Reduce Visual Stimulation", value=st.session_state.get('reduced_stimulation', False)):
            st.session_state.reduced_stimulation = True
        else:
            st.session_state.reduced_stimulation = False
        
        if st.sidebar.checkbox("High Contrast Mode", value=st.session_state.get('high_contrast', False)):
            st.session_state.high_contrast = True
        else:
            st.session_state.high_contrast = False
        
        if st.sidebar.checkbox("Large Text Mode", value=st.session_state.get('large_text', False)):
            st.session_state.large_text = True
        else:
            st.session_state.large_text = False
        
        # Communication preferences
        communication_style = st.sidebar.selectbox(
            "Communication Style",
            ["Direct", "Gentle", "Detailed", "Concise"],
            index=st.session_state.get('communication_style_index', 0)
        )
        st.session_state.communication_style = communication_style
        st.session_state.communication_style_index = ["Direct", "Gentle", "Detailed", "Concise"].index(communication_style)
        
        # Quick mood check
        st.sidebar.markdown("### Quick Mood Check")
        mood = st.sidebar.slider("How are you feeling right now?", 1, 10, st.session_state.current_mood)
        if mood != st.session_state.current_mood:
            st.session_state.current_mood = mood
            self.storage_manager.save_mood_entry(st.session_state.user_id, mood)
            st.sidebar.success(f"Mood updated: {mood}/10")
        
        # Crisis resources
        st.sidebar.markdown("### Crisis Resources")
        if st.sidebar.button("Emergency Support"):
            st.session_state.crisis_detected = True
            st.rerun()
        
        st.sidebar.markdown("**National Suicide Prevention Lifeline:** 988")
        st.sidebar.markdown("**Crisis Text Line:** Text HOME to 741741")
        
        # System status
        st.sidebar.markdown("### System Status")
        if st.session_state.agents_initialized:
            st.sidebar.success("All agents online")
        else:
            st.sidebar.error("Agents offline")
    
    def render_main_interface(self):
        """Render the main ChatGPT-like interface"""
        # Main header
        st.markdown("""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; text-align: center;">Mental Health & Neurodiversity Assessment</h1>
            <p style="color: white; margin: 10px 0 0 0; text-align: center; opacity: 0.9;">AI-powered conversational assessment for mental health and neurodevelopmental traits</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main conversation area
        self.render_chat_interface()
    
    def render_chat_interface(self):
        """Render ChatGPT-like conversation interface"""
        # Initialize conversation state
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
            st.session_state.assessment_started = False
            st.session_state.assessment_complete = False
            st.session_state.mental_health_report = None
        
        # Display conversation
        chat_container = st.container()
        with chat_container:
            if not st.session_state.conversation:
                # Start conversation with adaptive system
                if self.conversation_manager:
                    welcome_message = self.conversation_manager.start_conversation(
                        st.session_state.get('user_id', 'anonymous')
                    )
                    
                    # Add welcome message to conversation
                    st.session_state.conversation.append({
                        'role': 'assistant',
                        'content': welcome_message,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Display welcome message
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #667eea;">
                        <h3 style="margin: 0 0 10px 0; color: #333;">Assessment Agent:</h3>
                        <p style="margin: 0; color: #666;">{welcome_message}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Fallback welcome message
                    st.markdown("""
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #667eea;">
                        <h3 style="margin: 0 0 10px 0; color: #333;">Welcome! I'm here to help you explore your mental health and neurodiversity.</h3>
                        <p style="margin: 0; color: #666;">This is a safe, non-judgmental space where we can have a natural conversation about your experiences, thoughts, and feelings. I'll ask questions to understand how your mind works and what you might be experiencing.</p>
                        <p style="margin: 10px 0 0 0; color: #666;"><strong>Just start typing below to begin our conversation!</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Display conversation history
                for message in st.session_state.conversation:
                    if message['role'] == 'assistant':
                        st.markdown(f"""
                        <div style="background: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #667eea;">
                            <strong>Assessment Agent:</strong> {message['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #2196f3;">
                            <strong>You:</strong> {message['content']}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Show mental health report if complete
        if st.session_state.assessment_complete and st.session_state.mental_health_report:
            st.markdown("---")
            st.markdown("## Mental Health Assessment Report")
            self.display_mental_health_report(st.session_state.mental_health_report)
        
        # Text input for user
        if not st.session_state.assessment_complete:
            # Use a form for better Enter key handling
            with st.form(key="user_input_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Your response:",
                    placeholder="Type your response here... (Press Enter to send)",
                    height=100,
                    help="Press Enter to send your message"
                )
                
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    submit_button = st.form_submit_button("Send", type="primary", use_container_width=True)
                
                with col2:
                    clear_button = st.form_submit_button("Clear", use_container_width=True)
            
            # Handle form submission
            if submit_button and user_input.strip():
                self.process_user_message(user_input.strip())
                st.rerun()
            
            if clear_button:
                st.session_state.conversation = []
                st.session_state.assessment_started = False
                st.session_state.assessment_complete = False
                st.session_state.mental_health_report = None
                st.rerun()
    
    def process_user_message(self, user_message: str):
        """Process user message and generate AI response using adaptive conversation system"""
        user_id = st.session_state.get('user_id', 'anonymous')
        
        # Add user message to conversation
        st.session_state.conversation.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Check for conversation completion triggers
        if hasattr(self, 'conversation_manager') and self.conversation_manager and hasattr(self.conversation_manager, 'check_completion_triggers'):
            if self.conversation_manager.check_completion_triggers(user_message):
                # Generate comprehensive mental health report
                self.generate_comprehensive_mental_health_report()
                return
        
        # Use adaptive conversation system
        if hasattr(self, 'conversation_manager') and self.conversation_manager:
            try:
                # Generate adaptive response
                import asyncio
                response_data = asyncio.run(self.conversation_manager.generate_adaptive_response(
                    user_input=user_message,
                    user_id=user_id
                ))
                
                ai_response = response_data['response']
                analysis = response_data['analysis']
                
                # Coordinate with agents if available
                if self.agent_communication_manager:
                    try:
                        agent_coordination = asyncio.run(
                            self.agent_communication_manager.coordinate_assessment(
                                user_id=user_id,
                                user_input=user_message,
                                conversation_context={
                                    'conversation_history': st.session_state.conversation,
                                    'assessment_focus': self.conversation_manager.assessment_focus,
                                    'analysis': analysis
                                }
                            )
                        )
                        
                        # Log agent coordination
                        logger.log_system_event("agent_coordination", {
                            "user_id": user_id,
                            "agents_consulted": len(agent_coordination.get('agents_involved', [])),
                            "overall_urgency": agent_coordination.get('overall_urgency', 'low')
                        })
                        
                    except Exception as e:
                        logger.log_system_event("agent_coordination_error", {"error": str(e)})
                
            except Exception as e:
                logger.log_system_event("adaptive_conversation_error", {"error": str(e)})
                # Fallback to simple response
                ai_response = self.generate_ai_response(user_message)
        else:
            # Fallback to simple response if adaptive system not available
            ai_response = self.generate_ai_response(user_message)
        
        # Add AI response to conversation
        st.session_state.conversation.append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().isoformat()
        })
    
    def generate_comprehensive_mental_health_report(self):
        """Generate comprehensive mental health report using adaptive conversation system"""
        user_id = st.session_state.get('user_id', 'anonymous')
        
        if self.conversation_manager:
            try:
                # Generate report using adaptive conversation system
                report = self.conversation_manager.generate_final_report(user_id)
                
                # Coordinate with agents for additional insights
                if self.agent_communication_manager:
                    try:
                        agent_insights = asyncio.run(
                            self.agent_communication_manager.generate_comprehensive_report(
                                user_id=user_id,
                                conversation_history=st.session_state.conversation,
                                agent_responses={}  # Will be populated by agent communication
                            )
                        )
                        
                        # Merge agent insights with conversation report
                        report['agent_insights'] = agent_insights
                        
                    except Exception as e:
                        logger.log_system_event("agent_report_error", {"error": str(e)})
                
                st.session_state.mental_health_report = report
                st.session_state.assessment_complete = True
                
                # Add completion message to conversation
                st.session_state.conversation.append({
                    'role': 'assistant',
                    'content': "Thank you for sharing your experiences with me. I've analyzed our conversation and prepared a comprehensive mental health assessment report for you. Please review it below.",
                    'timestamp': datetime.now().isoformat()
                })
                
                # Log report generation
                logger.log_system_event("comprehensive_report_generated", {
                    "user_id": user_id,
                    "conversation_length": len(st.session_state.conversation),
                    "report_sections": list(report.keys())
                })
                
            except Exception as e:
                logger.log_system_event("report_generation_error", {"error": str(e)})
                # Fallback to simple report
                self.generate_mental_health_report()
        else:
            # Fallback to simple report
            self.generate_mental_health_report()
    
    def generate_ai_response(self, user_message: str) -> str:
        """Generate AI response based on user message"""
        # This is a simplified version - in a real implementation, you'd use the agents
        conversation_length = len(st.session_state.conversation)
        
        if conversation_length == 1:  # First response
            return "Thank you for sharing that with me. I'd like to understand more about how you experience the world. Can you tell me about a typical day in your life? What activities do you enjoy, and what challenges do you face?"
        
        elif conversation_length <= 3:  # Early conversation
            return "That's really helpful to know. I'm curious about your social interactions - how do you feel when you're around other people? Do you prefer one-on-one conversations or group settings?"
        
        elif conversation_length <= 5:  # Mid conversation
            return "I appreciate you being so open with me. Let's talk about your emotions and how you handle stress. When you're feeling overwhelmed or anxious, what do you typically do to cope?"
        
        elif conversation_length <= 7:  # Later conversation
            return "Thank you for sharing those experiences. I'm interested in understanding your attention and focus. Do you find it easy to concentrate on tasks, or do your thoughts often wander to other things?"
        
        else:  # Extended conversation
            return "You've shared a lot with me, and I really appreciate your openness. Is there anything else about your experiences, thoughts, or feelings that you'd like to discuss? Or do you feel like you've covered everything you wanted to share?"
    
    def generate_mental_health_report(self):
        """Generate comprehensive mental health report based on conversation"""
        conversation_text = " ".join([msg['content'] for msg in st.session_state.conversation if msg['role'] == 'user'])
        
        # Analyze conversation for mental health patterns
        report = {
            'conversation_summary': self.analyze_conversation_patterns(conversation_text),
            'potential_conditions': self.identify_potential_conditions(conversation_text),
            'strengths': self.identify_strengths(conversation_text),
            'recommendations': self.generate_recommendations(conversation_text),
            'resources': self.suggest_resources(conversation_text),
            'next_steps': self.suggest_next_steps(conversation_text)
        }
        
        st.session_state.mental_health_report = report
        st.session_state.assessment_complete = True
        
        # Add completion message to conversation
        st.session_state.conversation.append({
            'role': 'assistant',
            'content': "Thank you for sharing your experiences with me. I've analyzed our conversation and prepared a comprehensive mental health assessment report for you. Please review it below.",
            'timestamp': datetime.now().isoformat()
        })
    
    def analyze_conversation_patterns(self, text: str) -> str:
        """Analyze conversation for mental health patterns"""
        text_lower = text.lower()
        
        patterns = []
        
        # Anxiety indicators
        if any(word in text_lower for word in ['anxious', 'worry', 'nervous', 'panic', 'fear', 'stress']):
            patterns.append("Anxiety-related concerns")
        
        # Depression indicators
        if any(word in text_lower for word in ['sad', 'depressed', 'hopeless', 'empty', 'tired', 'unmotivated']):
            patterns.append("Mood-related concerns")
        
        # ADHD indicators
        if any(word in text_lower for word in ['distracted', 'focus', 'concentrate', 'hyperactive', 'impulsive', 'restless']):
            patterns.append("Attention and focus challenges")
        
        # ASD indicators
        if any(word in text_lower for word in ['social', 'sensory', 'routine', 'overwhelmed', 'sensitive', 'pattern']):
            patterns.append("Sensory and social processing differences")
        
        # Sleep issues
        if any(word in text_lower for word in ['sleep', 'insomnia', 'tired', 'exhausted', 'rest']):
            patterns.append("Sleep-related concerns")
        
        if patterns:
            return f"Based on our conversation, I noticed several patterns: {', '.join(patterns)}. These patterns suggest areas where you might benefit from additional support or understanding."
        else:
            return "Our conversation revealed a range of experiences and perspectives that are unique to you. While no specific patterns stood out, your openness to discussing your experiences is valuable."
    
    def identify_potential_conditions(self, text: str) -> list:
        """Identify potential mental health conditions based on conversation"""
        text_lower = text.lower()
        conditions = []
        
        # Simple keyword-based analysis (in a real system, this would be more sophisticated)
        anxiety_keywords = ['anxious', 'worry', 'nervous', 'panic', 'fear', 'stress', 'overwhelmed']
        depression_keywords = ['sad', 'depressed', 'hopeless', 'empty', 'tired', 'unmotivated', 'worthless']
        adhd_keywords = ['distracted', 'focus', 'concentrate', 'hyperactive', 'impulsive', 'restless', 'forgetful']
        asd_keywords = ['social', 'sensory', 'routine', 'overwhelmed', 'sensitive', 'pattern', 'repetitive']
        
        if sum(1 for word in anxiety_keywords if word in text_lower) >= 3:
            conditions.append("Anxiety-related concerns")
        
        if sum(1 for word in depression_keywords if word in text_lower) >= 3:
            conditions.append("Depression-related concerns")
        
        if sum(1 for word in adhd_keywords if word in text_lower) >= 3:
            conditions.append("Attention and focus challenges (possible ADHD traits)")
        
        if sum(1 for word in asd_keywords if word in text_lower) >= 3:
            conditions.append("Sensory and social processing differences (possible ASD traits)")
        
        return conditions if conditions else ["No specific conditions identified - general mental health discussion"]
    
    def identify_strengths(self, text: str) -> list:
        """Identify personal strengths from conversation"""
        text_lower = text.lower()
        strengths = []
        
        if any(word in text_lower for word in ['creative', 'artistic', 'imaginative', 'innovative']):
            strengths.append("Creativity and innovation")
        
        if any(word in text_lower for word in ['helpful', 'caring', 'empathetic', 'supportive']):
            strengths.append("Empathy and caring nature")
        
        if any(word in text_lower for word in ['determined', 'persistent', 'resilient', 'strong']):
            strengths.append("Resilience and determination")
        
        if any(word in text_lower for word in ['curious', 'learning', 'interested', 'exploring']):
            strengths.append("Curiosity and love of learning")
        
        if any(word in text_lower for word in ['honest', 'authentic', 'genuine', 'real']):
            strengths.append("Authenticity and honesty")
        
        return strengths if strengths else ["Openness to self-reflection and growth"]
    
    def generate_recommendations(self, text: str) -> list:
        """Generate personalized recommendations"""
        recommendations = [
            "Consider speaking with a mental health professional for a comprehensive assessment",
            "Practice self-care activities that you enjoy",
            "Maintain a regular sleep schedule",
            "Engage in physical activity regularly",
            "Consider mindfulness or meditation practices"
        ]
        
        text_lower = text.lower()
        
        if 'social' in text_lower or 'lonely' in text_lower:
            recommendations.append("Consider joining social groups or activities that interest you")
        
        if 'stress' in text_lower or 'overwhelmed' in text_lower:
            recommendations.append("Learn stress management techniques like deep breathing or progressive muscle relaxation")
        
        if 'focus' in text_lower or 'distracted' in text_lower:
            recommendations.append("Try time management techniques like the Pomodoro method")
        
        return recommendations
    
    def suggest_resources(self, text: str) -> list:
        """Suggest helpful resources"""
        # Log Resource Recommendation Agent activity
        logger.log_system_event("resource_recommendation_agent_triggered", {
            "user_input": text,
            "agent": "Resource Recommendation Agent",
            "action": "suggesting_resources"
        })
        
        return [
            "National Suicide Prevention Lifeline: 988",
            "Crisis Text Line: Text HOME to 741741",
            "National Alliance on Mental Illness (NAMI): nami.org",
            "Mental Health America: mhanational.org",
            "ADHD Foundation: adhdfoundation.org",
            "Autism Society: autism-society.org"
        ]
    
    def suggest_next_steps(self, text: str) -> list:
        """Suggest next steps for the user"""
        return [
            "Schedule an appointment with a mental health professional",
            "Keep a journal of your thoughts and feelings",
            "Practice the recommended self-care activities",
            "Reach out to trusted friends or family members",
            "Consider joining a support group",
            "Follow up with this assessment in a few weeks"
        ]
    
    def display_mental_health_report(self, report: dict):
        """Display the mental health report"""
        st.markdown("## Mental Health Assessment Report")
        st.markdown("---")
        
        # Conversation Analysis
        if 'conversation_summary' in report:
            st.markdown("### Conversation Analysis")
            st.write(report['conversation_summary'])
        
        # Potential Areas of Focus
        if 'potential_conditions' in report and report['potential_conditions']:
            st.markdown("### Potential Areas of Focus")
            for condition in report['potential_conditions']:
                st.markdown(f"• {condition}")
        
        # Strengths
        if 'strengths' in report and report['strengths']:
            st.markdown("### Your Strengths")
            for strength in report['strengths']:
                st.markdown(f"• {strength}")
        elif 'strengths_identified' in report and report['strengths_identified']:
            st.markdown("### Your Strengths")
            for strength in report['strengths_identified']:
                st.markdown(f"• {strength}")
        
        # Recommendations
        if 'recommendations' in report and report['recommendations']:
            st.markdown("### Recommendations")
            for rec in report['recommendations']:
                st.markdown(f"• {rec}")
        
        # Resources
        if 'resources' in report and report['resources']:
            st.markdown("### Helpful Resources")
            for resource in report['resources']:
                st.markdown(f"• {resource}")
        
        # Next Steps
        if 'next_steps' in report and report['next_steps']:
            st.markdown("### Suggested Next Steps")
            for step in report['next_steps']:
                st.markdown(f"• {step}")
        
        # Urgency Level
        urgency = report.get('urgency_level', 'Low')
        urgency_color = {
            'Low': 'green',
            'Medium': 'orange', 
            'High': 'red'
        }.get(urgency, 'green')
        
        st.markdown(f"**Urgency Level:** :{urgency_color}[{urgency}]")
        
        # Professional Referral
        if report.get('professional_referral', False):
            st.markdown("### Professional Referral Recommended")
            st.warning("Based on our conversation, I recommend seeking professional mental health support. Please consider reaching out to a qualified mental health professional.")
        
        # Display any additional sections that might be present
        for key, value in report.items():
            if key not in ['conversation_summary', 'potential_conditions', 'strengths', 'strengths_identified', 
                          'recommendations', 'resources', 'next_steps', 'urgency_level', 'professional_referral']:
                if isinstance(value, (list, tuple)) and value:
                    st.markdown(f"### {key.replace('_', ' ').title()}")
                    for item in value:
                        st.markdown(f"• {item}")
                elif isinstance(value, str) and value:
                    st.markdown(f"### {key.replace('_', ' ').title()}")
                    st.write(value)
        
        st.markdown("---")
        st.info("**Important:** This assessment is for informational purposes only and should not replace professional medical advice. Please consult with a qualified mental health professional for a comprehensive evaluation.")
    
    def render_assessment_tab(self):
        """Render the initial assessment tab"""
        st.header("Welcome to Your Mental Health Assessment")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="therapy-card">
                <h3>Comprehensive Mental Health Screening</h3>
                <p>Our AI agents will work together to provide you with a thorough mental health assessment. 
                This includes validated screening tools and personalized recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Assessment form
            with st.form("assessment_form"):
                st.subheader("Assessment Information")
                
                # Basic info
                col_a, col_b = st.columns(2)
                with col_a:
                    age = st.number_input("Age", min_value=13, max_value=120, value=25)
                    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Non-binary", "Other"])
                
                with col_b:
                    primary_concern = st.selectbox("Primary Concern", [
                        "Depression", "Anxiety", "Stress", "Sleep Issues", 
                        "Relationship Problems", "Work/School Stress", "Other"
                    ])
                
                # Free-form input
                st.subheader("💭 Tell Us About Yourself")
                user_input = st.text_area(
                    "Please describe what's been on your mind lately, how you've been feeling, or any concerns you'd like to discuss:",
                    height=150,
                    placeholder="I've been feeling overwhelmed lately and having trouble sleeping..."
                )
                
                # File upload
                st.subheader("📄 Additional Information (Optional)")
                uploaded_files = st.file_uploader(
                    "Upload any documents, journal entries, or other materials you'd like to share:",
                    accept_multiple_files=True,
                    type=['txt', 'pdf', 'docx']
                )
                
                # Submit assessment
                submitted = st.form_submit_button("🚀 Start Assessment", use_container_width=True)
                
                if submitted and user_input:
                    with st.spinner("🤖 AI agents are analyzing your information..."):
                        # Process assessment
                        assessment_result = self.process_assessment(
                            user_input, age, gender, primary_concern, uploaded_files
                        )
                        
                        if assessment_result:
                            self.display_assessment_results(assessment_result)
        
        with col2:
            # Quick assessment tools
            st.subheader("Quick Tools")
            
            if st.button("PHQ-9 Depression Screen", use_container_width=True):
                self.run_quick_assessment("PHQ9")
            
            if st.button("GAD-7 Anxiety Screen", use_container_width=True):
                self.run_quick_assessment("GAD7")
            
            if st.button("Sleep Quality Check", use_container_width=True):
                self.run_quick_assessment("SLEEP")
            
            # Recent assessments
            st.subheader("Recent Assessments")
            recent_assessments = self.storage_manager.get_recent_assessments(st.session_state.user_id)
            for assessment in recent_assessments[:3]:
                assessment_type = assessment.get('primary_concern', 'General Assessment')
                assessment_date = assessment.get('timestamp', 'Unknown Date')
                st.info(f"**{assessment_type}** - {assessment_date}")
    
    def render_therapy_chat_tab(self):
        """Render the therapy chat interface"""
        st.header("💬 Therapy Chat")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Chat interface
            st.subheader("🤖 Chat with Your AI Therapist")
            
            # Display conversation history
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.conversation_history:
                    if message['role'] == 'user':
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**Therapist:** {message['content']}")
            
            # Chat input
            with st.form("chat_form"):
                user_message = st.text_area(
                    "Type your message here:",
                    placeholder="How are you feeling today? What would you like to talk about?",
                    height=100
                )
                
                col_a, col_b, col_c = st.columns([1, 1, 1])
                with col_a:
                    send_button = st.form_submit_button("💬 Send", use_container_width=True)
                with col_b:
                    voice_button = st.form_submit_button("🎤 Voice", use_container_width=True)
                with col_c:
                    clear_button = st.form_submit_button("🗑️ Clear", use_container_width=True)
            
            if send_button and user_message:
                self.process_chat_message(user_message)
            
            if clear_button:
                st.session_state.conversation_history = []
                st.rerun()
        
        with col2:
            # Therapy tools
            st.subheader("Therapy Tools")
            
            if st.button("Mindfulness Exercise", use_container_width=True):
                self.start_mindfulness_exercise()
            
            if st.button("Thought Journal", use_container_width=True):
                self.open_thought_journal()
            
            if st.button("Goal Setting", use_container_width=True):
                self.open_goal_setting()
            
            if st.button("Mood Check-in", use_container_width=True):
                self.open_mood_checkin()
            
            # Crisis detection status
            if st.session_state.crisis_detected:
                st.markdown("""
                <div class="crisis-alert">
                    <h4>Crisis Support Available</h4>
                    <p>We've detected that you might need immediate support. Please use the Crisis Support tab for resources.</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_neurodivergent_assessment_tab(self):
        """Render the neurodivergent assessment tab"""
        st.header("Neurodivergent Assessment")
        
        # Introduction with accessibility considerations
        communication_style = st.session_state.get('communication_style', 'Gentle')
        
        if communication_style == 'Direct':
            intro_text = """
            <div class="therapy-card neurodivergent-support">
                <h3>Neurodivergent Assessment</h3>
                <p>This assessment explores your unique cognitive and sensory processing patterns. 
                We'll discuss attention, social communication, sensory experiences, and behavioral preferences.</p>
                <p><strong>Purpose:</strong> Self-reflection tool to identify support needs and accommodations.</p>
            </div>
            """
        elif communication_style == 'Detailed':
            intro_text = """
            <div class="therapy-card neurodivergent-support">
                <h3>Understanding Your Neurodivergent Traits</h3>
                <p>This comprehensive assessment is designed to help you explore and understand your unique way of thinking, 
                processing information, and experiencing the world around you. Through a natural, conversational approach, 
                we'll discuss various aspects of your daily experiences including attention patterns, social interactions, 
                sensory processing preferences, and behavioral tendencies.</p>
                <p><strong>Important Note:</strong> This is not a diagnostic tool or medical assessment. It's a self-reflection 
                exercise designed to help you better understand yourself and identify areas where you might benefit from 
                additional support, accommodations, or resources.</p>
                <p><strong>Your Privacy:</strong> All responses are confidential and stored securely. You can stop or pause 
                the assessment at any time.</p>
            </div>
            """
        else:  # Gentle or Concise
            intro_text = """
            <div class="therapy-card neurodivergent-support">
                <h3>Understanding Your Neurodivergent Traits</h3>
                <p>This assessment is designed to help you explore your unique way of thinking, processing, and experiencing the world. 
                We'll have a natural conversation about your experiences with attention, social interactions, sensory processing, and more.</p>
                <p><strong>This is not a diagnostic tool</strong> - it's a self-reflection exercise to help you understand yourself better 
                and identify areas where you might benefit from support or accommodations.</p>
            </div>
            """
        
        st.markdown(intro_text, unsafe_allow_html=True)
        
        # Assessment type selection
        st.subheader("Choose Your Assessment Focus")
        
        col1, col2, col3 = st.columns(3)
        
        # Apply sensory-friendly styling to buttons
        button_style = "sensory-friendly-button" if st.session_state.get('reduced_stimulation', False) else ""
        
        with col1:
            if st.button("ADHD Traits", use_container_width=True, key="adhd_button"):
                st.session_state.assessment_type = AssessmentType.ADHD
                st.session_state.assessment_started = True
                st.rerun()
        
        with col2:
            if st.button("Autism Spectrum Traits", use_container_width=True, key="asd_button"):
                st.session_state.assessment_type = AssessmentType.ASD
                st.session_state.assessment_started = True
                st.rerun()
        
        with col3:
            if st.button("Combined Assessment", use_container_width=True, key="combined_button"):
                st.session_state.assessment_type = AssessmentType.COMBINED
                st.session_state.assessment_started = True
                st.rerun()
        
        # Add sensory accommodation options
        if st.session_state.get('reduced_stimulation', False):
            st.info("Sensory-friendly mode is active. Visual stimulation has been reduced.")
        
        if st.session_state.get('high_contrast', False):
            st.info("High contrast mode is active for better visibility.")
        
        if st.session_state.get('large_text', False):
            st.info("Large text mode is active for easier reading.")
        
        # Assessment conversation
        if st.session_state.get('assessment_started', False):
            self._render_assessment_conversation()
        
        # Previous assessments
        if st.session_state.get('neurodevelopmental_assessments'):
            st.subheader("Previous Assessments")
            for assessment in st.session_state.neurodevelopmental_assessments[:3]:
                st.info(f"**{assessment['assessment_type']}** - {assessment.get('completed_at', 'Unknown date')}")
    
    def _render_assessment_conversation(self):
        """Render the assessment conversation interface"""
        st.subheader("Assessment Conversation")
        
        # Initialize assessment if not already started
        if 'neurodevelopmental_assessment_id' not in st.session_state:
            if hasattr(self, 'neurodevelopmental_assessment') and self.neurodevelopmental_assessment is not None:
                import asyncio
                try:
                    initial_prompt = asyncio.run(self.neurodevelopmental_assessment.start_assessment(
                        user_id=st.session_state.user_id,
                        assessment_type=st.session_state.assessment_type
                    ))
                    st.session_state.neurodevelopmental_assessment_id = str(uuid.uuid4())
                    st.session_state.neurodevelopmental_conversation = [{
                        'role': 'assistant',
                        'content': initial_prompt,
                        'timestamp': datetime.now().isoformat()
                    }]
                except Exception as e:
                    st.error(f"Failed to start assessment: {str(e)}")
                    return
            else:
                st.error("Neurodevelopmental Assessment Agent not available")
                return
        
        # Display conversation history
        conversation = st.session_state.get('neurodevelopmental_conversation', [])
        for message in conversation:
            if message['role'] == 'assistant':
                st.markdown(f"""
                <div class="therapy-card">
                    <strong>Assessment Agent:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>You:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
        
        # User input
        with st.form("neurodevelopmental_assessment_form"):
            user_response = st.text_area(
                "Your response:",
                placeholder="Share your thoughts and experiences...",
                height=100
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                submit_button = st.form_submit_button("Send Response", use_container_width=True)
            with col2:
                if st.form_submit_button("Complete Assessment", use_container_width=True):
                    self._complete_neurodevelopmental_assessment()
                    return
            
            if submit_button and user_response:
                self._process_neurodevelopmental_response(user_response)
    
    def _process_neurodevelopmental_response(self, user_response: str):
        """Process user response in neurodevelopmental assessment"""
        if hasattr(self, 'neurodevelopmental_assessment') and self.neurodevelopmental_assessment is not None:
            import asyncio
            try:
                # Add user response to conversation
                st.session_state.neurodevelopmental_conversation.append({
                    'role': 'user',
                    'content': user_response,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Process with agent
                result = asyncio.run(self.neurodevelopmental_assessment.process_response(
                    st.session_state.neurodevelopmental_assessment_id,
                    user_response
                ))
                
                # Add agent response
                st.session_state.neurodevelopmental_conversation.append({
                    'role': 'assistant',
                    'content': result.get('next_prompt', 'Thank you for sharing that with me.'),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Check if assessment is complete
                if result.get('is_complete', False):
                    self._complete_neurodevelopmental_assessment()
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing response: {str(e)}")
        else:
            st.error("Neurodevelopmental Assessment Agent not available")
    
    def _complete_neurodevelopmental_assessment(self):
        """Complete the neurodevelopmental assessment"""
        st.success("Assessment completed! Thank you for sharing your experiences.")
        
        # Get assessment results
        if hasattr(self, 'neurodevelopmental_assessment') and self.neurodevelopmental_assessment is not None:
            import asyncio
            try:
                # This would get the actual results from the agent
                st.session_state.assessment_completed = True
                st.session_state.assessment_started = False
                st.rerun()
            except Exception as e:
                st.error(f"Error completing assessment: {str(e)}")
    
    def render_crisis_support_tab(self):
        """Render the crisis support tab"""
        st.header("Crisis Support")
        
        # Immediate crisis resources
        st.markdown("""
        <div class="crisis-alert">
            <h3>If you're in immediate danger, please call 911 or go to your nearest emergency room.</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Emergency Contacts")
            
            emergency_contacts = [
                ("National Suicide Prevention Lifeline", "988", "24/7 crisis support"),
                ("Crisis Text Line", "Text HOME to 741741", "24/7 text support"),
                ("SAMHSA National Helpline", "1-800-662-4357", "Mental health services"),
                ("Veterans Crisis Line", "1-800-273-8255", "Press 1 for veterans"),
                ("Disaster Distress Helpline", "1-800-985-5990", "Disaster-related stress")
            ]
            
            for name, number, description in emergency_contacts:
                st.markdown(f"**{name}**")
                st.markdown(f"**Phone:** {number}")
                st.markdown(f"**Info:** {description}")
                st.markdown("---")
        
        with col2:
            st.subheader("Safety Planning")
            
            if st.button("Create Safety Plan", use_container_width=True):
                self.create_safety_plan()
            
            if st.button("Contact Support Person", use_container_width=True):
                self.contact_support_person()
            
            if st.button("Find Local Resources", use_container_width=True):
                self.find_local_resources()
            
            # Crisis assessment
            st.subheader("Crisis Assessment")
            crisis_level = st.selectbox(
                "How would you describe your current crisis level?",
                ["I'm safe", "I'm struggling but safe", "I need immediate help", "I'm in danger"]
            )
            
            if crisis_level != "I'm safe":
                st.warning("Please consider reaching out to emergency services or a crisis hotline.")
                
                if st.button("Get Immediate Help"):
                    st.markdown("""
                    <div class="crisis-alert">
                        <h4>Emergency Resources Activated</h4>
                        <p>Please call 988 or text HOME to 741741 immediately.</p>
                        <p>If you're in immediate danger, call 911.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    def render_progress_tracking_tab(self):
        """Render the progress tracking tab"""
        st.header("Progress Tracking")
        
        # Mood tracking
        st.subheader("Mood Trends")
        
        # Get mood data
        mood_data = self.storage_manager.get_mood_history(st.session_state.user_id)
        
        if mood_data:
            df = pd.DataFrame(mood_data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Mood trend chart
            fig = px.line(df, x='date', y='mood', title='Mood Over Time')
            fig.update_layout(xaxis_title="Date", yaxis_title="Mood (1-10)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Mood statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Mood", f"{mood_data[-1]['mood']}/10")
            with col2:
                avg_mood = df['mood'].mean()
                st.metric("Average Mood", f"{avg_mood:.1f}/10")
            with col3:
                mood_trend = df['mood'].iloc[-7:].mean() - df['mood'].iloc[-14:-7].mean() if len(df) >= 14 else 0
                st.metric("7-Day Trend", f"{mood_trend:+.1f}")
            with col4:
                st.metric("Total Entries", len(mood_data))
        else:
            st.info("No mood data available yet. Start tracking your mood to see trends!")
        
        # Therapy progress
        st.subheader("Therapy Progress")
        
        # Get therapy sessions
        sessions = self.storage_manager.get_therapy_sessions(st.session_state.user_id)
        
        if sessions:
            # Session frequency
            session_dates = [session['date'] for session in sessions]
            session_df = pd.DataFrame({'date': session_dates})
            session_df['date'] = pd.to_datetime(session_df['date'])
            session_df['count'] = 1
            
            # Group by week
            session_df['week'] = session_df['date'].dt.to_period('W')
            weekly_sessions = session_df.groupby('week').size().reset_index(name='sessions')
            
            fig = px.bar(weekly_sessions, x='week', y='sessions', title='Therapy Sessions per Week')
            st.plotly_chart(fig, use_container_width=True)
            
            # Session insights
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Sessions", len(sessions))
                st.metric("This Week", len([s for s in sessions if (datetime.now() - datetime.fromisoformat(s['date'])).days <= 7]))
            with col2:
                st.metric("Average Session Length", f"{sum(s.get('duration', 0) for s in sessions) / len(sessions):.1f} min")
                st.metric("Last Session", sessions[-1]['date'] if sessions else "Never")
        else:
            st.info("No therapy sessions recorded yet. Start chatting to track your progress!")
        
        # Goals and milestones
        st.subheader("Goals & Milestones")
        
        goals = self.storage_manager.get_user_goals(st.session_state.user_id)
        
        if goals:
            for goal in goals:
                progress = goal.get('progress', 0)
                st.progress(progress / 100)
                st.write(f"**{goal['title']}** - {progress}% complete")
                st.write(f"*{goal['description']}*")
        else:
            st.info("No goals set yet. Set some goals to track your progress!")
            if st.button("Set New Goal"):
                self.set_new_goal()
    
    def render_developer_monitor_tab(self):
        """Render the developer monitoring tab"""
        st.header("Developer Monitor")
        
        # Agent status
        st.subheader("Agent Status")
        
        agents = [
            ("Primary Screening", st.session_state.get('primary_screening'), "Screening"),
            ("Crisis Detection", st.session_state.get('crisis_detection'), "Crisis"),
            ("Therapeutic Intervention", st.session_state.get('therapeutic_intervention'), "Therapy"),
            ("Care Coordination", st.session_state.get('care_coordination'), "Care"),
            ("Progress Analytics", st.session_state.get('progress_analytics'), "Analytics"),
            ("Neurodivergent Assessment", st.session_state.get('neurodevelopmental_assessment'), "Assessment")
        ]
        
        for name, agent, icon in agents:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.write(f"{icon} {name}")
            with col2:
                status = " Online" if agent is not None else " Offline"
                st.write(status)
            with col3:
                if st.button(f"Test {name}", key=f"test_{name}"):
                    if agent is not None:
                        self.test_agent(agent)
                    else:
                        st.warning(f"{name} agent not initialized")
        
        # Intent Orchestrator Status
        st.subheader("🎯 Intent Orchestrator")
        
        if hasattr(self.conversation_manager, 'intent_orchestrator') and self.conversation_manager.intent_orchestrator:
            st.success("✅ Intent orchestrator is active and routing requests to appropriate agents.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Show Agent Status"):
                    try:
                        import asyncio
                        status = asyncio.run(self.conversation_manager.intent_orchestrator.get_agent_status())
                        st.json(status)
                    except Exception as e:
                        st.error(f"Error getting agent status: {e}")
            
            with col2:
                if st.button("Test Orchestration"):
                    st.info("Orchestration is automatically triggered during conversations. Check the logs below for orchestration events.")
        else:
            st.warning("⚠️ Intent orchestrator is not available. A2A components need to be fully initialized.")
        
        # Real-time communication log
        st.subheader("📡 Agent Communications")
        
        # Get communication logs
        comm_logs = self.storage_manager.get_communication_logs()
        
        if comm_logs:
            for log in comm_logs[-10:]:  # Show last 10 communications
                st.markdown(f"""
                <div class="agent-communication">
                    <strong>{log['timestamp']}</strong> - {log['from_agent']} → {log['to_agent']}
                    <br><strong>Type:</strong> {log['message_type']}
                    <br><strong>Content:</strong> {log['content'][:200]}...
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No agent communications logged yet.")
        
        # System metrics
        st.subheader("System Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sessions = len(self.storage_manager.get_all_sessions())
            st.metric("Total Sessions", total_sessions)
        
        with col2:
            total_messages = len(st.session_state.conversation_history)
            st.metric("Messages Today", total_messages)
        
        with col3:
            crisis_alerts = len(self.storage_manager.get_crisis_alerts())
            st.metric("Crisis Alerts", crisis_alerts)
        
        with col4:
            active_agents = sum(1 for _, agent, _ in agents if agent)
            st.metric("Active Agents", active_agents)
        
        # File system status
        st.subheader("💾 File System Status")
        
        storage_stats = self.storage_manager.get_storage_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Files:** {storage_stats['total_files']}")
            st.write(f"**Total Size:** {storage_stats['total_size']} MB")
        with col2:
            st.write(f"**Sessions:** {storage_stats['sessions']}")
            st.write(f"**Logs:** {storage_stats['logs']}")
    
    def process_assessment(self, user_input, age, gender, primary_concern, uploaded_files):
        """Process a mental health assessment"""
        try:
            # Create assessment data
            assessment_data = {
                'user_id': st.session_state.user_id,
                'session_id': st.session_state.session_id,
                'age': age,
                'gender': gender,
                'primary_concern': primary_concern,
                'user_input': user_input,
                'uploaded_files': [f.name for f in uploaded_files] if uploaded_files else [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Save assessment data
            self.storage_manager.save_assessment(assessment_data)
            
            # Process with Primary Screening Agent
            if hasattr(self, 'primary_screening') and self.primary_screening is not None:
                import asyncio
                session_id = asyncio.run(self.primary_screening.start_screening_session(
                    user_id=st.session_state.user_id,
                    input_data=assessment_data
                ))
                
                # Wait a moment for processing and get results
                import time
                time.sleep(2)  # Give time for processing
                
                results = asyncio.run(self.primary_screening.get_assessment_results(session_id))
                result = {
                    'session_id': session_id,
                    'results': results,
                    'status': 'completed'
                }
            else:
                st.error("Primary Screening Agent not available. Please initialize agents first.")
                return None
            
            return result
            
        except Exception as e:
            st.error(f"Assessment processing failed: {str(e)}")
            return None
    
    def display_assessment_results(self, results):
        """Display assessment results"""
        st.success("Assessment completed successfully!")
        
        # Handle different result structures
        if isinstance(results, dict):
            if 'results' in results:
                # Results from process_initial_assessment
                assessment_results = results.get('results', [])
                session_id = results.get('session_id', 'Unknown')
                st.info(f"Session ID: {session_id}")
            else:
                # Direct results
                assessment_results = [results]
        elif isinstance(results, list):
            assessment_results = results
        else:
            st.error("Invalid results format")
            return
        
        # Display results
        if assessment_results:
            for result in assessment_results:
                if isinstance(result, dict):
                    st.markdown(f"""
                    <div class="therapy-card">
                        <h4>{result.get('assessment_type', 'Assessment')} Results</h4>
                        <p><strong>Score:</strong> {result.get('score', 'N/A')}</p>
                        <p><strong>Severity:</strong> {result.get('severity', 'N/A')}</p>
                        <p><strong>Risk Level:</strong> {result.get('risk_level', 'N/A')}</p>
                        <p><strong>Recommendations:</strong></p>
                        <ul>
                            {''.join([f'<li>{rec}</li>' for rec in result.get('recommendations', [])])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"Result: {result}")
        else:
            st.info("No assessment results available")
    
    def process_chat_message(self, message):
        """Process a chat message"""
        # Add user message to history
        st.session_state.conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Process with Therapeutic Intervention Agent
        if hasattr(self, 'therapeutic_intervention') and self.therapeutic_intervention is not None:
            import asyncio
            
            # Check if we have an active therapy session
            if 'therapy_session_id' not in st.session_state:
                # Start a new therapy session
                try:
                    from agents.therapeutic_intervention_agent import TherapySessionType, InterventionType
                    session_id = asyncio.run(self.therapeutic_intervention.start_therapy_session(
                        user_id=st.session_state.user_id,
                        session_type=TherapySessionType.INDIVIDUAL,
                        intervention_type=InterventionType.CBT,
                        context={'conversation_history': st.session_state.conversation_history}
                    ))
                    st.session_state.therapy_session_id = session_id
                except Exception as e:
                    st.error(f"Failed to start therapy session: {str(e)}")
                    response = "I'm here to help. Could you tell me more about what's on your mind?"
                    st.session_state.conversation_history.append({
                        'role': 'assistant',
                        'content': response,
                        'timestamp': datetime.now().isoformat()
                    })
                    return
            
            # Conduct the intervention
            try:
                intervention_result = asyncio.run(self.therapeutic_intervention.conduct_intervention(
                    session_id=st.session_state.therapy_session_id,
                    user_input=message,
                    context={'conversation_history': st.session_state.conversation_history}
                ))
                
                # Extract response from intervention result
                if isinstance(intervention_result, dict):
                    response = intervention_result.get('response', 'I understand. How does that make you feel?')
                else:
                    response = str(intervention_result)
                    
            except Exception as e:
                # Fallback to a more therapeutic response
                response = self._generate_therapeutic_response(message)
        else:
            response = "Therapeutic Intervention Agent not available. Please initialize agents first."
        
        # Add agent response to history
        st.session_state.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save conversation
        self.storage_manager.save_conversation(st.session_state.user_id, message, response)
        
        st.rerun()
    
    def _generate_therapeutic_response(self, message: str) -> str:
        """Generate a basic therapeutic response as fallback"""
        # Simple keyword-based responses for demonstration
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['tired', 'exhausted', 'fatigue']):
            return "I hear that you're feeling tired. It sounds like you might be experiencing some stress or overwhelm. Can you tell me more about what's contributing to this feeling?"
        
        elif any(word in message_lower for word in ['work', 'job', 'career', 'professional']):
            return "Work can be a significant source of stress. It sounds like you're dealing with some challenges there. What aspects of your work are most difficult right now?"
        
        elif any(word in message_lower for word in ['sad', 'depressed', 'down', 'blue']):
            return "I understand you're feeling down. These feelings can be really difficult to navigate. What's been weighing on your mind lately?"
        
        elif any(word in message_lower for word in ['anxious', 'worried', 'nervous', 'anxiety']):
            return "Anxiety can feel overwhelming. It sounds like you might be experiencing some worry or nervousness. What's making you feel most anxious right now?"
        
        elif any(word in message_lower for word in ['ok', 'okay', 'fine', 'good']):
            return "I appreciate you sharing that with me. Sometimes 'okay' can mean different things. How are you really feeling today?"
        
        else:
            return "Thank you for sharing that with me. I'm here to listen and help. Can you tell me more about what's on your mind or how you're feeling?"
    
    def run_quick_assessment(self, assessment_type):
        """Run a quick assessment"""
        st.info(f"Running {assessment_type} assessment...")
        # Implementation would go here
        pass
    
    def start_mindfulness_exercise(self):
        """Start a mindfulness exercise"""
        st.info("Starting mindfulness exercise...")
        
        # Add mindfulness exercise to conversation
        exercise_text = """
        **Mindfulness Breathing Exercise**
        
        Let's take a moment to focus on your breathing:
        
        1. **Find a comfortable position** - sit or stand in a way that feels natural
        2. **Close your eyes gently** or focus on a spot in front of you
        3. **Breathe in slowly** through your nose for 4 counts
        4. **Hold your breath** for 4 counts
        5. **Breathe out slowly** through your mouth for 6 counts
        6. **Repeat this cycle** 3-5 times
        
        Notice how your body feels with each breath. If your mind wanders, gently bring your attention back to your breathing.
        
        How does this feel for you?
        """
        
        st.session_state.conversation_history.append({
            'role': 'assistant',
            'content': exercise_text,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save to conversation
        self.storage_manager.save_conversation(
            st.session_state.user_id,
            "Mindfulness exercise requested",
            exercise_text
        )
        
        st.rerun()
    
    def open_thought_journal(self):
        """Open thought journal"""
        st.info("Opening thought journal...")
        
        # Add thought journal prompt to conversation
        journal_text = """
        **Thought Journal Exercise**
        
        Let's explore your thoughts and feelings through writing:
        
        **Prompt 1: What's on your mind right now?**
        Take a moment to write down whatever thoughts are present. Don't judge them, just observe and record.
        
        **Prompt 2: What emotions are you experiencing?**
        Notice what feelings are present in your body and mind. Name them if you can.
        
        **Prompt 3: What triggered these thoughts/feelings?**
        Consider what situation, event, or memory might have brought these up.
        
        **Prompt 4: How can you be kind to yourself right now?**
        What would you say to a good friend in this situation?
        
        Feel free to share any insights or thoughts that come up during this exercise.
        """
        
        st.session_state.conversation_history.append({
            'role': 'assistant',
            'content': journal_text,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save to conversation
        self.storage_manager.save_conversation(
            st.session_state.user_id,
            "Thought journal requested",
            journal_text
        )
        
        st.rerun()
    
    def open_goal_setting(self):
        """Open goal setting interface"""
        st.info("Opening goal setting...")
        
        # Add goal setting exercise to conversation
        goal_text = """
        **SMART Goal Setting Exercise**
        
        Let's work together to set some meaningful goals for your mental health and wellbeing:
        
        **S - Specific**: What exactly do you want to achieve?
        **M - Measurable**: How will you know when you've achieved it?
        **A - Achievable**: Is this goal realistic for you right now?
        **R - Relevant**: Why is this goal important to you?
        **T - Time-bound**: When would you like to achieve this?
        
        **Some areas to consider:**
        - Daily self-care routines
        - Stress management techniques
        - Social connections
        - Physical health
        - Work-life balance
        - Personal growth
        
        What's one small goal you'd like to work on this week?
        """
        
        st.session_state.conversation_history.append({
            'role': 'assistant',
            'content': goal_text,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save to conversation
        self.storage_manager.save_conversation(
            st.session_state.user_id,
            "Goal setting requested",
            goal_text
        )
        
        st.rerun()
    
    def open_mood_checkin(self):
        """Open mood check-in"""
        st.info("Opening mood check-in...")
        # Implementation would go here
        pass
    
    def create_safety_plan(self):
        """Create a safety plan"""
        st.info("Creating safety plan...")
        # Implementation would go here
        pass
    
    def contact_support_person(self):
        """Contact support person"""
        st.info("Contacting support person...")
        # Implementation would go here
        pass
    
    def find_local_resources(self):
        """Find local resources"""
        st.info("Finding local resources...")
        # Implementation would go here
        pass
    
    def set_new_goal(self):
        """Set a new goal"""
        st.info("Setting new goal...")
        # Implementation would go here
        pass
    
    def test_agent(self, agent):
        """Test an agent"""
        st.info(f"Testing {agent.__class__.__name__}...")
        # Implementation would go here
        pass
    
    def run(self):
        """Run the main application"""
        # Initialize agents
        self.setup_agents()
        
        # Render main interface (no sidebar)
        self.render_main_interface()

# Main execution
if __name__ == "__main__":
    st.set_page_config(
        page_title="Mental Health Assessment",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    app = MentalHealthApp()
    app.run()
