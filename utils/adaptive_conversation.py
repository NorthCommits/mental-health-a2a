"""
Adaptive conversation system using OpenAI for mental health assessment
"""
import openai
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from .logger import logger
from .intent_orchestrator import IntentOrchestrator, IntentType

class AdaptiveConversationManager:
    """Manages adaptive conversations for mental health assessment"""
    
    def __init__(self, openai_api_key: str, a2a_communicator=None, agent_discovery=None):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.conversation_history: List[Dict[str, Any]] = []
        self.assessment_focus: Dict[str, Any] = {}
        
        # Initialize intent orchestrator if A2A components are available
        self.intent_orchestrator = None
        if a2a_communicator and agent_discovery:
            self.intent_orchestrator = IntentOrchestrator(a2a_communicator, agent_discovery)
        
    def start_conversation(self, user_id: str) -> str:
        """Start a new adaptive conversation"""
        self.conversation_history = []
        self.assessment_focus = {
            "user_id": user_id,
            "started_at": datetime.utcnow().isoformat(),
            "areas_explored": [],
            "potential_concerns": [],
            "conversation_stage": "introduction"
        }
        
        initial_prompt = self._create_initial_prompt()
        
        # Log conversation start
        logger.log_assessment_progress(
            user_id=user_id,
            stage="introduction",
            user_input="",
            ai_response=initial_prompt
        )
        
        return initial_prompt
    
    def _create_initial_prompt(self) -> str:
        """Create initial conversation prompt"""
        return """Hello! I'm here to have a conversation with you about your mental health and well-being. This is a safe, non-judgmental space where we can talk about your experiences, thoughts, and feelings.

I'll ask you questions to better understand how you're doing, but feel free to share whatever feels important to you. There are no right or wrong answers - I'm just here to listen and learn about your unique experiences.

What would you like to talk about today? You can start with anything that's on your mind, whether it's about your daily life, relationships, work, or anything else that feels relevant to you."""
    
    async def generate_adaptive_response(self, 
                                 user_input: str, 
                                 user_id: str) -> Dict[str, Any]:
        """Generate adaptive response based on conversation context"""
        
        # Ensure assessment_focus is initialized
        if not hasattr(self, 'assessment_focus') or not self.assessment_focus:
            self.assessment_focus = {
                "user_id": user_id,
                "started_at": datetime.utcnow().isoformat(),
                "areas_explored": [],
                "potential_concerns": [],
                "conversation_stage": "introduction"
            }
        
        # Add user input to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Analyze user input for mental health patterns
        analysis = self._analyze_user_input(user_input)
        
        # Update assessment focus
        self._update_assessment_focus(analysis)
        
        # Generate adaptive response
        response = self._generate_llm_response(user_input, analysis)
        
        # Add AI response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Use intent orchestrator if available
        orchestration_result = None
        if self.intent_orchestrator:
            try:
                orchestration_result = await self.intent_orchestrator.orchestrate_conversation(
                    user_input, 
                    {
                        "conversation_stage": self.assessment_focus.get("conversation_stage", "introduction"),
                        "user_id": user_id,
                        "conversation_history": self.conversation_history,
                        "assessment_focus": self.assessment_focus
                    }
                )
                
                # Log orchestration results
                logger.log_system_event("intent_orchestration", {
                    "user_id": user_id,
                    "intent_type": orchestration_result.get("intent_type"),
                    "confidence": orchestration_result.get("confidence"),
                    "agents_contacted": len(orchestration_result.get("agent_responses", [])),
                    "agent_ids": [r["agent_id"] for r in orchestration_result.get("agent_responses", [])]
                })
                
            except Exception as e:
                logger.log_system_event("orchestration_error", {
                    "user_id": user_id,
                    "error": str(e)
                })
        
        # Fallback: Log agent triggers (legacy method)
        if not orchestration_result:
            triggered_agents = []
            if analysis.get('crisis_indicators') or analysis.get('urgency_level') == 'high':
                triggered_agents.append("Crisis Detection Agent")
            if analysis.get('depression_indicators') or analysis.get('anxiety_indicators'):
                triggered_agents.append("Primary Screening Agent")
            if analysis.get('adhd_indicators'):
                triggered_agents.append("Neurodevelopmental Assessment Agent (ADHD)")
            if analysis.get('asd_indicators'):
                triggered_agents.append("Neurodevelopmental Assessment Agent (ASD)")
            if analysis.get('trauma_indicators'):
                triggered_agents.append("Trauma Assessment Agent")
            
            # Always active agents
            always_active = ["Conversation Analysis Agent", "Response Generation Agent"]
            
            # Log agent activity
            if triggered_agents:
                logger.log_system_event("agents_triggered", {
                    "user_id": user_id,
                    "triggered_agents": triggered_agents,
                    "always_active": always_active,
                    "analysis": analysis
                })
        
        # Log the interaction
        logger.log_assessment_progress(
            user_id=user_id,
            stage=self.assessment_focus["conversation_stage"],
            user_input=user_input,
            ai_response=response,
            insights=analysis
        )
        
        return {
            "response": response,
            "analysis": analysis,
            "conversation_stage": self.assessment_focus["conversation_stage"],
            "areas_explored": self.assessment_focus.get("areas_explored", []),
            "potential_concerns": self.assessment_focus.get("potential_concerns", [])
        }
    
    def _analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for mental health patterns"""
        
        # Use OpenAI to analyze the input
        analysis_prompt = f"""
        Analyze the following user input for mental health assessment patterns. 
        Look for indicators of anxiety, depression, ADHD, ASD, trauma, or other mental health concerns.
        Also identify strengths, coping mechanisms, and areas of resilience.
        
        User input: "{user_input}"
        
        IMPORTANT: If the user mentions suicidal thoughts, self-harm, or wanting to die, set urgency_level to "high" and include crisis indicators.
        
        Provide analysis in JSON format with these fields:
        - anxiety_indicators: list of anxiety-related patterns
        - depression_indicators: list of depression-related patterns
        - adhd_indicators: list of ADHD-related patterns (restlessness, focus issues, hyperactivity)
        - asd_indicators: list of ASD-related patterns (sensory sensitivity, social difficulties)
        - trauma_indicators: list of trauma-related patterns
        - crisis_indicators: list of crisis-related patterns (suicidal thoughts, self-harm, etc.)
        - strengths: list of personal strengths mentioned
        - coping_strategies: list of coping mechanisms mentioned
        - emotional_state: overall emotional state
        - urgency_level: low/medium/high (set to "high" for crisis situations)
        - suggested_focus_areas: areas to explore further
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a mental health assessment AI. Analyze user input for mental health patterns and provide structured JSON responses."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            # Extract JSON from response
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                analysis_json = json.loads(analysis_text[json_start:json_end])
                return analysis_json
            else:
                return self._fallback_analysis(user_input)
                
        except Exception as e:
            logger.log_system_event("openai_error", {"error": str(e), "user_input": user_input})
            return self._fallback_analysis(user_input)
    
    def _fallback_analysis(self, user_input: str) -> Dict[str, Any]:
        """Fallback analysis when OpenAI fails"""
        text_lower = user_input.lower()
        
        return {
            "anxiety_indicators": [word for word in ['anxious', 'worry', 'nervous', 'panic', 'fear', 'stress'] if word in text_lower],
            "depression_indicators": [word for word in ['sad', 'depressed', 'hopeless', 'empty', 'tired', 'unmotivated'] if word in text_lower],
            "adhd_indicators": [word for word in ['distracted', 'focus', 'concentrate', 'hyperactive', 'impulsive', 'restless'] if word in text_lower],
            "asd_indicators": [word for word in ['social', 'sensory', 'routine', 'overwhelmed', 'sensitive', 'pattern'] if word in text_lower],
            "trauma_indicators": [word for word in ['trauma', 'ptsd', 'flashback', 'trigger', 'nightmare'] if word in text_lower],
            "strengths": [],
            "coping_strategies": [],
            "emotional_state": "neutral",
            "urgency_level": "low",
            "suggested_focus_areas": ["general_wellbeing"]
        }
    
    def _update_assessment_focus(self, analysis: Dict[str, Any]):
        """Update assessment focus based on analysis"""
        # Ensure areas_explored exists
        if "areas_explored" not in self.assessment_focus:
            self.assessment_focus["areas_explored"] = []
        
        # Ensure potential_concerns exists
        if "potential_concerns" not in self.assessment_focus:
            self.assessment_focus["potential_concerns"] = []
        
        # Update areas explored
        for area in analysis.get("suggested_focus_areas", []):
            if area not in self.assessment_focus["areas_explored"]:
                self.assessment_focus["areas_explored"].append(area)
        
        # Update potential concerns
        for concern_type in ["anxiety_indicators", "depression_indicators", "adhd_indicators", "asd_indicators", "trauma_indicators"]:
            if analysis.get(concern_type):
                concern = concern_type.replace("_indicators", "")
                if concern not in self.assessment_focus["potential_concerns"]:
                    self.assessment_focus["potential_concerns"].append(concern)
        
        # Update conversation stage
        if len(self.conversation_history) < 4:
            self.assessment_focus["conversation_stage"] = "introduction"
        elif len(self.conversation_history) < 8:
            self.assessment_focus["conversation_stage"] = "exploration"
        elif len(self.conversation_history) < 12:
            self.assessment_focus["conversation_stage"] = "deep_dive"
        else:
            self.assessment_focus["conversation_stage"] = "conclusion"
    
    def _generate_llm_response(self, user_input: str, analysis: Dict[str, Any]) -> str:
        """Generate adaptive response using OpenAI"""
        
        # Create context for the AI
        context = self._create_conversation_context(analysis)
        
        # Check for crisis indicators
        is_crisis = analysis.get('urgency_level') == 'high' or analysis.get('crisis_indicators')
        
        if is_crisis:
            response_prompt = f"""
            You are a compassionate mental health companion responding to someone in crisis.
            
            Context: {context}
            User's input: "{user_input}"
            
            The user has expressed thoughts of self-harm or suicide. Respond with:
            1. Immediate empathy and validation of their feelings
            2. Gentle exploration of their current safety
            3. Offer to continue the conversation and provide support
            4. Suggest crisis resources but don't end the conversation abruptly
            5. Maintain a warm, caring tone while being appropriately concerned
            
            Be supportive and understanding. Don't be clinical or cold.
            """
        else:
            response_prompt = f"""
            You are a compassionate, supportive mental health companion. 
            You're having a conversation with someone about their mental health and well-being.
            
            Context: {context}
            User's input: "{user_input}"
            
            Generate a thoughtful, adaptive response that:
            1. Acknowledges what they've shared with empathy
            2. Asks gentle follow-up questions to better understand their experiences
            3. Shows genuine care and understanding
            4. Continues the conversation naturally without being pushy
            5. Maintains a warm, supportive, non-judgmental tone
            
            Keep your response conversational and natural, like a caring friend or therapist would speak.
            Be supportive and encouraging. Don't be overly clinical or formal.
            Continue the conversation rather than ending it prematurely.
            """
        
        try:
            system_message = "You are a compassionate, supportive mental health companion. You listen with empathy, ask thoughtful questions, and provide gentle guidance. You're here to support and understand, not to diagnose or redirect to professionals unless there's a clear crisis. Continue conversations naturally and warmly."
            
            if is_crisis:
                system_message = "You are a compassionate mental health companion responding to someone in crisis. You provide immediate empathy, validate their feelings, and offer support while being appropriately concerned about their safety. You're warm, understanding, and don't end conversations abruptly."
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": response_prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.log_system_event("openai_response_error", {"error": str(e)})
            return self._fallback_response(user_input, analysis)
    
    def _create_conversation_context(self, analysis: Dict[str, Any]) -> str:
        """Create context for the AI based on conversation history"""
        context_parts = []
        
        # Add conversation stage
        context_parts.append(f"Conversation stage: {self.assessment_focus['conversation_stage']}")
        
        # Add areas explored
        if self.assessment_focus["areas_explored"]:
            context_parts.append(f"Areas explored: {', '.join(self.assessment_focus['areas_explored'])}")
        
        # Add potential concerns
        if self.assessment_focus["potential_concerns"]:
            context_parts.append(f"Potential concerns identified: {', '.join(self.assessment_focus['potential_concerns'])}")
        
        # Add recent conversation history
        recent_history = self.conversation_history[-4:]  # Last 4 exchanges
        if recent_history:
            context_parts.append("Recent conversation:")
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "You"
                context_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def _fallback_response(self, user_input: str, analysis: Dict[str, Any]) -> str:
        """Fallback response when OpenAI fails"""
        responses = [
            "Thank you for sharing that with me. Can you tell me more about how that makes you feel?",
            "I appreciate you being so open with me. What else would you like to discuss about your experiences?",
            "That's really helpful to understand. How do you typically handle situations like that?",
            "Thank you for trusting me with that information. What other aspects of your life would you like to talk about?",
            "I can see that's important to you. Can you help me understand more about your perspective on this?"
        ]
        
        # Simple response selection based on conversation length
        response_index = len(self.conversation_history) % len(responses)
        return responses[response_index]
    
    def check_completion_triggers(self, user_input: str) -> bool:
        """Check if user wants to end the conversation"""
        completion_triggers = [
            "i don't have anything else to say",
            "nothing more",
            "that's all",
            "i'm done",
            "nothing else",
            "that's everything",
            "i have nothing more to add",
            "i'm finished",
            "that's it",
            "nothing more to say",
            "i think that's everything",
            "i'm good",
            "that's all i have to say"
        ]
        
        return any(trigger in user_input.lower() for trigger in completion_triggers)
    
    def generate_final_report(self, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive mental health report"""
        
        # Use OpenAI to generate comprehensive report
        report_prompt = f"""
        Based on the following conversation history, generate a comprehensive mental health assessment report.
        
        Conversation History:
        {json.dumps(self.conversation_history, indent=2)}
        
        Assessment Focus:
        {json.dumps(self.assessment_focus, indent=2)}
        
        Generate a detailed report in JSON format with these sections:
        - conversation_summary: Brief summary of the conversation
        - identified_patterns: Mental health patterns identified
        - potential_conditions: Potential mental health conditions or concerns
        - strengths_identified: Personal strengths and positive qualities
        - recommendations: Specific recommendations for the user
        - resources: Helpful resources and next steps
        - urgency_level: Assessment of urgency (low/medium/high)
        - professional_referral: Whether professional help is recommended
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a mental health assessment AI. Generate comprehensive, professional assessment reports in valid JSON format."},
                    {"role": "user", "content": report_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            report_text = response.choices[0].message.content.strip()
            
            # Try to parse as JSON directly first
            try:
                report_json = json.loads(report_text)
            except json.JSONDecodeError:
                # Extract JSON from response if it's embedded in text
                json_start = report_text.find('{')
                json_end = report_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    try:
                        report_json = json.loads(report_text[json_start:json_end])
                    except json.JSONDecodeError:
                        report_json = self._fallback_report()
                else:
                    report_json = self._fallback_report()
            
            # Log report generation
            logger.log_assessment_progress(
                user_id=user_id,
                stage="report_generation",
                user_input="",
                ai_response="Final report generated",
                insights={"report_generated": True}
            )
            
            return report_json
            
        except Exception as e:
            logger.log_system_event("report_generation_error", {"error": str(e)})
            return self._fallback_report()
    
    def _fallback_report(self) -> Dict[str, Any]:
        """Fallback report when OpenAI fails"""
        return {
            "conversation_summary": "Conversation completed successfully",
            "identified_patterns": self.assessment_focus.get("potential_concerns", []),
            "potential_conditions": ["General mental health discussion"],
            "strengths_identified": ["Openness to self-reflection"],
            "recommendations": [
                "Consider speaking with a mental health professional",
                "Practice self-care activities",
                "Maintain social connections"
            ],
            "resources": [
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741"
            ],
            "urgency_level": "low",
            "professional_referral": "Considered"
        }
