"""
Therapeutic modules and evidence-based interventions
Stores therapeutic content as JSON configuration files
"""

import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

class TherapeuticModules:
    """Manages therapeutic modules and evidence-based interventions"""
    
    def __init__(self, modules_dir: str = "data/therapeutic_modules"):
        self.modules_dir = Path(modules_dir)
        self.modules_dir.mkdir(parents=True, exist_ok=True)
        self.initialize_modules()
    
    def initialize_modules(self):
        """Initialize default therapeutic modules"""
        if not (self.modules_dir / "modules.json").exists():
            self.create_default_modules()
    
    def create_default_modules(self):
        """Create default therapeutic modules"""
        modules = {
            "cbt_techniques": {
                "name": "Cognitive Behavioral Therapy Techniques",
                "description": "Evidence-based CBT interventions for mental health",
                "techniques": [
                    {
                        "id": "thought_challenging",
                        "name": "Thought Challenging",
                        "description": "Identify and challenge negative thought patterns",
                        "steps": [
                            "Identify the negative thought",
                            "Write down evidence for and against the thought",
                            "Consider alternative explanations",
                            "Develop a balanced perspective"
                        ],
                        "prompts": [
                            "What evidence do you have that supports this thought?",
                            "What evidence contradicts this thought?",
                            "What would you tell a friend in this situation?",
                            "How likely is this worst-case scenario?"
                        ]
                    },
                    {
                        "id": "behavioral_activation",
                        "name": "Behavioral Activation",
                        "description": "Increase engagement in positive activities",
                        "steps": [
                            "Identify activities you used to enjoy",
                            "Start with small, manageable activities",
                            "Schedule activities in advance",
                            "Track mood before and after activities"
                        ],
                        "prompts": [
                            "What activities used to bring you joy?",
                            "What small step could you take today?",
                            "How did you feel after completing this activity?"
                        ]
                    },
                    {
                        "id": "problem_solving",
                        "name": "Problem Solving",
                        "description": "Systematic approach to solving problems",
                        "steps": [
                            "Define the problem clearly",
                            "Brainstorm possible solutions",
                            "Evaluate pros and cons of each solution",
                            "Choose and implement the best solution",
                            "Review and adjust as needed"
                        ],
                        "prompts": [
                            "What exactly is the problem you're facing?",
                            "What are all the possible ways to address this?",
                            "What are the pros and cons of each option?",
                            "Which solution feels most manageable?"
                        ]
                    }
                ]
            },
            "mindfulness_exercises": {
                "name": "Mindfulness and Meditation",
                "description": "Mindfulness practices for stress reduction and emotional regulation",
                "exercises": [
                    {
                        "id": "breathing_meditation",
                        "name": "Breathing Meditation",
                        "duration": "5-10 minutes",
                        "description": "Focus on your breath to calm the mind",
                        "instructions": [
                            "Find a comfortable seated position",
                            "Close your eyes or soften your gaze",
                            "Focus on your natural breathing",
                            "When your mind wanders, gently return to your breath",
                            "Continue for 5-10 minutes"
                        ],
                        "benefits": ["Reduces stress", "Improves focus", "Calms the nervous system"]
                    },
                    {
                        "id": "body_scan",
                        "name": "Body Scan Meditation",
                        "duration": "10-15 minutes",
                        "description": "Systematically scan your body for tension",
                        "instructions": [
                            "Lie down comfortably or sit in a chair",
                            "Start at the top of your head",
                            "Slowly scan down through your body",
                            "Notice any tension or discomfort",
                            "Breathe into areas of tension",
                            "Continue until you reach your toes"
                        ],
                        "benefits": ["Reduces physical tension", "Improves body awareness", "Promotes relaxation"]
                    },
                    {
                        "id": "loving_kindness",
                        "name": "Loving-Kindness Meditation",
                        "duration": "10-15 minutes",
                        "description": "Cultivate compassion for yourself and others",
                        "instructions": [
                            "Sit comfortably and close your eyes",
                            "Start by directing loving-kindness to yourself",
                            "Repeat phrases like 'May I be happy, may I be healthy'",
                            "Gradually extend to loved ones, acquaintances, and all beings",
                            "Feel the warmth and compassion in your heart"
                        ],
                        "benefits": ["Increases compassion", "Reduces negative emotions", "Improves relationships"]
                    }
                ]
            },
            "crisis_interventions": {
                "name": "Crisis Intervention Techniques",
                "description": "Immediate support strategies for crisis situations",
                "interventions": [
                    {
                        "id": "safety_planning",
                        "name": "Safety Planning",
                        "description": "Create a plan for staying safe during crisis",
                        "steps": [
                            "Identify warning signs that indicate you're in crisis",
                            "List coping strategies that help you feel better",
                            "Identify people you can contact for support",
                            "List professional resources and emergency contacts",
                            "Create a safe environment by removing means of self-harm",
                            "Plan for what to do if you feel unsafe"
                        ],
                        "resources": [
                            "National Suicide Prevention Lifeline: 988",
                            "Crisis Text Line: Text HOME to 741741",
                            "Emergency Services: 911"
                        ]
                    },
                    {
                        "id": "grounding_techniques",
                        "name": "Grounding Techniques",
                        "description": "Techniques to help you stay present and calm",
                        "techniques": [
                            {
                                "name": "5-4-3-2-1 Technique",
                                "steps": [
                                    "Name 5 things you can see",
                                    "Name 4 things you can touch",
                                    "Name 3 things you can hear",
                                    "Name 2 things you can smell",
                                    "Name 1 thing you can taste"
                                ]
                            },
                            {
                                "name": "Box Breathing",
                                "steps": [
                                    "Breathe in for 4 counts",
                                    "Hold your breath for 4 counts",
                                    "Breathe out for 4 counts",
                                    "Hold for 4 counts",
                                    "Repeat 4 times"
                                ]
                            },
                            {
                                "name": "Progressive Muscle Relaxation",
                                "steps": [
                                    "Start with your toes",
                                    "Tense the muscles for 5 seconds",
                                    "Release and notice the relaxation",
                                    "Move up through your body",
                                    "End with your head and face"
                                ]
                            }
                        ]
                    }
                ]
            },
            "mood_tracking": {
                "name": "Mood Tracking and Monitoring",
                "description": "Tools for tracking mood and identifying patterns",
                "tools": [
                    {
                        "id": "daily_mood_check",
                        "name": "Daily Mood Check-in",
                        "description": "Quick daily assessment of mood and energy",
                        "questions": [
                            "On a scale of 1-10, how is your mood today?",
                            "How is your energy level?",
                            "How well did you sleep last night?",
                            "What's one thing you're grateful for today?",
                            "What's one challenge you're facing?"
                        ]
                    },
                    {
                        "id": "mood_patterns",
                        "name": "Mood Pattern Analysis",
                        "description": "Identify patterns in mood changes",
                        "factors": [
                            "Sleep quality and duration",
                            "Exercise and physical activity",
                            "Social interactions",
                            "Work or school stress",
                            "Weather and seasonal changes",
                            "Medication adherence",
                            "Alcohol or substance use"
                        ]
                    }
                ]
            },
            "communication_skills": {
                "name": "Communication and Social Skills",
                "description": "Skills for improving relationships and communication",
                "skills": [
                    {
                        "id": "active_listening",
                        "name": "Active Listening",
                        "description": "Fully engage with what others are saying",
                        "techniques": [
                            "Give your full attention",
                            "Maintain eye contact",
                            "Nod and use verbal acknowledgments",
                            "Ask clarifying questions",
                            "Reflect back what you heard",
                            "Avoid interrupting or planning your response"
                        ]
                    },
                    {
                        "id": "assertive_communication",
                        "name": "Assertive Communication",
                        "description": "Express your needs clearly and respectfully",
                        "techniques": [
                            "Use 'I' statements",
                            "Be specific about your needs",
                            "Stay calm and composed",
                            "Listen to the other person's perspective",
                            "Find win-win solutions when possible"
                        ]
                    }
                ]
            }
        }
        
        # Save modules to file
        with open(self.modules_dir / "modules.json", 'w') as f:
            json.dump(modules, f, indent=2)
    
    def get_module(self, module_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific therapeutic module"""
        modules = self.load_modules()
        return modules.get(module_id)
    
    def get_all_modules(self) -> Dict[str, Any]:
        """Get all therapeutic modules"""
        return self.load_modules()
    
    def load_modules(self) -> Dict[str, Any]:
        """Load modules from file"""
        modules_file = self.modules_dir / "modules.json"
        if modules_file.exists():
            with open(modules_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_technique(self, module_id: str, technique_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific technique from a module"""
        module = self.get_module(module_id)
        if not module:
            return None
        
        if 'techniques' in module:
            for technique in module['techniques']:
                if technique['id'] == technique_id:
                    return technique
        
        if 'exercises' in module:
            for exercise in module['exercises']:
                if exercise['id'] == technique_id:
                    return exercise
        
        if 'interventions' in module:
            for intervention in module['interventions']:
                if intervention['id'] == technique_id:
                    return intervention
        
        return None
    
    def get_random_technique(self, module_id: str) -> Optional[Dict[str, Any]]:
        """Get a random technique from a module"""
        import random
        
        module = self.get_module(module_id)
        if not module:
            return None
        
        techniques = []
        if 'techniques' in module:
            techniques.extend(module['techniques'])
        if 'exercises' in module:
            techniques.extend(module['exercises'])
        if 'interventions' in module:
            techniques.extend(module['interventions'])
        
        if techniques:
            return random.choice(techniques)
        
        return None
    
    def search_techniques(self, query: str) -> List[Dict[str, Any]]:
        """Search for techniques matching a query"""
        modules = self.load_modules()
        results = []
        
        for module_id, module in modules.items():
            techniques = []
            if 'techniques' in module:
                techniques.extend(module['techniques'])
            if 'exercises' in module:
                techniques.extend(module['exercises'])
            if 'interventions' in module:
                techniques.extend(module['interventions'])
            
            for technique in techniques:
                if (query.lower() in technique.get('name', '').lower() or 
                    query.lower() in technique.get('description', '').lower()):
                    technique['module_id'] = module_id
                    results.append(technique)
        
        return results
    
    def get_crisis_resources(self) -> List[Dict[str, str]]:
        """Get crisis resources and emergency contacts"""
        return [
            {
                "name": "National Suicide Prevention Lifeline",
                "number": "988",
                "description": "24/7 crisis support and suicide prevention",
                "type": "phone"
            },
            {
                "name": "Crisis Text Line",
                "number": "Text HOME to 741741",
                "description": "24/7 crisis support via text message",
                "type": "text"
            },
            {
                "name": "SAMHSA National Helpline",
                "number": "1-800-662-4357",
                "description": "Mental health and substance abuse services",
                "type": "phone"
            },
            {
                "name": "Veterans Crisis Line",
                "number": "1-800-273-8255",
                "description": "Crisis support for veterans (Press 1)",
                "type": "phone"
            },
            {
                "name": "Disaster Distress Helpline",
                "number": "1-800-985-5990",
                "description": "Support for disaster-related stress",
                "type": "phone"
            }
        ]
    
    def get_safety_plan_template(self) -> Dict[str, Any]:
        """Get a safety plan template"""
        return {
            "warning_signs": [
                "Feeling hopeless or worthless",
                "Increased alcohol or drug use",
                "Withdrawing from friends and family",
                "Giving away possessions",
                "Talking about death or suicide"
            ],
            "coping_strategies": [
                "Call a trusted friend or family member",
                "Go for a walk or exercise",
                "Practice deep breathing or meditation",
                "Listen to calming music",
                "Write in a journal"
            ],
            "support_contacts": [
                "Family member or close friend",
                "Therapist or counselor",
                "Support group member",
                "Spiritual advisor"
            ],
            "professional_contacts": [
                "Primary care doctor",
                "Psychiatrist",
                "Therapist",
                "Crisis counselor"
            ],
            "emergency_contacts": [
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "Emergency Services: 911"
            ],
            "environmental_safety": [
                "Remove or secure means of self-harm",
                "Stay with a trusted person",
                "Go to a safe, public place",
                "Avoid alcohol and drugs"
            ]
        }
    
    def add_custom_module(self, module_id: str, module_data: Dict[str, Any]):
        """Add a custom therapeutic module"""
        modules = self.load_modules()
        modules[module_id] = module_data
        
        with open(self.modules_dir / "modules.json", 'w') as f:
            json.dump(modules, f, indent=2)
    
    def update_module(self, module_id: str, updates: Dict[str, Any]):
        """Update an existing therapeutic module"""
        modules = self.load_modules()
        if module_id in modules:
            modules[module_id].update(updates)
            
            with open(self.modules_dir / "modules.json", 'w') as f:
                json.dump(modules, f, indent=2)
    
    def delete_module(self, module_id: str):
        """Delete a therapeutic module"""
        modules = self.load_modules()
        if module_id in modules:
            del modules[module_id]
            
            with open(self.modules_dir / "modules.json", 'w') as f:
                json.dump(modules, f, indent=2)
