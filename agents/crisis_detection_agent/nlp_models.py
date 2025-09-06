"""
NLP Models for Crisis Detection Agent

Implements natural language processing models for detecting crisis indicators,
sentiment analysis, and intent classification in mental health contexts.
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import torch
from pydantic import BaseModel, Field

# Optional transformers import to prevent crashes
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Crisis detection features will be limited.")


class CrisisNLPModel:
    """
    Main NLP model for crisis detection using multiple approaches
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        
        # Initialize sentiment analyzer only if transformers is available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
            except Exception as e:
                print(f"Warning: Could not initialize sentiment analyzer: {e}")
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
            
        self.crisis_keywords = self._load_crisis_keywords()
        self.risk_patterns = self._load_risk_patterns()
        
    def _load_crisis_keywords(self) -> Dict[str, List[str]]:
        """Load crisis-related keywords and phrases"""
        return {
            "suicidal_ideation": [
                "kill myself", "end it all", "not worth living", "better off dead",
                "want to die", "suicide", "end my life", "not want to live",
                "wish I was dead", "hurt myself", "self harm", "cut myself",
                "overdose", "jump off", "hang myself", "shoot myself"
            ],
            "self_harm": [
                "cut myself", "hurt myself", "self harm", "burn myself",
                "hit myself", "punish myself", "bleeding", "scars",
                "wounds", "injuries", "self inflicted"
            ],
            "substance_abuse": [
                "drinking too much", "using drugs", "overdose", "high",
                "drunk", "addicted", "withdrawal", "relapse",
                "alcohol", "pills", "medication abuse"
            ],
            "psychotic_symptoms": [
                "hearing voices", "seeing things", "paranoid", "delusions",
                "hallucinations", "not real", "conspiracy", "being watched",
                "mind control", "aliens", "government"
            ],
            "severe_depression": [
                "can't get out of bed", "hopeless", "worthless", "empty",
                "numb", "nothing matters", "no point", "give up",
                "tired of living", "burden", "failure"
            ],
            "panic_attack": [
                "panic attack", "can't breathe", "heart racing", "chest pain",
                "dizzy", "fainting", "losing control", "going crazy",
                "dying", "emergency", "ambulance"
            ],
            "homicidal_ideation": [
                "hurt someone", "kill someone", "revenge", "angry",
                "violent", "rage", "hate", "destroy"
            ],
            "isolation": [
                "alone", "no one cares", "isolated", "lonely",
                "no friends", "abandoned", "rejected", "left out"
            ],
            "hopelessness": [
                "hopeless", "no future", "nothing will change", "stuck",
                "trapped", "no way out", "pointless", "useless"
            ]
        }
    
    def _load_risk_patterns(self) -> List[Dict[str, Any]]:
        """Load regex patterns for crisis detection"""
        return [
            {
                "pattern": r"i (want to|wish i could|am going to) (die|kill myself|end it all)",
                "risk_factor": "suicidal_ideation",
                "severity": 0.9
            },
            {
                "pattern": r"(cut|hurt|harm) (myself|my self)",
                "risk_factor": "self_harm",
                "severity": 0.8
            },
            {
                "pattern": r"i (hear|see) (voices|things) (that|which) (are not real|don't exist)",
                "risk_factor": "psychotic_symptoms",
                "severity": 0.7
            },
            {
                "pattern": r"i (can't|can not) (breathe|stop|control) (my|this) (breathing|panic|anxiety)",
                "risk_factor": "panic_attack",
                "severity": 0.6
            },
            {
                "pattern": r"i (am|feel) (hopeless|worthless|useless|a failure)",
                "risk_factor": "severe_depression",
                "severity": 0.5
            }
        ]
    
    async def analyze_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze text for crisis indicators
        
        Args:
            text: Text to analyze
            context: Additional context for analysis
            
        Returns:
            Analysis results with crisis indicators
        """
        try:
            # Basic sentiment analysis (if available)
            if self.sentiment_analyzer:
                sentiment_result = self.sentiment_analyzer(text)[0]
            else:
                sentiment_result = {"label": "NEUTRAL", "score": 0.5}
            
            # Keyword-based analysis
            keyword_analysis = self._analyze_keywords(text)
            
            # Pattern-based analysis
            pattern_analysis = self._analyze_patterns(text)
            
            # Context-aware analysis
            context_analysis = self._analyze_context(text, context)
            
            # Combine results
            crisis_indicators = self._combine_analysis_results(
                sentiment_result, keyword_analysis, pattern_analysis, context_analysis
            )
            
            return {
                "text": text,
                "sentiment": sentiment_result,
                "crisis_indicators": crisis_indicators,
                "risk_score": crisis_indicators["overall_risk_score"],
                "confidence": crisis_indicators["confidence"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return {
                "text": text,
                "error": str(e),
                "risk_score": 0.0,
                "confidence": 0.0
            }
    
    def _analyze_keywords(self, text: str) -> Dict[str, Any]:
        """Analyze text for crisis-related keywords"""
        text_lower = text.lower()
        keyword_scores = {}
        
        for risk_type, keywords in self.crisis_keywords.items():
            score = 0
            matches = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    matches.append(keyword)
            
            # Normalize score
            normalized_score = min(1.0, score / len(keywords))
            keyword_scores[risk_type] = {
                "score": normalized_score,
                "matches": matches,
                "count": score
            }
        
        return keyword_scores
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text using regex patterns"""
        pattern_scores = {}
        
        for pattern_info in self.risk_patterns:
            pattern = pattern_info["pattern"]
            risk_factor = pattern_info["risk_factor"]
            severity = pattern_info["severity"]
            
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pattern_scores[risk_factor] = {
                    "score": severity,
                    "matches": matches,
                    "pattern": pattern
                }
        
        return pattern_scores
    
    def _analyze_context(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze context for additional risk factors"""
        context_analysis = {
            "urgency_indicators": 0,
            "time_indicators": 0,
            "location_indicators": 0,
            "social_indicators": 0
        }
        
        if not context:
            return context_analysis
        
        # Analyze urgency indicators
        urgency_words = ["now", "immediately", "right now", "today", "tonight"]
        text_lower = text.lower()
        context_analysis["urgency_indicators"] = sum(1 for word in urgency_words if word in text_lower)
        
        # Analyze time context
        if "time" in context:
            # Check if it's during high-risk hours (late night, early morning)
            hour = context["time"].hour if hasattr(context["time"], 'hour') else 0
            if hour < 6 or hour > 22:  # Late night or early morning
                context_analysis["time_indicators"] = 1
        
        # Analyze location context
        if "location" in context:
            # Check if user is in a high-risk location
            location = context["location"].lower()
            high_risk_locations = ["bridge", "highway", "tall building", "isolation"]
            if any(loc in location for loc in high_risk_locations):
                context_analysis["location_indicators"] = 1
        
        # Analyze social context
        if "social_context" in context:
            social = context["social_context"]
            if social.get("isolation", False) or social.get("recent_loss", False):
                context_analysis["social_indicators"] = 1
        
        return context_analysis
    
    def _combine_analysis_results(
        self,
        sentiment: Dict[str, Any],
        keywords: Dict[str, Any],
        patterns: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine all analysis results into final crisis indicators"""
        
        # Calculate individual risk scores
        keyword_risk = max([data["score"] for data in keywords.values()], default=0.0)
        pattern_risk = max([data["score"] for data in patterns.values()], default=0.0)
        
        # Context risk
        context_risk = min(1.0, sum(context.values()) / len(context))
        
        # Sentiment risk (negative sentiment increases risk)
        sentiment_risk = 0.0
        if sentiment["label"] == "NEGATIVE":
            sentiment_risk = sentiment["score"]
        elif sentiment["label"] == "POSITIVE":
            sentiment_risk = 1.0 - sentiment["score"]
        
        # Overall risk score (weighted average)
        overall_risk = (
            keyword_risk * 0.4 +
            pattern_risk * 0.3 +
            context_risk * 0.2 +
            sentiment_risk * 0.1
        )
        
        # Confidence based on consistency of indicators
        indicators = [keyword_risk, pattern_risk, context_risk, sentiment_risk]
        confidence = 1.0 - np.std(indicators) if len(indicators) > 1 else 0.5
        
        return {
            "overall_risk_score": overall_risk,
            "confidence": confidence,
            "keyword_risk": keyword_risk,
            "pattern_risk": pattern_risk,
            "context_risk": context_risk,
            "sentiment_risk": sentiment_risk,
            "risk_factors": list(keywords.keys()) + list(patterns.keys()),
            "detected_patterns": patterns,
            "context_indicators": context
        }


class SentimentAnalyzer:
    """Specialized sentiment analyzer for mental health contexts"""
    
    def __init__(self):
        # Initialize pipelines only if transformers is available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base"
                )
            except Exception as e:
                print(f"Warning: Could not initialize sentiment/emotion pipelines: {e}")
                self.sentiment_pipeline = None
                self.emotion_pipeline = None
        else:
            self.sentiment_pipeline = None
            self.emotion_pipeline = None
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment and emotion in text"""
        try:
            # Basic sentiment (if available)
            if self.sentiment_pipeline:
                sentiment = self.sentiment_pipeline(text)[0]
            else:
                sentiment = {"label": "NEUTRAL", "score": 0.5}
            
            # Emotion analysis (if available)
            if self.emotion_pipeline:
                emotion = self.emotion_pipeline(text)[0]
            else:
                emotion = {"label": "neutral", "score": 0.5}
            
            # Mental health specific sentiment indicators
            mental_health_sentiment = self._analyze_mental_health_sentiment(text)
            
            return {
                "sentiment": sentiment,
                "emotion": emotion,
                "mental_health_sentiment": mental_health_sentiment,
                "overall_sentiment_score": self._calculate_sentiment_score(sentiment, emotion)
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {"error": str(e)}
    
    def _analyze_mental_health_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze mental health specific sentiment indicators"""
        text_lower = text.lower()
        
        # Positive mental health indicators
        positive_indicators = [
            "better", "improving", "hopeful", "grateful", "proud",
            "accomplished", "confident", "optimistic", "motivated"
        ]
        
        # Negative mental health indicators
        negative_indicators = [
            "worse", "hopeless", "worthless", "useless", "failure",
            "burden", "empty", "numb", "stuck", "trapped"
        ]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
        
        return {
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "net_sentiment": positive_count - negative_count,
            "sentiment_balance": (positive_count - negative_count) / max(1, positive_count + negative_count)
        }
    
    def _calculate_sentiment_score(self, sentiment: Dict[str, Any], emotion: Dict[str, Any]) -> float:
        """Calculate overall sentiment score"""
        # Base sentiment score
        sentiment_score = sentiment["score"] if sentiment["label"] == "POSITIVE" else 1.0 - sentiment["score"]
        
        # Emotion adjustment
        emotion_score = emotion["score"]
        
        # Combine scores
        return (sentiment_score + emotion_score) / 2.0


class IntentClassifier:
    """Classifies user intent in mental health contexts"""
    
    def __init__(self):
        self.intent_patterns = {
            "crisis_help": [
                "help", "crisis", "emergency", "urgent", "immediate",
                "can't cope", "overwhelmed", "breaking down"
            ],
            "general_support": [
                "support", "talk", "listen", "advice", "guidance",
                "counseling", "therapy", "treatment"
            ],
            "information_seeking": [
                "what", "how", "why", "when", "where", "explain",
                "understand", "learn", "know"
            ],
            "assessment_request": [
                "assessment", "evaluation", "screening", "test",
                "check", "evaluate", "diagnose"
            ],
            "crisis_denial": [
                "fine", "okay", "good", "better", "improving",
                "not a problem", "handling it", "coping"
            ]
        }
    
    async def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify user intent from text"""
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            normalized_score = min(1.0, score / len(patterns))
            intent_scores[intent] = {
                "score": normalized_score,
                "matches": [pattern for pattern in patterns if pattern in text_lower]
            }
        
        # Determine primary intent
        primary_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]["score"])
        
        return {
            "primary_intent": primary_intent,
            "intent_scores": intent_scores,
            "confidence": intent_scores[primary_intent]["score"]
        }
