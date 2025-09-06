"""
Session management for the Mental Health A2A system
Handles user sessions, conversation continuity, and state management
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

class SessionManager:
    """Manages user sessions and conversation continuity"""
    
    def __init__(self, storage_manager):
        self.storage_manager = storage_manager
        self.active_sessions = {}
    
    def create_session(self, user_id: str, session_data: Optional[Dict[str, Any]] = None) -> str:
        """Create a new session for a user"""
        session_id = str(uuid.uuid4())
        
        session_info = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'status': 'active',
            'conversation_count': 0,
            'assessment_count': 0,
            'crisis_alerts': 0,
            'data': session_data or {}
        }
        
        # Save session
        self.storage_manager.save_session(session_id, session_info)
        
        # Add to active sessions
        self.active_sessions[session_id] = session_info
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Load from storage
        session_data = self.storage_manager.load_session(session_id)
        if session_data:
            self.active_sessions[session_id] = session_data
        
        return session_data
    
    def update_session(self, session_id: str, updates: Dict[str, Any]):
        """Update session information"""
        session = self.get_session(session_id)
        if session:
            session.update(updates)
            session['last_activity'] = datetime.now().isoformat()
            
            # Save updated session
            self.storage_manager.save_session(session_id, session)
            
            # Update active sessions
            self.active_sessions[session_id] = session
    
    def end_session(self, session_id: str):
        """End a session"""
        session = self.get_session(session_id)
        if session:
            session['status'] = 'ended'
            session['ended_at'] = datetime.now().isoformat()
            
            # Save updated session
            self.storage_manager.save_session(session_id, session)
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        all_sessions = self.storage_manager.get_all_sessions()
        return [s for s in all_sessions if s['user_id'] == user_id]
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions"""
        return list(self.active_sessions.values())
    
    def cleanup_inactive_sessions(self, hours: int = 24):
        """Clean up inactive sessions"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        sessions_to_end = []
        for session_id, session in self.active_sessions.items():
            last_activity = datetime.fromisoformat(session['last_activity'])
            if last_activity < cutoff_time:
                sessions_to_end.append(session_id)
        
        for session_id in sessions_to_end:
            self.end_session(session_id)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        all_sessions = self.storage_manager.get_all_sessions()
        
        stats = {
            'total_sessions': len(all_sessions),
            'active_sessions': len(self.active_sessions),
            'ended_sessions': len([s for s in all_sessions if s.get('status') == 'ended']),
            'sessions_today': len([s for s in all_sessions 
                                 if datetime.fromisoformat(s['created_at']).date() == datetime.now().date()]),
            'average_session_duration': 0,
            'total_conversations': 0,
            'total_assessments': 0
        }
        
        if all_sessions:
            # Calculate average session duration
            durations = []
            for session in all_sessions:
                if 'ended_at' in session:
                    created = datetime.fromisoformat(session['created_at'])
                    ended = datetime.fromisoformat(session['ended_at'])
                    duration = (ended - created).total_seconds() / 60  # minutes
                    durations.append(duration)
            
            if durations:
                stats['average_session_duration'] = sum(durations) / len(durations)
            
            # Calculate totals
            stats['total_conversations'] = sum(s.get('conversation_count', 0) for s in all_sessions)
            stats['total_assessments'] = sum(s.get('assessment_count', 0) for s in all_sessions)
        
        return stats
    
    def create_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """Create conversation context for a session"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # Get recent conversation history
        conversation_history = self.storage_manager.get_conversation_history(session['user_id'])
        
        # Get recent mood data
        mood_history = self.storage_manager.get_mood_history(session['user_id'])
        
        # Get recent assessments
        recent_assessments = self.storage_manager.get_recent_assessments(session['user_id'], limit=3)
        
        context = {
            'session_id': session_id,
            'user_id': session['user_id'],
            'conversation_count': session.get('conversation_count', 0),
            'recent_messages': conversation_history[-5:] if conversation_history else [],
            'current_mood': mood_history[-1]['mood'] if mood_history else 5,
            'mood_trend': self._calculate_mood_trend(mood_history),
            'recent_assessments': recent_assessments,
            'crisis_alerts': session.get('crisis_alerts', 0),
            'session_duration': self._calculate_session_duration(session)
        }
        
        return context
    
    def _calculate_mood_trend(self, mood_history: List[Dict[str, Any]]) -> str:
        """Calculate mood trend from history"""
        if len(mood_history) < 2:
            return "insufficient_data"
        
        recent_moods = [m['mood'] for m in mood_history[-7:]]  # Last 7 entries
        if len(recent_moods) < 2:
            return "insufficient_data"
        
        # Simple trend calculation
        first_half = sum(recent_moods[:len(recent_moods)//2]) / (len(recent_moods)//2)
        second_half = sum(recent_moods[len(recent_moods)//2:]) / (len(recent_moods) - len(recent_moods)//2)
        
        if second_half > first_half + 0.5:
            return "improving"
        elif second_half < first_half - 0.5:
            return "declining"
        else:
            return "stable"
    
    def _calculate_session_duration(self, session: Dict[str, Any]) -> int:
        """Calculate session duration in minutes"""
        created = datetime.fromisoformat(session['created_at'])
        last_activity = datetime.fromisoformat(session['last_activity'])
        return int((last_activity - created).total_seconds() / 60)
    
    def increment_conversation_count(self, session_id: str):
        """Increment conversation count for a session"""
        session = self.get_session(session_id)
        if session:
            current_count = session.get('conversation_count', 0)
            self.update_session(session_id, {'conversation_count': current_count + 1})
    
    def increment_assessment_count(self, session_id: str):
        """Increment assessment count for a session"""
        session = self.get_session(session_id)
        if session:
            current_count = session.get('assessment_count', 0)
            self.update_session(session_id, {'assessment_count': current_count + 1})
    
    def increment_crisis_alerts(self, session_id: str):
        """Increment crisis alert count for a session"""
        session = self.get_session(session_id)
        if session:
            current_count = session.get('crisis_alerts', 0)
            self.update_session(session_id, {'crisis_alerts': current_count + 1})
    
    def get_session_insights(self, session_id: str) -> Dict[str, Any]:
        """Get insights for a specific session"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # Get conversation history
        conversation_history = self.storage_manager.get_conversation_history(session['user_id'])
        
        # Get mood data
        mood_history = self.storage_manager.get_mood_history(session['user_id'])
        
        # Get assessments
        assessments = self.storage_manager.get_user_assessments(session['user_id'])
        
        insights = {
            'session_id': session_id,
            'user_id': session['user_id'],
            'duration_minutes': self._calculate_session_duration(session),
            'conversation_count': session.get('conversation_count', 0),
            'assessment_count': session.get('assessment_count', 0),
            'crisis_alerts': session.get('crisis_alerts', 0),
            'mood_data_points': len(mood_history),
            'average_mood': sum(m['mood'] for m in mood_history) / len(mood_history) if mood_history else 0,
            'mood_trend': self._calculate_mood_trend(mood_history),
            'assessment_types': [a.get('assessment_type', 'unknown') for a in assessments],
            'last_activity': session['last_activity'],
            'status': session['status']
        }
        
        return insights
