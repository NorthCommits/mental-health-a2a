"""
File-based storage management for the Mental Health A2A system
Handles all data persistence using JSON files and directory structures
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import shutil

class FileStorageManager:
    """Manages file-based storage for the Mental Health A2A system"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        
        # Directory paths
        self.sessions_dir = self.base_dir / "sessions"
        self.assessments_dir = self.base_dir / "assessments"
        self.conversations_dir = self.base_dir / "conversations"
        self.mood_tracking_dir = self.base_dir / "mood_tracking"
        self.crisis_alerts_dir = self.base_dir / "crisis_alerts"
        self.therapy_sessions_dir = self.base_dir / "therapy_sessions"
        self.goals_dir = self.base_dir / "goals"
        self.agent_communications_dir = self.base_dir / "agent_communications"
        self.logs_dir = self.base_dir / "logs"
        self.therapeutic_modules_dir = self.base_dir / "therapeutic_modules"
        self.config_dir = self.base_dir / "config"
        self.appointments_dir = self.base_dir / "appointments"
        self.providers_dir = self.base_dir / "providers"
        self.care_plans_dir = self.base_dir / "care_plans"
        self.progress_data_dir = self.base_dir / "progress_data"
        self.trends_dir = self.base_dir / "trends"
        self.reports_dir = self.base_dir / "reports"
    
    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            "sessions",
            "assessments", 
            "conversations",
            "mood_tracking",
            "crisis_alerts",
            "therapy_sessions",
            "goals",
            "agent_communications",
            "logs",
            "therapeutic_modules",
            "config",
            "appointments",
            "providers",
            "care_plans",
            "progress_data",
            "trends",
            "reports"
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def save_session(self, session_id: str, session_data: Dict[str, Any]):
        """Save session data"""
        session_file = self.base_dir / "sessions" / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data"""
        session_file = self.base_dir / "sessions" / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all session data"""
        sessions = []
        sessions_dir = self.base_dir / "sessions"
        for session_file in sessions_dir.glob("*.json"):
            with open(session_file, 'r') as f:
                sessions.append(json.load(f))
        return sessions
    
    def save_assessment(self, assessment_data: Dict[str, Any]):
        """Save assessment data"""
        user_id = assessment_data['user_id']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create user-specific directory
        user_dir = self.base_dir / "assessments" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Save assessment
        assessment_file = user_dir / f"assessment_{timestamp}.json"
        with open(assessment_file, 'w') as f:
            json.dump(assessment_data, f, indent=2, default=str)
        
        # Update user's assessment index
        self._update_assessment_index(user_id, assessment_file.name)
    
    def get_user_assessments(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all assessments for a user"""
        user_dir = self.base_dir / "assessments" / user_id
        if not user_dir.exists():
            return []
        
        assessments = []
        for assessment_file in user_dir.glob("assessment_*.json"):
            with open(assessment_file, 'r') as f:
                assessments.append(json.load(f))
        
        return sorted(assessments, key=lambda x: x['timestamp'], reverse=True)
    
    def get_recent_assessments(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent assessments for a user"""
        assessments = self.get_user_assessments(user_id)
        return assessments[:limit]
    
    def save_conversation(self, user_id: str, user_message: str, agent_response: str):
        """Save conversation data"""
        conversation_data = {
            'user_id': user_id,
            'user_message': user_message,
            'agent_response': agent_response,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create user-specific directory
        user_dir = self.base_dir / "conversations" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Append to conversation log
        conversation_file = user_dir / "conversation_log.jsonl"
        with open(conversation_file, 'a') as f:
            f.write(json.dumps(conversation_data) + '\n')
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        conversation_file = self.base_dir / "conversations" / user_id / "conversation_log.jsonl"
        if not conversation_file.exists():
            return []
        
        conversations = []
        with open(conversation_file, 'r') as f:
            for line in f:
                conversations.append(json.loads(line.strip()))
        
        return conversations
    
    def save_mood_entry(self, user_id: str, mood: int, notes: str = ""):
        """Save mood tracking entry"""
        mood_data = {
            'user_id': user_id,
            'mood': mood,
            'notes': notes,
            'date': datetime.now().isoformat()
        }
        
        # Create user-specific directory
        user_dir = self.base_dir / "mood_tracking" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Append to mood log
        mood_file = user_dir / "mood_log.jsonl"
        with open(mood_file, 'a') as f:
            f.write(json.dumps(mood_data) + '\n')
    
    def get_mood_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get mood history for a user"""
        mood_file = self.base_dir / "mood_tracking" / user_id / "mood_log.jsonl"
        if not mood_file.exists():
            return []
        
        moods = []
        with open(mood_file, 'r') as f:
            for line in f:
                moods.append(json.loads(line.strip()))
        
        return moods
    
    def save_crisis_alert(self, alert_data: Dict[str, Any]):
        """Save crisis alert data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_file = self.base_dir / "crisis_alerts" / f"alert_{timestamp}.json"
        
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2, default=str)
    
    def get_crisis_alerts(self) -> List[Dict[str, Any]]:
        """Get all crisis alerts"""
        alerts = []
        alerts_dir = self.base_dir / "crisis_alerts"
        for alert_file in alerts_dir.glob("alert_*.json"):
            with open(alert_file, 'r') as f:
                alerts.append(json.load(f))
        
        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def save_therapy_session(self, session_data: Dict[str, Any]):
        """Save therapy session data"""
        user_id = session_data['user_id']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create user-specific directory
        user_dir = self.base_dir / "therapy_sessions" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Save session
        session_file = user_dir / f"session_{timestamp}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
    
    def get_therapy_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get therapy sessions for a user"""
        user_dir = self.base_dir / "therapy_sessions" / user_id
        if not user_dir.exists():
            return []
        
        sessions = []
        for session_file in user_dir.glob("session_*.json"):
            with open(session_file, 'r') as f:
                sessions.append(json.load(f))
        
        return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)
    
    def save_user_goal(self, user_id: str, goal_data: Dict[str, Any]):
        """Save user goal"""
        goals_file = self.base_dir / "goals" / f"{user_id}_goals.json"
        
        # Load existing goals
        goals = []
        if goals_file.exists():
            with open(goals_file, 'r') as f:
                goals = json.load(f)
        
        # Add new goal
        goal_data['id'] = len(goals) + 1
        goal_data['created_at'] = datetime.now().isoformat()
        goals.append(goal_data)
        
        # Save updated goals
        with open(goals_file, 'w') as f:
            json.dump(goals, f, indent=2, default=str)
    
    def get_user_goals(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user goals"""
        goals_file = self.base_dir / "goals" / f"{user_id}_goals.json"
        if goals_file.exists():
            with open(goals_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_agent_communication(self, comm_data: Dict[str, Any]):
        """Save agent communication log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comm_file = self.base_dir / "agent_communications" / f"comm_{timestamp}.json"
        
        with open(comm_file, 'w') as f:
            json.dump(comm_data, f, indent=2, default=str)
    
    def get_communication_logs(self) -> List[Dict[str, Any]]:
        """Get agent communication logs"""
        logs = []
        comm_dir = self.base_dir / "agent_communications"
        for comm_file in comm_dir.glob("comm_*.json"):
            with open(comm_file, 'r') as f:
                logs.append(json.load(f))
        
        return sorted(logs, key=lambda x: x['timestamp'], reverse=True)
    
    def save_system_log(self, log_data: Dict[str, Any]):
        """Save system log entry"""
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = self.base_dir / "logs" / f"system_{timestamp}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            'total_files': 0,
            'total_size': 0,
            'sessions': 0,
            'logs': 0
        }
        
        # Count files and calculate size
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                stats['total_files'] += 1
                stats['total_size'] += os.path.getsize(file_path)
        
        # Convert to MB
        stats['total_size'] = round(stats['total_size'] / (1024 * 1024), 2)
        
        # Count specific file types
        stats['sessions'] = len(list((self.base_dir / "sessions").glob("*.json")))
        stats['logs'] = len(list((self.base_dir / "logs").glob("*.jsonl")))
        
        return stats
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data files"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Clean up old logs
        logs_dir = self.base_dir / "logs"
        for log_file in logs_dir.glob("*.jsonl"):
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                log_file.unlink()
        
        # Clean up old communications
        comm_dir = self.base_dir / "agent_communications"
        for comm_file in comm_dir.glob("comm_*.json"):
            if datetime.fromtimestamp(comm_file.stat().st_mtime) < cutoff_date:
                comm_file.unlink()
    
    def backup_data(self, backup_dir: str):
        """Create a backup of all data"""
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"mental_health_backup_{timestamp}"
        backup_dest = backup_path / backup_name
        
        shutil.copytree(self.base_dir, backup_dest)
        
        return str(backup_dest)
    
    def restore_data(self, backup_path: str):
        """Restore data from backup"""
        backup_dir = Path(backup_path)
        if not backup_dir.exists():
            raise ValueError("Backup directory does not exist")
        
        # Remove existing data
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
        
        # Restore from backup
        shutil.copytree(backup_dir, self.base_dir)
    
    def _update_assessment_index(self, user_id: str, filename: str):
        """Update user's assessment index"""
        index_file = self.base_dir / "assessments" / user_id / "index.json"
        
        # Load existing index
        index = []
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
        
        # Add new assessment
        index.append({
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save updated index
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2, default=str)
    
    async def save_appointment(self, appointment_id: str, appointment_data: Dict[str, Any]) -> bool:
        """Save appointment data"""
        try:
            file_path = self.appointments_dir / f"{appointment_id}.json"
            with open(file_path, 'w') as f:
                json.dump(appointment_data, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving appointment: {e}")
            return False
    
    async def save_provider(self, provider_id: str, provider_data: Dict[str, Any]) -> bool:
        """Save provider data"""
        try:
            file_path = self.providers_dir / f"{provider_id}.json"
            with open(file_path, 'w') as f:
                json.dump(provider_data, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving provider: {e}")
            return False
    
    async def save_care_plan(self, plan_id: str, plan_data: Dict[str, Any]) -> bool:
        """Save care plan data"""
        try:
            file_path = self.care_plans_dir / f"{plan_id}.json"
            with open(file_path, 'w') as f:
                json.dump(plan_data, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving care plan: {e}")
            return False
    
    async def save_progress_data(self, data_point_id: str, data_point: Dict[str, Any]) -> bool:
        """Save progress data point"""
        try:
            file_path = self.progress_data_dir / f"{data_point_id}.json"
            with open(file_path, 'w') as f:
                json.dump(data_point, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving progress data: {e}")
            return False
    
    async def save_progress_trend(self, trend_id: str, trend_data: Dict[str, Any]) -> bool:
        """Save progress trend data"""
        try:
            file_path = self.trends_dir / f"{trend_id}.json"
            with open(file_path, 'w') as f:
                json.dump(trend_data, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving progress trend: {e}")
            return False
    
    async def save_progress_report(self, report_id: str, report_data: Dict[str, Any]) -> bool:
        """Save progress report data"""
        try:
            file_path = self.reports_dir / f"{report_id}.json"
            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving progress report: {e}")
            return False
