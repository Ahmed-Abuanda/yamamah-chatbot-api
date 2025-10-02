import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

class ChatHistoryManager:
    def __init__(self, chat_history_dir: str = "chat_history"):
        """
        Initialize the chat history manager.
        
        Args:
            chat_history_dir: Directory to store chat history files
        """
        self.chat_history_dir = Path(chat_history_dir)
        self.chat_history_dir.mkdir(exist_ok=True)
    
    def _get_session_file_path(self, session_id: str) -> Path:
        """Get the file path for a session's chat history."""
        return self.chat_history_dir / f"{session_id}.json"
    
    def load_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load chat history for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            List of chat messages, empty list if no history exists
        """
        session_file = self._get_session_file_path(session_id)
        
        if not session_file.exists():
            return []
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def save_chat_history(self, session_id: str, chat_history: List[Dict[str, Any]]) -> None:
        """
        Save chat history for a session.
        
        Args:
            session_id: The session identifier
            chat_history: List of chat messages to save
        """
        session_file = self._get_session_file_path(session_id)
        
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(chat_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving chat history for session {session_id}: {e}")
    
    def add_user_message(self, session_id: str, message: str) -> None:
        """
        Add a user message to the chat history.
        
        Args:
            session_id: The session identifier
            message: The user's message
        """
        chat_history = self.load_chat_history(session_id)
        
        user_message = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        
        chat_history.append(user_message)
        self.save_chat_history(session_id, chat_history)
    
    def add_assistant_message(self, session_id: str, message: str, 
                            map_action: Optional[str] = None,
                            region_id: Optional[str] = None,
                            city_id: Optional[str] = None,
                            district_id: Optional[str] = None,
                            map_data: Optional[Dict[str, Any]] = None,
                            chart_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an assistant message to the chat history.
        
        Args:
            session_id: The session identifier
            message: The assistant's response message
            map_action: Optional map action performed
            region_id: Optional region ID
            city_id: Optional city ID
            district_id: Optional district ID
            map_data: Optional map data
            chart_data: Optional chart data
        """
        chat_history = self.load_chat_history(session_id)
        
        assistant_message = {
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "map_action": map_action,
            "region_id": region_id,
            "city_id": city_id,
            "district_id": district_id,
            "map_data": map_data,
            "chart_data": chart_data
        }
        
        chat_history.append(assistant_message)
        self.save_chat_history(session_id, chat_history)
    
    def get_chat_context(self, session_id: str, max_messages: int = 10) -> str:
        """
        Get formatted chat context for LLM prompts.
        
        Args:
            session_id: The session identifier
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Formatted string of chat history for LLM context
        """
        chat_history = self.load_chat_history(session_id)
        
        # Get the most recent messages
        recent_messages = chat_history[-max_messages:] if len(chat_history) > max_messages else chat_history
        
        if not recent_messages:
            return ""
        
        context_parts = ["CHAT HISTORY:"]
        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            
            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")
        
        return "\n".join(context_parts)
    
    def clear_chat_history(self, session_id: str) -> None:
        """
        Clear chat history for a session.
        
        Args:
            session_id: The session identifier
        """
        session_file = self._get_session_file_path(session_id)
        if session_file.exists():
            session_file.unlink()
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics about a session's chat history.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Dictionary with session statistics
        """
        chat_history = self.load_chat_history(session_id)
        
        user_messages = [msg for msg in chat_history if msg.get("role") == "user"]
        assistant_messages = [msg for msg in chat_history if msg.get("role") == "assistant"]
        
        return {
            "session_id": session_id,
            "total_messages": len(chat_history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "first_message_time": chat_history[0].get("timestamp") if chat_history else None,
            "last_message_time": chat_history[-1].get("timestamp") if chat_history else None
        }
