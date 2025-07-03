# ai/memory.py - Conversation memory management
import time
from config import MAX_HISTORY_LENGTH, DEBUG

# Global conversation storage
conversation_history = {}

def add_to_conversation_history(username, user_message, ai_response):
    """Add conversation to memory"""
    try:
        if username not in conversation_history:
            conversation_history[username] = []
        
        conversation_history[username].append({
            "user": user_message,
            "assistant": ai_response,
            "timestamp": time.time()
        })
        
        # Keep only recent history
        if len(conversation_history[username]) > MAX_HISTORY_LENGTH:
            conversation_history[username] = conversation_history[username][-MAX_HISTORY_LENGTH:]
        
        if DEBUG:
            print(f"[Buddy V2] 💭 Added to memory for {username}")
            
    except Exception as e:
        if DEBUG:
            print(f"[Buddy V2] Memory error: {e}")

def get_conversation_context(username):
    """Get conversation context for LLM"""
    try:
        if username not in conversation_history or not conversation_history[username]:
            return ""
        
        context_parts = []
        for exchange in conversation_history[username][-2:]:  # Last 2 exchanges
            user_msg = exchange["user"][:100]
            ai_msg = exchange["assistant"][:100]
            context_parts.append(f"Human: {user_msg}")
            context_parts.append(f"Assistant: {ai_msg}")
        
        return "\n".join(context_parts)
        
    except Exception as e:
        if DEBUG:
            print(f"[Buddy V2] Context error: {e}")
        return ""