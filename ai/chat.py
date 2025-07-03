# ai/chat.py - LLM chat integration
import re
import requests
from ai.memory import get_conversation_context
from config import *

def ask_kobold(messages, max_tokens=MAX_TOKENS):
    """Simple KoboldCpp request"""
    payload = {
        "model": "llama3",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False
    }
    
    try:
        response = requests.post(KOBOLD_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return "I'm having trouble thinking right now."
            
    except Exception as e:
        print(f"[Buddy V2] KoboldCpp error: {e}")
        return "Sorry, I'm having connection issues."

def generate_response(question, username, lang=DEFAULT_LANG):
    """Generate AI response"""
    try:
        # Build conversation context
        context = get_conversation_context(username)
        
        # Create system message
        context_text = f"Recent conversation context:\n{context}" if context else ""
        
        system_msg = f"""You are Buddy, {username}'s helpful AI assistant. You're friendly, casual, and conversational.
Current date/time: {CURRENT_TIMESTAMP} UTC
Current user: {SYSTEM_USER}
Always respond in {"English" if lang == "en" else "Polish" if lang == "pl" else "Italian"}.
Keep responses natural and concise (1-2 sentences unless more detail needed).
Never use markdown, emoji, or special formatting - just plain conversational text.

{context_text}"""

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question}
        ]
        
        response = ask_kobold(messages)
        
        # Clean response
        response = re.sub(r'^(Buddy:|Assistant:|Human:)\s*', '', response, flags=re.IGNORECASE)
        response = response.strip()
        
        if DEBUG:
            print(f"[Buddy V2] 🧠 Generated response: {response[:50]}...")
        
        return response
        
    except Exception as e:
        print(f"[Buddy V2] Response generation error: {e}")
        return "Sorry, I'm having trouble thinking right now."