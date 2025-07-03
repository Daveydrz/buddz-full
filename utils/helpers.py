# utils/helpers.py - Helper functions
def should_end_conversation(text):
    """Check if user wants to end conversation"""
    end_phrases = ["bye", "goodbye", "quit", "exit", "stop", "that's all"]
    return any(phrase in text.lower() for phrase in end_phrases)

def is_noise_or_gibberish(text):
    """Filter out noise and gibberish - FIXED"""
    if not text or len(text.strip()) < 1:  # Allow single words
        return True
    
    # Don't filter out short but valid responses
    valid_short_responses = ["no", "yes", "ok", "hi", "hey", "what", "why", "how", "who", "when", "where"]
    text_lower = text.lower().strip()
    
    if text_lower in valid_short_responses:
        return False
    
    # Only filter if really short AND not alphabetic
    if len(text.strip()) < 2 and not text.strip().isalpha():
        return True
    
    return False

def setup_logging():
    """Setup logging configuration"""
    pass