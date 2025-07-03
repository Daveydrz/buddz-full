# voice/database.py - Voice profile database management
import json
import os
from config import KNOWN_USERS_PATH, DEBUG

# Global voice database
known_users = {}

def load_known_users():
    """Load voice database"""
    global known_users
    if os.path.exists(KNOWN_USERS_PATH):
        try:
            with open(KNOWN_USERS_PATH, "r", encoding="utf-8") as f:
                known_users = json.load(f)
            print(f"[Buddy V2] 📚 Loaded {len(known_users)} voice profiles")
        except Exception as e:
            print(f"[Buddy V2] ❌ Error loading voices: {e}")
            known_users = {}
    else:
        known_users = {}
        print("[Buddy V2] 📁 No voice database found, starting fresh")

def save_known_users():
    """Save voice database"""
    try:
        with open(KNOWN_USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(known_users, f, indent=2, ensure_ascii=False)
        print(f"[Buddy V2] 💾 Saved {len(known_users)} voice profiles")
    except Exception as e:
        print(f"[Buddy V2] ❌ Error saving voices: {e}")

def debug_voice_database():
    """Debug function to check voice database"""
    try:
        print(f"\n[DEBUG] 🔍 Voice Database Check:")
        print(f"[DEBUG] Database path: {KNOWN_USERS_PATH}")
        print(f"[DEBUG] File exists: {os.path.exists(KNOWN_USERS_PATH)}")
        
        if known_users:
            print(f"[DEBUG] Users in memory: {len(known_users)}")
            for name, embedding in known_users.items():
                if isinstance(embedding, list):
                    print(f"[DEBUG] - '{name}': {len(embedding)} dimensions")
                else:
                    print(f"[DEBUG] - '{name}': invalid type {type(embedding)}")
        else:
            print(f"[DEBUG] No users in memory")
            
    except Exception as e:
        print(f"[DEBUG] Error: {e}")

def clean_voice_database():
    """Clean up corrupted voice database"""
    global known_users
    
    print("[CLEAN] 🧹 Cleaning voice database...")
    
    # Remove invalid entries
    valid_users = {}
    for name, embedding in known_users.items():
        if isinstance(embedding, list) and len(embedding) == 256:
            # Only keep properly named users (not single letters like "To")
            if len(name) >= 2 and name.replace(" ", "").isalpha():
                valid_users[name] = embedding
                print(f"[CLEAN] ✅ Kept: '{name}'")
            else:
                print(f"[CLEAN] ❌ Removed invalid name: '{name}'")
        else:
            print(f"[CLEAN] ❌ Removed invalid embedding for: '{name}'")
    
    known_users = valid_users
    save_known_users()
    print(f"[CLEAN] ✅ Database cleaned. {len(valid_users)} valid users remain.")