"""
User Management Module for HeraAI

Handles user identification, profile management, and user-specific settings.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
from ..config.settings import MEMORY_CONFIG


class UserManager:
    """Manages user identification and profiles"""
    
    def __init__(self):
        """Initialize the user manager"""
        self.current_user = None
        self.user_data_dir = MEMORY_CONFIG["user_data_dir"]
        os.makedirs(self.user_data_dir, exist_ok=True)
    
    def identify_user(self) -> str:
        """
        Interactive user identification process
        
        Returns:
            str: The identified username
        """
        print("\n👤 === User Identification ===")
        
        # List existing users
        users = self.list_users()
        if users:
            print("📋 Existing users:")
            for i, user in enumerate(users[:5], 1):  # Show first 5 users
                last_seen = user.get("last_seen", "Never")
                if last_seen != "Never":
                    try:
                        last_seen_date = datetime.fromisoformat(last_seen)
                        last_seen = last_seen_date.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
                print(f"   {i}. {user['name']} (Last seen: {last_seen})")
            
            if len(users) > 5:
                print(f"   ... and {len(users) - 5} more users")
            
            print("\n💭 Options:")
            print("   • Enter a number to select an existing user")
            print("   • Type a name to create a new user")
            print("   • Type 'list' to see all users")
            
            choice = input("\n👤 Enter your choice: ").strip()
            
            # Handle numeric selection
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(users):
                    username = users[index]['name']
                    print(f"✅ Welcome back, {username}!")
                    self.current_user = username
                    self._update_last_seen(username)
                    return username
                else:
                    print("❌ Invalid selection. Please try again.")
                    return self.identify_user()
            
            # Handle special commands
            elif choice.lower() == 'list':
                print("\n📋 All users:")
                for user in users:
                    last_seen = user.get("last_seen", "Never")
                    print(f"   • {user['name']} (Last seen: {last_seen})")
                return self.identify_user()
            
            # Handle new username
            else:
                username = choice
                if self._is_valid_username(username):
                    if any(user['name'].lower() == username.lower() for user in users):
                        print(f"✅ Welcome back, {username}!")
                        self.current_user = username
                        self._update_last_seen(username)
                        return username
                    else:
                        print(f"🆕 Creating new user profile for {username}")
                        self._create_user(username)
                        self.current_user = username
                        return username
                else:
                    print("❌ Invalid username. Please use only letters, numbers, and underscores.")
                    return self.identify_user()
        
        else:
            # No existing users
            print("🆕 No existing users found. Let's create your profile!")
            username = input("👤 What's your name? ").strip()
            
            if self._is_valid_username(username):
                print(f"🎉 Welcome to HeraAI, {username}!")
                self._create_user(username)
                self.current_user = username
                return username
            else:
                print("❌ Invalid username. Please use only letters, numbers, and underscores.")
                return self.identify_user()
    
    def _is_valid_username(self, username: str) -> bool:
        """
        Validate username format
        
        Args:
            username: The username to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not username or len(username.strip()) == 0:
            return False
        
        username = username.strip()
        if len(username) < 2 or len(username) > 50:
            return False
        
        # Allow letters, numbers, spaces, hyphens, and underscores
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_')
        return all(char in allowed_chars for char in username)
    
    def _create_user(self, username: str) -> None:
        """
        Create a new user profile
        
        Args:
            username: The username for the new user
        """
        profile = {
            "name": username,
            "created_at": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat(),
            "preferences": {},
            "settings": {
                "voice_sensitivity": "medium",
                "memory_retention": "high",
                "personality": "helpful"
            },
            "statistics": {
                "total_conversations": 0,
                "total_memories": 0,
                "favorite_topics": []
            }
        }
        
        # Save user profile
        profile_file = os.path.join(self.user_data_dir, f"user_profile_{self._get_user_id(username)}.json")
        try:
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            print(f"✅ User profile created successfully")
        except Exception as e:
            print(f"⚠️ Error creating user profile: {e}")
    
    def _update_last_seen(self, username: str) -> None:
        """
        Update the last seen timestamp for a user
        
        Args:
            username: The username to update
        """
        user_id = self._get_user_id(username)
        profile_file = os.path.join(self.user_data_dir, f"user_profile_{user_id}.json")
        
        try:
            if os.path.exists(profile_file):
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                
                profile["last_seen"] = datetime.utcnow().isoformat()
                
                with open(profile_file, 'w', encoding='utf-8') as f:
                    json.dump(profile, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Error updating last seen: {e}")
    
    def list_users(self) -> List[Dict[str, str]]:
        """
        List all users
        
        Returns:
            List[Dict[str, str]]: List of user information
        """
        users = []
        
        try:
            for filename in os.listdir(self.user_data_dir):
                if filename.startswith("user_profile_") and filename.endswith(".json"):
                    profile_file = os.path.join(self.user_data_dir, filename)
                    
                    try:
                        with open(profile_file, 'r', encoding='utf-8') as f:
                            profile = json.load(f)
                        
                        users.append({
                            "name": profile.get("name", "Unknown"),
                            "created_at": profile.get("created_at", ""),
                            "last_seen": profile.get("last_seen", "Never")
                        })
                    except Exception as e:
                        print(f"⚠️ Error reading user profile {filename}: {e}")
                        continue
            
            # Sort by last seen (most recent first)
            users.sort(key=lambda x: x.get("last_seen", ""), reverse=True)
            
        except Exception as e:
            print(f"⚠️ Error listing users: {e}")
        
        return users
    
    def get_user_profile(self, username: str = None) -> Dict[str, Any]:
        """
        Get user profile information
        
        Args:
            username: Username to get profile for (defaults to current user)
            
        Returns:
            Dict[str, Any]: User profile data
        """
        if not username:
            username = self.current_user
        
        if not username:
            return {}
        
        user_id = self._get_user_id(username)
        profile_file = os.path.join(self.user_data_dir, f"user_profile_{user_id}.json")
        
        try:
            if os.path.exists(profile_file):
                with open(profile_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading user profile: {e}")
        
        return {}
    
    def update_user_profile(self, updates: Dict[str, Any], username: str = None) -> bool:
        """
        Update user profile information
        
        Args:
            updates: Dictionary of updates to apply
            username: Username to update (defaults to current user)
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if not username:
            username = self.current_user
        
        if not username:
            return False
        
        user_id = self._get_user_id(username)
        profile_file = os.path.join(self.user_data_dir, f"user_profile_{user_id}.json")
        
        try:
            profile = self.get_user_profile(username)
            if not profile:
                return False
            
            # Apply updates
            profile.update(updates)
            profile["last_updated"] = datetime.utcnow().isoformat()
            
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error updating user profile: {e}")
            return False
    
    def _get_user_id(self, username: str) -> str:
        """
        Generate a user ID from username
        
        Args:
            username: The username
            
        Returns:
            str: The user ID hash
        """
        import hashlib
        return hashlib.md5(username.encode()).hexdigest()[:16]
    
    def get_current_user(self) -> Optional[str]:
        """
        Get the current user
        
        Returns:
            Optional[str]: Current username or None
        """
        return self.current_user
    
    def display_user_stats(self, username: str = None) -> None:
        """
        Display user statistics
        
        Args:
            username: Username to display stats for (defaults to current user)
        """
        profile = self.get_user_profile(username)
        if not profile:
            print("❌ No user profile found")
            return
        
        print(f"\n📊 === User Statistics for {profile.get('name', 'Unknown')} ===")
        print(f"👤 Created: {profile.get('created_at', 'Unknown')}")
        print(f"🕒 Last seen: {profile.get('last_seen', 'Never')}")
        
        stats = profile.get('statistics', {})
        print(f"💬 Total conversations: {stats.get('total_conversations', 0)}")
        print(f"🧠 Total memories: {stats.get('total_memories', 0)}")
        
        settings = profile.get('settings', {})
        print(f"⚙️ Voice sensitivity: {settings.get('voice_sensitivity', 'medium')}")
        print(f"🎭 Personality: {settings.get('personality', 'helpful')}") 