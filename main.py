import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import requests
import speech_recognition as sr
import pyttsx3
import asyncio
import edge_tts
import pygame
import time
import threading
import sys
from typing import List, Dict
from memory import (save_memory, retrieve_memories, get_conversation_context, 
                   get_memory_stats, retrieve_memories_advanced, set_user,
                   get_user_profile, update_user_profile, list_users)

# Import advanced memory features
try:
    from memory_enhancements import (
        get_ai_memory_insights, create_memory_visualizations, 
        predict_future_memories, search_memories_advanced,
        generate_memory_report
    )
    ADVANCED_MEMORY_AVAILABLE = True
    print("✅ Advanced memory features loaded")
except ImportError as e:
    print(f"⚠️ Advanced memory features not available: {e}")
    ADVANCED_MEMORY_AVAILABLE = False

# Initialize recognizer and TTS engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Global variable to control TTS interruption
interrupt_tts = False

def listen():
    """Enhanced listening with natural pause detection"""
    with sr.Microphone() as source:
        # Adjust for ambient noise
        print("🎤 Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        
        # Configure for natural speech detection
        recognizer.pause_threshold = 1.0  # Pause threshold for natural flow
        recognizer.energy_threshold = 300  # Slightly higher to avoid false triggers
        recognizer.dynamic_energy_threshold = True  # Automatically adjust to environment
        recognizer.dynamic_energy_adjustment_damping = 0.15
        recognizer.dynamic_energy_ratio = 1.5
        recognizer.phrase_time_limit = 30  # Maximum 30 seconds per phrase
        recognizer.non_speaking_duration = 0.5  # Must be less than pause_threshold
        
        print("🗣️ Listening... (speak naturally, I'll detect when you finish)")
        
        try:
            # Listen with timeout but allow for natural pauses
            audio = recognizer.listen(source, timeout=2, phrase_time_limit=30)
        except sr.WaitTimeoutError:
            print("⏰ No speech detected. Please try again.")
            return None
    
    try:
        print("🔄 Processing speech...")
        text = recognizer.recognize_google(audio)
        print(f"✅ You said: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Sorry, I couldn't understand that. Please speak more clearly.")
        return None
    except sr.RequestError as e:
        print(f"❌ Speech recognition error: {e}")
        return None

def detect_voice_interrupt():
    """Detect if user starts speaking to interrupt TTS"""
    global interrupt_tts
    
    try:
        with sr.Microphone() as source:
            # Quick ambient noise adjustment
            recognizer.adjust_for_ambient_noise(source, duration=0.1)
            
            # Configure for interrupt detection (more sensitive)
            recognizer.pause_threshold = 0.5  # Very short pause for quick detection
            recognizer.energy_threshold = 150  # Lower threshold for interrupt detection
            recognizer.non_speaking_duration = 0.2  # Must be less than pause_threshold
            
            try:
                # Short listen period to detect voice start
                audio = recognizer.listen(source, timeout=0.05, phrase_time_limit=0.5)
                # If we get here, voice was detected
                interrupt_tts = True
                print("🛑 Voice detected - interrupting AI...")
                return True
            except sr.WaitTimeoutError:
                # No voice detected, continue
                return False
    except Exception:
        return False

def ask_gpt(prompt):
    # Use Ollama's local API
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",  # Change to your preferred model if needed
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "Sorry, I couldn't generate a response.")
    except Exception as e:
        print(f"Ollama API error: {e}")
        return "Sorry, I couldn't connect to the local AI model."

def speak(text):
    """Enhanced TTS with voice-based interruption"""
    global interrupt_tts
    interrupt_tts = False
    
    print(f"🤖 AI: {text}")
    
    # Use Edge TTS for realistic speech
    voice = "en-US-JennyNeural"  # You can change to any supported voice
    
    async def _speak():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save("output.mp3")
    
    asyncio.run(_speak())

    # Shared variables for thread communication
    audio_playing = threading.Event()
    audio_finished = threading.Event()

    def play_audio():
        global interrupt_tts
        try:
            pygame.mixer.init()
            pygame.mixer.music.load("output.mp3")
            pygame.mixer.music.play()
            audio_playing.set()
            
            while pygame.mixer.music.get_busy() and not interrupt_tts:
                time.sleep(0.05)
            
            if interrupt_tts:
                pygame.mixer.music.stop()
                print("🛑 Audio interrupted by voice")
            
            pygame.mixer.quit()
        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            audio_finished.set()
            if os.path.exists("output.mp3"):
                try:
                    os.remove("output.mp3")
                except:
                    pass

    def monitor_voice_interrupt():
        """Monitor for voice interruption while TTS is playing"""
        global interrupt_tts
        
        # Wait for audio to start playing
        audio_playing.wait(timeout=2)
        
        while not audio_finished.is_set() and not interrupt_tts:
            if detect_voice_interrupt():
                break
            time.sleep(0.1)

    # Start audio playback
    audio_thread = threading.Thread(target=play_audio)
    audio_thread.start()
    
    # Start voice interrupt monitoring
    print("🎤 Say something to interrupt...")
    interrupt_monitor = threading.Thread(target=monitor_voice_interrupt)
    interrupt_monitor.daemon = True  # Dies when main thread dies
    interrupt_monitor.start()
    
    # Wait for audio to finish or be interrupted
    audio_thread.join()

def identify_user():
    """Identify or set up the current user"""
    print("\n👤 === User Identification ===")
    
    # List existing users
    users = list_users()
    if users:
        print("📋 Existing users:")
        for i, user in enumerate(users[:5], 1):  # Show first 5 users
            last_seen = user.get("last_seen", "Never")
            if last_seen != "Never":
                from datetime import datetime
                try:
                    last_seen_dt = datetime.fromisoformat(last_seen)
                    last_seen = last_seen_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            print(f"{i}. {user['name']} (Last seen: {last_seen})")
        
        if len(users) > 5:
            print(f"... and {len(users) - 5} more users")
    
    print("\nHow would you like to proceed?")
    print("1. Say your name for voice identification")
    print("2. Type your name")
    if users:
        print("3. Select from existing users")
    
    choice = input("\nEnter your choice (1/2" + ("/3" if users else "") + "): ").strip()
    
    user_name = None
    
    if choice == "1":
        print("\n🎤 Please say your name clearly...")
        user_name = listen()
        if user_name:
            print(f"Heard: {user_name}")
            confirm = input("Is this correct? (y/n): ").strip().lower()
            if confirm != 'y':
                user_name = None
    
    elif choice == "2":
        user_name = input("Enter your name: ").strip()
    
    elif choice == "3" and users:
        try:
            selection = int(input(f"Select user (1-{min(len(users), 5)}): ")) - 1
            if 0 <= selection < min(len(users), 5):
                user_name = users[selection]["name"]
                print(f"Selected: {user_name}")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")
    
    if not user_name:
        print("❌ User identification failed. Please try again.")
        return identify_user()
    
    # Set the user in memory system
    user_id = set_user(user_name)
    
    # Get user profile
    profile = get_user_profile()
    if profile.get("name"):
        print(f"\n✅ Welcome back, {profile['name']}!")
        if profile.get("last_seen"):
            try:
                from datetime import datetime
                last_seen = datetime.fromisoformat(profile["last_seen"])
                print(f"Last interaction: {last_seen.strftime('%Y-%m-%d at %H:%M')}")
            except:
                pass
    else:
        print(f"\n🎉 Welcome, {user_name}! This is your first time with HeraAI.")
        print("I'll remember our conversations and learn your preferences over time.")
    
    return user_name

def display_user_memory_stats():
    """Display user-specific memory statistics"""
    stats = get_memory_stats()
    if not stats:
        print("No memory statistics available.")
        return
        
    print(f"\n=== Memory Stats for {stats.get('user_name', 'Unknown')} ===")
    print(f"Total Memories: {stats.get('total_memories_in_db', 0)}")
    print(f"Memory Retrievals: {stats.get('retrievals', 0)}")
    print(f"Conversation Sessions: {stats.get('sessions', 0)}")
    if stats.get('first_interaction'):
        try:
            from datetime import datetime
            first = datetime.fromisoformat(stats['first_interaction'])
            print(f"First Interaction: {first.strftime('%Y-%m-%d')}")
        except:
            pass
    print("="*50)

def show_user_profile():
    """Display current user's profile"""
    profile = get_user_profile()
    if not profile:
        print("No user profile available.")
        return
        
    print(f"\n=== Profile for {profile.get('name', 'Unknown')} ===")
    print(f"User ID: {profile.get('user_id', 'N/A')}")
    
    if profile.get('interests'):
        print(f"Interests: {', '.join(profile['interests'][:3])}...")
    
    if profile.get('goals'):
        print(f"Goals: {', '.join(profile['goals'][:2])}...")
    
    if profile.get('preferences'):
        recent_prefs = list(profile['preferences'].values())[-2:]
        print(f"Recent Preferences: {'; '.join(recent_prefs)}")
    
    print("="*50)

def show_advanced_context(query):
    """Show advanced memory context retrieval"""
    print("\n--- Advanced Memory Analysis ---")
    advanced_memories = retrieve_memories_advanced(query, top_k=3)
    
    if advanced_memories:
        for i, mem in enumerate(advanced_memories, 1):
            mem_type = mem['metadata'].get('type', 'conversation')
            importance = mem['importance_score']
            age_days = 0
            try:
                from datetime import datetime
                mem_time = datetime.fromisoformat(mem['metadata']['timestamp'])
                age_days = (datetime.utcnow() - mem_time).days
            except:
                pass
                
            print(f"{i}. [{mem['metadata']['role'].upper()}-{mem_type.upper()}] {mem['text'][:80]}...")
            print(f"   Score: {mem['composite_score']:.3f} | Importance: {importance:.2f} | "
                  f"Age: {age_days} days | Time Decay: {mem['time_decay']:.2f}")
    else:
        print("No relevant memories found.")
    print("-"*60)

def main():
    print("🎤 === HeraAI Voice Chat with Natural Speech Detection === 🎤")
    if ADVANCED_MEMORY_AVAILABLE:
        print("🧠 Enhanced with Ultra-Advanced Memory System & AI Insights")
    else:
        print("🧠 Enhanced with Voice-Based Interaction and Long-Term Memory")
    
    # User identification
    current_user = identify_user()
    
    # Display initial memory stats
    display_user_memory_stats()
    
    conversation = []  # Store current conversation
    turn_count = 0
    
    print(f"\n🗣️ Starting conversation with {current_user}...")
    print("✨ Voice Features:")
    print("- Speak naturally - I'll detect when you finish")
    print("- Interrupt me by speaking while I'm talking")
    print("- No need to press any buttons!")
    print("\n📋 Voice Commands:")
    print("- 'memory stats' - Show memory statistics")
    print("- 'show profile' - Display user profile") 
    print("- 'show context [query]' - Analyze memory context")
    print("- 'list users' - Show all users")
    
    if ADVANCED_MEMORY_AVAILABLE:
        print("\n🚀 Advanced Memory Commands:")
        print("- 'ai insights' - Get AI's self-analysis of memory patterns")
        print("- 'memory visualize' - Create visual memory maps")
        print("- 'predict memories [context]' - Predict relevant future memories")
        print("- 'search memories [query]' - Advanced natural language memory search")
        print("- 'memory report' - Generate comprehensive memory analysis")
        print("- 'memory health' - Check memory system health")
    
    print("- 'exit/quit/stop' - End conversation")
    
    while True:
        print(f"\n--- Turn {turn_count + 1} ---")
        user_input = listen()
        
        if user_input:
            # Handle special commands
            if user_input.lower() in ["exit", "quit", "stop"]:
                speak(f"Goodbye {current_user}! I'll remember our conversation for next time.")
                print("Goodbye!")
                display_user_memory_stats()
                break
            elif user_input.lower() == "memory stats":
                display_user_memory_stats()
                continue
            elif user_input.lower() == "show profile":
                show_user_profile()
                continue
            elif user_input.lower().startswith("show context"):
                query = user_input[12:].strip() or "general conversation"
                show_advanced_context(query)
                continue
            elif user_input.lower() == "list users":
                users = list_users()
                print(f"\n📋 All Users ({len(users)} total):")
                for user in users[:10]:  # Show first 10
                    print(f"- {user['name']} (Last seen: {user.get('last_seen', 'Never')[:10]})")
                continue
            
            # Advanced memory commands
            elif ADVANCED_MEMORY_AVAILABLE and user_input.lower() == "ai insights":
                handle_ai_insights()
                continue
            elif ADVANCED_MEMORY_AVAILABLE and user_input.lower() == "memory visualize":
                handle_memory_visualization()
                continue
            elif ADVANCED_MEMORY_AVAILABLE and user_input.lower().startswith("predict memories"):
                context = user_input[15:].strip() or "current conversation"
                handle_memory_prediction(context, conversation)
                continue
            elif ADVANCED_MEMORY_AVAILABLE and user_input.lower().startswith("search memories"):
                query = user_input[14:].strip()
                if query:
                    handle_advanced_memory_search(query)
                else:
                    print("Please provide a search query after 'search memories'")
                continue
            elif ADVANCED_MEMORY_AVAILABLE and user_input.lower() == "memory report":
                handle_memory_report()
                continue
            elif ADVANCED_MEMORY_AVAILABLE and user_input.lower() == "memory health":
                handle_memory_health()
                continue
                
            # Add to current conversation
            conversation.append({"role": "user", "content": user_input})
            
            # Save to long-term memory with enhanced type detection
            mem_type = "conversation"
            text_lower = user_input.lower()
            
            # Auto-detect memory type for better organization
            if any(indicator in text_lower for indicator in ["my name is", "i am", "call me"]):
                mem_type = "personal"
            elif any(indicator in text_lower for indicator in ["i like", "i love", "my favorite", "i prefer"]):
                mem_type = "preference"
            elif any(indicator in text_lower for indicator in ["remember", "important", "don't forget"]):
                mem_type = "fact"
            elif any(indicator in text_lower for indicator in ["my goal", "i want to", "i plan"]):
                mem_type = "personal"
            
            save_memory(user_input, "user", mem_type=mem_type)
            
            print("🧠 Thinking with long-term user memory...")
            
            # Use advanced context building with longer context for better continuity
            ltm_context = get_conversation_context(user_input, max_length=2000)
            
            # Check if we actually have meaningful memory context
            has_memory_context = bool(ltm_context and ltm_context.strip())
            
            if has_memory_context:
                print("📝 [Retrieved Long-Term Memories]")
                context_lines = ltm_context.split('\n')
                for line in context_lines[:4]:  # Show first 4 for more insight
                    print(f"   {line}")
                if len(context_lines) > 4:
                    print(f"   ... and {len(context_lines) - 4} more memories")
            else:
                print("📝 [No relevant memories found - treating as new conversation]")
            
            # Build comprehensive context for the AI
            recent_conversation = "\n".join([
                f"{'User' if msg['role']=='user' else 'AI'}: {msg['content']}" 
                for msg in conversation[-8:]  # More recent context
            ])
            
            # Conditional system prompt based on memory availability
            if has_memory_context:
                # System prompt when we have actual memory context
                system_prompt = f"""You are Hera AI, a helpful and engaging voice assistant with long-term memory capabilities. You are currently talking with {current_user}. 

You have access to background information about this user from previous conversations. Use this information naturally to:
- Remember details that are explicitly mentioned in the background context
- Maintain continuity based on what's actually provided
- Reference past discussions only if they appear in the background information
- Be helpful and engaging while staying grounded in the actual context provided

IMPORTANT: Only reference information that is explicitly provided in the background context below. Do not assume or make up details about the user that aren't mentioned."""
            else:
                # System prompt when we have no memory context
                system_prompt = f"""You are Hera AI, a helpful and engaging voice assistant. You are currently talking with {current_user}. 

This appears to be a new conversation or you don't have relevant background information about this user yet. Respond naturally and helpfully:
- Be friendly and engaging
- Ask questions to learn about the user
- Don't reference past conversations that you don't actually have information about
- Focus on the current conversation and building rapport

You should respond as if you're meeting this user for the first time or continuing a fresh conversation."""
            
            # Clean and organize background context only if it exists
            background_info = ""
            if has_memory_context and ltm_context:
                # Process LTM to extract essential information while preserving important details
                background_lines = ltm_context.split('\n')
                processed_background = []
                for line in background_lines:
                    if line.strip():
                        # Keep some structure for important memory types
                        if any(mem_type in line for mem_type in ["PERSONAL", "FACT", "PREFERENCE"]):
                            # Keep these as-is for importance
                            processed_background.append(line)
                        else:
                            # Clean regular conversation lines
                            clean_line = line.replace('[USER]:', '').replace('[AI]:', '').strip()
                            if clean_line and len(clean_line) > 15:
                                processed_background.append(f"Previous context: {clean_line}")
                
                if processed_background:
                    background_info = f"\nRelevant background about {current_user}:\n" + "\n".join(processed_background[:8])
            
            # Construct the final prompt - only include background if it exists
            if has_memory_context and background_info:
                full_context = f"""{system_prompt}
{background_info}

Recent conversation:
{recent_conversation}

User: {user_input}
AI:"""
            else:
                full_context = f"""{system_prompt}

Current conversation:
{recent_conversation}

User: {user_input}
AI:"""
            
            # Get AI response
            ai_response = ask_gpt(full_context)
            
            # Add AI response to conversation
            conversation.append({"role": "ai", "content": ai_response})
            
            # Save AI response to long-term memory
            save_memory(ai_response, "ai", mem_type="conversation")
            
            # Speak the response with voice interruption capability
            speak(ai_response)
            
            turn_count += 1
            
            # Show memory stats every 10 turns for long-term tracking
            if turn_count % 10 == 0:
                display_user_memory_stats()
                
            # Show advanced insights every 20 turns if available
            if ADVANCED_MEMORY_AVAILABLE and turn_count % 20 == 0:
                print("\n🔍 Automatic AI Memory Insights:")
                try:
                    insights = get_ai_memory_insights()
                    personality = insights.get('memory_personality_analysis', {})
                    print(f"   Memory Personality: {personality.get('communication_style', 'balanced')}")
                    print(f"   Primary Interest: {personality.get('primary_interest_area', 'general')}")
                    print(f"   Emotional Pattern: {personality.get('dominant_emotional_pattern', 'neutral')}")
                except Exception as e:
                    print(f"   Could not generate insights: {e}")

def handle_ai_insights():
    """Handle AI memory insights command"""
    try:
        print("\n🤖 === AI Memory Self-Analysis ===")
        insights = get_ai_memory_insights()
        
        # Display personality analysis
        personality = insights.get('memory_personality_analysis', {})
        print(f"🧠 Memory Personality Traits:")
        print(f"   - Communication Style: {personality.get('communication_style', 'Unknown')}")
        print(f"   - Primary Interest Area: {personality.get('primary_interest_area', 'Unknown')}")
        print(f"   - Dominant Emotional Pattern: {personality.get('dominant_emotional_pattern', 'Unknown')}")
        print(f"   - Topic Breadth: {personality.get('topic_breadth', 0)} different topics")
        print(f"   - Emotional Diversity: {personality.get('emotional_diversity', 0)} emotion types")
        
        # Display conversation patterns
        conversation = insights.get('conversation_patterns', {})
        print(f"\n💬 Conversation Analysis:")
        print(f"   - Average Conversation Length: {conversation.get('avg_conversation_length', 0):.1f} exchanges")
        print(f"   - Topic Transition Rate: {conversation.get('topic_transition_rate', 0):.2f}")
        print(f"   - Conversation Depth Score: {conversation.get('conversation_depth_score', 0):.2f}")
        print(f"   - Total Sessions: {conversation.get('total_unique_sessions', 0)}")
        
        # Display engagement patterns
        engagement = conversation.get('engagement_patterns', {})
        print(f"\n🎯 User Engagement Analysis:")
        print(f"   - Average User Message Length: {engagement.get('avg_user_message_length', 0):.0f} characters")
        print(f"   - Average AI Message Length: {engagement.get('avg_ai_message_length', 0):.0f} characters")
        print(f"   - User Engagement Score: {engagement.get('user_engagement_score', 0):.2f}/1.0")
        print(f"   - Communication Balance: {engagement.get('communication_balance', 0):.2f}")
        
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error generating AI insights: {e}")

def handle_memory_visualization():
    """Handle memory visualization command"""
    try:
        print("\n📊 Creating memory visualizations...")
        visualizations = create_memory_visualizations()
        
        if visualizations:
            print("✅ Memory visualizations created:")
            for viz_type, path in visualizations.items():
                print(f"   - {viz_type.replace('_', ' ').title()}: {path}")
            print("\nYou can open these files to view your memory patterns!")
        else:
            print("❌ Could not create visualizations. Make sure you have enough memories.")
            
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")

def handle_memory_prediction(context: str, conversation: List[Dict[str, str]]):
    """Handle memory prediction command"""
    try:
        print(f"\n🔮 Predicting relevant memories for: '{context}'")
        
        # Extract conversation flow
        conversation_flow = [msg['content'] for msg in conversation[-10:]]
        
        predictions = predict_future_memories(context, conversation_flow)
        
        if predictions:
            print(f"📈 Found {len(predictions)} potential memory predictions:")
            for i, pred in enumerate(predictions[:5], 1):
                print(f"\n{i}. Confidence: {pred['confidence']:.2f}")
                print(f"   Strategy: {pred['prediction_type']}")
                print(f"   Reasoning: {pred['reasoning']}")
                print(f"   Content: {pred['content_preview']}")
        else:
            print("🔍 No memory predictions available yet. Continue conversing to build patterns!")
            
    except Exception as e:
        print(f"❌ Error predicting memories: {e}")

def handle_advanced_memory_search(query: str):
    """Handle advanced memory search command"""
    try:
        print(f"\n🔍 Searching memories for: '{query}'")
        
        search_results = search_memories_advanced(query)
        
        print(f"📊 Search Results ({search_results['total_found']} found):")
        print(f"   Search Type: {search_results['search_type']}")
        
        if search_results['results']:
            for i, result in enumerate(search_results['results'][:5], 1):
                print(f"\n{i}. Relevance: {result['relevance_score']:.2f}")
                print(f"   Content: {result['content'][:100]}...")
                print(f"   Match Reason: {result['match_reason']}")
        
        # Show suggestions
        if search_results['suggestions']:
            print(f"\n💡 Search Suggestions:")
            for suggestion in search_results['suggestions']:
                print(f"   - {suggestion}")
        
        # Show insights
        insights = search_results['search_insights']
        print(f"\n📈 Search Insights:")
        print(f"   - Result Quality: {insights['result_quality']}")
        print(f"   - Search Effectiveness: {insights['search_effectiveness']:.1%}")
        if insights['time_span']:
            print(f"   - Time Span Covered: {insights['time_span']}")
        if insights['dominant_themes']:
            print(f"   - Key Themes: {', '.join(insights['dominant_themes'])}")
            
    except Exception as e:
        print(f"❌ Error searching memories: {e}")

def handle_memory_report():
    """Handle comprehensive memory report generation"""
    try:
        print("\n📋 Generating comprehensive memory report...")
        
        report = generate_memory_report()
        
        print("="*60)
        print("📊 COMPREHENSIVE MEMORY SYSTEM REPORT")
        print("="*60)
        
        # Basic stats
        print(f"\n📈 System Overview:")
        print(f"   - Total Memories: {report['total_memories']}")
        print(f"   - Memory Clusters: {report['total_clusters']}")
        print(f"   - Memory Relationships: {report['total_relationships']}")
        print(f"   - Report Generated: {report['generated_at'][:19]}")
        
        # System analytics
        analytics = report['system_analytics']
        basic_stats = analytics['basic_stats']
        print(f"\n📊 Usage Statistics:")
        print(f"   - Total Retrievals: {basic_stats['total_retrievals']}")
        print(f"   - Successful Retrievals: {basic_stats['successful_retrievals']}")
        print(f"   - Success Rate: {(basic_stats['successful_retrievals']/max(1,basic_stats['total_retrievals']))*100:.1f}%")
        
        # Memory distribution
        distribution = analytics['memory_distribution']
        print(f"\n🗂️ Memory Distribution:")
        print(f"   By Type: {distribution['by_type']}")
        print(f"   By Role: {distribution['by_role']}")
        
        # AI insights summary
        ai_insights = report['ai_insights']
        personality = ai_insights.get('memory_personality_analysis', {})
        print(f"\n🤖 AI Personality Analysis:")
        print(f"   - Communication Style: {personality.get('communication_style', 'Unknown')}")
        print(f"   - Primary Interest: {personality.get('primary_interest_area', 'Unknown')}")
        print(f"   - Emotional Pattern: {personality.get('dominant_emotional_pattern', 'Unknown')}")
        
        # Validation results
        validation = report['validation_results']
        print(f"\n✅ Memory Health Check:")
        print(f"   - Validation Score: {validation['validation_score']:.2f}/1.0")
        print(f"   - Conflicts Detected: {validation['conflicts_detected']}")
        print(f"   - Duplicate Candidates: {len(validation['duplicate_candidates'])}")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error generating memory report: {e}")

def handle_memory_health():
    """Handle memory system health check"""
    try:
        print("\n🏥 Memory System Health Check...")
        
        from memory import memory_system
        validation = memory_system.validate_memory_consistency()
        
        print("="*50)
        print("🏥 MEMORY SYSTEM HEALTH REPORT")
        print("="*50)
        
        # Overall health score
        health_score = validation['validation_score']
        if health_score >= 0.9:
            health_status = "🟢 Excellent"
        elif health_score >= 0.7:
            health_status = "🟡 Good"
        else:
            health_status = "🔴 Needs Attention"
        
        print(f"Overall Health: {health_status} ({health_score:.2f}/1.0)")
        print(f"Total Memories: {validation['total_memories']}")
        print(f"Conflicts Detected: {validation['conflicts_detected']}")
        print(f"Duplicate Candidates: {len(validation['duplicate_candidates'])}")
        
        # Show issues if any
        if validation['inconsistencies']:
            print(f"\n⚠️ Inconsistencies Found:")
            for issue in validation['inconsistencies'][:3]:
                print(f"   - {issue['type']}: {issue['fact']} has conflicting values")
        
        if validation['duplicate_candidates']:
            print(f"\n🔍 Potential Duplicates:")
            for dup in validation['duplicate_candidates'][:3]:
                print(f"   - Similarity {dup['similarity']:.3f}: '{dup['content1'][:50]}...' vs '{dup['content2'][:50]}...'")
        
        # Memory performance metrics
        analytics = memory_system.get_memory_analytics()
        cluster_analysis = analytics.get('cluster_analysis', {})
        
        print(f"\n📊 Performance Metrics:")
        print(f"   - Memory Clusters: {cluster_analysis.get('total_clusters', 0)}")
        print(f"   - Average Cluster Size: {cluster_analysis.get('avg_cluster_size', 0):.1f}")
        print(f"   - Memory Relationships: {analytics.get('relationship_analysis', {}).get('total_relationships', 0)}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if validation['conflicts_detected'] > 0:
            print("   - Review and resolve memory conflicts")
        if len(validation['duplicate_candidates']) > 5:
            print("   - Consider memory consolidation")
        if validation['total_memories'] > 1000:
            print("   - Regular memory compression recommended")
        if validation['validation_score'] > 0.95:
            print("   - Memory system is operating optimally!")
        
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error checking memory health: {e}")

if __name__ == "__main__":
    main()
