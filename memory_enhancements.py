"""
Advanced Memory Enhancement Features for HeraAI
=====================================================

This module provides cutting-edge memory capabilities including:
- AI Memory Reflection & Self-Analysis
- Memory Visualization & Mapping
- Predictive Memory Retrieval
- Natural Language Memory Search
- Memory Insights & Pattern Recognition
- Adaptive Learning from User Feedback
- Memory Health Monitoring
- Advanced Memory Operations
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from memory import memory_system

class AIMemoryReflection:
    """AI's ability to reflect on and analyze its own memory patterns"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
        
    def generate_memory_insights(self) -> Dict[str, Any]:
        """Generate AI insights about its own memory patterns"""
        insights = {
            'memory_personality_analysis': self.analyze_memory_personality(),
            'conversation_patterns': self.analyze_conversation_patterns(),
            'learning_evolution': self.analyze_learning_evolution(),
            'memory_strengths_weaknesses': self.identify_memory_strengths_weaknesses(),
            'user_relationship_analysis': self.analyze_user_relationship(),
            'prediction_accuracy': self.analyze_prediction_accuracy()
        }
        return insights
    
    def analyze_memory_personality(self) -> Dict[str, Any]:
        """Analyze the AI's memory personality traits"""
        memory_nodes = self.memory_system.memory_nodes
        
        # Analyze emotional patterns in memories
        emotion_patterns = defaultdict(float)
        topic_interests = defaultdict(int)
        response_styles = defaultdict(int)
        
        for node in memory_nodes.values():
            # Emotional analysis
            for emotion, intensity in node.emotional_context.items():
                emotion_patterns[emotion] += intensity
            
            # Topic analysis
            content_lower = node.content.lower()
            topics = ['technology', 'personal', 'creativity', 'learning', 'relationships', 'goals']
            for topic in topics:
                if topic in content_lower:
                    topic_interests[topic] += 1
            
            # Response style analysis (if AI response)
            if node.metadata.get('role') == 'ai':
                if any(word in content_lower for word in ['question', '?', 'what', 'how', 'why']):
                    response_styles['curious'] += 1
                if any(word in content_lower for word in ['help', 'assist', 'support']):
                    response_styles['helpful'] += 1
                if any(word in content_lower for word in ['think', 'believe', 'opinion']):
                    response_styles['thoughtful'] += 1
        
        # Determine personality traits
        dominant_emotion = max(emotion_patterns.items(), key=lambda x: x[1])[0] if emotion_patterns else 'neutral'
        dominant_topic = max(topic_interests.items(), key=lambda x: x[1])[0] if topic_interests else 'general'
        dominant_style = max(response_styles.items(), key=lambda x: x[1])[0] if response_styles else 'balanced'
        
        return {
            'dominant_emotional_pattern': dominant_emotion,
            'primary_interest_area': dominant_topic,
            'communication_style': dominant_style,
            'emotional_diversity': len(emotion_patterns),
            'topic_breadth': len(topic_interests),
            'emotional_intensity': sum(emotion_patterns.values()) / len(emotion_patterns) if emotion_patterns else 0
        }
    
    def analyze_conversation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in conversations"""
        memory_nodes = self.memory_system.memory_nodes
        
        conversation_lengths = []
        topic_transitions = []
        response_quality_indicators = []
        
        # Group memories by session
        sessions = defaultdict(list)
        for node in memory_nodes.values():
            session_id = node.metadata.get('session_id', 'unknown')
            sessions[session_id].append(node)
        
        for session_memories in sessions.values():
            if len(session_memories) > 1:
                conversation_lengths.append(len(session_memories))
                
                # Analyze topic flow
                topics = []
                for memory in sorted(session_memories, key=lambda x: x.metadata.get('timestamp', '')):
                    content = memory.content.lower()
                    if 'personal' in content or 'my' in content:
                        topics.append('personal')
                    elif any(tech in content for tech in ['ai', 'technology', 'computer', 'code']):
                        topics.append('technology')
                    elif any(learn in content for learn in ['learn', 'understand', 'know', 'study']):
                        topics.append('learning')
                    else:
                        topics.append('general')
                
                # Count topic transitions
                transitions = 0
                for i in range(1, len(topics)):
                    if topics[i] != topics[i-1]:
                        transitions += 1
                topic_transitions.append(transitions / len(topics) if len(topics) > 1 else 0)
        
        return {
            'avg_conversation_length': np.mean(conversation_lengths) if conversation_lengths else 0,
            'topic_transition_rate': np.mean(topic_transitions) if topic_transitions else 0,
            'total_unique_sessions': len(sessions),
            'conversation_depth_score': self.calculate_conversation_depth(),
            'engagement_patterns': self.analyze_engagement_patterns()
        }
    
    def calculate_conversation_depth(self) -> float:
        """Calculate how deep conversations typically go"""
        memory_nodes = self.memory_system.memory_nodes
        depth_indicators = ['because', 'why', 'explain', 'understand', 'feel', 'think', 'believe']
        
        total_depth = 0
        total_memories = 0
        
        for node in memory_nodes.values():
            content_lower = node.content.lower()
            depth_score = sum(1 for indicator in depth_indicators if indicator in content_lower)
            total_depth += depth_score
            total_memories += 1
        
        return total_depth / total_memories if total_memories > 0 else 0
    
    def analyze_engagement_patterns(self) -> Dict[str, Any]:
        """Analyze user engagement patterns"""
        memory_nodes = self.memory_system.memory_nodes
        
        user_message_lengths = []
        ai_message_lengths = []
        interruption_patterns = []
        
        for node in memory_nodes.values():
            content_length = len(node.content)
            role = node.metadata.get('role', 'unknown')
            
            if role == 'user':
                user_message_lengths.append(content_length)
            elif role == 'ai':
                ai_message_lengths.append(content_length)
        
        return {
            'avg_user_message_length': np.mean(user_message_lengths) if user_message_lengths else 0,
            'avg_ai_message_length': np.mean(ai_message_lengths) if ai_message_lengths else 0,
            'user_engagement_score': self.calculate_engagement_score(user_message_lengths),
            'communication_balance': len(user_message_lengths) / (len(user_message_lengths) + len(ai_message_lengths)) if user_message_lengths or ai_message_lengths else 0
        }
    
    def calculate_engagement_score(self, message_lengths: List[int]) -> float:
        """Calculate user engagement score based on message patterns"""
        if not message_lengths:
            return 0.0
        
        # Higher scores for more varied message lengths and longer messages
        avg_length = np.mean(message_lengths)
        length_variety = np.std(message_lengths) if len(message_lengths) > 1 else 0
        
        # Normalize to 0-1 scale
        engagement = min(1.0, (avg_length / 100) * 0.7 + (length_variety / 50) * 0.3)
        return engagement

class MemoryVisualization:
    """Create visual representations of memory patterns and connections"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
    
    def create_memory_network_graph(self, save_path: str = "memory_network.png") -> str:
        """Create a network graph of memory relationships"""
        try:
            G = nx.Graph()
            
            # Add nodes (memories)
            for memory_id, node in self.memory_system.memory_nodes.items():
                # Use importance as node size
                size = max(100, node.importance_score * 1000)
                node_type = node.metadata.get('type', 'conversation')
                
                G.add_node(memory_id, 
                          size=size, 
                          type=node_type,
                          content=node.content[:50] + "..." if len(node.content) > 50 else node.content)
            
            # Add edges (relationships)
            for source_id, targets in self.memory_system.relationship_graph.items():
                for target_id in targets:
                    if source_id in G.nodes and target_id in G.nodes:
                        G.add_edge(source_id, target_id)
            
            # Create visualization
            plt.figure(figsize=(15, 12))
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Color nodes by type
            color_map = {
                'conversation': 'lightblue',
                'personal': 'lightcoral',
                'fact': 'lightgreen',
                'preference': 'lightyellow',
                'summary': 'lightpink'
            }
            
            node_colors = [color_map.get(G.nodes[node].get('type', 'conversation'), 'lightgray') 
                          for node in G.nodes()]
            node_sizes = [G.nodes[node].get('size', 100) for node in G.nodes()]
            
            nx.draw(G, pos, 
                   node_color=node_colors,
                   node_size=node_sizes,
                   with_labels=False,
                   edge_color='gray',
                   alpha=0.7)
            
            plt.title("Memory Network Graph", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Memory network graph saved to {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error creating memory network graph: {e}")
            return None
    
    def create_memory_timeline(self, save_path: str = "memory_timeline.png") -> str:
        """Create a timeline visualization of memories"""
        try:
            memory_data = []
            
            for node in self.memory_system.memory_nodes.values():
                timestamp = node.metadata.get('timestamp')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        memory_data.append({
                            'date': dt,
                            'importance': node.importance_score,
                            'type': node.metadata.get('type', 'conversation'),
                            'content': node.content[:30] + "..." if len(node.content) > 30 else node.content
                        })
                    except:
                        continue
            
            if not memory_data:
                print("⚠️ No timeline data available")
                return None
            
            df = pd.DataFrame(memory_data)
            df = df.sort_values('date')
            
            plt.figure(figsize=(15, 8))
            
            # Create scatter plot
            type_colors = {
                'conversation': 'blue',
                'personal': 'red',
                'fact': 'green',
                'preference': 'orange',
                'summary': 'purple'
            }
            
            for mem_type in df['type'].unique():
                type_data = df[df['type'] == mem_type]
                plt.scatter(type_data['date'], type_data['importance'], 
                           c=type_colors.get(mem_type, 'gray'),
                           label=mem_type, alpha=0.7, s=50)
            
            plt.xlabel('Date')
            plt.ylabel('Importance Score')
            plt.title('Memory Timeline - Importance Over Time')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Memory timeline saved to {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error creating memory timeline: {e}")
            return None
    
    def create_memory_wordcloud(self, save_path: str = "memory_wordcloud.png") -> str:
        """Create a word cloud from memory content"""
        try:
            # Collect all memory content
            all_text = []
            for node in self.memory_system.memory_nodes.values():
                # Weight text by importance
                weight = max(1, int(node.importance_score * 5))
                all_text.extend([node.content] * weight)
            
            if not all_text:
                print("⚠️ No text data available for word cloud")
                return None
            
            text_corpus = " ".join(all_text)
            
            # Create word cloud
            wordcloud = WordCloud(
                width=1200, height=800,
                background_color='white',
                max_words=200,
                colormap='viridis',
                relative_scaling=0.5
            ).generate(text_corpus)
            
            plt.figure(figsize=(15, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Memory Content Word Cloud', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Memory word cloud saved to {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error creating word cloud: {e}")
            return None

class PredictiveMemory:
    """Predict what memories might be relevant based on context and patterns"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.prediction_history = []
    
    def predict_relevant_memories(self, current_context: str, 
                                 conversation_flow: List[str] = None) -> List[Dict[str, Any]]:
        """Predict memories that might become relevant"""
        predictions = []
        
        try:
            # Analyze current context
            context_embedding = self.memory_system.model.encode(current_context).tolist()
            
            # Get conversation patterns
            if conversation_flow:
                flow_patterns = self.analyze_conversation_flow(conversation_flow)
            else:
                flow_patterns = {}
            
            # Predict based on different strategies
            predictions.extend(self.predict_by_semantic_similarity(context_embedding))
            predictions.extend(self.predict_by_temporal_patterns())
            predictions.extend(self.predict_by_relationship_chains(context_embedding))
            predictions.extend(self.predict_by_user_patterns())
            
            # Remove duplicates and sort by confidence
            seen_ids = set()
            unique_predictions = []
            for pred in predictions:
                if pred['memory_id'] not in seen_ids:
                    unique_predictions.append(pred)
                    seen_ids.add(pred['memory_id'])
            
            # Sort by prediction confidence
            unique_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return unique_predictions[:10]  # Top 10 predictions
            
        except Exception as e:
            print(f"❌ Error in predictive memory: {e}")
            return []
    
    def predict_by_semantic_similarity(self, context_embedding: List[float]) -> List[Dict[str, Any]]:
        """Predict based on semantic similarity with looser thresholds"""
        predictions = []
        
        for memory_id, node in self.memory_system.memory_nodes.items():
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([context_embedding], [node.embedding])[0][0]
            
            # Lower threshold for predictions (might become relevant)
            if 0.3 < similarity < 0.6:  # Not immediately relevant but potentially so
                confidence = similarity * 0.7  # Lower confidence for predictions
                predictions.append({
                    'memory_id': memory_id,
                    'confidence': confidence,
                    'prediction_type': 'semantic_potential',
                    'reasoning': f"Semantic similarity suggests potential relevance (score: {similarity:.3f})",
                    'content_preview': node.content[:100] + "..." if len(node.content) > 100 else node.content
                })
        
        return predictions
    
    def predict_by_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Predict based on temporal access patterns"""
        predictions = []
        current_time = datetime.utcnow()
        
        # Find memories accessed at similar times historically
        time_patterns = defaultdict(list)
        
        for node in self.memory_system.memory_nodes.values():
            try:
                timestamp = datetime.fromisoformat(node.metadata.get('timestamp', ''))
                hour = timestamp.hour
                day_of_week = timestamp.weekday()
                time_key = f"{day_of_week}_{hour}"
                time_patterns[time_key].append(node)
            except:
                continue
        
        # Current time pattern
        current_key = f"{current_time.weekday()}_{current_time.hour}"
        if current_key in time_patterns:
            for node in time_patterns[current_key][:5]:  # Top 5 from this time pattern
                confidence = 0.4 + (node.access_count * 0.1)  # Base + access boost
                predictions.append({
                    'memory_id': node.id,
                    'confidence': min(confidence, 0.8),
                    'prediction_type': 'temporal_pattern',
                    'reasoning': f"Similar time pattern (accessed during {current_key})",
                    'content_preview': node.content[:100] + "..." if len(node.content) > 100 else node.content
                })
        
        return predictions
    
    def predict_by_relationship_chains(self, context_embedding: List[float]) -> List[Dict[str, Any]]:
        """Predict memories through relationship chains"""
        predictions = []
        
        # Find memories with strong relationships that might become relevant
        for memory_id, relationships in self.memory_system.relationship_graph.items():
            if len(relationships) > 2:  # Well-connected memories
                node = self.memory_system.memory_nodes.get(memory_id)
                if node:
                    # Calculate relationship strength
                    relationship_strength = len(relationships) / 10  # Normalize
                    
                    # Check if any related memories are semantically close to context
                    max_related_similarity = 0
                    for related_id in relationships:
                        related_node = self.memory_system.memory_nodes.get(related_id)
                        if related_node:
                            from sklearn.metrics.pairwise import cosine_similarity
                            similarity = cosine_similarity([context_embedding], [related_node.embedding])[0][0]
                            max_related_similarity = max(max_related_similarity, similarity)
                    
                    if max_related_similarity > 0.4:  # Related memory is somewhat relevant
                        confidence = relationship_strength * 0.6 + max_related_similarity * 0.4
                        predictions.append({
                            'memory_id': memory_id,
                            'confidence': min(confidence, 0.8),
                            'prediction_type': 'relationship_chain',
                            'reasoning': f"Connected to {len(relationships)} related memories, max similarity: {max_related_similarity:.3f}",
                            'content_preview': node.content[:100] + "..." if len(node.content) > 100 else node.content
                        })
        
        return predictions
    
    def predict_by_user_patterns(self) -> List[Dict[str, Any]]:
        """Predict based on user behavior patterns"""
        predictions = []
        
        # Find frequently accessed memories that might be referenced again
        high_access_memories = []
        for node in self.memory_system.memory_nodes.values():
            if node.access_count > 3:  # Frequently accessed
                high_access_memories.append(node)
        
        # Sort by access count and recency
        high_access_memories.sort(key=lambda x: (x.access_count, x.last_accessed), reverse=True)
        
        for i, node in enumerate(high_access_memories[:5]):
            confidence = 0.5 + (node.access_count * 0.05) - (i * 0.05)  # Decrease for lower rank
            predictions.append({
                'memory_id': node.id,
                'confidence': min(confidence, 0.8),
                'prediction_type': 'user_pattern',
                'reasoning': f"Frequently accessed memory (count: {node.access_count})",
                'content_preview': node.content[:100] + "..." if len(node.content) > 100 else node.content
            })
        
        return predictions
    
    def analyze_conversation_flow(self, conversation_flow: List[str]) -> Dict[str, Any]:
        """Analyze the flow of conversation to predict direction"""
        # Simple pattern analysis
        topics = []
        sentiment_trend = []
        
        for message in conversation_flow[-5:]:  # Last 5 messages
            # Simple topic detection
            if any(word in message.lower() for word in ['personal', 'family', 'life']):
                topics.append('personal')
            elif any(word in message.lower() for word in ['work', 'job', 'career']):
                topics.append('professional')
            elif any(word in message.lower() for word in ['tech', 'ai', 'computer']):
                topics.append('technical')
            else:
                topics.append('general')
        
        return {
            'dominant_topics': Counter(topics).most_common(2),
            'topic_consistency': len(set(topics)) / len(topics) if topics else 0,
            'conversation_length': len(conversation_flow)
        }

class NaturalLanguageMemorySearch:
    """Advanced natural language search for memories"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
    
    def search_memories_natural(self, query: str) -> Dict[str, Any]:
        """Search memories using natural language queries"""
        search_results = {
            'query': query,
            'search_type': self.classify_search_query(query),
            'results': [],
            'suggestions': [],
            'total_found': 0,
            'search_insights': {}
        }
        
        try:
            # Classify and process the query
            search_type = search_results['search_type']
            
            if search_type == 'temporal':
                results = self.search_by_time(query)
            elif search_type == 'emotional':
                results = self.search_by_emotion(query)
            elif search_type == 'factual':
                results = self.search_by_facts(query)
            elif search_type == 'relational':
                results = self.search_by_relationships(query)
            else:
                results = self.search_semantic_enhanced(query)
            
            search_results['results'] = results
            search_results['total_found'] = len(results)
            search_results['suggestions'] = self.generate_search_suggestions(query, results)
            search_results['search_insights'] = self.generate_search_insights(query, results)
            
        except Exception as e:
            print(f"❌ Error in natural language search: {e}")
        
        return search_results
    
    def classify_search_query(self, query: str) -> str:
        """Classify the type of search query"""
        query_lower = query.lower()
        
        # Temporal indicators
        if any(word in query_lower for word in ['when', 'yesterday', 'last week', 'ago', 'recent', 'old', 'before', 'after']):
            return 'temporal'
        
        # Emotional indicators
        if any(word in query_lower for word in ['happy', 'sad', 'excited', 'frustrated', 'emotional', 'feel', 'feeling']):
            return 'emotional'
        
        # Factual indicators
        if any(word in query_lower for word in ['fact', 'information', 'detail', 'specific', 'exactly', 'remember about']):
            return 'factual'
        
        # Relational indicators
        if any(word in query_lower for word in ['related to', 'connected', 'similar', 'like', 'about']):
            return 'relational'
        
        return 'semantic'
    
    def search_by_time(self, query: str) -> List[Dict[str, Any]]:
        """Search memories by temporal criteria"""
        results = []
        query_lower = query.lower()
        
        # Parse temporal expressions
        now = datetime.utcnow()
        target_date = None
        
        if 'yesterday' in query_lower:
            target_date = now - timedelta(days=1)
        elif 'last week' in query_lower:
            target_date = now - timedelta(weeks=1)
        elif 'recent' in query_lower:
            target_date = now - timedelta(days=7)
        
        for node in self.memory_system.memory_nodes.values():
            try:
                memory_date = datetime.fromisoformat(node.metadata.get('timestamp', ''))
                
                if target_date:
                    # Check if memory is within the time range
                    time_diff = abs((memory_date - target_date).days)
                    if time_diff <= 1:  # Within 1 day of target
                        results.append({
                            'memory_id': node.id,
                            'content': node.content,
                            'metadata': node.metadata,
                            'relevance_score': 1.0 - (time_diff / 7),  # Higher score for closer matches
                            'match_reason': f"Memory from {memory_date.strftime('%Y-%m-%d %H:%M')}"
                        })
                
            except:
                continue
        
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)
    
    def search_by_emotion(self, query: str) -> List[Dict[str, Any]]:
        """Search memories by emotional content"""
        results = []
        query_lower = query.lower()
        
        # Extract emotion from query
        target_emotions = []
        emotion_keywords = {
            'joy': ['happy', 'excited', 'glad', 'pleased', 'joyful'],
            'sadness': ['sad', 'disappointed', 'upset', 'down'],
            'anger': ['angry', 'mad', 'frustrated', 'annoyed'],
            'fear': ['scared', 'worried', 'anxious', 'nervous'],
            'surprise': ['surprised', 'shocked', 'amazed']
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                target_emotions.append(emotion)
        
        for node in self.memory_system.memory_nodes.values():
            emotional_match = 0.0
            
            # Check emotional context
            for emotion in target_emotions:
                if emotion in node.emotional_context:
                    emotional_match += node.emotional_context[emotion]
            
            # Check content for emotional words
            content_lower = node.content.lower()
            for emotion in target_emotions:
                for keyword in emotion_keywords[emotion]:
                    if keyword in content_lower:
                        emotional_match += 0.3
            
            if emotional_match > 0:
                results.append({
                    'memory_id': node.id,
                    'content': node.content,
                    'metadata': node.metadata,
                    'relevance_score': min(emotional_match, 1.0),
                    'match_reason': f"Emotional match for: {', '.join(target_emotions)}"
                })
        
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)
    
    def search_semantic_enhanced(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced semantic search with query expansion"""
        results = []
        
        # Use the existing advanced retrieval
        advanced_results = self.memory_system.retrieve_memories_ultra_advanced(query, top_k=15)
        
        for result in advanced_results:
            results.append({
                'memory_id': result['id'],
                'content': result['text'],
                'metadata': result['metadata'],
                'relevance_score': result['composite_score'],
                'match_reason': f"Semantic similarity: {result['semantic_similarity']:.3f}, Strategy: {result['retrieval_strategy']}"
            })
        
        return results
    
    def generate_search_suggestions(self, query: str, results: List[Dict[str, Any]]) -> List[str]:
        """Generate search suggestions based on results"""
        suggestions = []
        
        if not results:
            suggestions.extend([
                "Try searching for specific topics or keywords",
                "Search by time: 'recent memories' or 'yesterday'",
                "Search by emotion: 'happy memories' or 'when I was excited'",
                "Search by type: 'personal information' or 'my preferences'"
            ])
        else:
            # Analyze results to suggest related searches
            content_words = []
            for result in results[:5]:
                words = result['content'].lower().split()
                content_words.extend([word for word in words if len(word) > 4])
            
            word_counts = Counter(content_words)
            common_words = [word for word, count in word_counts.most_common(3) if count > 1]
            
            if common_words:
                suggestions.extend([f"More about '{word}'" for word in common_words])
        
        return suggestions[:5]
    
    def generate_search_insights(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights about the search results"""
        insights = {
            'result_quality': 'high' if results and results[0]['relevance_score'] > 0.7 else 'medium' if results else 'low',
            'time_span': None,
            'dominant_themes': [],
            'search_effectiveness': len(results) / 10 if len(results) <= 10 else 1.0
        }
        
        if results:
            # Analyze time span
            timestamps = []
            for result in results:
                timestamp = result['metadata'].get('timestamp')
                if timestamp:
                    try:
                        timestamps.append(datetime.fromisoformat(timestamp))
                    except:
                        pass
            
            if len(timestamps) > 1:
                time_span = (max(timestamps) - min(timestamps)).days
                insights['time_span'] = f"{time_span} days"
            
            # Analyze themes
            all_content = " ".join([result['content'] for result in results[:5]])
            words = all_content.lower().split()
            word_counts = Counter([word for word in words if len(word) > 4])
            insights['dominant_themes'] = [word for word, count in word_counts.most_common(3)]
        
        return insights

# Global instances
ai_reflection = AIMemoryReflection(memory_system)
memory_viz = MemoryVisualization(memory_system)
predictive_memory = PredictiveMemory(memory_system)
nl_search = NaturalLanguageMemorySearch(memory_system)

# Enhanced API functions
def get_ai_memory_insights() -> Dict[str, Any]:
    """Get AI's insights about its own memory patterns"""
    return ai_reflection.generate_memory_insights()

def create_memory_visualizations() -> Dict[str, str]:
    """Create visual representations of memory data"""
    results = {}
    
    network_path = memory_viz.create_memory_network_graph()
    if network_path:
        results['network_graph'] = network_path
    
    timeline_path = memory_viz.create_memory_timeline()
    if timeline_path:
        results['timeline'] = timeline_path
    
    wordcloud_path = memory_viz.create_memory_wordcloud()
    if wordcloud_path:
        results['wordcloud'] = wordcloud_path
    
    return results

def predict_future_memories(context: str, conversation_flow: List[str] = None) -> List[Dict[str, Any]]:
    """Predict what memories might become relevant"""
    return predictive_memory.predict_relevant_memories(context, conversation_flow)

def search_memories_advanced(query: str) -> Dict[str, Any]:
    """Advanced natural language memory search"""
    return nl_search.search_memories_natural(query)

def generate_memory_report() -> Dict[str, Any]:
    """Generate comprehensive memory system report"""
    return {
        'system_analytics': memory_system.get_memory_analytics(),
        'ai_insights': get_ai_memory_insights(),
        'validation_results': memory_system.validate_memory_consistency(),
        'total_memories': len(memory_system.memory_nodes),
        'total_clusters': len(memory_system.clusters),
        'total_relationships': sum(len(rels) for rels in memory_system.relationship_graph.values()),
        'generated_at': datetime.utcnow().isoformat()
    } 