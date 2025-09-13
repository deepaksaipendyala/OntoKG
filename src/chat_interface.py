"""
Conversational Chat Interface for Materials Ontology Expansion
Provides natural language interaction with the knowledge graph and AI systems
"""

import streamlit as st
import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

# Import our enhanced modules
from advanced_llm_integration import EnhancedHypothesisGenerator
from enhanced_validation import EnhancedMaterialsValidator
from discovery_analytics import MaterialsDiscoveryEngine
from knowledge_graph import MaterialsKG, Hypothesis
from config import load_config

class ChatMessage:
    """Represents a chat message"""
    def __init__(self, role: str, content: str, metadata: Optional[Dict] = None):
        self.role = role  # 'user', 'assistant', 'system'
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

class MaterialsChatBot:
    """Intelligent chat bot for materials science queries"""
    
    def __init__(self, kg, llm, validator, analytics):
        self.kg = kg
        self.llm = llm
        self.validator = validator
        self.analytics = analytics
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # System prompt for materials science context
        self.system_prompt = """
        You are MatBot, an expert AI assistant specializing in materials science and discovery.
        You have access to a comprehensive materials knowledge graph and advanced AI tools.
        
        Your capabilities include:
        - Answering questions about materials, properties, and applications
        - Suggesting new materials for specific applications
        - Validating material-application relationships
        - Providing insights from materials databases
        - Running discovery analytics and pattern recognition
        
        Always provide scientifically accurate information and cite your sources when possible.
        If you're uncertain, acknowledge the limitations and suggest further research.
        """
        
        # Intent classification patterns
        self.intent_patterns = {
            'query_material': [
                r'what.*properties.*of\s+(\w+)',
                r'tell me about\s+(\w+)',
                r'properties.*of\s+(\w+)',
                r'(\w+)\s+properties'
            ],
            'query_application': [
                r'materials.*for\s+(\w+)',
                r'what.*materials.*used.*in\s+(\w+)',
                r'(\w+)\s+materials',
                r'materials.*(\w+)\s+application'
            ],
            'suggest_materials': [
                r'suggest.*materials.*for\s+(\w+)',
                r'recommend.*materials.*(\w+)',
                r'find.*materials.*(\w+)',
                r'best.*materials.*for\s+(\w+)'
            ],
            'validate_hypothesis': [
                r'validate.*(\w+).*for\s+(\w+)',
                r'check.*if\s+(\w+).*good.*for\s+(\w+)',
                r'is\s+(\w+).*suitable.*for\s+(\w+)',
                r'can\s+(\w+).*be.*used.*for\s+(\w+)'
            ],
            'discovery_analytics': [
                r'analyze.*trends',
                r'discovery.*insights',
                r'patterns.*in.*materials',
                r'analytics.*report'
            ]
        }

    def _normalize_kg_data_for_analytics(self, data: Any) -> Dict[str, Any]:
        """Normalize KG data to {'nodes': {Label: [..]}, 'edges': [...]} for analytics."""
        normalized = {'nodes': {}, 'edges': []}
        try:
            if isinstance(data, dict):
                nodes_obj = data.get('nodes', {})
                if isinstance(nodes_obj, dict):
                    # Already keyed by label
                    for t, lst in nodes_obj.items():
                        normalized['nodes'][t] = list(lst) if isinstance(lst, list) else []
                elif isinstance(nodes_obj, list):
                    for n in nodes_obj:
                        if isinstance(n, dict):
                            t = n.get('type', 'Unknown')
                            normalized['nodes'].setdefault(t, []).append(n)
                edges_obj = data.get('edges', [])
                if isinstance(edges_obj, list):
                    normalized['edges'] = edges_obj
            elif isinstance(data, list):
                # Merge chunks
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    nobj = item.get('nodes')
                    if isinstance(nobj, dict):
                        for t, lst in nobj.items():
                            normalized['nodes'].setdefault(t, [])
                            if isinstance(lst, list):
                                normalized['nodes'][t].extend(lst)
                    elif isinstance(nobj, list):
                        for n in nobj:
                            if isinstance(n, dict):
                                t = n.get('type', 'Unknown')
                                normalized['nodes'].setdefault(t, []).append(n)
                    eobj = item.get('edges')
                    if isinstance(eobj, list):
                        normalized['edges'].extend(eobj)
        except Exception:
            pass
        return normalized

    def _normalize_application_name(self, text: str) -> str:
        """Normalize user-provided application names to canonical KG slugs."""
        t = (text or '').strip().lower()
        t = t.replace('-', ' ').replace('/', ' ').replace('\t', ' ')
        # common plural to singular
        t = t.replace('cells', 'cell').replace('devices', 'device').replace('capacitors', 'capacitor')
        # collapse spaces
        t = ' '.join(t.split())
        # synonyms/canonicalization
        synonyms = {
            'thermoelectric': 'thermoelectric_device',
            'thermoelectric device': 'thermoelectric_device',
            'thermoelectric_device': 'thermoelectric_device',
            'solar cell': 'solar_cell',
            'solar_cell': 'solar_cell',
            'capacitor': 'capacitor'
        }
        if t in synonyms:
            return synonyms[t]
        # default: make underscore slug
        return t.replace(' ', '_')
    
    def classify_intent(self, message: str) -> tuple:
        """Classify user intent and extract entities"""
        message_lower = message.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, message_lower)
                if match:
                    entities = match.groups()
                    return intent, entities
        
        return 'general_query', ()

    def _dynamic_vocab(self) -> Dict[str, set]:
        try:
            kg_data = self.kg.export_graph_data()
        except Exception:
            kg_data = {"nodes": {}, "edges": []}
        materials, applications = set(), set()
        # Normalize kg_data into nodes_dict and edges_list robustly
        nodes_dict: Dict[str, List[Dict[str, Any]]] = {}
        edges_list: List[Dict[str, Any]] = []
        if isinstance(kg_data, dict):
            nodes_obj = kg_data.get('nodes', {})
            if isinstance(nodes_obj, dict):
                nodes_dict = nodes_obj
            elif isinstance(nodes_obj, list):
                # Legacy shape: list of nodes with 'type' field
                for n in nodes_obj:
                    if isinstance(n, dict):
                        t = n.get('type') or 'Unknown'
                        nodes_dict.setdefault(t, []).append(n)
            edges_obj = kg_data.get('edges', [])
            if isinstance(edges_obj, list):
                edges_list = edges_obj
        elif isinstance(kg_data, list):
            # Composite list of chunks
            for item in kg_data:
                if not isinstance(item, dict):
                    continue
                nobj = item.get('nodes')
                if isinstance(nobj, dict):
                    for t, lst in nobj.items():
                        nodes_dict.setdefault(t, [])
                        if isinstance(lst, list):
                            nodes_dict[t].extend(lst)
                elif isinstance(nobj, list):
                    for n in nobj:
                        if isinstance(n, dict):
                            t = n.get('type') or 'Unknown'
                            nodes_dict.setdefault(t, []).append(n)
                eobj = item.get('edges')
                if isinstance(eobj, list):
                    edges_list.extend(eobj)

        for n in nodes_dict.get('Material', []) or []:
            if isinstance(n, dict) and n.get('name'):
                materials.add(n['name'])
        for n in nodes_dict.get('Application', []) or []:
            if isinstance(n, dict) and n.get('name'):
                applications.add(n['name'])

        for e in edges_list:
            if isinstance(e, dict) and str(e.get('relationship', '')).upper() == 'USED_IN':
                s = e.get('source') or e.get('from')
                t = e.get('target') or e.get('to')
                if s:
                    materials.add(s)
                if t:
                    applications.add(t)
        return {"materials": materials, "applications": applications}

    def _find_entity(self, text: str, candidates: set) -> Optional[str]:
        t = (text or '').lower()
        best = ''
        for name in candidates:
            n = str(name).lower()
            if n and n in t and len(n) > len(best):
                best = name
        return best or None
    
    def process_message(self, user_message: str) -> str:
        """Process user message and generate response"""
        try:
            # Dynamic routing using KG entities (application/material) without rigid keywords
            vocab = self._dynamic_vocab()
            found_mat = self._find_entity(user_message, vocab['materials'])
            found_app = self._find_entity(user_message, vocab['applications'])
            msg_l = user_message.lower()
            if found_mat and found_app:
                return self._handle_validation(found_mat, found_app)
            if found_app:
                if any(k in msg_l for k in ['suggest', 'recommend', 'propose']):
                    return self._handle_material_suggestion(found_app)
                return self._handle_application_query(found_app)
            if found_mat:
                return self._handle_material_query(found_mat)
            if any(k in msg_l for k in ['analytics', 'trends', 'insights']):
                return self._handle_analytics_query()

            # Fallback to existing pattern classifier
            intent, entities = self.classify_intent(user_message)
            if intent == 'query_material':
                return self._handle_material_query(entities[0] if entities else user_message)
            
            elif intent == 'query_application':
                return self._handle_application_query(entities[0] if entities else user_message)
            
            elif intent == 'suggest_materials':
                return self._handle_material_suggestion(entities[0] if entities else user_message)
            
            elif intent == 'validate_hypothesis':
                if len(entities) >= 2:
                    return self._handle_validation(entities[0], entities[1])
                else:
                    return "I need both a material and application to validate. Please specify both."
            
            elif intent == 'discovery_analytics':
                return self._handle_analytics_query()
            
            else:
                return self._handle_general_query(user_message)
 
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question."
    
    def _handle_material_query(self, material: str) -> str:
        """Handle queries about specific materials"""
        material = material.strip()
        
        try:
            # Get material properties from knowledge graph
            with self.kg.driver.session() as session:
                result = session.run("""
                    MATCH (m:Material)
                    WHERE toLower(m.name) = toLower($material)
                    MATCH (m)-[r:HAS_PROPERTY]->(p:Property)
                    RETURN p.name as property, r.value as value
                """, material=material)
                
                properties = []
                for record in result:
                    properties.append(f"- {record['property']}: {record['value']}")
                
                # Get applications
                app_result = session.run("""
                    MATCH (m:Material)
                    WHERE toLower(m.name) = toLower($material)
                    MATCH (m)-[r:USED_IN]->(a:Application)
                    RETURN a.name as application, r.confidence as confidence
                    ORDER BY coalesce(r.confidence, 0) DESC
                """, material=material)
                
                applications = []
                for record in app_result:
                    conf_pct = record['confidence'] * 100
                    applications.append(f"- {record['application']} ({conf_pct:.0f}% confidence)")
            
            if properties or applications:
                response = f"📊 **{material} Information:**\n\n"
                
                if properties:
                    response += "**Properties:**\n" + "\n".join(properties) + "\n\n"
                
                if applications:
                    response += "**Applications:**\n" + "\n".join(applications)
                
                return response
            else:
                return f"I don't have specific information about {material} in the knowledge graph yet. Try loading seed data from the sidebar or ask for analytics/suggestions."
        
        except Exception as e:
            return f"I had trouble accessing information about {material}. Error: {str(e)}"
    
    def _handle_application_query(self, application: str) -> str:
        """Handle queries about materials for specific applications"""
        application = self._normalize_application_name(application)
        
        try:
            # Use robust KG helper that includes fallbacks and case-insensitive matching
            results = self.kg.get_materials_for_application(application)
            materials = []
            for rec in results:
                try:
                    conf = float(rec.get('confidence', rec.get('r.confidence', 0)))
                except Exception:
                    conf = 0.0
                conf_pct = conf * 100
                name = rec.get('material') or rec.get('material_name') or ''
                if name:
                    materials.append(f"- **{name}** ({conf_pct:.0f}% confidence)")
        
            app_display = application.replace('_', ' ').title()
            if materials:
                response = f"🧪 **Materials for {app_display} Applications:**\n\n"
                response += "\n".join(materials)
                response += f"\n\nWould you like me to suggest additional materials or validate a specific material for {app_display}?"
                return response
            else:
                return f"I don't have materials listed for {app_display} yet. Try loading seed data from the sidebar, or ask for AI-generated suggestions."
        
        except Exception as e:
            return f"I had trouble finding materials for {application}. Error: {str(e)}"
    
    def _handle_material_suggestion(self, application: str) -> str:
        """Handle material suggestion requests"""
        application = self._normalize_application_name(application)
        
        try:
            # Get current materials for context using robust helper
            current = self.kg.get_materials_for_application(application)
            current_materials = [m.get('material') for m in current if m.get('material')] or ["BaTiO3", "SrTiO3"]
            
            context = {"known_materials": current_materials}
            
            if application == 'capacitor':
                response_obj = self.llm.generate_capacitor_hypotheses(current_materials, context)
            elif application == 'thermoelectric_device':
                response_obj = self.llm.generate_thermoelectric_hypotheses(current_materials, context)
            elif application == 'solar_cell':
                response_obj = self.llm.generate_solar_hypotheses(current_materials, context)
            else:
                response_obj = self.llm.generate_capacitor_hypotheses(current_materials, context)
            
            if getattr(response_obj, 'hypotheses', None):
                response = f"**AI-Generated Material Suggestions for {application.replace('_',' ').title()}:**\n\n"
                response += f"Based on {len(current_materials)} known materials.\n\n"
                
                for i, hyp in enumerate(response_obj.hypotheses[:3], 1):
                    material = hyp.get('material', 'Unknown')
                    try:
                        confidence = float(hyp.get('confidence', 0)) * 100
                    except Exception:
                        confidence = 0.0
                    rationale = hyp.get('rationale', 'No rationale provided')
                    response += f"{i}. {material} ({confidence:.0f}% confidence)\n"
                    response += f"   Rationale: {rationale[:140]}...\n\n"
                
                response += "Would you like me to validate any of these suggestions?"
                return response
            else:
                return f"I wasn't able to generate material suggestions for {application.replace('_',' ').title()} right now. Please ensure the model is running or load seed data and try again."
        
        except Exception as e:
            return f"I encountered an error while generating suggestions: {str(e)}"
    
    def _handle_validation(self, material: str, application: str) -> str:
        """Handle material-application validation requests"""
        material = material.strip().title()
        application = application.strip().lower()
        
        try:
            # Prepare material properties for validation
            material_properties = {
                'n_atoms': len(material.replace('(', '').replace(')', '')) // 3,
                'volume': 50.0,
                'electronegativity_diff': 1.5,
            }
            
            # Run basic validation
            validation_result = self.validator.validate_hypothesis(material, application)
            
            response = f"✅ **Validation Results: {material} → {application}**\n\n"
            
            # Overall assessment
            status_emoji = "✅" if validation_result.is_valid else "❌"
            status_text = "VALIDATED" if validation_result.is_valid else "NOT VALIDATED"
            
            response += f"**Status:** {status_emoji} {status_text}\n"
            response += f"**Confidence:** {validation_result.confidence:.1%}\n"
            response += f"**Source:** {validation_result.source}\n\n"
            
            # Evidence
            if validation_result.evidence:
                response += "**Evidence:**\n"
                for key, value in validation_result.evidence.items():
                    response += f"- {key}: {value}\n"
                response += "\n"
            
            if validation_result.is_valid:
                response += "💡 This material shows promise for the specified application!"
            else:
                response += "💡 This combination may not be optimal. Consider alternative materials or applications."
            
            return response
        
        except Exception as e:
            return f"I encountered an error during validation: {str(e)}"
    
    def _handle_analytics_query(self) -> str:
        """Handle discovery analytics queries"""
        try:
            # Get knowledge graph data
            kg_data_raw = self.kg.export_graph_data()
            kg_data = self._normalize_kg_data_for_analytics(kg_data_raw)
            
            # Generate sample discovery history
            discovery_history = []
            for i in range(10):
                discovery_history.append({
                    'timestamp': (datetime.now()).isoformat(),
                    'material': f'Material_{i}',
                    'application': ['capacitor', 'thermoelectric_device', 'solar_cell'][i % 3],
                    'confidence': 0.6 + (i % 5) * 0.08,
                    'validated': i % 3 == 0
                })
            
            # Run analytics
            analysis = self.analytics.analyze_discovery_landscape(kg_data, discovery_history)
            
            response = "📊 **Discovery Analytics Report:**\n\n"
            
            # Metrics
            if 'metrics' in analysis:
                metrics = analysis['metrics']
                response += "**System Metrics:**\n"
                response += f"- Total Materials: {metrics.get('total_materials', 0)}\n"
                response += f"- Active Applications: {metrics.get('total_applications', 0)}\n"
                response += f"- Validated Relationships: {metrics.get('total_relationships', 0)}\n"
                response += f"- Discovery Readiness: {metrics.get('discovery_readiness_score', 0):.1%}\n\n"
            
            # Top insights
            insights = analysis.get('insights', [])
            if insights:
                response += "**🔥 Top Discovery Insights:**\n"
                for i, insight in enumerate(insights[:3], 1):
                    impact_emoji = "🔥" if insight['impact_score'] > 0.8 else "⚡" if insight['impact_score'] > 0.6 else "💡"
                    response += f"{i}. {impact_emoji} {insight['title']}\n"
                    response += f"   *Impact: {insight['impact_score']:.1%} | Type: {insight['insight_type'].replace('_', ' ').title()}*\n\n"
            
            # Recommendations
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                response += "**🎯 Recommendations:**\n"
                for i, rec in enumerate(recommendations[:3], 1):
                    response += f"{i}. {rec}\n"
                response += "\n"
            
            response += "💡 Would you like me to dive deeper into any of these insights?"
            
            return response
        
        except Exception as e:
            return f"I encountered an error during analytics: {str(e)}"
    
    def _handle_general_query(self, message: str) -> str:
        """Handle general queries about materials science"""
        
        message_lower = message.lower()
        
        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return (
                "Hello! I'm MatBot, your AI assistant for materials science. "
                "Ask about materials, properties, applications, AI suggestions, or analytics."
            )
        
        elif any(help_word in message_lower for help_word in ['help', 'what can you do', 'capabilities']):
            return (
                "I can query the knowledge graph, suggest materials for applications, validate hypotheses, "
                "and show analytics (clusters, trends). Try: 'Materials for capacitors' or 'Analyze discovery patterns'."
            )
        
        # If the message mentions analytics explicitly, route to analytics
        if 'analytics' in message_lower or 'discovery analytics' in message_lower:
            return self._handle_analytics_query()

        # Try a best-effort application query fallback if the message mentions 'for <app>'
        m = re.search(r"for\s+([a-zA-Z0-9_\- ]+)$", message_lower)
        if m:
            app = m.group(1).strip()
            return self._handle_application_query(app)
        
        return (
            f"I understood: '{message}'. Please ask about a material, property, or application. "
            "Examples: 'Tell me about BaTiO3', 'What materials are used in solar_cell?', "
            "or 'Suggest materials for thermoelectric_device'."
        )

def render_chat_interface(kg, llm, validator, analytics):
    """Render the chat interface in Streamlit"""
    
    st.markdown("### 💬 MatBot - Your AI Materials Science Assistant")
    
    # Initialize chat bot
    if 'chat_bot' not in st.session_state:
        st.session_state.chat_bot = MaterialsChatBot(kg, llm, validator, analytics)
    
    chat_bot = st.session_state.chat_bot
    
    # Chat interface styling
    st.markdown("""
    <style>
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-bottom: 1rem;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
        text-align: right;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
    }
    .system-message {
        background: #e8f4f8;
        color: #2c3e50;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-style: italic;
        text-align: center;
        border-left: 4px solid #3498db;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="system-message">
                👋 Welcome to MatBot! I'm your AI assistant for materials science.<br>
                Ask me about materials, properties, applications, or request AI-generated suggestions!
            </div>
            """, unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message.role == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message.content}
                </div>
                """, unsafe_allow_html=True)
            
            elif message.role == 'assistant':
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>MatBot:</strong> {message.content}
                </div>
                """, unsafe_allow_html=True)
    
    # Input section
    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask MatBot anything about materials science:",
            placeholder="e.g., 'What materials are best for capacitors?' or 'Suggest materials for solar cells'",
            key="chat_input"
        )
    
    with col2:
        send_button = st.button("Send 🚀", type="primary")
    
    # Quick action buttons
    st.markdown("**Quick Actions:**")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("🧪 Material Info"):
            user_input = "Tell me about BaTiO3"
            send_button = True
    
    with quick_col2:
        if st.button("🔍 Find Materials"):
            user_input = "What materials are used for capacitors?"
            send_button = True
    
    with quick_col3:
        if st.button("🤖 AI Suggestions"):
            user_input = "Suggest materials for thermoelectric devices"
            send_button = True
    
    with quick_col4:
        if st.button("📊 Analytics"):
            user_input = "Show me discovery analytics"
            send_button = True
    
    # Process message
    if (send_button and user_input) or (user_input and st.session_state.get('auto_send', False)):
        # Add user message to history
        user_msg = ChatMessage('user', user_input)
        st.session_state.chat_history.append(user_msg)
        
        # Show thinking indicator
        with st.spinner("🤖 MatBot is thinking..."):
            try:
                # Process message synchronously
                response = chat_bot.process_message(user_input)
                
                # Add assistant response to history
                assistant_msg = ChatMessage('assistant', response)
                st.session_state.chat_history.append(assistant_msg)
                
            except Exception as e:
                error_response = f"I apologize, but I encountered an error: {str(e)}. Please try again."
                assistant_msg = ChatMessage('assistant', error_response)
                st.session_state.chat_history.append(assistant_msg)
        
        # Rerun to show new messages (don't modify session state directly)
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Chat statistics
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown(f"**Chat Stats:** {len(st.session_state.chat_history)} messages | "
                   f"Last activity: {st.session_state.chat_history[-1].timestamp.strftime('%H:%M:%S')}")

# Example usage functions for testing
def get_sample_chat_responses():
    """Get sample responses for testing"""
    return {
        "material_query": "BaTiO3 is a perovskite ceramic with excellent dielectric properties (ε ≈ 1500) and is widely used in capacitor applications.",
        "application_query": "Materials commonly used for capacitors include BaTiO3, SrTiO3, and other high-κ dielectrics.",
        "suggestion": "For thermoelectric applications, I recommend considering SnSe (ZT ≈ 2.6), Bi2Te3, and other chalcogenide materials.",
        "validation": "CaTiO3 shows good potential for capacitor applications with moderate dielectric constant (ε ≈ 170) and stable perovskite structure."
    }
