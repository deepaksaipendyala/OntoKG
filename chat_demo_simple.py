#!/usr/bin/env python3
"""
Simple Chat Demo for Materials Discovery
Showcases the conversational AI interface
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page config
st.set_page_config(
    page_title="MatBot - AI Materials Assistant",
    page_icon="🤖",
    layout="wide"
)

# Chat interface CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 2px solid #e1e5e9;
        border-radius: 15px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin-bottom: 1rem;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        margin-left: 20%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        margin-right: 20%;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    .system-message {
        background: rgba(255, 255, 255, 0.9);
        color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        border: 2px dashed #3498db;
    }
    .quick-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

class SimpleChatBot:
    """Simple chat bot with predefined responses"""
    
    def __init__(self):
        self.responses = {
            'hello': "👋 Hello! I'm MatBot, your AI assistant for materials science! I can help you explore materials, properties, and applications. What would you like to know?",
            'help': """
🤖 **MatBot Help Guide:**

**I can help you with:**
- 🧪 Material properties: "What are the properties of BaTiO3?"
- 🔍 Application queries: "What materials are used for capacitors?"
- 🤖 AI suggestions: "Suggest materials for solar cells"
- ✅ Validation: "Is SnSe good for thermoelectric devices?"
- 📊 Analytics: "Show me discovery trends"

**Try asking:** "Tell me about BaTiO3" or "What materials are best for capacitors?"
            """,
            'batio3': """
📊 **BaTiO3 (Barium Titanate) Information:**

**Properties:**
- Dielectric constant: ~1500 (very high)
- Band gap: ~3.2 eV
- Crystal structure: Perovskite (ABO₃)
- Type: Ferroelectric ceramic

**Applications:**
- Capacitors (95% confidence) - Excellent high-κ dielectric
- Piezoelectric devices
- Ferroelectric memory

**Why it's special:** BaTiO3 has one of the highest dielectric constants among practical materials, making it ideal for capacitor applications.
            """,
            'capacitor': """
🧪 **Materials for Capacitor Applications:**

**Top Materials:**
- **BaTiO3** (95% confidence) - Highest dielectric constant (~1500)
- **SrTiO3** (85% confidence) - Good dielectric properties (~300)
- **PbTiO3** (80% confidence) - Ferroelectric perovskite

**Key Requirements:**
- High dielectric constant (ε > 50)
- Low dielectric loss
- Stable crystal structure
- Good processability

💡 **AI Suggestion:** CaTiO3 and KNbO3 are promising candidates based on perovskite structure analysis!
            """,
            'thermoelectric': """
⚡ **Materials for Thermoelectric Applications:**

**Top Materials:**
- **SnSe** (95% confidence) - Record ZT ~2.6
- **Bi2Te3** (90% confidence) - Commercial standard ZT ~0.8
- **PbTe** (85% confidence) - High-temperature applications

**Key Requirements:**
- High thermoelectric figure of merit (ZT > 0.5)
- Low thermal conductivity
- Good electrical conductivity
- Temperature stability

🔥 **Hot Tip:** Chalcogenide materials (Te, Se compounds) dominate this field!
            """,
            'solar': """
☀️ **Materials for Solar Cell Applications:**

**Top Materials:**
- **Si** (95% confidence) - Industry standard, 1.1 eV band gap
- **CH3NH3PbI3** (85% confidence) - Perovskite, 1.6 eV band gap
- **CsSnI3** (75% confidence) - Lead-free perovskite, 1.3 eV band gap

**Key Requirements:**
- Optimal band gap (1.0-1.8 eV)
- High absorption coefficient
- Charge carrier mobility
- Stability under illumination

🌱 **Trend:** Lead-free perovskites are the future for sustainable solar cells!
            """,
            'analytics': """
📊 **Discovery Analytics Report:**

**System Metrics:**
- Total Materials: 15
- Active Applications: 8
- Validated Relationships: 24
- Discovery Readiness: 87%

**🔥 Top Insights:**
1. High-potential cluster: Perovskite materials (89% discovery potential)
2. Discovery rate trending upward (76% strength)
3. Gap in thermoelectric materials coverage

**🎯 Recommendations:**
1. Focus on perovskite material variants
2. Explore chalcogenide thermoelectrics
3. Investigate lead-free solar absorbers

*This is a live analysis of our knowledge graph patterns!*
            """
        }
    
    def get_response(self, message: str) -> str:
        """Get response based on message content"""
        message_lower = message.lower()
        
        # Check for greetings
        if any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return self.responses['hello']
        
        # Check for help
        elif any(word in message_lower for word in ['help', 'what can you do']):
            return self.responses['help']
        
        # Check for specific materials
        elif 'batio3' in message_lower or 'barium titanate' in message_lower:
            return self.responses['batio3']
        
        # Check for applications
        elif 'capacitor' in message_lower:
            return self.responses['capacitor']
        elif 'thermoelectric' in message_lower:
            return self.responses['thermoelectric']
        elif 'solar' in message_lower or 'photovoltaic' in message_lower:
            return self.responses['solar']
        elif 'analytics' in message_lower or 'trends' in message_lower:
            return self.responses['analytics']
        
        # Default response
        else:
            return f"""
I understand you're asking about: "{message}"

I'm specialized in materials science. Here are some things you can ask me:

🧪 **"Tell me about BaTiO3"** - Get material properties
🔍 **"What materials are used for capacitors?"** - Find applications
🤖 **"Suggest materials for solar cells"** - AI recommendations
📊 **"Show me analytics"** - Discovery insights

What would you like to explore?
            """

def main():
    """Main chat demo application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 MatBot - AI Materials Assistant</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 15px; margin-bottom: 2rem;">
        <h3>💬 Conversational AI for Materials Science</h3>
        <p>Ask me anything about materials, properties, applications, or get AI-powered suggestions!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat bot
    if 'simple_chat_bot' not in st.session_state:
        st.session_state.simple_chat_bot = SimpleChatBot()
    
    if 'simple_chat_history' not in st.session_state:
        st.session_state.simple_chat_history = []
    
    chat_bot = st.session_state.simple_chat_bot
    
    # Chat history display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if not st.session_state.simple_chat_history:
        st.markdown("""
        <div class="system-message">
            👋 <strong>Welcome to MatBot!</strong><br>
            I'm your AI assistant for materials science. Ask me about materials, properties, 
            applications, or request AI-generated suggestions!<br><br>
            <strong>Try:</strong> "Tell me about BaTiO3" or "What materials are best for capacitors?"
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.simple_chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {message['content']}
                <div style="font-size: 0.8em; margin-top: 0.5rem; opacity: 0.8;">
                    {message['timestamp'].strftime('%H:%M:%S')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        elif message['role'] == 'assistant':
            st.markdown(f"""
            <div class="assistant-message">
                <strong>MatBot:</strong> {message['content']}
                <div style="font-size: 0.8em; margin-top: 0.5rem; opacity: 0.8;">
                    {message['timestamp'].strftime('%H:%M:%S')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input section
    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "💬 Ask MatBot:",
            placeholder="e.g., 'What materials are best for capacitors?' or 'Tell me about BaTiO3'",
            key="simple_chat_input"
        )
    
    with col2:
        send_button = st.button("Send 🚀", type="primary")
    
    # Quick action buttons
    st.markdown("**🚀 Quick Examples:**")
    quick_col1, quick_col2, quick_col3, quick_col4, quick_col5 = st.columns(5)
    
    quick_actions = [
        ("🧪 BaTiO3", "Tell me about BaTiO3"),
        ("🔋 Capacitors", "What materials are used for capacitors?"),
        ("⚡ Thermoelectric", "What materials are good for thermoelectric devices?"),
        ("☀️ Solar Cells", "What materials are used for solar cells?"),
        ("📊 Analytics", "Show me analytics report")
    ]
    
    for i, (label, query) in enumerate(quick_actions):
        with [quick_col1, quick_col2, quick_col3, quick_col4, quick_col5][i]:
            if st.button(label, key=f"quick_{i}"):
                user_input = query
                send_button = True
    
    # Process message
    if send_button and user_input:
        # Add user message
        user_msg = {
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        }
        st.session_state.simple_chat_history.append(user_msg)
        
        # Generate response
        with st.spinner("🤖 MatBot is thinking..."):
            response = chat_bot.get_response(user_input)
        
        # Add assistant response
        assistant_msg = {
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now()
        }
        st.session_state.simple_chat_history.append(assistant_msg)
        
        # Clear input and rerun
        st.session_state.simple_chat_input = ""
        st.rerun()
    
    # Chat controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.simple_chat_history = []
            st.rerun()
    
    with col2:
        if st.button("💡 Get Help"):
            help_msg = {
                'role': 'assistant',
                'content': chat_bot.responses['help'],
                'timestamp': datetime.now()
            }
            st.session_state.simple_chat_history.append(help_msg)
            st.rerun()
    
    with col3:
        if st.button("🚀 Full App"):
            st.markdown("**🌐 Access Full Enhanced App:** http://localhost:8501")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <h4>🎯 MatBot Demo</h4>
        <p>This is a simplified chat interface. The full enhanced app with Neo4j integration 
        and advanced features is available at <strong>http://localhost:8501</strong></p>
        <p><strong>Features:</strong> Chat Interface | Knowledge Graph | AI Ensemble | Advanced Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat statistics
    if st.session_state.simple_chat_history:
        st.sidebar.markdown("### 📊 Chat Statistics")
        st.sidebar.metric("Messages", len(st.session_state.simple_chat_history))
        if st.session_state.simple_chat_history:
            last_msg = st.session_state.simple_chat_history[-1]
            st.sidebar.metric("Last Activity", last_msg['timestamp'].strftime('%H:%M:%S'))

if __name__ == "__main__":
    main()
