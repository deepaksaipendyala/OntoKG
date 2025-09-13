"""
Streamlit Web Application for Materials Ontology Expansion
Interactive interface for demonstrating LLM-guided knowledge graph expansion
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import os
from typing import Dict, List, Any
import networkx as nx
import pyvis
from io import StringIO

# Import our modules
from knowledge_graph import MaterialsKG, Hypothesis
from llm_integration import OllamaHypothesisGenerator
from validation import MaterialsValidator
from config import load_config

# Page configuration
st.set_page_config(
    page_title="Materials Ontology Expansion",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_components():
    """Initialize knowledge graph and other components"""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize components with configuration
        kg = MaterialsKG(config)
        llm = OllamaHypothesisGenerator(config.ollama_url, config.ollama_model)
        validator = MaterialsValidator(config)
        
        return kg, llm, validator, config
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return None, None, None, None

def display_knowledge_graph_stats(kg: MaterialsKG):
    """Display knowledge graph statistics"""
    try:
        stats = kg.get_graph_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Materials", stats.get("materials", 0))
        with col2:
            st.metric("Properties", stats.get("properties", 0))
        with col3:
            st.metric("Applications", stats.get("applications", 0))
        with col4:
            st.metric("Relationships", stats.get("relationships", 0))
            
    except Exception as e:
        st.error(f"Error getting stats: {e}")

def display_materials_for_application(kg: MaterialsKG, application: str):
    """Display materials used in a specific application"""
    try:
        materials = kg.get_materials_for_application(application)
        
        if not materials:
            st.warning(f"No materials found for application: {application}")
            return
        
        # Create DataFrame
        df = pd.DataFrame(materials)
        
        # Display table
        st.subheader(f"Materials used in {application}")
        st.dataframe(df, use_container_width=True)
        
        # Create confidence distribution chart
        if len(df) > 1:
            fig = px.bar(df, x='material', y='r.confidence', 
                        title=f'Confidence Scores for {application} Materials',
                        color='r.confidence',
                        color_continuous_scale='viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying materials: {e}")

def display_material_properties(kg: MaterialsKG, material: str):
    """Display properties of a specific material"""
    try:
        properties = kg.get_properties_for_material(material)
        
        if not properties:
            st.warning(f"No properties found for material: {material}")
            return
        
        # Create DataFrame
        df = pd.DataFrame(properties)
        
        # Display table
        st.subheader(f"Properties of {material}")
        st.dataframe(df, use_container_width=True)
        
        # Create property visualization
        if len(df) > 1:
            fig = px.scatter(df, x='property', y='value', 
                           title=f'Properties of {material}',
                           size='confidence',
                           hover_data=['unit', 'source'])
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying properties: {e}")

def run_hypothesis_expansion(kg: MaterialsKG, llm: OllamaHypothesisGenerator, 
                           validator: MaterialsValidator, application: str):
    """Run the hypothesis expansion process"""
    
    st.subheader(f"🤖 LLM-Guided Expansion for {application}")
    
    # Get current materials for the application
    current_materials = kg.get_materials_for_application(application)
    material_names = [m['material'] for m in current_materials]
    
    if not material_names:
        st.warning(f"No current materials found for {application}. Please initialize the knowledge graph first.")
        return
    
    # Display current state
    st.info(f"Current {application} materials: {', '.join(material_names)}")
    
    # Test LLM connection
    if not llm.test_connection():
        st.error("❌ Cannot connect to Ollama. Please ensure Ollama is running.")
        return
    
    st.success("✅ Connected to Ollama")
    
    # Get context for LLM
    context = {}
    for material in material_names:
        neighbors = kg.get_neighbors(material)
        for entity_type, names in neighbors.items():
            if entity_type not in context:
                context[entity_type] = []
            context[entity_type].extend(names)
    
    # Generate hypotheses
    with st.spinner("🤖 Generating hypotheses with LLM..."):
        try:
            response = llm.generate_hypotheses(application, context, material_names)
            
            if not response.hypotheses:
                st.warning("No hypotheses generated by LLM")
                return
            
            st.success(f"✅ Generated {len(response.hypotheses)} hypotheses")
            
            # Display LLM reasoning
            with st.expander("🧠 LLM Reasoning"):
                st.write(response.reasoning)
            
            # Validate hypotheses
            st.subheader("🔍 Validating Hypotheses")
            
            validated_hypotheses = []
            
            for i, hypothesis_data in enumerate(response.hypotheses):
                material = hypothesis_data.get('material', '')
                rationale = hypothesis_data.get('rationale', '')
                confidence = hypothesis_data.get('confidence', 0.5)
                
                if not material:
                    continue
                
                # Validate hypothesis
                validation_result = validator.validate_hypothesis(material, application)
                
                # Create hypothesis object
                hypothesis = Hypothesis(
                    material=material,
                    application=application,
                    relationship="USED_IN",
                    confidence=validation_result.confidence * confidence,
                    source="LLM",
                    validated_by=validation_result.source,
                    rationale=rationale
                )
                
                # Display validation result
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{material}** → {application}")
                    st.write(f"*Rationale:* {rationale}")
                
                with col2:
                    if validation_result.is_valid:
                        st.success(f"✅ Valid\nConfidence: {validation_result.confidence:.2f}")
                    else:
                        st.error(f"❌ Invalid\nConfidence: {validation_result.confidence:.2f}")
                
                with col3:
                    if validation_result.is_valid and validation_result.confidence >= 0.6:
                        if st.button(f"Add to KG", key=f"add_{i}"):
                            kg.add_hypothesis(hypothesis)
                            st.success(f"✅ Added {material} to knowledge graph!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.info("Not added")
                
                if validation_result.is_valid and validation_result.confidence >= 0.6:
                    validated_hypotheses.append(hypothesis)
                
                st.divider()
            
            # Summary
            if validated_hypotheses:
                st.success(f"🎉 Successfully validated {len(validated_hypotheses)} hypotheses!")
            else:
                st.warning("No hypotheses met validation criteria")
                
        except Exception as e:
            st.error(f"Error during hypothesis generation: {e}")

def create_network_visualization(kg: MaterialsKG):
    """Create a network visualization of the knowledge graph"""
    try:
        graph_data = kg.export_graph_data()
        
        if not graph_data['nodes'] or not graph_data['edges']:
            st.warning("No data available for visualization")
            return
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for node in graph_data['nodes']:
            G.add_node(node['name'], type=node['type'])
        
        # Add edges
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], 
                      relationship=edge['relationship'],
                      confidence=edge.get('confidence', 1.0))
        
        # Create visualization with pyvis - WHITE BACKGROUND
        net = pyvis.network.Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # Configure physics and interaction for better visualization
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100},
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 0.5
                }
            },
            "interaction": {
                "hover": true,
                "hoverConnectedEdges": true,
                "selectConnectedEdges": false,
                "dragNodes": true,
                "dragView": true,
                "zoomView": true
            },
            "nodes": {
                "font": {"size": 16, "color": "black", "strokeWidth": 2, "strokeColor": "white"},
                "borderWidth": 3,
                "shadow": {"enabled": true, "color": "rgba(0,0,0,0.3)", "size": 5, "x": 2, "y": 2}
            },
            "edges": {
                "font": {"size": 14, "color": "black", "strokeWidth": 2, "strokeColor": "white"},
                "smooth": {"type": "continuous"},
                "shadow": {"enabled": true, "color": "rgba(0,0,0,0.2)", "size": 3}
            }
        }
        """)
        
        # Add nodes with colors by type - CLEARER COLORS
        colors = {
            'Material': '#FF4444',      # Bright Red for Materials
            'Property': '#44AA44',      # Bright Green for Properties  
            'Application': '#4444FF',    # Bright Blue for Applications
            'Unknown': '#888888'         # Gray for Unknown
        }
        
        for node in G.nodes(data=True):
            node_type = node[1].get('type', 'Unknown')
            color = colors.get(node_type, '#888888')
            net.add_node(node[0], 
                        label=node[0], 
                        color=color, 
                        title=f"<b>Type:</b> {node_type}<br><b>Name:</b> {node[0]}",
                        font={"size": 16, "color": "black", "strokeWidth": 2, "strokeColor": "white"})
        
        # Add edges with RELATIONSHIP LABELS
        for edge in G.edges(data=True):
            confidence = edge[2].get('confidence', 1.0)
            relationship = edge[2].get('relationship', 'RELATED_TO')
            
            # Color edges by relationship type
            edge_colors = {
                'USED_IN': '#FF6B6B',           # Red for USED_IN
                'HAS_PROPERTY': '#4ECDC4',      # Teal for HAS_PROPERTY
                'RELATED_TO': '#95A5A6'         # Gray for other relationships
            }
            edge_color = edge_colors.get(relationship, '#95A5A6')
            
            net.add_edge(edge[0], edge[1], 
                        label=relationship,  # SHOW RELATIONSHIP NAME ON EDGE
                        color=edge_color,
                        title=f"<b>Relationship:</b> {relationship}<br><b>Confidence:</b> {confidence:.2f}<br><b>From:</b> {edge[0]}<br><b>To:</b> {edge[1]}",
                        width=confidence * 3 + 2,
                        font={"size": 12, "color": "black", "strokeWidth": 1, "strokeColor": "white"})
        
        # Save and display
        net.save_graph("temp_graph.html")
        
        with open("temp_graph.html", "r") as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=600)
        
        # Add legend
        st.markdown("""
        ### 🎨 **Visualization Legend**
        - **🔴 Red Nodes**: Materials (e.g., BaTiO3, Si, CdTe)
        - **🟢 Green Nodes**: Properties (e.g., band_gap, dielectric_constant)  
        - **🔵 Blue Nodes**: Applications (e.g., solar_cell, capacitor)
        - **Edge Labels**: Show relationship types (USED_IN, HAS_PROPERTY)
        - **Edge Colors**: Red = USED_IN, Teal = HAS_PROPERTY, Gray = Other
        """)
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧬 Materials Ontology Expansion</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>About:</strong> This system uses Large Language Models to expand materials science knowledge graphs 
    by generating hypotheses about material-application relationships and validating them against known databases.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    kg, llm, validator, config = initialize_components()
    
    if kg is None:
        st.error("Failed to initialize. Please check your Neo4j connection.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Display current stats
    st.sidebar.subheader("📊 Knowledge Graph Stats")
    try:
        stats = kg.get_graph_stats()
        st.sidebar.metric("Materials", stats.get("materials", 0))
        st.sidebar.metric("Applications", stats.get("applications", 0))
        st.sidebar.metric("Relationships", stats.get("relationships", 0))
        
        # Show data source information
        if kg.data_manager:
            st.sidebar.subheader("📁 Data Sources")
            data_stats = kg.data_manager.get_data_source_statistics()
            for source_name, source_stats in data_stats.items():
                if isinstance(source_stats, dict) and 'error' not in source_stats:
                    st.sidebar.write(f"**{source_name}**: ✅")
                else:
                    st.sidebar.write(f"**{source_name}**: ❌")
    except:
        st.sidebar.error("Cannot connect to knowledge graph")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query KG", "🤖 Expand KG", "📊 Visualize", "⚙️ Settings"])
    
    with tab1:
        st.header("Query Knowledge Graph")
        
        # Query options
        query_type = st.selectbox("Query Type", 
                                 ["Materials by Application", 
                                  "Properties by Material", 
                                  "Search Materials"])
        
        if query_type == "Materials by Application":
            application = st.selectbox("Application", 
                                     ["capacitor", "thermoelectric_device", "solar_cell"])
            if st.button("Query"):
                display_materials_for_application(kg, application)
        
        elif query_type == "Properties by Material":
            # Get list of materials
            try:
                materials_query = "MATCH (m:Material) RETURN m.name as name ORDER BY m.name"
                with kg.driver.session() as session:
                    result = session.run(materials_query)
                    materials = [record["name"] for record in result]
                
                material = st.selectbox("Material", materials)
                if st.button("Query"):
                    display_material_properties(kg, material)
            except Exception as e:
                st.error(f"Error getting materials: {e}")
        
        elif query_type == "Search Materials":
            search_term = st.text_input("Search term")
            if st.button("Search") and search_term:
                try:
                    results = kg.search_materials(search_term)
                    if results:
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No materials found")
                except Exception as e:
                    st.error(f"Error searching: {e}")
    
    with tab2:
        st.header("LLM-Guided Knowledge Graph Expansion")
        
        # Check Ollama connection
        if llm.test_connection():
            st.success("✅ Ollama connection active")
            
            # Select application for expansion
            application = st.selectbox("Select Application to Expand", 
                                     ["capacitor", "thermoelectric_device", "solar_cell"],
                                     key="expansion_app")
            
            if st.button("🚀 Start Expansion Process"):
                run_hypothesis_expansion(kg, llm, validator, application)
        else:
            st.error("❌ Ollama not connected")
            st.info("Please ensure Ollama is running and a model is available")
    
    with tab3:
        st.header("Knowledge Graph Visualization")
        
        if st.button("🔄 Refresh Visualization"):
            create_network_visualization(kg)
    
    with tab4:
        st.header("Settings & Configuration")
        
        # Neo4j settings
        st.subheader("Neo4j Configuration")
        neo4j_uri = st.text_input("Neo4j URI", value=os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
        neo4j_user = st.text_input("Neo4j User", value=os.getenv('NEO4J_USER', 'neo4j'))
        neo4j_password = st.text_input("Neo4j Password", type="password", 
                                      value=os.getenv('NEO4J_PASSWORD', ''))
        
        # Ollama settings
        st.subheader("Ollama Configuration")
        ollama_url = st.text_input("Ollama Base URL", value=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
        ollama_model = st.text_input("Ollama Model", value=os.getenv('OLLAMA_MODEL', 'llama3.1'))
        
        # Actions
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Test Connections"):
                # Test Neo4j
                try:
                    test_kg = MaterialsKG(neo4j_uri, neo4j_user, neo4j_password)
                    test_kg.close()
                    st.success("✅ Neo4j connection successful")
                except Exception as e:
                    st.error(f"❌ Neo4j connection failed: {e}")
                
                # Test Ollama
                test_llm = OllamaHypothesisGenerator(ollama_url, ollama_model)
                if test_llm.test_connection():
                    st.success("✅ Ollama connection successful")
                else:
                    st.error("❌ Ollama connection failed")
        
        with col2:
            if st.button("🗑️ Clear Knowledge Graph"):
                if st.checkbox("Are you sure? This will delete all data!"):
                    try:
                        kg.clear_database()
                        st.success("✅ Knowledge graph cleared")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error clearing graph: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    Materials Ontology Expansion System | Powered by Neo4j, Ollama, and Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
