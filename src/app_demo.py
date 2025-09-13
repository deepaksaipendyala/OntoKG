"""
Streamlit Web Application for Materials Ontology Expansion - Demo Version
Works without Neo4j by using in-memory data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from typing import Dict, List, Any
import networkx as nx
import pyvis
import streamlit.components.v1 as components

# Import our modules
from validation import MaterialsValidator
from llm_integration import OllamaHypothesisGenerator

# Page configuration
st.set_page_config(
    page_title="Materials Ontology Expansion - Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DemoKnowledgeGraph:
    """Demo version that works without Neo4j"""
    
    def __init__(self):
        # Load complete seed data
        self._load_seed_data()
        
    def _load_seed_data(self):
        """Load complete seed data from JSON file"""
        try:
            with open('data/seed_data.json', 'r') as f:
                seed_data = json.load(f)
            
            # Load materials
            self.materials = {}
            for mat in seed_data['materials']:
                self.materials[mat['name']] = {
                    'type': mat['type'],
                    'formula': mat['formula']
                }
            
            # Load properties
            self.properties = {}
            for prop in seed_data['properties']:
                self.properties[prop['name']] = prop['description']
            
            # Load applications
            self.applications = {}
            for app in seed_data['applications']:
                self.applications[app['name']] = app['description']
            
            # Load material-property relationships
            self.material_properties = {}
            for rel in seed_data['has_property_relations']:
                key = (rel['material'], rel['property'])
                self.material_properties[key] = {
                    'value': rel['value'],
                    'unit': rel['unit'],
                    'confidence': rel['confidence']
                }
            
            # Load material-application relationships
            self.material_applications = {}
            for rel in seed_data['used_in_relations']:
                key = (rel['material'], rel['application'])
                self.material_applications[key] = {
                    'confidence': rel['confidence'],
                    'source': rel['source']
                }
                
        except Exception as e:
            st.error(f"Error loading seed data: {e}")
            # Fallback to minimal data
            self.materials = {"BaTiO3": {"type": "perovskite", "formula": "BaTiO3"}}
            self.properties = {"dielectric_constant": "Relative permittivity"}
            self.applications = {"capacitor": "Energy storage device"}
            self.material_properties = {}
            self.material_applications = {}
    
    def get_graph_stats(self):
        return {
            "materials": len(self.materials),
            "properties": len(self.properties),
            "applications": len(self.applications),
            "relationships": len(self.material_properties) + len(self.material_applications)
        }
    
    def get_materials_for_application(self, application):
        results = []
        for (material, app), data in self.material_applications.items():
            if app == application:
                results.append({
                    "material": material,
                    "confidence": data["confidence"],
                    "source": data["source"],
                    "validated_by": "curated"
                })
        return results
    
    def get_properties_for_material(self, material):
        results = []
        for (mat, prop), data in self.material_properties.items():
            if mat == material:
                results.append({
                    "property": prop,
                    "value": data["value"],
                    "unit": data["unit"],
                    "confidence": data["confidence"],
                    "source": "curated"
                })
        return results
    
    def add_material_application(self, material, application, confidence, source="LLM"):
        self.material_applications[(material, application)] = {
            "confidence": confidence,
            "source": source
        }

def display_knowledge_graph_stats(kg):
    """Display knowledge graph statistics"""
    stats = kg.get_graph_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Materials", stats["materials"])
    with col2:
        st.metric("Properties", stats["properties"])
    with col3:
        st.metric("Applications", stats["applications"])
    with col4:
        st.metric("Relationships", stats["relationships"])

def display_materials_for_application(kg, application):
    """Display materials used in a specific application"""
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
        fig = px.bar(df, x='material', y='confidence', 
                    title=f'Confidence Scores for {application} Materials',
                    color='confidence',
                    color_continuous_scale='viridis')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def display_material_properties(kg, material):
    """Display properties of a specific material"""
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

def run_hypothesis_expansion(kg, llm, validator, application):
    """Run the hypothesis expansion process"""
    
    st.subheader(f"🤖 LLM-Guided Expansion for {application}")
    
    # Get current materials for the application
    current_materials = kg.get_materials_for_application(application)
    material_names = [m['material'] for m in current_materials]
    
    if not material_names:
        st.warning(f"No current materials found for {application}")
        return
    
    # Display current state
    st.info(f"Current {application} materials: {', '.join(material_names)}")
    
    # Test LLM connection
    if not llm.test_connection():
        st.error("❌ Cannot connect to Ollama. Please ensure Ollama is running.")
        return
    
    st.success("✅ Connected to Ollama")
    
    # Generate hypotheses
    with st.spinner("🤖 Generating hypotheses with LLM..."):
        try:
            # Create context
            context = {"Property": list(kg.properties.keys()), "Application": list(kg.applications.keys())}
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
            
            validated_count = 0
            
            for i, hypothesis_data in enumerate(response.hypotheses):
                material = hypothesis_data.get('material', '')
                rationale = hypothesis_data.get('rationale', '')
                confidence = hypothesis_data.get('confidence', 0.5)
                
                if not material:
                    continue
                
                # Validate hypothesis
                validation_result = validator.validate_hypothesis(material, application)
                
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
                            final_confidence = validation_result.confidence * confidence
                            kg.add_material_application(material, application, final_confidence)
                            st.success(f"✅ Added {material} to knowledge graph!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.info("Not added")
                
                if validation_result.is_valid and validation_result.confidence >= 0.6:
                    validated_count += 1
                
                st.divider()
            
            # Summary
            if validated_count > 0:
                st.success(f"🎉 Successfully validated {validated_count} hypotheses!")
            else:
                st.warning("No hypotheses met validation criteria")
                
        except Exception as e:
            st.error(f"Error during hypothesis generation: {e}")

def create_interactive_neo4j_style_visualization(kg):
    """Create an interactive Neo4j-style network visualization"""
    try:
        # Create Pyvis network
        net = pyvis.network.Network(
            height="700px", 
            width="100%", 
            bgcolor="#ffffff", 
            font_color="black",
            directed=False
        )
        
        # Configure physics for smooth interaction
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
        
        # Add material nodes (larger, bright red with clear borders)
        for material, data in kg.materials.items():
            net.add_node(
                material, 
                label=material,
                color={"background": "#FF6B6B", "border": "#E53E3E", "highlight": {"background": "#FF8E8E", "border": "#FC8181"}},
                size=30,
                title=f"""
                <div style="background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
                <b style="color: #E53E3E; font-size: 16px;">{material}</b><br>
                <b>Type:</b> Material<br>
                <b>Formula:</b> {data['formula']}<br>
                <b>Structure:</b> {data['type']}<br>
                <i style="color: #666;">Click and drag to move</i>
                </div>
                """,
                group="materials"
            )
        
        # Add application nodes (medium, bright blue)
        for app in kg.applications.keys():
            net.add_node(
                app, 
                label=app.replace('_', ' ').title(),
                color={"background": "#4299E1", "border": "#3182CE", "highlight": {"background": "#63B3ED", "border": "#4299E1"}},
                size=25,
                title=f"""
                <div style="background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
                <b style="color: #3182CE; font-size: 16px;">{app.replace('_', ' ').title()}</b><br>
                <b>Type:</b> Application<br>
                <b>Description:</b> {kg.applications[app]}<br>
                <i style="color: #666;">Click and drag to move</i>
                </div>
                """,
                group="applications"
            )
        
        # Add property nodes (smaller, bright green)
        for prop in kg.properties.keys():
            net.add_node(
                prop, 
                label=prop.replace('_', ' ').title(),
                color={"background": "#48BB78", "border": "#38A169", "highlight": {"background": "#68D391", "border": "#48BB78"}},
                size=20,
                title=f"""
                <div style="background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
                <b style="color: #38A169; font-size: 16px;">{prop.replace('_', ' ').title()}</b><br>
                <b>Type:</b> Property<br>
                <b>Description:</b> {kg.properties[prop]}<br>
                <i style="color: #666;">Click and drag to move</i>
                </div>
                """,
                group="properties"
            )
        
        # Add material-application edges
        for (material, app), data in kg.material_applications.items():
            confidence = data["confidence"]
            source = data["source"]
            
            # Color edges based on source
            edge_color = "#FFA500" if source == "LLM" else "#22C55E"  # Orange for LLM, Green for curated
            
            net.add_edge(
                material, app,
                label=f"{confidence:.2f}",
                color={"color": edge_color, "highlight": "#FFD700", "hover": "#FF6B6B"},
                width=confidence * 4 + 2,
                title=f"""
                <div style="background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
                <b style="color: #333; font-size: 16px;">USED_IN Relationship</b><br>
                <b>Material:</b> {material}<br>
                <b>Application:</b> {app}<br>
                <b>Confidence:</b> {confidence:.2f}<br>
                <b>Source:</b> {source}<br>
                <i style="color: #666;">Hover to see details</i>
                </div>
                """
            )
        
        # Add material-property edges
        for (material, prop), data in kg.material_properties.items():
            value = data["value"]
            unit = data["unit"]
            confidence = data["confidence"]
            
            net.add_edge(
                material, prop,
                label=f"{value} {unit}",
                color={"color": "#8B5CF6", "highlight": "#A78BFA", "hover": "#C4B5FD"},
                width=confidence * 3 + 2,
                title=f"""
                <div style="background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
                <b style="color: #333; font-size: 16px;">HAS_PROPERTY Relationship</b><br>
                <b>Material:</b> {material}<br>
                <b>Property:</b> {prop}<br>
                <b>Value:</b> {value} {unit}<br>
                <b>Confidence:</b> {confidence:.2f}<br>
                <i style="color: #666;">Hover to see details</i>
                </div>
                """
            )
        
        # Generate HTML
        net.save_graph("temp_graph.html")
        
        # Read and display the HTML
        with open("temp_graph.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Display the interactive graph
        components.html(html_content, height=650)
        
        # Add legend
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid #e9ecef;">
        <h4 style="color: #495057; margin-top: 0;">📊 Graph Legend:</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
        <div>
        <p><span style="color: #FF6B6B; font-size: 20px;">●</span> <strong style="color: #495057;">Materials</strong><br><small style="color: #6c757d;">Chemical compounds and elements</small></p>
        <p><span style="color: #4299E1; font-size: 20px;">●</span> <strong style="color: #495057;">Applications</strong><br><small style="color: #6c757d;">End-use applications</small></p>
        <p><span style="color: #48BB78; font-size: 20px;">●</span> <strong style="color: #495057;">Properties</strong><br><small style="color: #6c757d;">Physical/chemical properties</small></p>
        </div>
        <div>
        <p><span style="color: #FFA500; font-size: 18px;">—</span> <strong style="color: #495057;">LLM Discovered</strong><br><small style="color: #6c757d;">Relationships found by AI</small></p>
        <p><span style="color: #22C55E; font-size: 18px;">—</span> <strong style="color: #495057;">Curated</strong><br><small style="color: #6c757d;">Known relationships</small></p>
        <p><span style="color: #8B5CF6; font-size: 18px;">—</span> <strong style="color: #495057;">Properties</strong><br><small style="color: #6c757d;">Material has property</small></p>
        </div>
        </div>
        <hr style="margin: 15px 0; border: none; border-top: 1px solid #dee2e6;">
        <p style="color: #495057; margin-bottom: 0;"><strong>🖱️ Interactions:</strong> Drag nodes to move them, scroll to zoom, hover for details!</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        st.write("Falling back to simple visualization...")
        create_simple_visualization(kg)

def create_simple_visualization(kg):
    """Fallback simple visualization"""
    try:
        # Create a simple network representation
        st.subheader("Knowledge Graph Structure")
        
        # Show materials
        st.write("**Materials:**")
        for material, data in kg.materials.items():
            st.write(f"• {material} ({data['type']})")
        
        # Show applications
        st.write("**Applications:**")
        for app, desc in kg.applications.items():
            st.write(f"• {app.replace('_', ' ').title()}")
        
        # Show relationships
        st.write("**Relationships:**")
        for (material, app), data in kg.material_applications.items():
            st.write(f"• {material} → {app} (confidence: {data['confidence']:.2f})")
            
    except Exception as e:
        st.error(f"Error creating simple visualization: {e}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🧬 Materials Ontology Expansion - Demo Version")
    
    st.info("""
    **Demo Version:** This version works without Neo4j and shows the complete functionality 
    using in-memory data. You can see the knowledge graph, run LLM expansions, and visualize results.
    """)
    
    # Initialize components
    kg = DemoKnowledgeGraph()
    llm = OllamaHypothesisGenerator(model="llama3:latest")
    validator = MaterialsValidator()
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Display current stats
    st.sidebar.subheader("📊 Knowledge Graph Stats")
    stats = kg.get_graph_stats()
    st.sidebar.metric("Materials", stats["materials"])
    st.sidebar.metric("Applications", stats["applications"])
    st.sidebar.metric("Relationships", stats["relationships"])
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Query KG", "🤖 Expand KG", "📊 Visualize"])
    
    with tab1:
        st.header("Query Knowledge Graph")
        
        # Query options
        query_type = st.selectbox("Query Type", 
                                 ["Materials by Application", 
                                  "Properties by Material"])
        
        if query_type == "Materials by Application":
            application = st.selectbox("Application", 
                                     ["capacitor", "thermoelectric_device", "solar_cell"])
            if st.button("Query"):
                display_materials_for_application(kg, application)
        
        elif query_type == "Properties by Material":
            material = st.selectbox("Material", list(kg.materials.keys()))
            if st.button("Query"):
                display_material_properties(kg, material)
    
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
        
        if st.button("🔄 Refresh Interactive Graph"):
            create_interactive_neo4j_style_visualization(kg)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    Materials Ontology Expansion System - Demo Version | Powered by Ollama and Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
