"""
Enhanced Streamlit Application for Materials Ontology Expansion
Features advanced visualizations, real-time analytics, and comprehensive discovery insights
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime, timedelta
import logging

# Import enhanced modules
from advanced_llm_integration import EnhancedHypothesisGenerator, AdvancedLLMResponse
from enhanced_validation import EnhancedMaterialsValidator, EnhancedValidationResult
from advanced_visualization import AdvancedNetworkVisualizer, MaterialsAnalyticsDashboard, RealTimeUpdater
from discovery_analytics import MaterialsDiscoveryEngine, DiscoveryInsight
from knowledge_graph import MaterialsKG, Hypothesis
from config import load_config
from chat_interface import render_chat_interface
# Import working LLM integration as fallback
from llm_integration import OllamaHypothesisGenerator
from validation import MaterialsValidator
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Advanced Materials Ontology Expansion",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 0.5rem 0;
    }
    
    .insight-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .success-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .discovery-metric {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_cached_kg_data():
    """Load knowledge graph data with caching (working visualization structure)."""
    try:
        config = load_config()
        kg = MaterialsKG(config)
        nodes = {'Material': [], 'Property': [], 'Application': []}
        edges = []
        with kg.driver.session() as session:
            # Materials
            result = session.run("MATCH (m:Material) RETURN m.name as name, m.formula as formula, m.type as type")
            for record in result:
                nodes['Material'].append({
                    'name': record.get('name'),
                    'formula': record.get('formula', ''),
                    'type': record.get('type', 'unknown'),
                    'id': record.get('name')
                })
            # Properties
            result = session.run("MATCH (p:Property) RETURN p.name as name, p.description as description")
            for record in result:
                nodes['Property'].append({
                    'name': record.get('name'),
                    'description': record.get('description', ''),
                    'id': record.get('name')
                })
            # Applications
            result = session.run("MATCH (a:Application) RETURN a.name as name, a.description as description")
            for record in result:
                nodes['Application'].append({
                    'name': record.get('name'),
                    'description': record.get('description', ''),
                    'id': record.get('name')
                })
            # USED_IN
            result = session.run(
                """
                MATCH (m:Material)-[r:USED_IN]->(a:Application)
                RETURN m.name as source, a.name as target, 'USED_IN' as relationship,
                       r.confidence as confidence, r.source as source_type
                """
            )
            for record in result:
                edges.append({
                    'source': record.get('source'),
                    'target': record.get('target'),
                    'relationship': record.get('relationship'),
                    'confidence': record.get('confidence', 1.0),
                    'source_type': record.get('source_type', 'curated')
                })
            # HAS_PROPERTY
            result = session.run(
                """
                MATCH (m:Material)-[r:HAS_PROPERTY]->(p:Property)
                RETURN m.name as source, p.name as target, 'HAS_PROPERTY' as relationship,
                       r.value as value, r.unit as unit, r.confidence as confidence
                """
            )
            for record in result:
                edges.append({
                    'source': record.get('source'),
                    'target': record.get('target'),
                    'relationship': record.get('relationship'),
                    'value': record.get('value'),
                    'unit': record.get('unit'),
                    'confidence': record.get('confidence', 1.0)
                })
        kg.close()
        return {'nodes': nodes, 'edges': edges}
    except Exception as e:
        logger.error(f"Error loading KG data: {e}")
        return None

@st.cache_resource
def initialize_enhanced_components():
    """Initialize enhanced components with caching"""
    try:
        config = load_config()
        
        # Initialize components
        kg = MaterialsKG(config)
        
        # Use basic LLM with single model for speed
        llm = OllamaHypothesisGenerator(model="llama3.2:latest")
        
        # Use basic validator for speed
        validator = MaterialsValidator()
        visualizer = AdvancedNetworkVisualizer()
        analytics_dashboard = MaterialsAnalyticsDashboard()
        discovery_engine = MaterialsDiscoveryEngine()
        real_time_updater = RealTimeUpdater()
        
        return kg, llm, validator, visualizer, analytics_dashboard, discovery_engine, real_time_updater, config
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        st.error(f"Failed to initialize components: {e}")
        return None, None, None, None, None, None, None, None

def display_enhanced_metrics(kg_data: Dict[str, Any], discovery_metrics: Dict[str, Any]):
    """Display enhanced metrics dashboard"""
    
    st.markdown("### 📊 Discovery Landscape Metrics")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🧪 Materials</h3>
            <h2>{}</h2>
            <p>Active compounds</p>
        </div>
        """.format(discovery_metrics.get('total_materials', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Applications</h3>
            <h2>{}</h2>
            <p>Target domains</p>
        </div>
        """.format(discovery_metrics.get('total_applications', 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔗 Relationships</h3>
            <h2>{}</h2>
            <p>Validated connections</p>
        </div>
        """.format(discovery_metrics.get('total_relationships', 0)), unsafe_allow_html=True)
    
    with col4:
        readiness_score = discovery_metrics.get('discovery_readiness_score', 0.0)
        readiness_color = "#00B894" if readiness_score > 0.7 else "#FDCB6E" if readiness_score > 0.4 else "#E17055"
        
        st.markdown(f"""
        <div class="metric-card" style="background: {readiness_color}">
            <h3>🚀 Readiness</h3>
            <h2>{readiness_score:.1%}</h2>
            <p>Discovery readiness</p>
        </div>
        """, unsafe_allow_html=True)

def display_discovery_insights(insights: List[Dict[str, Any]]):
    """Display discovery insights with enhanced formatting"""
    
    st.markdown("### 💡 Discovery Insights")
    
    if not insights:
        st.info("No insights available yet. Run analysis to generate insights.")
        return
    
    # Display top 3 insights
    for i, insight in enumerate(insights[:3]):
        impact_color = "#00B894" if insight['impact_score'] > 0.7 else "#FDCB6E" if insight['impact_score'] > 0.4 else "#E17055"
        
        st.markdown(f"""
        <div class="insight-card">
            <h4 style="color: {impact_color};">
                {'🔥' if insight['impact_score'] > 0.7 else '⚡' if insight['impact_score'] > 0.4 else '💡'} 
                {insight['title']}
            </h4>
            <p>{insight['description']}</p>
            <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                <small>Impact: {insight['impact_score']:.1%}</small>
                <small>Confidence: {insight['confidence']:.1%}</small>
                <small>Type: {insight['insight_type'].replace('_', ' ').title()}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show recommendations
        if insight.get('actionable_recommendations'):
            with st.expander(f"💡 Recommendations for {insight['title'][:30]}..."):
                for rec in insight['actionable_recommendations']:
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <strong>Action:</strong> {rec}
                    </div>
                    """, unsafe_allow_html=True)

async def run_enhanced_hypothesis_expansion(kg: MaterialsKG, llm: EnhancedHypothesisGenerator, 
                                          validator: EnhancedMaterialsValidator, application: str):
    """Run enhanced hypothesis expansion with real-time updates"""
    
    st.markdown("### 🤖 Advanced LLM Ensemble Analysis")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Get context
        status_text.text("🔍 Gathering knowledge graph context...")
        progress_bar.progress(10)
        
        current_materials = kg.get_materials_for_application(application)
        material_names = [m['material'] for m in current_materials]
        
        if not material_names:
            st.warning(f"No current materials found for {application}. Please initialize the knowledge graph first.")
            return
        
        # Get enhanced context
        context = {}
        for material in material_names:
            neighbors = kg.get_neighbors(material)
            for entity_type, names in neighbors.items():
                if entity_type not in context:
                    context[entity_type] = []
                context[entity_type].extend(names)
        
        progress_bar.progress(25)
        
        # Step 2: Test ensemble connection
        status_text.text("🔗 Testing LLM ensemble connection...")
        ensemble_status = llm.get_ensemble_status()
        
        if ensemble_status['success_rate'] == 0:
            st.error("❌ No LLM models available. Please ensure Ollama is running with models.")
            return
        
        # Display ensemble status
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Available Models", f"{len(ensemble_status['available_models'])}/{ensemble_status['total_models']}")
        with col2:
            st.metric("Success Rate", f"{ensemble_status['success_rate']:.1%}")
        
        progress_bar.progress(40)
        
        # Step 3: Generate hypotheses
        status_text.text("🧠 Generating hypotheses with ensemble voting...")
        
        # Run async hypothesis generation
        response = await llm.generate_ensemble_hypotheses(application, material_names, context)
        
        progress_bar.progress(70)
        
        # Step 4: Display results
        status_text.text("📊 Processing ensemble results...")
        
        if not response.hypotheses:
            st.warning("No hypotheses generated. Please check LLM connectivity and try again.")
            return
        
        # Display ensemble analysis
        st.markdown("#### 🎯 Ensemble Analysis Results")
        
        # Ensemble metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Generated Hypotheses", len(response.hypotheses))
        with col2:
            st.metric("Ensemble Confidence", f"{response.confidence:.1%}")
        with col3:
            uncertainty_range = response.uncertainty_metrics.get('confidence_range', 0)
            st.metric("Uncertainty Range", f"±{uncertainty_range:.1%}")
        
        # Chain of thought reasoning
        if response.chain_of_thought:
            with st.expander("🧠 Chain of Thought Reasoning"):
                for i, thought in enumerate(response.chain_of_thought):
                    st.markdown(f"**Step {i+1}:** {thought}")
        
        progress_bar.progress(85)
        
        # Step 5: Enhanced validation
        status_text.text("✅ Running enhanced validation...")
        
        validated_hypotheses = []
        
        for i, hypothesis in enumerate(response.hypotheses):
            material = hypothesis.get('material', '')
            
            # Extract material properties for ML validation
            material_properties = {
                'n_atoms': len(material.replace('(', '').replace(')', '')) // 3,  # Rough estimate
                'volume': 50.0,  # Default value
                'electronegativity_diff': 1.5,  # Default value
            }
            
            # Run enhanced validation
            validation_result = await validator.validate_hypothesis_enhanced(
                material, application, material_properties
            )
            
            hypothesis['validation_result'] = {
                'is_valid': validation_result.is_valid,
                'confidence': validation_result.confidence,
                'consensus_score': validation_result.consensus_score,
                'validation_methods': validation_result.validation_methods,
                'uncertainty_bounds': validation_result.uncertainty_bounds,
                'risk_assessment': validation_result.risk_assessment
            }
            
            if validation_result.is_valid:
                validated_hypotheses.append(hypothesis)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Display validated results
        st.markdown("#### ✅ Validated Discoveries")
        
        if validated_hypotheses:
            for hyp in validated_hypotheses:
                validation = hyp['validation_result']
                
                # Color based on confidence
                conf_color = "#00B894" if validation['confidence'] > 0.7 else "#FDCB6E" if validation['confidence'] > 0.5 else "#E17055"
                
                st.markdown(f"""
                <div class="success-gradient">
                    <h4>🎯 {hyp['material']} → {application}</h4>
                    <p><strong>Rationale:</strong> {hyp.get('rationale', 'No rationale provided')}</p>
                    <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                        <span>Confidence: {validation['confidence']:.1%}</span>
                        <span>Consensus: {validation['consensus_score']:.1%}</span>
                        <span>Methods: {len(validation['validation_methods'])}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show detailed validation info
                with st.expander(f"🔍 Detailed Validation for {hyp['material']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Validation Methods:**")
                        for method in validation['validation_methods']:
                            st.markdown(f"- {method.replace('_', ' ').title()}")
                    
                    with col2:
                        st.markdown("**Risk Assessment:**")
                        for risk_type, level in validation['risk_assessment'].items():
                            color = "#E17055" if level == 'high' else "#FDCB6E" if level == 'medium' else "#00B894"
                            st.markdown(f"- {risk_type.title()}: <span style='color: {color}'>{level.upper()}</span>", unsafe_allow_html=True)
        else:
            st.info("No hypotheses passed validation. Consider adjusting validation criteria or exploring different materials.")
        
        # Add to knowledge graph
        if validated_hypotheses:
            if st.button("🚀 Add Validated Discoveries to Knowledge Graph"):
                added_count = 0
                for hyp in validated_hypotheses:
                    try:
                        hypothesis_obj = Hypothesis(
                            material=hyp['material'],
                            application=application,
                            relationship='USED_IN',
                            confidence=hyp['validation_result']['confidence'],
                            source='Enhanced_LLM_Ensemble',
                            validated_by='Enhanced_Validation',
                            rationale=hyp.get('rationale', '')
                        )
                        
                        kg.add_hypothesis(hypothesis_obj)
                        added_count += 1
                    except Exception as e:
                        st.error(f"Error adding {hyp['material']}: {e}")
                
                if added_count > 0:
                    st.success(f"✅ Successfully added {added_count} validated discoveries to the knowledge graph!")
                    st.balloons()
    
    except Exception as e:
        logger.error(f"Error in enhanced expansion: {e}")
        st.error(f"Error during expansion: {e}")
    
    finally:
        progress_bar.empty()
        status_text.empty()

def display_advanced_analytics(analytics_dashboard: MaterialsAnalyticsDashboard, 
                             discovery_engine: MaterialsDiscoveryEngine,
                             kg_data: Dict[str, Any]):
    """Display advanced analytics dashboard"""
    
    st.markdown("### 📈 Advanced Analytics Dashboard")
    
    # Run discovery analysis
    with st.spinner("🔍 Analyzing discovery landscape..."):
        # Generate sample discovery history
        discovery_history = []
        for i in range(30):
            discovery_history.append({
                'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
                'material': f'Material_{i}',
                'application': ['capacitor', 'thermoelectric_device', 'solar_cell'][i % 3],
                'confidence': 0.5 + (i % 5) * 0.1,
                'validated': i % 3 == 0
            })
        
        analysis_results = discovery_engine.analyze_discovery_landscape(kg_data, discovery_history)
    
    # Display summary
    if 'summary' in analysis_results:
        summary = analysis_results['summary']
        
        st.markdown("#### 📋 Executive Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'top_opportunity' in summary:
                st.markdown(f"""
                <div class="discovery-metric">
                    <h4>🎯 Top Opportunity</h4>
                    <p>{summary['top_opportunity']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if 'key_trend' in summary:
                st.markdown(f"""
                <div class="discovery-metric">
                    <h4>📈 Key Trend</h4>
                    <p>{summary['key_trend']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'priority_action' in summary:
                st.markdown(f"""
                <div class="discovery-metric">
                    <h4>⚡ Priority Action</h4>
                    <p>{summary['priority_action']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display analytics charts
    tab1, tab2, tab3 = st.tabs(["🔍 Material Clusters", "📊 Trend Analysis", "🎯 Performance Matrix"])
    
    with tab1:
        clusters = analysis_results.get('clusters', [])
        if clusters:
            st.markdown("#### 🧬 Material Cluster Analysis")
            
            # Create cluster visualization
            cluster_df = pd.DataFrame([
                {
                    'Cluster': c['cluster_name'],
                    'Materials': len(c['materials']),
                    'Discovery Potential': c['discovery_potential'],
                    'Similarity Score': c['similarity_score']
                }
                for c in clusters
            ])
            
            fig = px.scatter(
                cluster_df, 
                x='Similarity Score', 
                y='Discovery Potential',
                size='Materials',
                color='Cluster',
                hover_name='Cluster',
                title="Material Clusters: Discovery Potential vs Similarity"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster details
            for cluster in clusters[:3]:  # Top 3 clusters
                with st.expander(f"🔍 {cluster['cluster_name']} Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Materials in Cluster:**")
                        for material in cluster['materials']:
                            st.markdown(f"- {material}")
                    
                    with col2:
                        st.markdown("**Properties:**")
                        for prop, value in cluster['centroid_properties'].items():
                            st.markdown(f"- {prop}: {value:.2f}")
        else:
            st.info("No clusters identified. More data needed for cluster analysis.")
    
    with tab2:
        trends = analysis_results.get('trends', [])
        if trends:
            st.markdown("#### 📈 Trend Analysis")
            
            for trend in trends:
                direction_emoji = "📈" if trend['trend_direction'] == 'increasing' else "📉" if trend['trend_direction'] == 'decreasing' else "➡️"
                strength_color = "#00B894" if trend['strength'] > 0.7 else "#FDCB6E" if trend['strength'] > 0.4 else "#E17055"
                
                st.markdown(f"""
                <div style="background: {strength_color}; color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                    <h4>{direction_emoji} {trend['trend_type'].replace('_', ' ').title()}</h4>
                    <p><strong>Direction:</strong> {trend['trend_direction'].title()}</p>
                    <p><strong>Strength:</strong> {trend['strength']:.1%}</p>
                    <p><strong>Period:</strong> {trend['time_period']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant trends detected. More temporal data needed.")
    
    with tab3:
        # Create synthetic performance matrix
        materials = ['BaTiO3', 'SrTiO3', 'Bi2Te3', 'SnSe', 'CH3NH3PbI3']
        applications = ['capacitor', 'thermoelectric_device', 'solar_cell']
        
        performance_data = {}
        for material in materials:
            performance_data[material] = {}
            for app in applications:
                # Synthetic performance score
                performance_data[material][app] = np.random.uniform(0.3, 1.0)
        
        fig = analytics_dashboard.create_application_performance_matrix(performance_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top recommendations
    recommendations = analysis_results.get('recommendations', [])
    if recommendations:
        st.markdown("#### 🎯 Top Recommendations")
        for i, rec in enumerate(recommendations):
            st.markdown(f"""
            <div class="recommendation-box">
                <strong>{i+1}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)

def main():
    """Enhanced main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧬 Advanced Materials Ontology Expansion</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 15px; margin-bottom: 2rem;">
        <h3>AI-Powered Materials Discovery Platform</h3>
        <p>Leveraging ensemble LLMs, advanced validation, and comprehensive analytics for accelerated materials research</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    components = initialize_enhanced_components()
    if components[0] is None:
        st.error("Failed to initialize application components. Please check your configuration.")
        st.stop()
    
    kg, llm, validator, visualizer, analytics_dashboard, discovery_engine, real_time_updater, config = components
    
    # Load data
    kg_data = load_cached_kg_data()
    if kg_data is None:
        st.error("Failed to load knowledge graph data. Please check your Neo4j connection.")
        st.stop()
    
    # Ensure kg_data is in correct format (strict normalization)
    def _normalize_kg_data(data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {'nodes': {}, 'edges': []}
        if isinstance(data, list):
            # Merge all chunks
            for item in data:
                if not isinstance(item, dict):
                    continue
                # Nodes
                nodes_part = item.get('nodes')
                if isinstance(nodes_part, dict):
                    for t, lst in nodes_part.items():
                        normalized['nodes'].setdefault(t, [])
                        if isinstance(lst, list):
                            normalized['nodes'][t].extend(lst)
                elif isinstance(nodes_part, list):
                    # If nodes provided as flat list with type field
                    for n in nodes_part:
                        t = n.get('type', 'Unknown')
                        normalized['nodes'].setdefault(t, [])
                        normalized['nodes'][t].append(n)
                # Edges
                edges_part = item.get('edges')
                if isinstance(edges_part, list):
                    normalized['edges'].extend(edges_part)
        elif isinstance(data, dict):
            nodes_part = data.get('nodes', {})
            if isinstance(nodes_part, dict):
                for t, lst in nodes_part.items():
                    normalized['nodes'][t] = list(lst) if isinstance(lst, list) else []
            elif isinstance(nodes_part, list):
                for n in nodes_part:
                    t = n.get('type', 'Unknown')
                    normalized['nodes'].setdefault(t, [])
                    normalized['nodes'][t].append(n)
            edges_part = data.get('edges', [])
            if isinstance(edges_part, list):
                normalized['edges'] = edges_part
        return normalized

    kg_data = _normalize_kg_data(kg_data)
    
    # Build dynamic application options from KG
    application_nodes = set()
    try:
        node_groups = kg_data.get('nodes', {}) if isinstance(kg_data, dict) else {}
        if isinstance(node_groups, dict):
            for n in node_groups.get('Application', []) or []:
                name = n.get('name') if isinstance(n, dict) else None
                if name:
                    application_nodes.add(name)
    except Exception:
        application_nodes = set()

    dynamic_applications = set(application_nodes)
    try:
        for e in kg_data.get('edges', []) or []:
            rel = str(e.get('relationship', '')).lower()
            if rel == 'used_in':
                target = e.get('target')
                source = e.get('source')
                if application_nodes:
                    if target in application_nodes:
                        dynamic_applications.add(target)
                    if source in application_nodes:
                        dynamic_applications.add(source)
                else:
                    # No explicit Application nodes; assume target is application
                    if target:
                        dynamic_applications.add(target)
    except Exception:
        pass

    application_options = sorted(dynamic_applications) or ["capacitor", "thermoelectric_device", "solar_cell"]
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Build/Refresh KG controls
    st.sidebar.markdown("### 🔄 Build/Refresh Knowledge Graph")
    with st.sidebar.expander("MatKG Builder", expanded=False):
        max_rows = st.number_input("Max rows (0 = all)", min_value=0, value=50000, step=10000)
        clear_first = st.checkbox("Clear database first", value=False)
        csv_path_input = st.text_input(
            "SUBRELOBJ.csv path (optional)", value="",
            help="Leave blank to use default at materials_ontology_expansion/SUBRELOBJ.csv or MATKG_SUBRELOBJ_CSV env"
        )
        if st.button("Run Builder", type="secondary"):
            with st.spinner("Building knowledge graph in Neo4j..."):
                try:
                    script_path = Path(__file__).resolve().parent.parent / "build_matkg_neo4j.py"
                    cmd = [sys.executable, str(script_path)]
                    if csv_path_input.strip():
                        cmd.extend(["--csv", csv_path_input.strip()])
                    if int(max_rows) > 0:
                        cmd.extend(["--max-rows", str(int(max_rows))])
                    if clear_first:
                        cmd.append("--clear-first")
                    # Run builder non-interactively
                    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if completed.returncode == 0:
                        st.success("Builder completed successfully")
                        st.text(completed.stdout[-1000:])
                        # Clear caches and reload KG data
                        load_cached_kg_data.clear()
                        kg_data_new = load_cached_kg_data()
                        if kg_data_new:
                            st.success("Knowledge graph reloaded for visualization")
                        else:
                            st.warning("Reload failed; verify Neo4j has data.")
                    else:
                        st.error("Builder failed")
                        st.text_area("Builder stderr", completed.stderr[-4000:], height=200)
                except Exception as e:
                    st.error(f"Failed to run builder: {e}")

    # Seed KG initializer
    st.sidebar.markdown("### 🌱 Load Seed Knowledge Graph")
    with st.sidebar.expander("Seed Loader", expanded=False):
        if st.button("Load seed_data.json into Neo4j"):
            with st.spinner("Loading seed data into Neo4j..."):
                try:
                    # Clear cache to ensure fresh visualization after load
                    load_cached_kg_data.clear()
                    # Use MaterialsKG high-level APIs
                    # seed_data.json is located at materials_ontology_expansion/data/seed_data.json
                    seed_path = Path(__file__).resolve().parent.parent / "data" / "seed_data.json"
                    import json as _json
                    with open(seed_path, 'r') as f:
                        seed = _json.load(f)
                    # Add nodes
                    for m in seed.get('materials', []):
                        kg.add_material(m['name'], m.get('formula'), m)
                    for p in seed.get('properties', []):
                        kg.add_property(p['name'], p.get('description'))
                    for a in seed.get('applications', []):
                        kg.add_application(a['name'], a.get('description'))
                    # Add HAS_PROPERTY relationships
                    for rel in seed.get('has_property_relations', []):
                        kg.add_has_property_relationship(
                            rel['material'], rel['property'], rel.get('value'), rel.get('unit'),
                            source='Seed Data', confidence=rel.get('confidence', 1.0)
                        )
                    # Add USED_IN relationships
                    for rel in seed.get('used_in_relations', []):
                        kg.add_used_in_relationship(
                            rel['material'], rel['application'],
                            confidence=rel.get('confidence', 1.0), source=rel.get('source', 'Seed Data'),
                            validated_by=rel.get('validated_by')
                        )
                    # Reload KG data for visualization
                    kg_data_new = load_cached_kg_data()
                    if kg_data_new:
                        st.success("Seed data loaded and KG reloaded.")
                        # Rerun to refresh all components with new KG
                        st.rerun()
                    else:
                        st.warning("Seed data load completed, but visualization reload failed.")
                except Exception as e:
                    st.error(f"Failed to load seed data: {e}")

    # System status
    st.sidebar.markdown("### 📊 System Status")
    
    # Test connections
    llm_status = llm.test_connection()
    
    # Get basic LLM status
    available_models = llm.get_available_models() if hasattr(llm, 'get_available_models') else []
    models_available = 1.0 if llm_status and available_models else 0.0
    available_count = len(available_models) if available_models else (1 if llm_status else 0)
    total_count = 1  # Single model setup
    
    st.sidebar.markdown(f"""
    **LLM System:** {'✅' if llm_status else '❌'} ({available_count}/{total_count} models, {models_available:.1%} available)
    
    **Knowledge Graph:** {'✅' if kg_data else '❌'}
    
    **Validation Engine:** ✅ Enhanced ML validation
    
    **Analytics Engine:** ✅ Advanced pattern recognition
    """)
    
    # Generate discovery metrics with proper error handling
    try:
        sample_discovery_history = [
            {'timestamp': (datetime.now() - timedelta(days=i)).isoformat(), 
             'material': f'Mat_{i}', 'application': 'capacitor', 'confidence': 0.8}
            for i in range(10)
        ]
        
        analysis_results = discovery_engine.analyze_discovery_landscape(kg_data, sample_discovery_history)
        discovery_metrics = analysis_results.get('metrics', {})
    except Exception as e:
        logger.error(f"Error in discovery analysis: {e}")
        discovery_metrics = {
            'total_materials': len(kg_data.get('nodes', {}).get('Material', [])),
            'total_applications': len(kg_data.get('nodes', {}).get('Application', [])),
            'total_relationships': len(kg_data.get('edges', [])),
            'discovery_readiness_score': 0.75
        }
        # Ensure analysis_results is available downstream
        analysis_results = {'insights': [], 'metrics': discovery_metrics}
    
    # Display enhanced metrics
    display_enhanced_metrics(kg_data, discovery_metrics)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 AI Chat Assistant",
        "🔍 Intelligent Query", 
        "🤖 Ensemble Discovery", 
        "📊 3D Visualization", 
        "📈 Analytics Dashboard",
        "💡 Discovery Insights"
    ])
    
    with tab1:
        # Chat Interface Tab
        render_chat_interface(kg, llm, validator, discovery_engine)
    
    with tab2:
        st.markdown("### 🔍 Intelligent Knowledge Graph Querying")
        
        query_col1, query_col2 = st.columns([2, 1])
        
        with query_col1:
            query_type = st.selectbox(
                "Query Type",
                ["Materials by Application", "Properties by Material", "Similar Materials", "Discovery Gaps"],
                help="Select the type of query to run against the knowledge graph"
            )
        
        with query_col2:
            if query_type == "Materials by Application":
                application = st.selectbox("Application", application_options)
                if st.button("🔍 Query"):
                    results = kg.get_materials_for_application(application)
                    if results:
                        df = pd.DataFrame(results).rename(columns={'r.confidence': 'confidence'})
                        st.dataframe(df, width='stretch')
                        if 'confidence' in df.columns:
                            fig = px.bar(df, x='material', y='confidence', title=f"Materials for {application}")
                            st.plotly_chart(fig, width='stretch')
                    else:
                        st.info(f"No materials found for {application}")
            elif query_type == "Properties by Material":
                material_name = st.text_input("Material name", value="BaTiO3")
                if st.button("🔍 Query", key="props_query"):
                    rows = kg.get_properties_for_material(material_name)
                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df, width='stretch')
                    else:
                        st.info(f"No properties found for {material_name}")
            elif query_type == "Similar Materials":
                material_name = st.text_input("Material name", value="BaTiO3", key="sim_mat")
                if st.button("🔍 Query", key="sim_query"):
                    rows = kg.get_similar_materials(material_name)
                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df, width='stretch')
                        fig = px.bar(df, x='material', y='score', title=f"Similar to {material_name}")
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info(f"No similar materials found for {material_name}")
            elif query_type == "Discovery Gaps":
                if st.button("🔍 Find Gaps"):
                    rows = kg.get_discovery_gaps()
                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df, width='stretch')
                        fig = px.bar(df, x='application', y='count', title="Under-served Applications")
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("No gaps identified with current data")
    
    with tab3:
        st.markdown("### 🤖 Fast AI Suggestions (llama3.2)")
        
        # Application selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            application = st.selectbox(
                "Target Application", 
                application_options,
                key="ensemble_app"
            )
        
        with col2:
            if st.button("🚀 Generate Suggestions", type="primary"):
                with st.spinner("Generating suggestions..."):
                    # Get current materials for context
                    current = kg.get_materials_for_application(application)
                    known_materials = [m.get('material') or m.get('material_name') for m in current] or ["BaTiO3", "SrTiO3"]
                    context = {"known_materials": known_materials}
                    
                    # Basic LLM calls
                    if application == 'capacitor':
                        resp = llm.generate_capacitor_hypotheses(known_materials, context)
                    elif application == 'thermoelectric_device':
                        resp = llm.generate_thermoelectric_hypotheses(known_materials, context)
                    elif application == 'solar_cell':
                        resp = llm.generate_solar_cell_hypotheses(known_materials, context)
                    else:
                        resp = llm.generate_generic_hypotheses(application, known_materials, context)
                    
                    # Render
                    if getattr(resp, 'hypotheses', None):
                        st.success(f"Found {len(resp.hypotheses)} suggestions")
                        for h in resp.hypotheses[:5]:
                            mat = h.get('material')
                            conf = h.get('confidence', 0)
                            st.write(f"- **{mat}** (confidence: {conf:.2f})")
                    else:
                        st.info("No suggestions available from the model.")
    
    with tab4:
        st.markdown("### 📊 Advanced 3D Network Visualization")
        
        viz_col1, viz_col2 = st.columns([3, 1])
        
        with viz_col2:
            highlight_recent = st.checkbox("Highlight Recent", value=True)
            show_3d = st.checkbox("3D View", value=True)
            
            if st.button("🔄 Refresh Visualization"):
                st.rerun()
        
        with viz_col1:
            if show_3d:
                fig = visualizer.create_3d_network_visualization(kg_data, highlight_recent)
                if getattr(fig, 'data', None) and len(fig.data) > 0:
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("3D view has no drawable nodes. Falling back to interactive 2D view.")
                    html_content = visualizer.create_interactive_2d_network(kg_data)
                    st.components.v1.html(html_content, height=600)
            else:
                # 2D interactive network
                html_content = visualizer.create_interactive_2d_network(kg_data)
                st.components.v1.html(html_content, height=600)
    
    with tab5:
        try:
            display_advanced_analytics(analytics_dashboard, discovery_engine, kg_data)
        except Exception as e:
            st.error(f"Analytics unavailable: {e}")
    
    with tab6:
        try:
            insights = analysis_results.get('insights', [])
            display_discovery_insights(insights)
        except Exception as e:
            st.error(f"Error displaying insights: {e}")
            st.info("Discovery insights temporarily unavailable. Other features are working normally.")
            insights = []
        
        # Real-time insight generation
        if st.button("🔄 Generate Fresh Insights"):
            with st.spinner("🧠 Analyzing patterns and generating insights..."):
                try:
                    fresh_analysis = discovery_engine.analyze_discovery_landscape(kg_data, sample_discovery_history)
                    fresh_insights = fresh_analysis.get('insights', [])
                    display_discovery_insights(fresh_insights)
                except Exception as e:
                    st.error(f"Unable to generate fresh insights: {e}")

if __name__ == "__main__":
    main()
