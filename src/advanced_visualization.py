"""
Advanced Visualization and Analytics for Materials Ontology Expansion
Features interactive 3D networks, real-time updates, and comprehensive analytics dashboards
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, timedelta
import pyvis.network as pvnet
import streamlit.components.v1 as components
from dataclasses import dataclass
import colorsys
import math

@dataclass
class VisualizationConfig:
    """Configuration for visualization components"""
    node_size_range: Tuple[int, int] = (10, 50)
    edge_width_range: Tuple[int, int] = (1, 5)
    color_scheme: str = "viridis"
    layout_algorithm: str = "spring"
    animation_duration: int = 1000
    show_labels: bool = True
    interactive: bool = True

class AdvancedNetworkVisualizer:
    """Advanced network visualization with 3D support and real-time updates"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.color_maps = self._initialize_color_maps()
        
    def _initialize_color_maps(self) -> Dict[str, Dict[str, str]]:
        """Initialize color maps for different node types"""
        return {
            'node_types': {
                'Material': '#FF6B6B',      # Red
                'Property': '#4ECDC4',      # Teal
                'Application': '#45B7D1',   # Blue
                'Hypothesis': '#FFA07A',    # Light salmon
                'Validated': '#98D8C8'      # Mint green
            },
            'applications': {
                'capacitor': '#FF6B6B',
                'thermoelectric_device': '#4ECDC4',
                'solar_cell': '#45B7D1',
                'battery': '#96CEB4',
                'sensor': '#FFEAA7'
            },
            'confidence': {
                'high': '#00B894',      # Green
                'medium': '#FDCB6E',    # Yellow
                'low': '#E17055'        # Red
            }
        }
    
    def create_3d_network_visualization(self, kg_data: Dict[str, Any], 
                                      highlight_recent: bool = True) -> go.Figure:
        """Create advanced 3D network visualization"""
        
        # Build NetworkX graph
        G = self._build_networkx_graph(kg_data)
        
        # Calculate 3D layout
        pos_3d = self._calculate_3d_layout(G)
        
        # Create node traces
        node_traces = self._create_3d_node_traces(G, pos_3d, highlight_recent)
        
        # Create edge traces
        edge_traces = self._create_3d_edge_traces(G, pos_3d)
        
        # Combine traces
        fig = go.Figure(data=node_traces + edge_traces)
        
        # Update layout for MUCH BETTER VISIBILITY
        fig.update_layout(
            title={
                'text': "3D Materials Knowledge Graph",
                'x': 0.5,
                'font': {'size': 28, 'color': '#2E86AB', 'family': 'Arial Black'}
            },
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, showticklabels=False, showbackground=True, backgroundcolor='rgba(230,230,250,0.3)'),
                yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, showticklabels=False, showbackground=True, backgroundcolor='rgba(230,230,250,0.3)'),
                zaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, showticklabels=False, showbackground=True, backgroundcolor='rgba(230,230,250,0.3)'),
                bgcolor='rgba(245,245,245,0.8)',  # Light background for contrast
                camera=dict(
                    eye=dict(x=2.0, y=2.0, z=1.8),  # Better viewing angle
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='cube'  # Equal aspect ratio
            ),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='gray',
                borderwidth=1,
                font=dict(size=14, family='Arial')
            ),
            margin=dict(l=0, r=0, t=60, b=0),
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=700  # Taller for better view
        )
        
        return fig
    
    def _build_networkx_graph(self, kg_data: Dict[str, Any]) -> nx.Graph:
        """Build NetworkX graph from knowledge graph data"""
        G = nx.Graph()
        
        # Handle both dict and list formats for kg_data
        if isinstance(kg_data, list):
            # Convert list format to dict format
            nodes_dict = {}
            edges_list = []
            
            for item in kg_data:
                if isinstance(item, dict):
                    if 'nodes' in item:
                        nodes_dict = item['nodes']
                    if 'edges' in item:
                        edges_list = item['edges']
            
            kg_data = {'nodes': nodes_dict, 'edges': edges_list}
        
        # Add nodes
        nodes_data = kg_data.get('nodes', {})
        if isinstance(nodes_data, dict):
            for node_type, nodes in nodes_data.items():
                for node in nodes:
                    node_id = node.get('id', node.get('name', ''))
                    G.add_node(node_id, 
                              type=node_type,
                              properties=node,
                              size=self._calculate_node_size(node),
                              color=self.color_maps['node_types'].get(node_type, '#CCCCCC'))
        
        # Add edges
        for edge in kg_data.get('edges', []):
            source = edge.get('source', edge.get('from'))
            target = edge.get('target', edge.get('to'))
            if source and target:
                G.add_edge(source, target, 
                          properties=edge,
                          weight=edge.get('confidence', 1.0),
                          width=self._calculate_edge_width(edge))
        
        return G
    
    def _calculate_3d_layout(self, G: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Calculate 3D layout positions for nodes"""
        
        if len(G.nodes()) == 0:
            return {}
        
        # Start with 2D spring layout
        pos_2d = nx.spring_layout(G, k=1, iterations=50)
        
        # Extend to 3D based on node types
        pos_3d = {}
        node_types = set(nx.get_node_attributes(G, 'type').values())
        type_z_levels = {node_type: i for i, node_type in enumerate(node_types)}
        
        for node, (x, y) in pos_2d.items():
            node_type = G.nodes[node].get('type', 'unknown')
            z = type_z_levels.get(node_type, 0) * 0.5
            
            # Add some randomness to z for visual appeal
            z += np.random.normal(0, 0.1)
            
            pos_3d[node] = (x, y, z)
        
        return pos_3d
    
    def _create_3d_node_traces(self, G: nx.Graph, pos_3d: Dict[str, Tuple[float, float, float]],
                              highlight_recent: bool) -> List[go.Scatter3d]:
        """Create 3D node traces"""
        traces = []
        
        # Group nodes by type
        node_groups = {}
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'unknown')
            if node_type not in node_groups:
                node_groups[node_type] = []
            node_groups[node_type].append((node, data))
        
        # Create trace for each node type
        for node_type, nodes in node_groups.items():
            x_coords, y_coords, z_coords = [], [], []
            node_text, node_sizes, node_colors = [], [], []
            
            for node, data in nodes:
                if node in pos_3d:
                    x, y, z = pos_3d[node]
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)
                    
                    # Node text with properties
                    properties = data.get('properties', {})
                    text_lines = [f"<b>{node}</b>", f"Type: {node_type}"]
                    
                    for key, value in properties.items():
                        if key not in ['id', 'name', 'type'] and value is not None:
                            text_lines.append(f"{key}: {value}")
                    
                    node_text.append("<br>".join(text_lines))
                    node_sizes.append(data.get('size', 20))
                    
                    # Color based on recency if highlighting
                    if highlight_recent and 'timestamp' in properties:
                        node_colors.append(self._get_recency_color(properties['timestamp']))
                    else:
                        node_colors.append(data.get('color', self.color_maps['node_types'].get(node_type, '#CCCCCC')))
            
            if x_coords:  # Only create trace if we have data
                trace = go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers+text',
                    name=node_type,
                    text=[node.split('_')[0][:8] if len(node) > 8 else node for node, _ in nodes],  # Show short labels
                    textposition="middle center",
                    textfont=dict(size=14, color='black', family='Arial Black'),  # Bigger text for bigger nodes
                    hovertemplate='%{text}<extra></extra>',
                    marker=dict(
                        size=[max(45, s) for s in node_sizes],  # Much bigger nodes to fit text
                        color=node_colors,
                        opacity=0.9,  # More opaque
                        line=dict(width=3, color='black'),  # Black border for contrast
                        symbol='circle'
                    )
                )
                traces.append(trace)
        
        return traces
    
    def _create_3d_edge_traces(self, G: nx.Graph, pos_3d: Dict[str, Tuple[float, float, float]]) -> List[go.Scatter3d]:
        """Create 3D edge traces"""
        edge_traces = []
        
        # Group edges by type/confidence
        edge_groups = {'high': [], 'medium': [], 'low': []}
        
        for edge in G.edges(data=True):
            source, target, data = edge
            if source in pos_3d and target in pos_3d:
                confidence = data.get('properties', {}).get('confidence', 0.5)
                
                # Handle None confidence values
                if confidence is None:
                    confidence = 0.5
                
                # Ensure confidence is a number
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.5
                
                if confidence > 0.8:
                    edge_groups['high'].append(edge)
                elif confidence > 0.5:
                    edge_groups['medium'].append(edge)
                else:
                    edge_groups['low'].append(edge)
        
        # Create traces for each confidence level
        colors = {'high': '#00B894', 'medium': '#FDCB6E', 'low': '#E17055'}
        widths = {'high': 8, 'medium': 6, 'low': 4}  # Much thicker, more visible lines
        
        for confidence_level, edges in edge_groups.items():
            if not edges:
                continue
                
            x_coords, y_coords, z_coords = [], [], []
            
            for source, target, data in edges:
                x0, y0, z0 = pos_3d[source]
                x1, y1, z1 = pos_3d[target]
                
                # Add edge coordinates
                x_coords.extend([x0, x1, None])
                y_coords.extend([y0, y1, None])
                z_coords.extend([z0, z1, None])
            
            trace = go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                name=f'{confidence_level.capitalize()} Confidence',
                line=dict(
                    color=colors[confidence_level],
                    width=widths[confidence_level]
                ),
                hoverinfo='none',
                showlegend=True
            )
            edge_traces.append(trace)
        
        return edge_traces
    
    def _calculate_node_size(self, node: Dict[str, Any]) -> int:
        """Calculate node size based on properties"""
        base_size = 20
        
        # Size based on connections (if available)
        connections = node.get('connections', 1)
        size_factor = min(math.log(connections + 1), 3)  # Cap at 3x
        
        # Size based on confidence
        confidence = node.get('confidence', 0.5)
        confidence_factor = 0.5 + confidence
        
        return int(base_size * size_factor * confidence_factor)
    
    def _calculate_edge_width(self, edge: Dict[str, Any]) -> int:
        """Calculate edge width based on confidence"""
        confidence = edge.get('confidence', 0.5)
        if confidence is None:
            confidence = 0.5
        
        min_width, max_width = self.config.edge_width_range
        
        width = min_width + (max_width - min_width) * confidence
        return int(width)
    
    def _get_recency_color(self, timestamp: str) -> str:
        """Get color based on recency of the data"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            now = datetime.now()
            age_hours = (now - dt).total_seconds() / 3600
            
            if age_hours < 24:
                return '#00B894'  # Green for very recent
            elif age_hours < 168:  # 1 week
                return '#FDCB6E'  # Yellow for recent
            else:
                return '#E17055'  # Red for older
        except:
            return '#CCCCCC'  # Gray for unknown
    
    def create_interactive_2d_network(self, kg_data: Dict[str, Any]) -> str:
        """Create interactive 2D network using PyVis"""
        
        net = pvnet.Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "hideEdgesOnDrag": true
            }
        }
        """)
        
        # Add nodes
        for node_type, nodes in kg_data.get('nodes', {}).items():
            for node in nodes:
                node_id = node.get('id', node.get('name', ''))
                
                # Create detailed tooltip
                tooltip_lines = [f"<b>{node_id}</b>", f"Type: {node_type}"]
                for key, value in node.items():
                    if key not in ['id', 'name'] and value is not None:
                        tooltip_lines.append(f"{key}: {value}")
                
                net.add_node(
                    node_id,
                    label=node_id,
                    title="<br>".join(tooltip_lines),
                    color=self.color_maps['node_types'].get(node_type, '#CCCCCC'),
                    size=self._calculate_node_size(node),
                    physics=True
                )
        
        # Add edges
        for edge in kg_data.get('edges', []):
            source = edge.get('source', edge.get('from'))
            target = edge.get('target', edge.get('to'))
            
            if source and target:
                confidence = edge.get('confidence', 0.5)
                
                net.add_edge(
                    source,
                    target,
                    width=self._calculate_edge_width(edge),
                    color=self._get_confidence_color(confidence),
                    title=f"Confidence: {confidence:.2f}"
                )
        
        # Generate HTML
        html = net.generate_html()
        return html
    
    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence level"""
        # Handle None confidence values
        if confidence is None:
            confidence = 0.5
        
        # Ensure confidence is a number
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        
        if confidence > 0.8:
            return '#00B894'  # Green
        elif confidence > 0.5:
            return '#FDCB6E'  # Yellow
        else:
            return '#E17055'  # Red

class MaterialsAnalyticsDashboard:
    """Comprehensive analytics dashboard for materials data"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        
    def create_property_distribution_chart(self, materials_data: List[Dict[str, Any]]) -> go.Figure:
        """Create property distribution charts"""
        
        df = pd.DataFrame(materials_data)
        
        # Identify numeric properties
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            # Fallback to simple bar chart
            return self._create_simple_bar_chart(df)
        
        # Create subplots
        n_cols = min(len(numeric_cols), 3)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=numeric_cols,
            specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            # Create histogram
            values = df[col].dropna()
            
            if len(values) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=values,
                        name=col,
                        nbinsx=min(20, len(values)),
                        opacity=0.7,
                        marker_color=self.color_palette[i % len(self.color_palette)]
                    ),
                    row=row, col=col_idx
                )
        
        fig.update_layout(
            title="Property Distributions",
            showlegend=False,
            height=300 * n_rows
        )
        
        return fig
    
    def create_materials_comparison_radar(self, materials: List[str], 
                                        properties_data: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create radar chart comparing materials"""
        
        if not materials or not properties_data:
            return go.Figure().add_annotation(
                text="No data available for comparison",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Get common properties
        all_properties = set()
        for material in materials:
            if material in properties_data:
                all_properties.update(properties_data[material].keys())
        
        common_properties = list(all_properties)[:8]  # Limit to 8 properties for readability
        
        fig = go.Figure()
        
        for i, material in enumerate(materials):
            if material in properties_data:
                values = []
                for prop in common_properties:
                    value = properties_data[material].get(prop, 0)
                    values.append(value)
                
                # Normalize values for radar chart
                if values:
                    max_val = max(values) if max(values) > 0 else 1
                    normalized_values = [v / max_val for v in values]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=normalized_values + [normalized_values[0]],  # Close the polygon
                        theta=common_properties + [common_properties[0]],
                        fill='toself',
                        name=material,
                        line_color=self.color_palette[i % len(self.color_palette)]
                    ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Materials Property Comparison"
        )
        
        return fig
    
    def create_discovery_timeline(self, discovery_data: List[Dict[str, Any]]) -> go.Figure:
        """Create timeline of discoveries/additions"""
        
        if not discovery_data:
            return go.Figure().add_annotation(
                text="No discovery data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        df = pd.DataFrame(discovery_data)
        
        # Ensure we have timestamp data
        if 'timestamp' not in df.columns:
            return self._create_simple_bar_chart(df)
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Group by date and count discoveries
        daily_counts = df.groupby('date').size().reset_index(name='count')
        
        fig = go.Figure()
        
        # Add bar chart for daily discoveries
        fig.add_trace(go.Bar(
            x=daily_counts['date'],
            y=daily_counts['count'],
            name='Daily Discoveries',
            marker_color='#4ECDC4'
        ))
        
        # Add cumulative line
        daily_counts['cumulative'] = daily_counts['count'].cumsum()
        
        fig.add_trace(go.Scatter(
            x=daily_counts['date'],
            y=daily_counts['cumulative'],
            mode='lines+markers',
            name='Cumulative Discoveries',
            yaxis='y2',
            line=dict(color='#FF6B6B', width=3)
        ))
        
        # Update layout
        fig.update_layout(
            title="Discovery Timeline",
            xaxis_title="Date",
            yaxis_title="Daily Discoveries",
            yaxis2=dict(
                title="Cumulative Discoveries",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified'
        )
        
        return fig
    
    def create_confidence_analysis(self, hypotheses_data: List[Dict[str, Any]]) -> go.Figure:
        """Create confidence analysis charts"""
        
        if not hypotheses_data:
            return go.Figure().add_annotation(
                text="No hypothesis data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        df = pd.DataFrame(hypotheses_data)
        
        # Create subplot with confidence distribution and validation status
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Confidence Distribution', 'Validation Status'),
            specs=[[{"type": "histogram"}, {"type": "pie"}]]
        )
        
        # Confidence distribution
        if 'confidence' in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df['confidence'],
                    nbinsx=20,
                    name='Confidence',
                    marker_color='#4ECDC4'
                ),
                row=1, col=1
            )
        
        # Validation status pie chart
        if 'validated' in df.columns:
            status_counts = df['validated'].value_counts()
            
            fig.add_trace(
                go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    name='Validation Status'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title="Hypothesis Confidence Analysis",
            showlegend=False
        )
        
        return fig
    
    def _create_simple_bar_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create simple bar chart fallback"""
        
        if 'name' in df.columns:
            fig = go.Figure(data=[
                go.Bar(x=df['name'], y=df.index, marker_color='#4ECDC4')
            ])
            fig.update_layout(title="Materials Count", xaxis_title="Materials", yaxis_title="Count")
        else:
            fig = go.Figure().add_annotation(
                text="Insufficient data for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return fig
    
    def create_application_performance_matrix(self, performance_data: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create heatmap showing material performance across applications"""
        
        if not performance_data:
            return go.Figure().add_annotation(
                text="No performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Convert to matrix format
        materials = list(performance_data.keys())
        applications = list(set().union(*(d.keys() for d in performance_data.values())))
        
        matrix = []
        for material in materials:
            row = []
            for app in applications:
                value = performance_data[material].get(app, 0)
                row.append(value)
            matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=applications,
            y=materials,
            colorscale='Viridis',
            colorbar=dict(title="Performance Score")
        ))
        
        fig.update_layout(
            title="Material-Application Performance Matrix",
            xaxis_title="Applications",
            yaxis_title="Materials"
        )
        
        return fig

class RealTimeUpdater:
    """Handle real-time updates for visualizations"""
    
    def __init__(self):
        self.update_callbacks = {}
        self.last_update = datetime.now()
    
    def register_callback(self, component_id: str, callback_func):
        """Register callback for real-time updates"""
        self.update_callbacks[component_id] = callback_func
    
    def trigger_update(self, component_id: str, data: Any):
        """Trigger update for specific component"""
        if component_id in self.update_callbacks:
            try:
                self.update_callbacks[component_id](data)
                self.last_update = datetime.now()
            except Exception as e:
                st.error(f"Error updating {component_id}: {e}")
    
    def get_update_status(self) -> Dict[str, Any]:
        """Get status of real-time updates"""
        return {
            'registered_components': list(self.update_callbacks.keys()),
            'last_update': self.last_update.isoformat(),
            'update_frequency': '30s'  # Could be configurable
        }