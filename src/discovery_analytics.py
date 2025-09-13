"""
Discovery Analytics and Insights Engine for Materials Ontology Expansion
Advanced pattern recognition, trend analysis, and scientific discovery insights
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from collections import Counter, defaultdict
import logging
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class DiscoveryInsight:
    """Represents a discovery insight"""
    insight_type: str
    title: str
    description: str
    confidence: float
    evidence: Dict[str, Any]
    impact_score: float
    novelty_score: float
    actionable_recommendations: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class TrendAnalysis:
    """Represents trend analysis results"""
    trend_type: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable', 'volatile'
    strength: float  # 0-1
    time_period: str
    key_drivers: List[str]
    predictions: Dict[str, float]
    confidence_interval: Tuple[float, float]

@dataclass
class MaterialCluster:
    """Represents a cluster of similar materials"""
    cluster_id: int
    materials: List[str]
    centroid_properties: Dict[str, float]
    cluster_name: str
    similarity_score: float
    common_applications: List[str]
    discovery_potential: float

class AdvancedPatternRecognizer:
    """Advanced pattern recognition for materials discovery"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.pattern_cache = {}
        
    def identify_material_clusters(self, materials_data: List[Dict[str, Any]], 
                                 n_clusters: Optional[int] = None) -> List[MaterialCluster]:
        """Identify clusters of similar materials"""
        
        if len(materials_data) < 3:
            return []
        
        try:
            # Prepare feature matrix
            feature_matrix, material_names = self._prepare_feature_matrix(materials_data)
            
            if feature_matrix is None or len(feature_matrix) < 3:
                return []
            
            # Determine optimal number of clusters
            if n_clusters is None:
                n_clusters = min(max(2, len(materials_data) // 3), 8)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix)
            
            # Create MaterialCluster objects
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_materials = [material_names[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_materials) < 2:
                    continue
                
                # Calculate centroid properties
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                centroid = np.mean(feature_matrix[cluster_indices], axis=0)
                
                # Extract centroid properties (simplified)
                centroid_properties = self._extract_centroid_properties(centroid, materials_data)
                
                # Calculate similarity score
                similarity_score = self._calculate_cluster_similarity(feature_matrix[cluster_indices])
                
                # Identify common applications
                common_applications = self._find_common_applications(cluster_materials, materials_data)
                
                # Generate cluster name
                cluster_name = self._generate_cluster_name(cluster_materials, centroid_properties)
                
                # Calculate discovery potential
                discovery_potential = self._calculate_discovery_potential(cluster_materials, materials_data)
                
                cluster = MaterialCluster(
                    cluster_id=cluster_id,
                    materials=cluster_materials,
                    centroid_properties=centroid_properties,
                    cluster_name=cluster_name,
                    similarity_score=similarity_score,
                    common_applications=common_applications,
                    discovery_potential=discovery_potential
                )
                
                clusters.append(cluster)
            
            # Sort by discovery potential
            clusters.sort(key=lambda x: x.discovery_potential, reverse=True)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error in material clustering: {e}")
            return []
    
    def _prepare_feature_matrix(self, materials_data: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], List[str]]:
        """Prepare feature matrix for clustering"""
        
        # Extract numeric features
        numeric_features = []
        material_names = []
        
        for material in materials_data:
            material_name = material.get('name', material.get('id', ''))
            if not material_name:
                continue
                
            features = []
            
            # Extract numeric properties
            for key, value in material.items():
                if key not in ['name', 'id', 'type', 'formula'] and isinstance(value, (int, float)):
                    features.append(value)
            
            if len(features) > 0:
                numeric_features.append(features)
                material_names.append(material_name)
        
        if len(numeric_features) < 2:
            return None, []
        
        # Pad features to same length
        max_length = max(len(f) for f in numeric_features)
        padded_features = []
        
        for features in numeric_features:
            padded = features + [0] * (max_length - len(features))
            padded_features.append(padded)
        
        feature_matrix = np.array(padded_features)
        
        # Scale features
        try:
            feature_matrix = self.scaler.fit_transform(feature_matrix)
        except:
            pass  # Continue with unscaled features if scaling fails
        
        return feature_matrix, material_names
    
    def _extract_centroid_properties(self, centroid: np.ndarray, 
                                   materials_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract meaningful properties from cluster centroid"""
        
        # This is simplified - in practice, you'd map back to original features
        properties = {}
        
        # Get common property names
        all_properties = set()
        for material in materials_data:
            for key, value in material.items():
                if isinstance(value, (int, float)) and key not in ['name', 'id']:
                    all_properties.add(key)
        
        # Map centroid values to properties (simplified)
        property_list = sorted(list(all_properties))
        for i, prop in enumerate(property_list[:len(centroid)]):
            properties[prop] = float(centroid[i])
        
        return properties
    
    def _calculate_cluster_similarity(self, cluster_features: np.ndarray) -> float:
        """Calculate similarity within cluster"""
        if len(cluster_features) < 2:
            return 0.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(cluster_features)):
            for j in range(i + 1, len(cluster_features)):
                sim = cosine_similarity([cluster_features[i]], [cluster_features[j]])[0][0]
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _find_common_applications(self, cluster_materials: List[str], 
                                materials_data: List[Dict[str, Any]]) -> List[str]:
        """Find common applications for materials in cluster"""
        
        application_counts = Counter()
        
        for material in materials_data:
            material_name = material.get('name', material.get('id', ''))
            if material_name in cluster_materials:
                applications = material.get('applications', [])
                if isinstance(applications, str):
                    applications = [applications]
                elif isinstance(applications, list):
                    pass
                else:
                    applications = []
                
                for app in applications:
                    application_counts[app] += 1
        
        # Return applications used by at least 50% of materials in cluster
        threshold = max(1, len(cluster_materials) // 2)
        common_apps = [app for app, count in application_counts.items() if count >= threshold]
        
        return common_apps
    
    def _generate_cluster_name(self, materials: List[str], 
                             centroid_properties: Dict[str, float]) -> str:
        """Generate descriptive name for cluster"""
        
        # Extract material types/families
        material_types = []
        for material in materials:
            if 'Ti' in material and 'O' in material:
                material_types.append('titanate')
            elif 'Te' in material or 'Se' in material:
                material_types.append('chalcogenide')
            elif 'Pb' in material and 'I' in material:
                material_types.append('perovskite')
            else:
                material_types.append('compound')
        
        # Find most common type
        type_counts = Counter(material_types)
        dominant_type = type_counts.most_common(1)[0][0] if type_counts else 'mixed'
        
        # Add property-based descriptor
        if 'dielectric_constant' in centroid_properties and centroid_properties['dielectric_constant'] > 100:
            return f"High-κ {dominant_type}s"
        elif 'zt' in centroid_properties and centroid_properties['zt'] > 1.0:
            return f"High-ZT {dominant_type}s"
        elif 'band_gap' in centroid_properties and 1.0 <= centroid_properties['band_gap'] <= 2.0:
            return f"Solar-active {dominant_type}s"
        else:
            return f"{dominant_type.capitalize()} cluster"
    
    def _calculate_discovery_potential(self, materials: List[str], 
                                     materials_data: List[Dict[str, Any]]) -> float:
        """Calculate discovery potential for cluster"""
        
        # Factors contributing to discovery potential:
        # 1. Cluster size (more materials = more patterns)
        # 2. Property diversity within cluster
        # 3. Number of unexplored applications
        # 4. Recency of additions
        
        size_factor = min(len(materials) / 5.0, 1.0)  # Normalize to max 5 materials
        
        # Property diversity (simplified)
        diversity_factor = 0.7  # Placeholder
        
        # Unexplored applications (simplified)
        exploration_factor = 0.8  # Placeholder
        
        # Recency factor
        recency_factor = 0.6  # Placeholder
        
        potential = (size_factor * 0.3 + diversity_factor * 0.3 + 
                    exploration_factor * 0.3 + recency_factor * 0.1)
        
        return potential

class TrendAnalyzer:
    """Analyze trends in materials discovery and properties"""
    
    def __init__(self):
        self.trend_cache = {}
        
    def analyze_discovery_trends(self, discovery_data: List[Dict[str, Any]], 
                               time_window: int = 30) -> List[TrendAnalysis]:
        """Analyze trends in materials discovery"""
        
        if len(discovery_data) < 5:
            return []
        
        try:
            df = pd.DataFrame(discovery_data)
            
            # Ensure timestamp column
            if 'timestamp' not in df.columns:
                return []
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            trends = []
            
            # Discovery rate trend
            rate_trend = self._analyze_discovery_rate_trend(df, time_window)
            if rate_trend:
                trends.append(rate_trend)
            
            # Confidence trend
            confidence_trend = self._analyze_confidence_trend(df, time_window)
            if confidence_trend:
                trends.append(confidence_trend)
            
            # Application diversity trend
            diversity_trend = self._analyze_application_diversity_trend(df, time_window)
            if diversity_trend:
                trends.append(diversity_trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return []
    
    def _analyze_discovery_rate_trend(self, df: pd.DataFrame, time_window: int) -> Optional[TrendAnalysis]:
        """Analyze trend in discovery rate"""
        
        # Group by day and count discoveries
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby('date').size()
        
        if len(daily_counts) < 3:
            return None
        
        # Calculate trend
        x = np.arange(len(daily_counts))
        y = daily_counts.values
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend direction
        if slope > 0.1:
            direction = 'increasing'
            strength = min(abs(slope), 1.0)
        elif slope < -0.1:
            direction = 'decreasing'
            strength = min(abs(slope), 1.0)
        else:
            direction = 'stable'
            strength = 1.0 - abs(slope)
        
        # Make prediction
        future_prediction = y[-1] + slope * 7  # 7 days ahead
        
        return TrendAnalysis(
            trend_type='discovery_rate',
            trend_direction=direction,
            strength=strength,
            time_period=f"Last {len(daily_counts)} days",
            key_drivers=['LLM hypothesis generation', 'Validation improvements'],
            predictions={'next_week': float(max(0, future_prediction))},
            confidence_interval=(float(max(0, future_prediction - 2)), 
                               float(future_prediction + 2))
        )
    
    def _analyze_confidence_trend(self, df: pd.DataFrame, time_window: int) -> Optional[TrendAnalysis]:
        """Analyze trend in validation confidence"""
        
        if 'confidence' not in df.columns:
            return None
        
        # Group by week and calculate average confidence
        df['week'] = df['timestamp'].dt.isocalendar().week
        weekly_confidence = df.groupby('week')['confidence'].mean()
        
        if len(weekly_confidence) < 2:
            return None
        
        # Calculate trend
        x = np.arange(len(weekly_confidence))
        y = weekly_confidence.values
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            direction = 'increasing'
            strength = min(abs(slope) * 10, 1.0)
        elif slope < -0.01:
            direction = 'decreasing'  
            strength = min(abs(slope) * 10, 1.0)
        else:
            direction = 'stable'
            strength = 1.0 - abs(slope) * 10
        
        return TrendAnalysis(
            trend_type='validation_confidence',
            trend_direction=direction,
            strength=strength,
            time_period=f"Last {len(weekly_confidence)} weeks",
            key_drivers=['Data quality improvements', 'Model refinements'],
            predictions={'trend_continuation': float(y[-1] + slope)},
            confidence_interval=(float(max(0, y[-1] + slope - 0.1)), 
                               float(min(1, y[-1] + slope + 0.1)))
        )
    
    def _analyze_application_diversity_trend(self, df: pd.DataFrame, time_window: int) -> Optional[TrendAnalysis]:
        """Analyze trend in application diversity"""
        
        if 'application' not in df.columns:
            return None
        
        # Calculate diversity over time
        df['week'] = df['timestamp'].dt.isocalendar().week
        weekly_diversity = df.groupby('week')['application'].nunique()
        
        if len(weekly_diversity) < 2:
            return None
        
        x = np.arange(len(weekly_diversity))
        y = weekly_diversity.values
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            direction = 'increasing'
        elif slope < -0.1:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        strength = min(abs(slope), 1.0)
        
        return TrendAnalysis(
            trend_type='application_diversity',
            trend_direction=direction,
            strength=strength,
            time_period=f"Last {len(weekly_diversity)} weeks",
            key_drivers=['Expanding research scope', 'Cross-domain discoveries'],
            predictions={'next_period': float(max(1, y[-1] + slope))},
            confidence_interval=(float(max(1, y[-1] + slope - 1)), 
                               float(y[-1] + slope + 1))
        )

class InsightGenerator:
    """Generate actionable insights from materials data"""
    
    def __init__(self):
        self.insight_templates = self._load_insight_templates()
        
    def _load_insight_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load templates for different types of insights"""
        return {
            'gap_analysis': {
                'title_template': "Potential gap in {application} materials",
                'description_template': "Analysis suggests limited materials for {application} with {property} > {threshold}",
                'impact_weight': 0.8
            },
            'cluster_opportunity': {
                'title_template': "Discovery opportunity in {cluster_name}",
                'description_template': "Material cluster '{cluster_name}' shows high discovery potential with {similarity:.1%} similarity",
                'impact_weight': 0.7
            },
            'trend_alert': {
                'title_template': "{trend_type} trend: {direction}",
                'description_template': "Detected {direction} trend in {trend_type} with {strength:.1%} strength",
                'impact_weight': 0.6
            },
            'validation_improvement': {
                'title_template': "Validation accuracy improvement detected",
                'description_template': "Recent validation improvements show {improvement:.1%} increase in accuracy",
                'impact_weight': 0.5
            }
        }
    
    def generate_insights(self, kg_data: Dict[str, Any], 
                         clusters: List[MaterialCluster],
                         trends: List[TrendAnalysis]) -> List[DiscoveryInsight]:
        """Generate comprehensive insights"""
        
        insights = []
        
        # Gap analysis insights
        gap_insights = self._generate_gap_insights(kg_data)
        insights.extend(gap_insights)
        
        # Cluster-based insights
        cluster_insights = self._generate_cluster_insights(clusters)
        insights.extend(cluster_insights)
        
        # Trend-based insights
        trend_insights = self._generate_trend_insights(trends)
        insights.extend(trend_insights)
        
        # Validation insights
        validation_insights = self._generate_validation_insights(kg_data)
        insights.extend(validation_insights)
        
        # Sort by impact score
        insights.sort(key=lambda x: x.impact_score, reverse=True)
        
        return insights[:10]  # Return top 10 insights
    
    def _generate_gap_insights(self, kg_data: Dict[str, Any]) -> List[DiscoveryInsight]:
        """Generate insights about gaps in materials coverage"""
        
        insights = []
        
        try:
            # Analyze application coverage
            applications = kg_data.get('applications', [])
            materials_per_app = {}
            
            for edge in kg_data.get('edges', []):
                if edge.get('relationship') == 'USED_IN':
                    app = edge.get('target', edge.get('to'))
                    if app not in materials_per_app:
                        materials_per_app[app] = 0
                    materials_per_app[app] += 1
            
            # Identify under-served applications
            avg_coverage = np.mean(list(materials_per_app.values())) if materials_per_app else 0
            
            for app, count in materials_per_app.items():
                if count < avg_coverage * 0.5:  # Less than 50% of average
                    insight = DiscoveryInsight(
                        insight_type='gap_analysis',
                        title=f"Limited materials for {app}",
                        description=f"Only {count} materials found for {app}, below average of {avg_coverage:.1f}",
                        confidence=0.8,
                        evidence={'material_count': count, 'average_coverage': avg_coverage},
                        impact_score=0.8,
                        novelty_score=0.6,
                        actionable_recommendations=[
                            f"Focus LLM hypothesis generation on {app} applications",
                            f"Search literature for additional {app} materials",
                            f"Explore similar applications for material transfer"
                        ],
                        related_entities=[app]
                    )
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating gap insights: {e}")
        
        return insights
    
    def _generate_cluster_insights(self, clusters: List[MaterialCluster]) -> List[DiscoveryInsight]:
        """Generate insights from material clusters"""
        
        insights = []
        
        for cluster in clusters[:3]:  # Top 3 clusters
            if cluster.discovery_potential > 0.6:
                insight = DiscoveryInsight(
                    insight_type='cluster_opportunity',
                    title=f"High-potential cluster: {cluster.cluster_name}",
                    description=f"Cluster of {len(cluster.materials)} materials with {cluster.discovery_potential:.1%} discovery potential",
                    confidence=cluster.similarity_score,
                    evidence={
                        'cluster_size': len(cluster.materials),
                        'similarity_score': cluster.similarity_score,
                        'common_applications': cluster.common_applications,
                        'centroid_properties': cluster.centroid_properties
                    },
                    impact_score=cluster.discovery_potential,
                    novelty_score=0.7,
                    actionable_recommendations=[
                        f"Generate hypotheses for materials similar to {cluster.cluster_name}",
                        f"Explore cross-application potential for {', '.join(cluster.materials[:3])}",
                        f"Investigate property optimization within {cluster.cluster_name}"
                    ],
                    related_entities=cluster.materials
                )
                insights.append(insight)
        
        return insights
    
    def _generate_trend_insights(self, trends: List[TrendAnalysis]) -> List[DiscoveryInsight]:
        """Generate insights from trend analysis"""
        
        insights = []
        
        for trend in trends:
            if trend.strength > 0.6:
                insight = DiscoveryInsight(
                    insight_type='trend_alert',
                    title=f"{trend.trend_type.replace('_', ' ').title()} trending {trend.trend_direction}",
                    description=f"Strong {trend.trend_direction} trend detected in {trend.trend_type} with {trend.strength:.1%} strength",
                    confidence=trend.strength,
                    evidence={
                        'trend_direction': trend.trend_direction,
                        'strength': trend.strength,
                        'time_period': trend.time_period,
                        'predictions': trend.predictions
                    },
                    impact_score=trend.strength * 0.6,
                    novelty_score=0.5,
                    actionable_recommendations=self._get_trend_recommendations(trend),
                    related_entities=trend.key_drivers
                )
                insights.append(insight)
        
        return insights
    
    def _generate_validation_insights(self, kg_data: Dict[str, Any]) -> List[DiscoveryInsight]:
        """Generate insights about validation performance"""
        
        insights = []
        
        try:
            # Analyze validation success rates
            edges = kg_data.get('edges', [])
            validated_edges = [e for e in edges if e.get('validated', False)]
            
            if len(edges) > 0:
                success_rate = len(validated_edges) / len(edges)
                
                if success_rate > 0.8:
                    insight = DiscoveryInsight(
                        insight_type='validation_improvement',
                        title="High validation success rate achieved",
                        description=f"Current validation success rate: {success_rate:.1%}",
                        confidence=0.9,
                        evidence={'success_rate': success_rate, 'total_hypotheses': len(edges)},
                        impact_score=0.7,
                        novelty_score=0.4,
                        actionable_recommendations=[
                            "Leverage high-confidence validation methods for new hypotheses",
                            "Document successful validation patterns for reuse",
                            "Consider expanding to more challenging materials"
                        ],
                        related_entities=[]
                    )
                    insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error generating validation insights: {e}")
        
        return insights
    
    def _get_trend_recommendations(self, trend: TrendAnalysis) -> List[str]:
        """Get recommendations based on trend type and direction"""
        
        recommendations = []
        
        if trend.trend_type == 'discovery_rate':
            if trend.trend_direction == 'increasing':
                recommendations.extend([
                    "Maintain current discovery momentum",
                    "Scale up validation resources to handle increased discoveries",
                    "Document successful discovery patterns"
                ])
            elif trend.trend_direction == 'decreasing':
                recommendations.extend([
                    "Investigate causes of declining discovery rate",
                    "Refresh LLM models or expand training data",
                    "Explore new application domains"
                ])
        
        elif trend.trend_type == 'validation_confidence':
            if trend.trend_direction == 'increasing':
                recommendations.extend([
                    "Document improved validation methods",
                    "Apply successful validation approaches to new domains"
                ])
            elif trend.trend_direction == 'decreasing':
                recommendations.extend([
                    "Review and update validation criteria",
                    "Investigate data quality issues"
                ])
        
        return recommendations

class MaterialsDiscoveryEngine:
    """Main engine combining all analytics components"""
    
    def __init__(self):
        self.pattern_recognizer = AdvancedPatternRecognizer()
        self.trend_analyzer = TrendAnalyzer()
        self.insight_generator = InsightGenerator()
        
    def analyze_discovery_landscape(self, kg_data: Dict[str, Any], 
                                  discovery_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive analysis of the discovery landscape"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'clusters': [],
            'trends': [],
            'insights': [],
            'recommendations': [],
            'metrics': {}
        }
        
        try:
            # Extract materials data
            materials_data = self._extract_materials_data(kg_data)
            
            # Identify material clusters
            clusters = self.pattern_recognizer.identify_material_clusters(materials_data)
            results['clusters'] = [self._cluster_to_dict(c) for c in clusters]
            
            # Analyze trends
            trends = self.trend_analyzer.analyze_discovery_trends(discovery_history)
            results['trends'] = [self._trend_to_dict(t) for t in trends]
            
            # Generate insights
            insights = self.insight_generator.generate_insights(kg_data, clusters, trends)
            results['insights'] = [self._insight_to_dict(i) for i in insights]
            
            # Calculate metrics
            results['metrics'] = self._calculate_discovery_metrics(kg_data, clusters, trends, insights)
            
            # Generate summary
            results['summary'] = self._generate_summary(clusters, trends, insights)
            
            # Generate recommendations
            results['recommendations'] = self._generate_top_recommendations(insights)
            
        except Exception as e:
            logger.error(f"Error in discovery landscape analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    def _extract_materials_data(self, kg_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract materials data from knowledge graph"""
        
        materials_data = []
        
        # Handle different data formats
        if isinstance(kg_data, list):
            # Convert list format
            nodes_dict = {}
            for item in kg_data:
                if isinstance(item, dict) and 'nodes' in item:
                    nodes_dict.update(item['nodes'])
            materials = nodes_dict.get('Material', [])
        else:
            # Get materials from nodes
            materials = kg_data.get('nodes', {}).get('Material', [])
        
        for material in materials:
            material_dict = material.copy()
            
            # Add properties from edges
            material_name = material.get('name', material.get('id'))
            for edge in kg_data.get('edges', []):
                if edge.get('source') == material_name and edge.get('relationship') == 'HAS_PROPERTY':
                    prop_name = edge.get('target')
                    prop_value = edge.get('value')
                    if prop_name and prop_value is not None:
                        material_dict[prop_name] = prop_value
            
            # Add applications
            applications = []
            for edge in kg_data.get('edges', []):
                if edge.get('source') == material_name and edge.get('relationship') == 'USED_IN':
                    applications.append(edge.get('target'))
            
            material_dict['applications'] = applications
            materials_data.append(material_dict)
        
        return materials_data
    
    def _cluster_to_dict(self, cluster: MaterialCluster) -> Dict[str, Any]:
        """Convert MaterialCluster to dictionary"""
        return {
            'cluster_id': cluster.cluster_id,
            'materials': cluster.materials,
            'centroid_properties': cluster.centroid_properties,
            'cluster_name': cluster.cluster_name,
            'similarity_score': cluster.similarity_score,
            'common_applications': cluster.common_applications,
            'discovery_potential': cluster.discovery_potential
        }
    
    def _trend_to_dict(self, trend: TrendAnalysis) -> Dict[str, Any]:
        """Convert TrendAnalysis to dictionary"""
        return {
            'trend_type': trend.trend_type,
            'trend_direction': trend.trend_direction,
            'strength': trend.strength,
            'time_period': trend.time_period,
            'key_drivers': trend.key_drivers,
            'predictions': trend.predictions,
            'confidence_interval': trend.confidence_interval
        }
    
    def _insight_to_dict(self, insight: DiscoveryInsight) -> Dict[str, Any]:
        """Convert DiscoveryInsight to dictionary"""
        return {
            'insight_type': insight.insight_type,
            'title': insight.title,
            'description': insight.description,
            'confidence': insight.confidence,
            'evidence': insight.evidence,
            'impact_score': insight.impact_score,
            'novelty_score': insight.novelty_score,
            'actionable_recommendations': insight.actionable_recommendations,
            'related_entities': insight.related_entities,
            'timestamp': insight.timestamp
        }
    
    def _calculate_discovery_metrics(self, kg_data: Dict[str, Any], 
                                   clusters: List[MaterialCluster],
                                   trends: List[TrendAnalysis],
                                   insights: List[DiscoveryInsight]) -> Dict[str, Any]:
        """Calculate key discovery metrics"""
        
        return {
            'total_materials': len(kg_data.get('nodes', {}).get('Material', [])),
            'total_applications': len(kg_data.get('nodes', {}).get('Application', [])),
            'total_relationships': len(kg_data.get('edges', [])),
            'identified_clusters': len(clusters),
            'high_potential_clusters': len([c for c in clusters if c.discovery_potential > 0.7]),
            'active_trends': len(trends),
            'actionable_insights': len([i for i in insights if len(i.actionable_recommendations) > 0]),
            'average_insight_impact': np.mean([i.impact_score for i in insights]) if insights else 0.0,
            'discovery_readiness_score': self._calculate_readiness_score(kg_data, clusters, trends)
        }
    
    def _calculate_readiness_score(self, kg_data: Dict[str, Any], 
                                 clusters: List[MaterialCluster],
                                 trends: List[TrendAnalysis]) -> float:
        """Calculate overall discovery readiness score"""
        
        # Factors: data completeness, cluster quality, trend strength
        data_completeness = min(len(kg_data.get('edges', [])) / 100, 1.0)  # Normalize to 100 edges
        
        cluster_quality = np.mean([c.discovery_potential for c in clusters]) if clusters else 0.0
        
        trend_strength = np.mean([t.strength for t in trends]) if trends else 0.0
        
        readiness = (data_completeness * 0.4 + cluster_quality * 0.4 + trend_strength * 0.2)
        
        return float(readiness)
    
    def _generate_summary(self, clusters: List[MaterialCluster],
                         trends: List[TrendAnalysis],
                         insights: List[DiscoveryInsight]) -> Dict[str, str]:
        """Generate executive summary"""
        
        summary = {}
        
        # Cluster summary
        if clusters:
            top_cluster = max(clusters, key=lambda x: x.discovery_potential)
            summary['top_opportunity'] = f"Highest discovery potential in {top_cluster.cluster_name} ({top_cluster.discovery_potential:.1%})"
        
        # Trend summary
        strong_trends = [t for t in trends if t.strength > 0.7]
        if strong_trends:
            summary['key_trend'] = f"Strong {strong_trends[0].trend_direction} trend in {strong_trends[0].trend_type}"
        
        # Insight summary
        if insights:
            top_insight = max(insights, key=lambda x: x.impact_score)
            summary['priority_action'] = top_insight.title
        
        return summary
    
    def _generate_top_recommendations(self, insights: List[DiscoveryInsight]) -> List[str]:
        """Generate top actionable recommendations"""
        
        all_recommendations = []
        
        for insight in insights:
            for rec in insight.actionable_recommendations:
                all_recommendations.append({
                    'recommendation': rec,
                    'impact_score': insight.impact_score,
                    'confidence': insight.confidence
                })
        
        # Sort by weighted score
        all_recommendations.sort(key=lambda x: x['impact_score'] * x['confidence'], reverse=True)
        
        # Return top 5 unique recommendations
        seen = set()
        top_recommendations = []
        
        for rec in all_recommendations:
            if rec['recommendation'] not in seen and len(top_recommendations) < 5:
                top_recommendations.append(rec['recommendation'])
                seen.add(rec['recommendation'])
        
        return top_recommendations
