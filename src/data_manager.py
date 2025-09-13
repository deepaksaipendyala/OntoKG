"""
Data Manager for Materials Ontology Expansion
Manages multiple data sources including MatKG, seed data, and external APIs
"""

import os
import json
from typing import Dict, List, Any, Optional, Iterator
from abc import ABC, abstractmethod
import logging

from config import SystemConfig, DataSourceConfig, MatKGConfig
from matkg_loader import MatKGLoader, MatKGEntity, MatKGRelationship
from knowledge_graph import MaterialsKG, Hypothesis

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.priority = config.priority
    
    @abstractmethod
    def load_entities(self) -> Iterator[Dict[str, Any]]:
        """Load entities from this data source"""
        pass
    
    @abstractmethod
    def load_relationships(self) -> Iterator[Dict[str, Any]]:
        """Load relationships from this data source"""
        pass
    
    @abstractmethod
    def validate_hypothesis(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        """Validate a hypothesis using this data source"""
        pass
    
    def get_priority(self) -> int:
        """Get priority of this data source"""
        return self.priority

class MatKGDataSource(DataSource):
    """MatKG data source implementation"""
    
    def __init__(self, config: DataSourceConfig, matkg_config: MatKGConfig):
        super().__init__(config)
        self.matkg_loader = MatKGLoader(matkg_config)
        self.entity_cache = {}
        self.relationship_cache = {}
    
    def load_entities(self) -> Iterator[Dict[str, Any]]:
        """Load entities from MatKG"""
        logger.info(f"Loading entities from {self.name}")
        
        for entity in self.matkg_loader.load_entities():
            yield {
                'name': entity.name,
                'uri': entity.uri,
                'type': entity.entity_type,
                'confidence': entity.confidence,
                'source': 'MatKG',
                'metadata': entity.metadata
            }
    
    def load_relationships(self) -> Iterator[Dict[str, Any]]:
        """Load relationships from MatKG"""
        logger.info(f"Loading relationships from {self.name}")
        
        for relationship in self.matkg_loader.load_relationships():
            yield {
                'subject': relationship.subject,
                'predicate': relationship.predicate,
                'object': relationship.object,
                'confidence': relationship.confidence,
                'source': 'MatKG',
                'source_papers': relationship.source_papers,
                'metadata': relationship.metadata
            }
    
    def validate_hypothesis(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        """Validate hypothesis against MatKG data"""
        # Search for the relationship in MatKG
        relationships = self.matkg_loader.get_relationships_for_entity(hypothesis.material)
        
        for rel in relationships:
            if (rel.predicate == hypothesis.relationship and 
                rel.object == hypothesis.application):
                return {
                    'is_valid': True,
                    'confidence': rel.confidence,
                    'source': 'MatKG',
                    'evidence': {
                        'matkg_confidence': rel.confidence,
                        'source_papers': rel.source_papers,
                        'relationship_found': True
                    }
                }
        
        return {
            'is_valid': False,
            'confidence': 0.0,
            'source': 'MatKG',
            'evidence': {
                'relationship_found': False,
                'reason': 'No matching relationship found in MatKG'
            }
        }

class SeedDataSource(DataSource):
    """Seed data source implementation"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.data_file = config.config.get('file', 'data/seed_data.json')
        self.data = self._load_seed_data()
    
    def _load_seed_data(self) -> Dict[str, Any]:
        """Load seed data from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Seed data file not found: {self.data_file}")
            return {'materials': [], 'properties': [], 'applications': [], 'relationships': []}
    
    def load_entities(self) -> Iterator[Dict[str, Any]]:
        """Load entities from seed data"""
        logger.info(f"Loading entities from {self.name}")
        
        # Load materials
        for material in self.data.get('materials', []):
            yield {
                'name': material['name'],
                'type': 'Material',
                'formula': material.get('formula'),
                'confidence': 1.0,
                'source': 'Seed Data',
                'metadata': material
            }
        
        # Load properties
        for property_data in self.data.get('properties', []):
            yield {
                'name': property_data['name'],
                'type': 'Property',
                'description': property_data.get('description'),
                'confidence': 1.0,
                'source': 'Seed Data',
                'metadata': property_data
            }
        
        # Load applications
        for app in self.data.get('applications', []):
            yield {
                'name': app['name'],
                'type': 'Application',
                'description': app.get('description'),
                'confidence': 1.0,
                'source': 'Seed Data',
                'metadata': app
            }
    
    def load_relationships(self) -> Iterator[Dict[str, Any]]:
        """Load relationships from seed data"""
        logger.info(f"Loading relationships from {self.name}")
        
        for rel in self.data.get('relationships', []):
            yield {
                'subject': rel['material'],
                'predicate': rel['relationship'],
                'object': rel['target'],
                'confidence': rel.get('confidence', 1.0),
                'source': 'Seed Data',
                'metadata': rel
            }
    
    def validate_hypothesis(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        """Validate hypothesis against seed data"""
        # Check if relationship exists in seed data
        for rel in self.data.get('relationships', []):
            if (rel['material'] == hypothesis.material and 
                rel['relationship'] == hypothesis.relationship and
                rel['target'] == hypothesis.application):
                return {
                    'is_valid': True,
                    'confidence': rel.get('confidence', 1.0),
                    'source': 'Seed Data',
                    'evidence': {
                        'relationship_found': True,
                        'seed_data': rel
                    }
                }
        
        return {
            'is_valid': False,
            'confidence': 0.0,
            'source': 'Seed Data',
            'evidence': {
                'relationship_found': False,
                'reason': 'No matching relationship found in seed data'
            }
        }

class MaterialsProjectDataSource(DataSource):
    """Materials Project API data source implementation"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.api_key = config.config.get('api_key')
        self.base_url = "https://api.materialsproject.org"
    
    def load_entities(self) -> Iterator[Dict[str, Any]]:
        """Load entities from Materials Project API"""
        if not self.api_key:
            logger.warning("Materials Project API key not provided")
            return
        
        logger.info(f"Loading entities from {self.name}")
        # Implementation would query Materials Project API
        # For now, return empty iterator
        return iter([])
    
    def load_relationships(self) -> Iterator[Dict[str, Any]]:
        """Load relationships from Materials Project API"""
        if not self.api_key:
            return
        
        logger.info(f"Loading relationships from {self.name}")
        # Implementation would query Materials Project API
        return iter([])
    
    def validate_hypothesis(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        """Validate hypothesis against Materials Project data"""
        if not self.api_key:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'source': 'Materials Project',
                'evidence': {'reason': 'API key not provided'}
            }
        
        # Implementation would query Materials Project API
        return {
            'is_valid': False,
            'confidence': 0.0,
            'source': 'Materials Project',
            'evidence': {'reason': 'Not implemented yet'}
        }

class DataManager:
    """Manages multiple data sources"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.data_sources = {}
        self._initialize_data_sources()
    
    def _initialize_data_sources(self):
        """Initialize all configured data sources"""
        from config import get_data_source_configs
        
        source_configs = get_data_source_configs()
        
        # Initialize MatKG data source
        if source_configs['matkg'].enabled:
            self.data_sources['matkg'] = MatKGDataSource(
                source_configs['matkg'], 
                self.config.matkg_config
            )
        
        # Initialize seed data source
        if source_configs['seed_data'].enabled:
            self.data_sources['seed_data'] = SeedDataSource(
                source_configs['seed_data']
            )
        
        # Initialize Materials Project data source
        if source_configs['materials_project'].enabled:
            self.data_sources['materials_project'] = MaterialsProjectDataSource(
                source_configs['materials_project']
            )
        
        logger.info(f"Initialized {len(self.data_sources)} data sources")
    
    def get_enabled_sources(self) -> List[DataSource]:
        """Get list of enabled data sources sorted by priority"""
        enabled_sources = [
            source for source in self.data_sources.values() 
            if source.enabled
        ]
        return sorted(enabled_sources, key=lambda x: x.get_priority(), reverse=True)
    
    def load_all_entities(self) -> Iterator[Dict[str, Any]]:
        """Load entities from all enabled data sources"""
        for source in self.get_enabled_sources():
            try:
                for entity in source.load_entities():
                    yield entity
            except Exception as e:
                logger.error(f"Error loading entities from {source.name}: {e}")
    
    def load_all_relationships(self) -> Iterator[Dict[str, Any]]:
        """Load relationships from all enabled data sources"""
        for source in self.get_enabled_sources():
            try:
                for relationship in source.load_relationships():
                    yield relationship
            except Exception as e:
                logger.error(f"Error loading relationships from {source.name}: {e}")
    
    def validate_hypothesis(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        """Validate hypothesis against all enabled data sources"""
        validation_results = {}
        
        for source in self.get_enabled_sources():
            try:
                result = source.validate_hypothesis(hypothesis)
                validation_results[source.name] = result
            except Exception as e:
                logger.error(f"Error validating hypothesis with {source.name}: {e}")
                validation_results[source.name] = {
                    'is_valid': False,
                    'confidence': 0.0,
                    'source': source.name,
                    'error': str(e)
                }
        
        # Combine validation results
        return self._combine_validation_results(validation_results)
    
    def _combine_validation_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine validation results from multiple sources"""
        valid_results = [r for r in results.values() if r.get('is_valid', False)]
        
        if not valid_results:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'source': 'Combined',
                'evidence': {
                    'all_sources': results,
                    'reason': 'No source validated the hypothesis'
                }
            }
        
        # Use highest confidence result
        best_result = max(valid_results, key=lambda x: x.get('confidence', 0.0))
        
        return {
            'is_valid': True,
            'confidence': best_result['confidence'],
            'source': best_result['source'],
            'evidence': {
                'primary_source': best_result,
                'all_sources': results
            }
        }
    
    def get_data_source_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all data sources"""
        stats = {}
        
        for name, source in self.data_sources.items():
            try:
                if isinstance(source, MatKGDataSource):
                    stats[name] = {
                        'entities': source.matkg_loader.get_entity_statistics(),
                        'relationships': source.matkg_loader.get_relationship_statistics()
                    }
                elif isinstance(source, SeedDataSource):
                    stats[name] = {
                        'materials': len(source.data.get('materials', [])),
                        'properties': len(source.data.get('properties', [])),
                        'applications': len(source.data.get('applications', [])),
                        'relationships': len(source.data.get('relationships', []))
                    }
                else:
                    stats[name] = {'status': 'Statistics not available'}
            except Exception as e:
                stats[name] = {'error': str(e)}
        
        return stats
