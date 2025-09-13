"""
MatKG Data Loader
Flexible system to load and process MatKG dataset
"""

import os
import pandas as pd
import pickle
import json
import gzip
import tarfile
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
import logging
from pathlib import Path

from config import MatKGConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatKGEntity:
    """Represents a MatKG entity"""
    uri: str
    name: str
    entity_type: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

@dataclass
class MatKGRelationship:
    """Represents a MatKG relationship"""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_papers: List[str] = None
    metadata: Dict[str, Any] = None

class MatKGLoader:
    """Loads and processes MatKG dataset"""
    
    def __init__(self, config: MatKGConfig):
        self.config = config
        self.data_path = Path(config.data_path)
        self.cache_path = Path(config.cache_path)
        self.entities_cache = {}
        self.relationships_cache = {}
        
        # Create cache directory if it doesn't exist
        if config.cache_enabled:
            self.cache_path.mkdir(parents=True, exist_ok=True)
    
    def download_matkg_data(self) -> bool:
        """Download MatKG data if not present"""
        # Check if data already exists
        required_files = [
            'SUBRELOBJ.csv',
            'ENTPTNERDOI.csv.tar.gz',
            'entity_uri_mapping.pickle'
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.data_path / file).exists():
                missing_files.append(file)
        
        if not missing_files:
            logger.info("MatKG data already present")
            return True
        
        logger.warning(f"Missing MatKG files: {missing_files}")
        logger.info("Please download MatKG data from: https://zenodo.org/record/10144972")
        logger.info("Extract files to: " + str(self.data_path))
        
        return False
    
    def load_entity_mapping(self) -> Dict[str, str]:
        """Load entity URI to name mapping"""
        cache_file = self.cache_path / 'entity_mapping.json'
        
        if self.config.cache_enabled and cache_file.exists():
            logger.info("Loading entity mapping from cache")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        mapping_file = self.data_path / 'entity_uri_mapping.pickle'
        if not mapping_file.exists():
            logger.error(f"Entity mapping file not found: {mapping_file}")
            return {}
        
        logger.info("Loading entity mapping from pickle file")
        with open(mapping_file, 'rb') as f:
            mapping_data = pickle.load(f)
        
        # Convert to simple dict format
        entity_mapping = {}
        for uri, name in mapping_data.items():
            entity_mapping[uri] = name
        
        # Cache the mapping
        if self.config.cache_enabled:
            with open(cache_file, 'w') as f:
                json.dump(entity_mapping, f)
        
        logger.info(f"Loaded {len(entity_mapping)} entity mappings")
        return entity_mapping
    
    def load_entities(self) -> Iterator[MatKGEntity]:
        """Load entities from SUBRELOBJ.csv"""
        cache_file = self.cache_path / 'entities.json'
        
        if self.config.cache_enabled and cache_file.exists():
            logger.info("Loading entities from cache")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                for entity_data in cached_data:
                    yield MatKGEntity(**entity_data)
                return
        
        entities_file = self.data_path / 'SUBRELOBJ.csv'
        if not entities_file.exists():
            logger.error(f"Entities file not found: {entities_file}")
            return
        
        logger.info("Loading entities from CSV file")
        entity_mapping = self.load_entity_mapping()
        
        entities = []
        chunk_size = 10000
        
        try:
            # Read CSV in chunks to handle large files
            for chunk in pd.read_csv(entities_file, chunksize=chunk_size):
                for _, row in chunk.iterrows():
                    entity = self._parse_entity_row(row, entity_mapping)
                    if entity and self._should_include_entity(entity):
                        entities.append(entity)
                        
                        if self.config.max_entities and len(entities) >= self.config.max_entities:
                            break
                
                if self.config.max_entities and len(entities) >= self.config.max_entities:
                    break
                    
        except Exception as e:
            logger.error(f"Error loading entities: {e}")
            return
        
        # Cache entities
        if self.config.cache_enabled:
            entity_data = [entity.__dict__ for entity in entities]
            with open(cache_file, 'w') as f:
                json.dump(entity_data, f)
        
        logger.info(f"Loaded {len(entities)} entities")
        for entity in entities:
            yield entity
    
    def load_relationships(self) -> Iterator[MatKGRelationship]:
        """Load relationships from ENTPTNERDOI.csv"""
        cache_file = self.cache_path / 'relationships.json'
        
        if self.config.cache_enabled and cache_file.exists():
            logger.info("Loading relationships from cache")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                for rel_data in cached_data:
                    yield MatKGRelationship(**rel_data)
                return
        
        relationships_file = self.data_path / 'ENTPTNERDOI.csv.tar.gz'
        if not relationships_file.exists():
            logger.error(f"Relationships file not found: {relationships_file}")
            return
        
        logger.info("Loading relationships from compressed CSV file")
        
        relationships = []
        chunk_size = 10000
        
        try:
            # Extract and read the tar.gz file
            with tarfile.open(relationships_file, 'r:gz') as tar:
                csv_member = tar.getmember('ENTPTNERDOI.csv')
                csv_file = tar.extractfile(csv_member)
                
                # Read CSV in chunks
                for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                    for _, row in chunk.iterrows():
                        relationship = self._parse_relationship_row(row)
                        if relationship and self._should_include_relationship(relationship):
                            relationships.append(relationship)
                    
                    # Limit for performance
                    if len(relationships) >= 100000:  # Reasonable limit for demo
                        break
                        
        except Exception as e:
            logger.error(f"Error loading relationships: {e}")
            return
        
        # Cache relationships
        if self.config.cache_enabled:
            rel_data = [rel.__dict__ for rel in relationships]
            with open(cache_file, 'w') as f:
                json.dump(rel_data, f)
        
        logger.info(f"Loaded {len(relationships)} relationships")
        for relationship in relationships:
            yield relationship
    
    def _parse_entity_row(self, row: pd.Series, entity_mapping: Dict[str, str]) -> Optional[MatKGEntity]:
        """Parse a single entity row from CSV"""
        try:
            uri = str(row.get('entity_uri', ''))
            if not uri or uri == 'nan':
                return None
            
            name = entity_mapping.get(uri, uri.split('/')[-1])  # Fallback to URI fragment
            entity_type = str(row.get('entity_type', 'Unknown'))
            confidence = float(row.get('confidence', 1.0))
            
            return MatKGEntity(
                uri=uri,
                name=name,
                entity_type=entity_type,
                confidence=confidence,
                metadata={'row_data': row.to_dict()}
            )
        except Exception as e:
            logger.warning(f"Error parsing entity row: {e}")
            return None
    
    def _parse_relationship_row(self, row: pd.Series) -> Optional[MatKGRelationship]:
        """Parse a single relationship row from CSV"""
        try:
            subject = str(row.get('subject', ''))
            predicate = str(row.get('predicate', ''))
            object_uri = str(row.get('object', ''))
            confidence = float(row.get('confidence', 0.5))
            
            if not all([subject, predicate, object_uri]):
                return None
            
            # Parse source papers if available
            source_papers = []
            if 'source_papers' in row and pd.notna(row['source_papers']):
                source_papers = str(row['source_papers']).split(';')
            
            return MatKGRelationship(
                subject=subject,
                predicate=predicate,
                object=object_uri,
                confidence=confidence,
                source_papers=source_papers,
                metadata={'row_data': row.to_dict()}
            )
        except Exception as e:
            logger.warning(f"Error parsing relationship row: {e}")
            return None
    
    def _should_include_entity(self, entity: MatKGEntity) -> bool:
        """Check if entity should be included based on configuration"""
        # Check entity type filter
        if self.config.entity_types and entity.entity_type not in self.config.entity_types:
            return False
        
        # Check confidence threshold
        if entity.confidence < self.config.confidence_threshold:
            return False
        
        return True
    
    def _should_include_relationship(self, relationship: MatKGRelationship) -> bool:
        """Check if relationship should be included based on configuration"""
        # Check relationship type filter
        if self.config.relationship_types and relationship.predicate not in self.config.relationship_types:
            return False
        
        # Check confidence threshold
        if relationship.confidence < self.config.confidence_threshold:
            return False
        
        return True
    
    def get_entity_statistics(self) -> Dict[str, int]:
        """Get statistics about loaded entities"""
        stats = {}
        
        for entity in self.load_entities():
            entity_type = entity.entity_type
            stats[entity_type] = stats.get(entity_type, 0) + 1
        
        return stats
    
    def get_relationship_statistics(self) -> Dict[str, int]:
        """Get statistics about loaded relationships"""
        stats = {}
        
        for relationship in self.load_relationships():
            predicate = relationship.predicate
            stats[predicate] = stats.get(predicate, 0) + 1
        
        return stats
    
    def search_entities(self, query: str, entity_type: str = None) -> List[MatKGEntity]:
        """Search for entities by name"""
        results = []
        query_lower = query.lower()
        
        for entity in self.load_entities():
            if query_lower in entity.name.lower():
                if entity_type is None or entity.entity_type == entity_type:
                    results.append(entity)
        
        return results
    
    def get_relationships_for_entity(self, entity_uri: str) -> List[MatKGRelationship]:
        """Get all relationships for a specific entity"""
        relationships = []
        
        for relationship in self.load_relationships():
            if relationship.subject == entity_uri or relationship.object == entity_uri:
                relationships.append(relationship)
        
        return relationships
    
    def clear_cache(self):
        """Clear all cached data"""
        if self.cache_path.exists():
            import shutil
            shutil.rmtree(self.cache_path)
            logger.info("Cleared MatKG cache")
