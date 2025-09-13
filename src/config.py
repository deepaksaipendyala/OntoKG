"""
Configuration system for Materials Ontology Expansion
Supports multiple data sources including MatKG
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    name: str
    enabled: bool = True
    priority: int = 1  # Higher number = higher priority
    config: Dict[str, Any] = None

@dataclass
class MatKGConfig:
    """MatKG specific configuration"""
    data_path: str = "data/matkg/"
    cache_enabled: bool = True
    cache_path: str = "data/matkg_cache/"
    max_entities: Optional[int] = None  # None = load all
    entity_types: list = None  # None = load all types
    relationship_types: list = None  # None = load all types
    confidence_threshold: float = 0.5

@dataclass
class SystemConfig:
    """Main system configuration"""
    # Database
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "123123123"
    
    # LLM
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    
    # Data sources
    use_matkg: bool = True
    use_seed_data: bool = True
    matkg_config: MatKGConfig = None
    
    # Validation
    validation_sources: list = None
    confidence_threshold: float = 0.6
    
    # Performance
    batch_size: int = 1000
    max_memory_usage: str = "2GB"

def load_config() -> SystemConfig:
    """Load configuration from environment variables and defaults"""
    
    # MatKG configuration
    matkg_config = MatKGConfig(
        data_path=os.getenv('MATKG_DATA_PATH', 'data/matkg/'),
        cache_enabled=os.getenv('MATKG_CACHE_ENABLED', 'true').lower() == 'true',
        cache_path=os.getenv('MATKG_CACHE_PATH', 'data/matkg_cache/'),
        max_entities=int(os.getenv('MATKG_MAX_ENTITIES', '0')) or None,
        confidence_threshold=float(os.getenv('MATKG_CONFIDENCE_THRESHOLD', '0.5'))
    )
    
    # Entity types to load (configurable)
    entity_types_str = os.getenv('MATKG_ENTITY_TYPES', '')
    if entity_types_str:
        matkg_config.entity_types = entity_types_str.split(',')
    
    # Relationship types to load (configurable)
    rel_types_str = os.getenv('MATKG_RELATIONSHIP_TYPES', '')
    if rel_types_str:
        matkg_config.relationship_types = rel_types_str.split(',')
    
    # Validation sources
    validation_sources_str = os.getenv('VALIDATION_SOURCES', 'matkg,property_thresholds')
    validation_sources = validation_sources_str.split(',')
    
    return SystemConfig(
        neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
        neo4j_password=os.getenv('NEO4J_PASSWORD', '123123123'),
        ollama_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        ollama_model=os.getenv('OLLAMA_MODEL', 'llama3.2:latest'),
        use_matkg=os.getenv('USE_MATKG', 'true').lower() == 'true',
        use_seed_data=os.getenv('USE_SEED_DATA', 'true').lower() == 'true',
        matkg_config=matkg_config,
        validation_sources=validation_sources,
        confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.6')),
        batch_size=int(os.getenv('BATCH_SIZE', '1000')),
        max_memory_usage=os.getenv('MAX_MEMORY_USAGE', '2GB')
    )

def get_data_source_configs() -> Dict[str, DataSourceConfig]:
    """Get configuration for all data sources"""
    return {
        'matkg': DataSourceConfig(
            name='MatKG',
            enabled=os.getenv('USE_MATKG', 'true').lower() == 'true',
            priority=3,
            config={'confidence_threshold': 0.5}
        ),
        'seed_data': DataSourceConfig(
            name='Seed Data',
            enabled=os.getenv('USE_SEED_DATA', 'true').lower() == 'true',
            priority=1,
            config={'file': 'data/seed_data.json'}
        ),
        'materials_project': DataSourceConfig(
            name='Materials Project',
            enabled=os.getenv('USE_MATERIALS_PROJECT', 'false').lower() == 'true',
            priority=2,
            config={'api_key': os.getenv('MP_API_KEY')}
        )
    }
