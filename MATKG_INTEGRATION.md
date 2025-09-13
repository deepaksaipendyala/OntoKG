# MatKG Integration Guide

This guide explains how to integrate MatKG (Materials Knowledge Graph) data into the Materials Ontology Expansion project.

## What is MatKG?

MatKG is the largest knowledge graph in materials science, containing:
- **150,000+ entities** (materials, properties, applications, etc.)
- **3.5 million relationships** extracted from 5+ million scientific papers
- **Structured data** with confidence scores and source papers

## Benefits of MatKG Integration

### 1. **Massive Scale Improvement**
- **Before**: ~50 materials, ~100 relationships
- **After**: 150,000+ entities, 3.5M+ relationships

### 2. **Real Scientific Data**
- Relationships extracted from actual literature
- Confidence scores based on statistical analysis
- Source paper references for verification

### 3. **Better LLM Context**
- Rich, real-world patterns for hypothesis generation
- Comprehensive coverage of materials science domains
- Authentic validation against literature

## Setup Instructions

### Step 1: Download MatKG Data

Run the setup script:
```bash
python setup_matkg.py
```

This will:
- Create necessary directories
- Download MatKG files (~4GB total)
- Set up environment configuration

### Step 2: Configure Environment

Copy the example environment file:
```bash
cp env_example.txt .env
```

Edit `.env` to configure:
```bash
# Enable MatKG
USE_MATKG=true

# MatKG Configuration
MATKG_DATA_PATH=data/matkg/
MATKG_CONFIDENCE_THRESHOLD=0.5
MATKG_ENTITY_TYPES=Material,Property,Application
MATKG_RELATIONSHIP_TYPES=USED_IN,HAS_PROPERTY
```

### Step 3: Initialize Knowledge Graph

Use the enhanced initialization script:
```bash
python src/init_kg_enhanced.py
```

This will:
- Load MatKG data into Neo4j
- Create entities and relationships
- Show progress and statistics

## Configuration Options

### Data Source Priority
```bash
# Priority order (higher number = higher priority)
USE_MATKG=true          # Priority 3
USE_SEED_DATA=true      # Priority 1  
USE_MATERIALS_PROJECT=false  # Priority 2
```

### MatKG Filtering
```bash
# Limit entities loaded (0 = load all)
MATKG_MAX_ENTITIES=0

# Filter by entity types
MATKG_ENTITY_TYPES=Material,Property,Application

# Filter by relationship types
MATKG_RELATIONSHIP_TYPES=USED_IN,HAS_PROPERTY

# Minimum confidence threshold
MATKG_CONFIDENCE_THRESHOLD=0.5
```

### Performance Tuning
```bash
# Batch processing size
BATCH_SIZE=1000

# Memory usage limit
MAX_MEMORY_USAGE=2GB

# Enable caching
MATKG_CACHE_ENABLED=true
MATKG_CACHE_PATH=data/matkg_cache/
```

## Usage Examples

### 1. Basic Initialization
```python
from config import load_config
from knowledge_graph import MaterialsKG

# Load configuration
config = load_config()

# Initialize with MatKG
kg = MaterialsKG(config)
kg.initialize_from_data_sources()

# Get statistics
stats = kg.get_graph_stats()
print(f"Loaded {stats['materials']} materials, {stats['relationships']} relationships")
```

### 2. Custom Configuration
```python
from config import SystemConfig, MatKGConfig

# Custom MatKG configuration
matkg_config = MatKGConfig(
    data_path="custom/path/to/matkg/",
    max_entities=10000,  # Limit for demo
    confidence_threshold=0.7,
    entity_types=["Material", "Property"]
)

# System configuration
config = SystemConfig(
    use_matkg=True,
    matkg_config=matkg_config
)

kg = MaterialsKG(config)
```

### 3. Validation with MatKG
```python
from validation import MaterialsValidator
from knowledge_graph import Hypothesis

validator = MaterialsValidator(config)

# Validate hypothesis against MatKG
hypothesis = Hypothesis(
    material="SrTiO3",
    application="capacitor",
    relationship="USED_IN"
)

result = validator.validate_hypothesis(hypothesis.material, hypothesis.application)
print(f"Valid: {result.is_valid}, Confidence: {result.confidence}")
```

## Data Structure

### MatKG Entities
```python
{
    'name': 'BaTiO3',
    'uri': 'http://example.org/BaTiO3',
    'type': 'Material',
    'confidence': 0.95,
    'source': 'MatKG',
    'metadata': {...}
}
```

### MatKG Relationships
```python
{
    'subject': 'BaTiO3',
    'predicate': 'USED_IN',
    'object': 'capacitor',
    'confidence': 0.87,
    'source': 'MatKG',
    'source_papers': ['10.1038/s41597-024-03039-z'],
    'metadata': {...}
}
```

## Troubleshooting

### Common Issues

1. **MatKG files not found**
   ```bash
   python setup_matkg.py --check-matkg
   ```

2. **Memory issues with large datasets**
   ```bash
   # Reduce batch size
   BATCH_SIZE=500
   
   # Limit entities
   MATKG_MAX_ENTITIES=50000
   ```

3. **Slow loading**
   ```bash
   # Enable caching
   MATKG_CACHE_ENABLED=true
   
   # Use SSD storage
   MATKG_DATA_PATH=/path/to/ssd/matkg/
   ```

### Performance Tips

1. **Use SSD storage** for MatKG data files
2. **Enable caching** for repeated operations
3. **Limit entity types** if you don't need all data
4. **Use batch processing** for large datasets

## Advanced Usage

### Custom Data Sources
```python
from data_manager import DataSource, DataManager

class CustomDataSource(DataSource):
    def load_entities(self):
        # Custom implementation
        pass
    
    def validate_hypothesis(self, hypothesis):
        # Custom validation logic
        pass

# Add to data manager
data_manager = DataManager(config)
data_manager.data_sources['custom'] = CustomDataSource(config)
```

### Query MatKG Directly
```python
from matkg_loader import MatKGLoader
from config import MatKGConfig

loader = MatKGLoader(MatKGConfig())

# Search for entities
entities = loader.search_entities("BaTiO3", entity_type="Material")

# Get relationships
relationships = loader.get_relationships_for_entity("BaTiO3")

# Get statistics
entity_stats = loader.get_entity_statistics()
```

## Integration Benefits

### For Researchers
- **Comprehensive coverage**: Access to 150K+ materials entities
- **Literature grounding**: All relationships backed by papers
- **Confidence scoring**: Statistical confidence for each relationship

### For Developers
- **Flexible configuration**: Easy to enable/disable data sources
- **Modular design**: Clean separation of concerns
- **Extensible**: Easy to add new data sources

### For the Project
- **Authentic validation**: Real scientific relationships
- **Scalable architecture**: Handles large datasets efficiently
- **Production ready**: Robust error handling and logging

## Next Steps

1. **Download MatKG data** using the setup script
2. **Configure environment** for your needs
3. **Initialize knowledge graph** with MatKG data
4. **Run LLM expansion** with rich context
5. **Validate hypotheses** against real literature

The MatKG integration transforms your project from a proof-of-concept to a production-ready system that materials researchers can actually use!
