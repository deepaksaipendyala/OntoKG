"""
Enhanced Knowledge Graph Initialization
Uses data manager to load from multiple sources including MatKG
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from config import load_config
from knowledge_graph import MaterialsKG
from data_manager import DataManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Initialize knowledge graph from configured data sources"""
    
    logger.info("Starting enhanced knowledge graph initialization...")
    
    # Load configuration
    config = load_config()
    logger.info(f"Configuration loaded: MatKG={config.use_matkg}, Seed Data={config.use_seed_data}")
    
    # Initialize knowledge graph
    kg = MaterialsKG(config)
    
    try:
        # Clear existing data
        logger.info("Clearing existing knowledge graph...")
        kg.clear_database()
        
        # Initialize from data sources
        logger.info("Initializing from data sources...")
        kg.initialize_from_data_sources()
        
        # Get final statistics
        stats = kg.get_graph_stats()
        logger.info(f"Knowledge graph initialized successfully!")
        logger.info(f"Final statistics: {stats}")
        
        # Get data source statistics
        if kg.data_manager:
            data_stats = kg.data_manager.get_data_source_statistics()
            logger.info("Data source statistics:")
            for source_name, source_stats in data_stats.items():
                logger.info(f"  {source_name}: {source_stats}")
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise
    finally:
        kg.close()
    
    logger.info("Initialization complete!")

def check_matkg_data():
    """Check if MatKG data is available"""
    config = load_config()
    
    if not config.use_matkg:
        logger.info("MatKG not enabled in configuration")
        return False
    
    matkg_path = Path(config.matkg_config.data_path)
    required_files = [
        'SUBRELOBJ.csv',
        'ENTPTNERDOI.csv.tar.gz',
        'entity_uri_mapping.pickle'
    ]
    
    missing_files = []
    for file in required_files:
        if not (matkg_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing MatKG files: {missing_files}")
        logger.info("To download MatKG data:")
        logger.info("1. Go to: https://zenodo.org/record/10144972")
        logger.info("2. Download the dataset")
        logger.info(f"3. Extract files to: {matkg_path}")
        logger.info("4. Run this script again")
        return False
    
    logger.info("MatKG data files found!")
    return True

def setup_matkg_directory():
    """Create MatKG data directory structure"""
    config = load_config()
    matkg_path = Path(config.matkg_config.data_path)
    cache_path = Path(config.matkg_config.cache_path)
    
    matkg_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created MatKG directories:")
    logger.info(f"  Data: {matkg_path}")
    logger.info(f"  Cache: {cache_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Materials Knowledge Graph")
    parser.add_argument("--check-matkg", action="store_true", 
                       help="Check if MatKG data is available")
    parser.add_argument("--setup-dirs", action="store_true",
                       help="Create MatKG directory structure")
    parser.add_argument("--force", action="store_true",
                       help="Force initialization even if MatKG data is missing")
    
    args = parser.parse_args()
    
    if args.check_matkg:
        check_matkg_data()
    elif args.setup_dirs:
        setup_matkg_directory()
    else:
        # Check MatKG data availability
        if not check_matkg_data() and not args.force:
            logger.error("MatKG data not available. Use --force to proceed with seed data only.")
            sys.exit(1)
        
        # Run initialization
        main()
