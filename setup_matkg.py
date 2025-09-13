"""
MatKG Setup Script
Helps users download and set up MatKG data
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for MatKG data"""
    data_dir = Path("data/matkg")
    cache_dir = Path("data/matkg_cache")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directories:")
    logger.info(f"  Data: {data_dir.absolute()}")
    logger.info(f"  Cache: {cache_dir.absolute()}")
    
    return data_dir

def download_matkg_data(data_dir: Path):
    """Download MatKG data from Zenodo"""
    
    # MatKG dataset URL
    zenodo_url = "https://zenodo.org/record/10144972/files/"
    
    files_to_download = [
        "SUBRELOBJ.csv",
        "ENTPTNERDOI.csv.tar.gz", 
        "entity_uri_mapping.pickle"
    ]
    
    logger.info("Starting MatKG data download...")
    logger.info("This may take a while as the dataset is large (~4GB total)")
    
    for filename in files_to_download:
        file_path = data_dir / filename
        url = zenodo_url + filename
        
        if file_path.exists():
            logger.info(f"File {filename} already exists, skipping...")
            continue
        
        logger.info(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r{filename}: {percent:.1f}%", end="", flush=True)
            
            print()  # New line after progress
            logger.info(f"Downloaded {filename} successfully")
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            if file_path.exists():
                file_path.unlink()  # Remove partial file
    
    logger.info("MatKG data download complete!")

def verify_matkg_data(data_dir: Path):
    """Verify that MatKG data files are present and valid"""
    required_files = [
        "SUBRELOBJ.csv",
        "ENTPTNERDOI.csv.tar.gz",
        "entity_uri_mapping.pickle"
    ]
    
    missing_files = []
    for filename in required_files:
        file_path = data_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
        else:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ {filename} ({size_mb:.1f} MB)")
    
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return False
    
    logger.info("✓ All MatKG files present and valid")
    return True

def setup_environment():
    """Set up environment configuration"""
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if not env_file.exists():
        if env_example.exists():
            logger.info("Creating .env file from example...")
            with open(env_example, 'r') as f:
                content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            logger.info("✓ Created .env file")
        else:
            logger.warning("No .env file found. Please create one manually.")
    else:
        logger.info("✓ .env file already exists")

def main():
    """Main setup function"""
    logger.info("MatKG Setup Script")
    logger.info("==================")
    
    # Create directories
    data_dir = create_directories()
    
    # Check if data already exists
    if verify_matkg_data(data_dir):
        logger.info("MatKG data is already set up!")
        return
    
    # Ask user if they want to download
    print("\nMatKG dataset is large (~4GB). Do you want to download it now?")
    print("1. Yes, download now")
    print("2. No, I'll download manually")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        download_matkg_data(data_dir)
        verify_matkg_data(data_dir)
    elif choice == "2":
        logger.info("Manual download instructions:")
        logger.info("1. Go to: https://zenodo.org/record/10144972")
        logger.info("2. Download the dataset")
        logger.info(f"3. Extract files to: {data_dir.absolute()}")
        logger.info("4. Run this script again to verify")
    else:
        logger.info("Setup cancelled")
        return
    
    # Set up environment
    setup_environment()
    
    logger.info("\nSetup complete!")
    logger.info("You can now run: python src/init_kg_enhanced.py")

if __name__ == "__main__":
    main()
