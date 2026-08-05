import os
import urllib.request
from pathlib import Path
from utils.logger import logger

def download_file(url: str, dest_path: str, force: bool = False) -> bool:
    """Download a file from a URL to a local path."""
    path = Path(dest_path)
    
    if path.exists() and not force:
        logger.debug(f"File {dest_path} already exists. Skipping download.")
        return True
        
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Downloading {url} to {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Downloaded successfully: {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False
