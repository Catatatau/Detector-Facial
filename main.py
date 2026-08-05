import sys
import os

# Ensure the app can find its own modules when run from the root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import logger
from database.session import init_db
from ui.app import run_app

def main():
    logger.info("Starting Enterprise Local Face Recognition Platform...")
    
    # Initialize the database
    init_db()
    
    # Launch the UI application
    run_app()

if __name__ == "__main__":
    main()
