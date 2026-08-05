from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config.settings import config
from utils.logger import logger
from database.models import Base

# Setup engine
engine = create_engine(config.database.location, echo=False)

# Setup session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create all tables if they don't exist yet."""
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized.")

def get_db():
    """Dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
