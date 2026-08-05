from datetime import datetime
import json
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    embeddings = relationship("Embedding", back_populates="user", cascade="all, delete-orphan")
    events = relationship("RecognitionEvent", back_populates="user")

class Embedding(Base):
    __tablename__ = 'embeddings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    # Storing embedding as a JSON list for SQLite simplicity.
    # In PostgreSQL we would use ARRAY(Float) or pgvector.
    vector_data = Column(Text, nullable=False) 
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="embeddings")

class RecognitionEvent(Base):
    __tablename__ = 'recognition_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)  # Nullable for "Unknown"
    confidence_score = Column(Float, nullable=True)
    processing_latency = Column(Float, nullable=True) # in ms
    camera_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="events")

class Device(Base):
    __tablename__ = 'devices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)

class Setting(Base):
    __tablename__ = 'settings'

    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)
