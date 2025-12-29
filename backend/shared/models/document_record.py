from sqlalchemy import Column, Integer, String, JSON, DateTime, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class DocumentRecord(Base):
    __tablename__ = 'documents'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    blob_url = Column(Text, nullable=True)
    
    # We store the full JSON output from parser here
    raw_extraction_data = Column(JSON, nullable=True)
    semantic_data = Column(JSON, nullable=True)
    
    status = Column(String(50), default="processed")
    created_at = Column(DateTime, default=datetime.utcnow)