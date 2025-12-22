import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
# from shared.models.document_record import Base, DocumentRecord
from shared.services.document_record import Base, DocumentRecord

class DatabaseManager:
    def __init__(self):
        # Read connection string from environment variables
        self.connection_string = os.environ.get("DB_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("DB_CONNECTION_STRING is not set.")

        # Create Engine
        self.engine = create_engine(self.connection_string)
        
        # Create Tables if they don't exist (Auto-migration for simplicity)
        Base.metadata.create_all(self.engine)
        
        # Session Factory
        self.Session = sessionmaker(bind=self.engine)

    def save_document(self, filename: str, blob_url: str, extraction_data: dict, semantic_data: dict):
        """Saves the parsed document data to Postgres."""
        session = self.Session()
        try:
            doc = DocumentRecord(
                filename=filename,
                blob_url=blob_url,
                raw_extraction_data=extraction_data,
                semantic_data=semantic_data,
            )
            session.add(doc)
            session.commit()
            return doc.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()