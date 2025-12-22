# backend/functions/rfp-extraction/function_app.py
"""
Azure Function entry point for blob-triggered PDF processing.
Core logic has been extracted to shared.core.processor for reusability.
"""

import azure.functions as func
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from config import config
from shared.core.processor import DocumentProcessor
from shared.extractors import ExtractorFactory

app = func.FunctionApp()

# Database Connection Setup
engine = create_engine(config.database_url)

# Extraction Strategy Configuration (from environment variables)
EXTRACTION_STRATEGY = os.getenv("EXTRACTION_STRATEGY", "simple")
# Options: simple, context_aware


@app.function_name(name="BlobTriggerPDF")
@app.blob_trigger(arg_name="myblob", path="input-pdfs/{name}", connection="AzureWebJobsStorage")
def main(myblob: func.InputStream):
    """
    Azure Function triggered by blob storage uploads.
    Processes PDF documents through OCR, semantic analysis, and database persistence.
    """
    logging.info(f"Processing file: {myblob.name} ({myblob.length} bytes)")

    session = Session(engine)

    try:
        # Read file content
        file_bytes = myblob.read()

        # Create extractor using configured strategy
        factory = ExtractorFactory()
        extractor = factory.create_qa_extractor(EXTRACTION_STRATEGY)
        logging.info(f"Using extraction strategy: {EXTRACTION_STRATEGY}")

        # Process using core processor
        processor = DocumentProcessor(session, extractor=extractor)
        result = processor.process_document(
            file_content=file_bytes,
            filename=myblob.name,
            client_name="Unknown Client",  # TODO: Extract from document or metadata
            source_type="Uploaded Document"
        )

        # Log result
        if result.success:
            logging.info(f"✓ {result.message}")
        else:
            logging.error(f"✗ {result.message}")
            if result.error:
                raise result.error

    except Exception as e:
        session.rollback()
        logging.error(f"Error processing document {myblob.name}: {str(e)}")
        raise
    finally:
        session.close()