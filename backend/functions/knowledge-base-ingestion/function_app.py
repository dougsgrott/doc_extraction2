# backend/functions/rfp-extraction/function_app_qa.py
"""
Azure Function for Q&A extraction from answered RFPs.

This function processes RFP documents that may already contain answers,
extracting both questions and their corresponding answers for knowledge base building.

Use this function for:
- Previously answered RFPs
- Knowledge base documents
- RFPs with pre-filled responses
"""

import azure.functions as func
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from config import config
from shared.core.processor import DocumentProcessor
from shared.extractors import ExtractorFactory
from shared.extractors.base.config import QAExtractorConfig
from shared.parsers import create_parser

app = func.FunctionApp()

# Database Connection Setup
engine = create_engine(config.database_url)

# Extraction Strategy Configuration
EXTRACTION_STRATEGY = os.getenv("EXTRACTION_STRATEGY", "simple")
# Options: simple (fast, basic), context_aware (better quality, structure-aware)


def load_config() -> QAExtractorConfig:
    """Load Q&A extraction configuration from environment variables."""
    return QAExtractorConfig(
        chunk_size=int(os.getenv("CHUNK_SIZE", "8000")),
        overlap=int(os.getenv("CHUNK_OVERLAP", "500")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
        deduplicate_pairs=os.getenv("ENABLE_DEDUP", "true").lower() == "true",
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.9")),
        # LLM config auto-loaded from base ExtractorConfig
    )


@app.function_name(name="ExtractQA")
@app.blob_trigger(
    arg_name="myblob",
    path="input-pdfs/{name}",
    connection="AzureWebJobsStorage"
)
def main(myblob: func.InputStream):
    """
    Azure Function triggered by blob storage uploads to extract Q&A pairs.

    This function:
    1. Parses the PDF document (using Azure DI or pymupdf)
    2. Extracts question-answer pairs using LLM
    3. Saves both questions and answers to the database

    Trigger: Upload PDF to 'input-pdfs' container
    """
    logging.info(f"[Q&A] Processing file: {myblob.name} ({myblob.length} bytes)")

    session = Session(engine)

    try:
        # Read file content
        file_bytes = myblob.read()

        # Create extractor with custom config
        factory = ExtractorFactory()
        config = load_config()
        extractor = factory.create_qa_extractor(strategy=EXTRACTION_STRATEGY, config=config)

        logging.info(f"[Q&A] Using extraction strategy: {EXTRACTION_STRATEGY}")

        # For context-aware strategy, inject structure extractor
        if EXTRACTION_STRATEGY == "context_aware":
            # Structure extractor needs its own config (not QAExtractorConfig)
            structure_extractor = factory.create_structure_extractor(
                strategy="structure_aware"
                # Uses default StructureExtractorConfig with LLM settings from env
            )
            # Context-aware QA extractor uses structure extractor internally
            extractor.set_structure_extractor(structure_extractor)
            logging.info("[Q&A] Structure extractor injected for context-aware extraction")

        # Create parser (defaults to env var DOCUMENT_PARSER)
        parser = create_parser()

        # Process using core processor
        processor = DocumentProcessor(session, parser=parser, extractor=extractor)
        result = processor.process_document(
            file_content=file_bytes,
            filename=myblob.name,
            client_name="Unknown Client",
            source_type="Q&A Document"
        )

        # Log result
        if result.success:
            logging.info(f"[Q&A] Success: {result.message}")
        else:
            logging.error(f"[Q&A] Failed: {result.message}")
            if result.error:
                raise result.error

    except Exception as e:
        session.rollback()
        logging.error(f"[Q&A] Error processing {myblob.name}: {str(e)}")
        raise
    finally:
        session.close()
