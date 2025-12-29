# backend/functions/email-extraction/function_app_questions.py
"""
Azure Function for question-only extraction from blank RFPs.

This function processes blank RFP documents to extract only the questions,
without attempting to extract answers. Questions are saved with answer_text=NULL.

Use this function for:
- New/blank RFPs that need to be answered
- RFIs (Request for Information)
- Documents where you only need questions extracted
"""

import azure.functions as func
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from config import config
from shared.extractors import ExtractorFactory
from shared.extractors.base.config import QuestionExtractorConfig
from shared.extractors.base import QuestionToQAAdapter
from shared.parsers import create_parser
from shared.core.processor import DocumentProcessor

app = func.FunctionApp()

# Database Connection Setup
engine = create_engine(config.database_url)

# Extraction Strategy Configuration
EXTRACTION_STRATEGY = os.getenv("EXTRACTION_STRATEGY", "context_aware")
# Options: simple_llm (fast, basic), context_aware (better quality, section-aware)


def load_config() -> QuestionExtractorConfig:
    """Load question extraction configuration from environment variables."""
    return QuestionExtractorConfig(
        chunk_size=int(os.getenv("CHUNK_SIZE", "6000")),
        overlap=int(os.getenv("CHUNK_OVERLAP", "500")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.6")),
        deduplicate_questions=os.getenv("ENABLE_DEDUP", "true").lower() == "true",
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.9")),
        # LLM config auto-loaded from base ExtractorConfig
    )


@app.function_name(name="ExtractQuestions")
@app.blob_trigger(
    arg_name="myblob",
    path="input-pdfs/{name}",
    connection="AzureWebJobsStorage"
)
def main(myblob: func.InputStream):
    """
    Azure Function triggered by blob storage uploads to extract questions only.

    This function:
    1. Parses the PDF document (using Azure DI or pymupdf)
    2. Extracts questions only using LLM (no answer extraction)
    3. Saves questions to the database with answer_text=NULL

    Trigger: Upload PDF to 'input-pdfs' container
    Note: Questions are saved with answer_text=NULL for later answering
    """
    logging.info(f"[Questions] Processing file: {myblob.name} ({myblob.length} bytes)")

    session = Session(engine)

    try:
        # Read file content
        file_bytes = myblob.read()

        # Create question extractor with custom config
        factory = ExtractorFactory()
        config = load_config()
        question_extractor = factory.create_question_extractor(
            strategy=EXTRACTION_STRATEGY,
            config=config
        )

        logging.info(f"[Questions] Using extraction strategy: {EXTRACTION_STRATEGY}")

        # For context-aware strategy, inject structure extractor
        if EXTRACTION_STRATEGY == "context_aware":
            # Structure extractor needs its own config (not QuestionExtractorConfig)
            structure_extractor = factory.create_structure_extractor(
                strategy="structure_aware"
                # Uses default StructureExtractorConfig with LLM settings from env
            )
            question_extractor.set_structure_extractor(structure_extractor)
            logging.info("[Questions] Structure extractor injected for context-aware extraction")

        # Wrap question extractor in adapter to make it compatible with DocumentProcessor
        # The adapter converts ExtractedQuestion → ExtractedQAPair with answer_text=None
        qa_extractor = QuestionToQAAdapter(question_extractor)

        # Create parser (defaults to env var DOCUMENT_PARSER)
        parser = create_parser()

        # Use DocumentProcessor for all orchestration and database logic
        processor = DocumentProcessor(
            session=session,
            parser=parser,
            extractor=qa_extractor
        )

        # Process document (handles parsing, extraction, and database persistence)
        result = processor.process_document(
            file_content=file_bytes,
            filename=myblob.name,
            client_name="Unknown Client",
            source_type="Blank RFP"
        )

        if result.success:
            logging.info(f"[Questions] Success: {result.message}")
        else:
            logging.warning(f"[Questions] Processing completed with issues: {result.message}")

    except Exception as e:
        session.rollback()
        logging.error(f"[Questions] Error processing {myblob.name}: {str(e)}", exc_info=True)
        raise
    finally:
        session.close()
