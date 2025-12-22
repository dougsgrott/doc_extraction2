# shared/core/processor.py
"""
Core document processing logic extracted from Azure Function.
This can be used both by Azure Functions and local testing scripts.
"""

import logging
from datetime import date
from typing import BinaryIO, Union
from sqlalchemy.orm import Session

from shared.services.doc_intelligence import DocumentParser
from shared.models.db import RFP
from shared.models.db.relationships import bulk_create_answers_with_rfp_update
from shared.models.converters import bulk_extracted_qa_to_answers
from shared.models.enums import SourceType, ProcessingStatus, RFPStatus


class ProcessingResult:
    """Result of document processing operation."""

    def __init__(self, success: bool, message: str, qa_count: int = 0, error: Exception = None):
        self.success = success
        self.message = message
        self.qa_count = qa_count
        self.error = error

    def __str__(self):
        return f"ProcessingResult(success={self.success}, message='{self.message}', qa_count={self.qa_count})"


class DocumentProcessor:
    """
    Core business logic for processing PDF documents.
    Orchestrates OCR, semantic analysis, and database persistence.
    """

    def __init__(
        self,
        session: Session,
        parser: DocumentParser = None,
        extractor = None
    ):
        """
        Initialize the document processor.

        Args:
            session: SQLAlchemy database session
            parser: DocumentParser instance (creates new if None)
            extractor: QuestionAnswerExtractor instance (new architecture)
        """
        self.session = session
        self.parser = parser or DocumentParser()
        self.extractor = extractor

    def process_document(
        self,
        file_content: Union[bytes, BinaryIO],
        filename: str,
        client_name: str = "Unknown Client",
        source_type: str = "Uploaded Document"
    ) -> ProcessingResult:
        """
        Process a PDF document through the full pipeline.

        Args:
            file_content: PDF file as bytes or file-like object
            filename: Name of the file being processed
            client_name: Client name for RFP record
            source_type: Type of source document

        Returns:
            ProcessingResult with status and details
        """
        try:
            logging.info(f"Processing file: {filename}")

            # Step 1: Parse PDF with Azure Document Intelligence (OCR)
            logging.info("Step 1: Extracting text with Azure Document Intelligence...")
            doc_result = self.parser.parse_stream(file_content)

            # Extract text from paragraphs
            full_text = ""
            if doc_result.paragraphs:
                full_text = "\n".join([p.content for p in doc_result.paragraphs])

            if not full_text:
                logging.warning(f"No text extracted from {filename}")
                return ProcessingResult(
                    success=False,
                    message="No text could be extracted from the document"
                )

            logging.info(f"Extracted {len(full_text)} characters of text")

            # Step 2: Semantic Analysis (Extract Questions & Answers)
            logging.info("Step 2: Identifying Questions and Answers with LLM...")

            extracted_data = self.extractor.extract(full_text)

            logging.info(f"Extracted {len(extracted_data)} potential Q&A pairs")

            if not extracted_data:
                logging.warning(f"No Q&A pairs extracted from {filename}")
                return ProcessingResult(
                    success=True,
                    message="Document processed but no Q&A pairs found",
                    qa_count=0
                )

            # Step 3: Save to Database
            logging.info("Step 3: Persisting to PostgreSQL...")
            self._save_to_database(
                filename=filename,
                client_name=client_name,
                source_type=source_type,
                extracted_data=extracted_data
            )

            logging.info(f"Successfully committed {len(extracted_data)} Q&A pairs to DB")

            return ProcessingResult(
                success=True,
                message=f"Successfully processed {len(extracted_data)} Q&A pairs",
                qa_count=len(extracted_data)
            )

        except Exception as e:
            logging.error(f"Error processing document {filename}: {str(e)}", exc_info=True)
            return ProcessingResult(
                success=False,
                message=f"Error processing document: {str(e)}",
                error=e
            )

    def _save_to_database(self, filename: str, client_name: str, source_type: str, extracted_data):
        """
        Save extracted Q&A data to the database.

        Args:
            filename: Name of the source file
            client_name: Client name for RFP
            source_type: Type of source document (unused in new schema)
            extracted_data: List of ExtractedQAPair objects
        """
        # Create RFP (no Source table in new schema)
        new_rfp = RFP(
            source_id=filename,  # Store filename as text identifier
            client_name=client_name,
            format="PDF",
            processing_status=ProcessingStatus.COMPLETED,
            status=RFPStatus.ACTIVE
        )
        self.session.add(new_rfp)
        self.session.flush()  # Flush to get new_rfp.id

        # Convert extracted Q&A pairs to Answer models
        answers = bulk_extracted_qa_to_answers(
            extracted_data,
            rfp_id=new_rfp.id,
            source=SourceType.RFP_EXTRACTION
        )

        # Create all answers and update RFP counters
        bulk_create_answers_with_rfp_update(
            self.session,
            answers,
            commit=True
        )

        # NOTE: Vector embeddings for question_embedding can be generated
        # in a separate batch process to avoid timeout issues with large documents
