"""
Configuration classes for extraction strategies.

This module defines configuration dataclasses that allow customization
of extractor behavior without modifying code.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ExtractorConfig:
    """
    Base configuration for all extractors.

    Contains common settings shared across all extraction strategies,
    particularly LLM connection details and rate limiting.
    """
    # LLM Configuration
    llm_endpoint: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_deployment: Optional[str] = None
    llm_api_version: str = "2024-08-01-preview"

    # Generation Parameters
    temperature: float = 0.0  # Deterministic by default
    max_tokens: Optional[int] = None

    # Rate Limiting
    rate_limit_delay: float = 0.2  # Seconds between API calls

    # Logging
    verbose: bool = False

    def __post_init__(self):
        """Load from environment if not explicitly set."""
        if self.llm_endpoint is None:
            self.llm_endpoint = os.getenv("OPENAI_ENDPOINT")
        if self.llm_api_key is None:
            self.llm_api_key = os.getenv("OPENAI_KEY")
        if self.llm_deployment is None:
            self.llm_deployment = os.getenv("OPENAI_DEPLOYMENT")

    def validate(self) -> bool:
        """Validate that required configuration is present."""
        return all([
            self.llm_endpoint,
            self.llm_api_key,
            self.llm_deployment
        ])


@dataclass
class StructureExtractorConfig(ExtractorConfig):
    """
    Configuration for document structure extractors.

    Extends base config with settings specific to structure detection,
    such as sampling strategy and confidence thresholds.
    """
    # Sampling Strategy
    num_samples: int = 5  # Number of document samples to analyze
    sample_size: int = 3000  # Characters per sample
    overlap_between_samples: int = 200  # Overlap to catch split sections

    # Analysis Temperature (slightly higher for structure detection)
    structure_analysis_temperature: float = 0.1
    section_detection_temperature: float = 0.0
    classification_temperature: float = 0.0

    # Section Boundaries
    min_section_length: int = 100  # Minimum chars for a valid section
    max_sections: int = 100  # Maximum sections to detect

    # Confidence Thresholds
    min_structure_confidence: float = 0.5
    min_section_confidence: float = 0.4

    # Fallback Behavior
    use_fallback_for_unstructured: bool = True


@dataclass
class QuestionExtractorConfig(ExtractorConfig):
    """
    Configuration for question extractors.

    Extends base config with settings specific to question extraction,
    including chunking parameters and confidence thresholds.
    """
    # Chunking Parameters
    chunk_size: int = 6000  # Characters per chunk
    overlap: int = 500  # Overlap between chunks

    # Extraction Parameters
    confidence_threshold: float = 0.6  # Minimum confidence to include
    extract_question_numbers: bool = True
    extract_categories: bool = True

    # Question Detection
    detect_imperative_questions: bool = True  # "Describe...", "Provide..."
    detect_confirmation_requests: bool = True  # "Confirm that..."
    detect_checkbox_items: bool = True

    # Section Type Filtering (for structure-aware extractors)
    allowed_section_types: Optional[List] = None  # None = default behavior (QUESTIONNAIRE, PRICING)
    # When set, only extract from these section types
    # Example: [SectionType.QUESTIONNAIRE, SectionType.PRICING]

    # Deduplication
    deduplicate_questions: bool = True
    similarity_threshold: float = 0.9  # For fuzzy matching


@dataclass
class QAExtractorConfig(ExtractorConfig):
    """
    Configuration for question+answer extractors.

    Extends base config with settings specific to Q&A pair extraction,
    including answer detection parameters.
    """
    # Chunking Parameters
    chunk_size: int = 8000  # Larger chunks to capture Q+A pairs together
    overlap: int = 500  # Overlap between chunks

    # Extraction Parameters
    extract_answers: bool = True  # Set to False to extract questions only
    confidence_threshold: float = 0.5  # Minimum confidence to include

    # Answer Detection
    detect_empty_answers: bool = True  # Detect "[Insert answer]" patterns
    empty_answer_patterns: list = field(default_factory=lambda: [
        "enter response here",
        "insert details",
        "to be completed",
        "[insert",
        "n/a",
        "tbd"
    ])

    # Question/Answer Matching
    extract_question_numbers: bool = True
    extract_categories: bool = True

    # Section Type Filtering (for structure-aware extractors)
    allowed_section_types: Optional[List] = None  # None = default behavior (QUESTIONNAIRE, PRICING)
    # When set, only extract from these section types
    # Example: [SectionType.QUESTIONNAIRE, SectionType.PRICING]

    # Deduplication
    deduplicate_pairs: bool = True
    similarity_threshold: float = 0.9


@dataclass
class StrategyConfig:
    """
    Complete configuration for an extraction strategy.

    This high-level config specifies which strategy to use and how to
    configure it, including dependencies on other extractors.
    """
    # Strategy Selection
    strategy_name: str  # "simple", "structure_aware", "context_aware", "agentic"
    extractor_type: str  # "structure", "question", "qa"

    # Extractor-Specific Configuration
    config: ExtractorConfig = field(default_factory=ExtractorConfig)

    # Dependent Extractors
    use_structure_detection: bool = False
    structure_strategy: Optional[str] = None  # Strategy for structure detection

    # Custom Parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    # def __post_init__(self):
    #     """Validate strategy configuration."""
    #     valid_strategies = ["simple", "structure_aware", "context_aware", "agentic"]
    #     valid_types = ["structure", "question", "qa"]

    #     if self.strategy_name not in valid_strategies:
    #         raise ValueError(
    #             f"Invalid strategy '{self.strategy_name}'. "
    #             f"Must be one of: {valid_strategies}"
    #         )

    #     if self.extractor_type not in valid_types:
    #         raise ValueError(
    #             f"Invalid extractor type '{self.extractor_type}'. "
    #             f"Must be one of: {valid_types}"
    #         )

    @classmethod
    def from_env(cls, strategy_name: str = None, extractor_type: str = None) -> "StrategyConfig":
        """
        Create strategy config from environment variables.

        Args:
            strategy_name: Override default strategy from env
            extractor_type: Override default extractor type from env

        Returns:
            StrategyConfig with values from environment
        """
        strategy = strategy_name or os.getenv("EXTRACTION_STRATEGY", "simple")
        etype = extractor_type or os.getenv("EXTRACTOR_TYPE", "qa")
        use_structure = os.getenv("USE_STRUCTURE_DETECTION", "false").lower() == "true"
        structure_strat = os.getenv("STRUCTURE_STRATEGY", "structure_aware")

        # Select appropriate config class based on extractor type
        if etype == "structure":
            config = StructureExtractorConfig()
        elif etype == "question":
            config = QuestionExtractorConfig()
        else:  # qa
            config = QAExtractorConfig()

        return cls(
            strategy_name=strategy,
            extractor_type=etype,
            config=config,
            use_structure_detection=use_structure,
            structure_strategy=structure_strat if use_structure else None
        )
