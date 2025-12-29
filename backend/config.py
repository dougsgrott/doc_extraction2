# config.py
"""
Centralized configuration management.
Supports both Azure Functions environment and local .env files.
"""

import os
from typing import Optional
from pathlib import Path


class Config:
    """Configuration manager that handles environment variables."""

    def __init__(self):
        """Initialize configuration by loading environment variables."""
        # Try to load .env.local if it exists (for local development)
        self._load_env_file()

    def _load_env_file(self):
        """Load environment variables from .env.local if it exists."""
        env_file = Path(__file__).parent / ".env.local"
        if env_file.exists():
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            if "=" in line:
                                key, value = line.split("=", 1)
                                # Only set if not already in environment
                                if key.strip() not in os.environ:
                                    os.environ[key.strip()] = value.strip()
            except Exception as e:
                print(f"Warning: Could not load .env.local: {e}")

    # Database Configuration
    @property
    def database_url(self) -> str:
        """Get database connection URL."""
        # Try DATABASE_URL first (standard), then DB_CONNECTION_STRING (Azure Functions)
        return os.environ.get("DATABASE_URL") or os.environ.get("DB_CONNECTION_STRING", "")

    # Azure Document Intelligence Configuration
    @property
    def di_endpoint(self) -> str:
        """Get Document Intelligence endpoint."""
        return os.environ.get("DI_ENDPOINT", "")

    @property
    def di_key(self) -> str:
        """Get Document Intelligence API key."""
        return os.environ.get("DI_KEY", "")

    # Azure OpenAI Configuration
    @property
    def openai_endpoint(self) -> str:
        """Get OpenAI endpoint."""
        return os.environ.get("OPENAI_ENDPOINT", "")

    @property
    def openai_key(self) -> str:
        """Get OpenAI API key."""
        return os.environ.get("OPENAI_KEY", "")

    @property
    def openai_deployment(self) -> str:
        """Get OpenAI deployment name."""
        return os.environ.get("OPENAI_DEPLOYMENT", "gpt-4o")

    @property
    def openai_api_version(self) -> str:
        """Get OpenAI API version."""
        return os.environ.get("OPENAI_API_VERSION", "2024-12-01-preview")

    # Azure Storage Configuration
    @property
    def azure_storage_connection(self) -> str:
        """Get Azure Storage connection string."""
        return os.environ.get("AzureWebJobsStorage", "")

    @property
    def input_container_name(self) -> str:
        """Get input container name."""
        return os.environ.get("INPUT_CONTAINER_NAME", "input-pdfs")

    # Local Testing Configuration
    @property
    def local_mode(self) -> bool:
        """Check if running in local mode."""
        return os.environ.get("LOCAL_MODE", "false").lower() in ("true", "1", "yes")

    @property
    def use_mock_services(self) -> bool:
        """Check if mock services should be used."""
        return os.environ.get("USE_MOCK_SERVICES", "false").lower() in ("true", "1", "yes")

    @property
    def dry_run(self) -> bool:
        """Check if running in dry-run mode (no DB writes)."""
        return os.environ.get("DRY_RUN", "false").lower() in ("true", "1", "yes")

    # Logging Configuration
    @property
    def log_level(self) -> str:
        """Get logging level."""
        return os.environ.get("LOG_LEVEL", "INFO")

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate that required configuration is present.

        Returns:
            Tuple of (is_valid, list of missing keys)
        """
        missing = []

        # Required for all modes
        if not self.database_url:
            missing.append("DATABASE_URL or DB_CONNECTION_STRING")

        # Required unless using mock services
        if not self.use_mock_services:
            if not self.di_endpoint:
                missing.append("DI_ENDPOINT")
            if not self.di_key:
                missing.append("DI_KEY")
            if not self.openai_endpoint:
                missing.append("OPENAI_ENDPOINT")
            if not self.openai_key:
                missing.append("OPENAI_KEY")

        return len(missing) == 0, missing

    def print_config(self, hide_secrets: bool = True):
        """Print current configuration (for debugging)."""
        print("=" * 60)
        print("Configuration:")
        print("=" * 60)
        print(f"Local Mode: {self.local_mode}")
        print(f"Use Mock Services: {self.use_mock_services}")
        print(f"Dry Run: {self.dry_run}")
        print(f"Log Level: {self.log_level}")
        print(f"Database URL: {self._mask(self.database_url) if hide_secrets else self.database_url}")
        print(f"DI Endpoint: {self.di_endpoint}")
        print(f"DI Key: {self._mask(self.di_key) if hide_secrets else self.di_key}")
        print(f"OpenAI Endpoint: {self.openai_endpoint}")
        print(f"OpenAI Key: {self._mask(self.openai_key) if hide_secrets else self.openai_key}")
        print(f"OpenAI Deployment: {self.openai_deployment}")
        print(f"OpenAI API Version: {self.openai_api_version}")
        print("=" * 60)

    @staticmethod
    def _mask(value: str, show_chars: int = 4) -> str:
        """Mask sensitive values for display."""
        if not value or len(value) <= show_chars:
            return "***"
        return value[:show_chars] + "..." + value[-show_chars:]


# Global config instance
config = Config()
