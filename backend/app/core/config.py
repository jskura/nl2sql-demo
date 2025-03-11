import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# BigQuery configuration
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "samples")

# Dataset configuration for each demo phase
BASIC_DEMO_DATASET = os.getenv("BASIC_DEMO_DATASET", BIGQUERY_DATASET)
ENHANCED_DEMO_DATASET = os.getenv("ENHANCED_DEMO_DATASET", BIGQUERY_DATASET)
AGENT_DEMO_DATASET = os.getenv("AGENT_DEMO_DATASET", BIGQUERY_DATASET)

# Vertex AI configuration
USE_MOCK_RESPONSES = os.getenv("USE_MOCK_RESPONSES", "true").lower() == "true"
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
VERTEX_AI_MODEL = os.getenv("VERTEX_AI_MODEL", "gemini-1.5-pro")
VERTEX_AI_MAX_OUTPUT_TOKENS = int(os.getenv("VERTEX_AI_MAX_OUTPUT_TOKENS", "1024"))
VERTEX_AI_TEMPERATURE = float(os.getenv("VERTEX_AI_TEMPERATURE", "0.2"))
VERTEX_AI_SAFETY_THRESHOLD = os.getenv("VERTEX_AI_SAFETY_THRESHOLD", "BLOCK_ONLY_HIGH")  # Options: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "standard")  # standard, detailed, json
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
LOG_FILE_MAX_SIZE = int(os.getenv("LOG_FILE_MAX_SIZE", "10485760"))  # 10MB
LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "10"))
LOG_JSON_ENABLED = os.getenv("LOG_JSON_ENABLED", "true").lower() == "true"

# Application configuration
APP_NAME = "NL2SQL Agent Demo"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# API configuration
API_PREFIX = "/api"
CORS_ORIGINS = ["*"]  # For demo purposes, in production limit this 