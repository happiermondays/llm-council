"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# Anthropic API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Council members - list of Anthropic model identifiers
COUNCIL_MODELS = [
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-fable-5-1",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "claude-opus-5-5"

# Model used for generating short conversation titles (fast and cheap)
TITLE_MODEL = "claude-haiku-5-5"

# Anthropic Messages API endpoint and version
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Max tokens per response
MAX_TOKENS = 4096

# Data directory for conversation storage
DATA_DIR = "data/conversations"
