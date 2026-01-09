"""
Configuration for Sessionizer Agent
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data-1766404178410.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sessions.json")

# Session Detection Thresholds (data-driven from ACTIVE events analysis)
TIME_GAP_THRESHOLD = 900        # 15 minutes (900 seconds) - gap larger than this = new session
MICRO_SWITCH_THRESHOLD = 20     # Not used anymore, kept for compatibility
MIN_SESSION_ACTIONS = 2         # Minimum actions to form a valid session

# Event Categories for Session Detection
# ACTIVE: User-initiated actions that define session boundaries
ACTIVE_EVENT_TYPES = [
    'field_input',        # User typing in fields
    'browser_activity',   # Web page visits
    'clipboard',          # Copy/paste actions
    'erp_activity',       # ERP system interactions
    'sap_interaction',    # SAP-specific interactions
]

# PASSIVE: Context events that attach to sessions (don't create boundaries)
PASSIVE_EVENT_TYPES = [
    'active_window_activity',  # Window focus changes (~32s heartbeat)
    'data_reconcilation',      # Background data sync
    'business_app_usage',      # App running heartbeat
]

# Event Type Filtering
# By default, process ALL event types. Add types here to exclude them.
EXCLUDE_EVENT_TYPES = []        # Empty = process all event types

# App Categories for Classification
APP_CATEGORIES = {
    "Communication": ["Microsoft Teams", "ms-teams.exe", "Outlook", "OUTLOOK.EXE", "Slack"],
    "Productivity": ["Excel", "EXCEL.EXE", "Word", "WINWORD.EXE", "PowerPoint", "POWERPNT.EXE"],
    "Browser": ["Chrome", "Google Chrome", "Edge", "Microsoft Edge", "Firefox"],
    "ERP": ["SAP", "Dynamics 365", "D365"],
    "Other": []
}

# Field types that indicate meaningful work
MEANINGFUL_FIELDS = [
    "Type a message",
    "Message",
    "Email address",
    "Search",
    "Enter code",
    "Subject",
    "Body"
]

# =============================================================================
# LLM SESSIONIZATION SETTINGS
# =============================================================================

# Local LLM API (LM Studio / Ollama / vLLM compatible)
LLM_API_URL = "http://192.168.0.101:1234/v1/chat/completions"
LLM_MODEL = "gemma-3-4b-it-qat"
LLM_TIMEOUT = 120  # seconds per request
LLM_MAX_RETRIES = 2

# OpenRouter (online)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_API_KEY = os.getenv(OPENROUTER_API_KEY_ENV, "")
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
OPENROUTER_FREE_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-2-9b-it:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "mistralai/mistral-7b-instruct:free"
]
OPENROUTER_REASONING_DEFAULT = True

# Batch Processing
LLM_BATCH_SIZE = 50  # events per batch (adjustable in UI)
LLM_BATCH_SIZE_MIN = 30
LLM_BATCH_SIZE_MAX = 200

# Prompt Template (user can customize in UI)
LLM_PROMPT_TEMPLATE = """You are a work session analyzer. Given a batch of telemetry events, group them into logical work sessions.

EVENTS:
{events}

Analyze these events and return ONLY valid JSON in this exact format:
{{
  "sessions": [
    {{
      "session_id": 1,
      "event_ids": [list of event IDs that belong together],
      "start_time": "earliest timestamp",
      "end_time": "latest timestamp",
      "primary_app": "most used app",
      "reasoning": "why these events are grouped",
      "intent": "what the user was trying to accomplish"
    }}
  ],
  "isolated_events": {{
    "event_ids": [IDs that don't fit any session],
    "reason": "why these couldn't be grouped"
  }}
}}

Return ONLY the JSON, no explanations."""

# =============================================================================
# INTENT EXTRACTION SETTINGS
# =============================================================================

INTENT_LLM_PROVIDER = "local"  # local, openai, anthropic, openrouter
INTENT_LLM_URL = "http://192.168.0.100:1234/v1/chat/completions"
INTENT_LLM_MODEL = "gemma-3-4b-it"
INTENT_OUTPUT_FILE = "output/session_intents.json"
INTENT_LLM_TIMEOUT = 30

# =============================================================================
# WORKFLOW EXTRACTION SETTINGS
# =============================================================================

WORKFLOW_GAP_THRESHOLD_HOURS = 4
WORKFLOW_MAX_SESSIONS = 10
WORKFLOW_MAX_SPAN_HOURS = 8
WORKFLOW_OUTPUT_FILE = "output/workflow_intents.json"

# =============================================================================
# SEMANTIC SESSIONIZER SETTINGS
# =============================================================================
SEMANTIC_HARD_BREAK_SEC = 300
SEMANTIC_SOFT_BREAK_SEC = 60
SEMANTIC_MICRO_SWITCH_SEC = 20
SEMANTIC_SIM_SAME = 0.58
SEMANTIC_SIM_SPLIT = 0.40

# Ollama Embedding Settings
OLLAMA_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "embeddinggemma:300m"

# =============================================================================
# APP CLASSIFIER SETTINGS
# =============================================================================

# Enable/disable auto-classification of unknown apps
APP_CLASSIFIER_ENABLED = True

# Auto-save classifications to pattern_classifications.json
APP_CLASSIFIER_AUTO_SAVE = True

# LLM provider for classification ("local" or "openrouter")
APP_CLASSIFIER_LLM_PROVIDER = "local"

# =============================================================================
# USER CONFIG OVERRIDE
# =============================================================================
# Load user-specific settings from user_config.py if it exists
# This allows you to override defaults without modifying config.py
try:
    from user_config import *
    print("[OK] Loaded user_config.py - using your custom settings")
except ImportError:
    pass  # user_config.py doesn't exist, use defaults
