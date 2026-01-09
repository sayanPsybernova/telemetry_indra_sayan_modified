"""
App Classifier - Auto-classify unknown apps using LLM

Features:
- LLM-powered classification of unknown apps
- JSON cache persistence to pattern_classifications.json
- Graceful fallback on errors (never blocks pipeline)
- Reuses LLMSessionizer infrastructure
"""

import json
import os
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Module-level cache (mirrors parser.py pattern)
_classification_cache = {}
_cache_loaded = False
_cache_file_path = None


def _get_cache_file_path() -> str:
    """Get path to pattern_classifications.json cache file."""
    global _cache_file_path

    if _cache_file_path:
        return _cache_file_path

    # Try multiple locations (same as parser.py lines 167-173)
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'output', 'pattern_classifications.json'),
        os.path.join(os.path.dirname(__file__), '..', 'pattern_classifications.json'),
    ]

    for path in possible_paths:
        path = os.path.normpath(path)
        if os.path.exists(path):
            _cache_file_path = path
            return path

    # Use first path if none exist
    _cache_file_path = os.path.normpath(possible_paths[0])
    return _cache_file_path


def _load_classification_cache() -> dict:
    """Load classification cache from JSON file."""
    global _classification_cache, _cache_loaded

    if _cache_loaded:
        return _classification_cache

    cache_path = _get_cache_file_path()

    if not os.path.exists(cache_path):
        _cache_loaded = True
        return {}

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            classifications = json.load(f)

        # Build lookup dict
        for entry in classifications:
            raw = entry.get('raw_pattern', '').lower().strip()
            if raw:
                _classification_cache[raw] = entry

        logger.info(f"Loaded {len(_classification_cache)} classifications from app_classifier cache")
        _cache_loaded = True
        return _classification_cache

    except Exception as e:
        logger.warning(f"Error loading app_classifier cache: {e}")
        _cache_loaded = True
        return {}


def get_cached_classification(app_name: str) -> Optional[Dict]:
    """
    Check if app_name has a cached classification.

    Args:
        app_name: Application name to look up

    Returns:
        Classification dict if found and accepted, None otherwise
    """
    if not app_name:
        return None

    cache = _load_classification_cache()
    app_lower = app_name.lower().strip()

    # Check exact match
    entry = cache.get(app_lower)
    if entry and entry.get('status') == 'accepted':
        return entry

    # Check without .exe suffix
    if app_lower.endswith('.exe'):
        app_without_exe = app_lower[:-4]
        entry = cache.get(app_without_exe)
        if entry and entry.get('status') == 'accepted':
            return entry

    return None


def save_classification(raw_app: str, classification: Dict) -> bool:
    """
    Save a new classification to the JSON cache.

    Args:
        raw_app: Raw application name from telemetry
        classification: Classification dict from LLM

    Returns:
        True if saved successfully, False otherwise
    """
    if not raw_app or not classification:
        return False

    try:
        from config import APP_CLASSIFIER_AUTO_SAVE
    except ImportError:
        logger.warning("Cannot import APP_CLASSIFIER_AUTO_SAVE from config")
        return False

    if not APP_CLASSIFIER_AUTO_SAVE:
        return False

    try:
        cache_path = _get_cache_file_path()

        # Load existing
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                classifications = json.load(f)
        else:
            classifications = []

        # Create new entry (match existing format)
        new_entry = {
            'raw_pattern': raw_app,
            'action_sample': '',
            'event_count': 1,
            'llm_category': classification.get('app_category'),
            'llm_work_related': classification.get('is_work_related'),
            'llm_suggested_name': classification.get('suggested_name'),
            'status': 'accepted',
            'classified_at': datetime.now().isoformat(),
            'source': 'app_classifier'
        }

        # Check for duplicates
        existing_idx = None
        for i, entry in enumerate(classifications):
            if entry.get('raw_pattern', '').lower() == raw_app.lower():
                existing_idx = i
                break

        if existing_idx is not None:
            classifications[existing_idx] = new_entry
            logger.info(f"Updated classification for {raw_app}")
        else:
            classifications.append(new_entry)
            logger.info(f"Added new classification for {raw_app}")

        # Save
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(classifications, f, indent=2, ensure_ascii=False)

        # Update in-memory cache
        _classification_cache[raw_app.lower()] = new_entry

        # Reload parser cache
        try:
            from src.parser import reload_classifications
            reload_classifications()
        except ImportError:
            logger.warning("Cannot reload parser classifications")

        return True

    except Exception as e:
        logger.error(f"Failed to save classification: {e}")
        return False


def _call_llm_for_classification(app_name: str, context: Optional[str] = None) -> Optional[Dict]:
    """
    Internal function to call LLM for app classification.

    Args:
        app_name: Application name to classify
        context: Optional context string

    Returns:
        Classification dict or None on failure
    """
    try:
        from src.llm_sessionizer import LLMSessionizer
        from config import (
            LLM_API_URL, LLM_MODEL, LLM_TIMEOUT,
            APP_CLASSIFIER_LLM_PROVIDER
        )

        # Determine LLM config
        if APP_CLASSIFIER_LLM_PROVIDER == "local":
            api_url = LLM_API_URL
            model = LLM_MODEL
            provider = "local"
            api_key = None
        else:
            from config import OPENROUTER_API_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL
            api_url = OPENROUTER_API_URL
            model = OPENROUTER_MODEL
            provider = "openrouter"
            api_key = OPENROUTER_API_KEY

        # Build prompt
        prompt = f"""Classify this application from user telemetry.

APPLICATION TO CLASSIFY: {app_name}
Context: {context or 'No context provided'}

IMPORTANT: Classify the APPLICATION "{app_name}", NOT the context.

Respond with JSON only:
{{
    "app_category": "Browser|Office|Communication|FileManager|System|ERP|Other",
    "is_work_related": "Yes|No|Maybe",
    "suggested_name": "clean normalized name"
}}

Examples:
- "chrome.exe" -> {{"app_category": "Browser", "is_work_related": "Yes", "suggested_name": "Chrome"}}
- "ShellExperienceHost.exe" -> {{"app_category": "System", "is_work_related": "No", "suggested_name": "Shell Experience Host"}}
- "TallyPrime.exe" -> {{"app_category": "ERP", "is_work_related": "Yes", "suggested_name": "Tally"}}
"""

        # Create LLM instance (reuse LLMSessionizer)
        llm = LLMSessionizer(
            api_url=api_url,
            model=model,
            prompt_template="",
            timeout=LLM_TIMEOUT,
            provider=provider,
            api_key=api_key
        )

        # Call LLM
        response_text, error = llm.call_llm(prompt)

        if error:
            logger.warning(f"LLM call failed for {app_name}: {error}")
            return None

        # Extract JSON (robust parsing)
        result = llm.extract_json(response_text)

        if result:
            logger.info(f"Classified {app_name} -> {result.get('suggested_name')}")
            return result
        else:
            logger.warning(f"Failed to parse LLM response for {app_name}")
            return None

    except Exception as e:
        logger.error(f"Error in LLM classification for {app_name}: {e}")
        return None


def classify_app(app_name: str, context: Optional[str] = None) -> Dict:
    """
    Classify an unknown app using LLM and cache the result.

    Main entry point called from normalize_app_name() in parser.py.

    Args:
        app_name: Application name to classify
        context: Optional context information

    Returns:
        Dict with keys:
            - success: bool
            - suggested_name: str (normalized app name)
            - from_cache: bool
            - error: str (optional, only on failure)
            - classification: dict (optional, full classification data)
    """
    try:
        from config import APP_CLASSIFIER_ENABLED
    except ImportError:
        logger.warning("Cannot import APP_CLASSIFIER_ENABLED from config")
        return {
            'success': False,
            'suggested_name': app_name,
            'from_cache': False,
            'error': 'Config import failed'
        }

    if not APP_CLASSIFIER_ENABLED:
        return {
            'success': False,
            'suggested_name': app_name,
            'from_cache': False,
            'error': 'Disabled by config'
        }

    if not app_name:
        return {
            'success': False,
            'suggested_name': app_name,
            'from_cache': False,
            'error': 'Empty app name'
        }

    # Check cache first
    cached = get_cached_classification(app_name)
    if cached:
        return {
            'success': True,
            'suggested_name': cached.get('llm_suggested_name', app_name),
            'from_cache': True,
            'classification': cached
        }

    # Call LLM
    classification = _call_llm_for_classification(app_name, context)

    if not classification:
        return {
            'success': False,
            'suggested_name': app_name,
            'from_cache': False,
            'error': 'LLM classification failed'
        }

    # Save to cache
    save_classification(app_name, classification)

    return {
        'success': True,
        'suggested_name': classification.get('suggested_name', app_name),
        'from_cache': False,
        'classification': classification
    }


# Test function
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Test cases
    test_apps = [
        "TallyPrime.exe",
        "ShellExperienceHost.exe",
        "UnknownApp123.exe"
    ]

    print("Testing App Classifier...")
    print("=" * 60)

    for app in test_apps:
        print(f"\nClassifying: {app}")
        result = classify_app(app)
        print(f"  Success: {result.get('success')}")
        print(f"  Suggested name: {result.get('suggested_name')}")
        print(f"  From cache: {result.get('from_cache')}")
        if not result.get('success'):
            print(f"  Error: {result.get('error')}")
