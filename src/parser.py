"""
Parser for extracting structured information from telemetry action field.

Supports multiple event types with resilient error handling:
1. field_input: "Context - Entered 'value' in field 'field_name' within 'window'"
2. browser_activity: "web_browsing page_visit on browser-title url"
3. active_window_activity: "app.exe: Window Title"
4. clipboard: "Copied type (N chars): content"
5. data_reconcilation: "Data reconciliation between App1, App2"
6. erp_activity: "SAP: action" or "ORACLE: action"
7. sap_interaction: ERP interaction events
8. business_app_usage: "app_running - office APP.EXE"

Error Handling Strategy:
- Each parser is wrapped in try/except
- If specific parser fails → fall back to generic parser
- If generic parser fails → return minimal result with error logged
- NEVER crash, NEVER lose data
"""
import re
import os
import json
import logging
from typing import Dict, Optional
from src.app_classifier import classify_app

# Setup logger
logger = logging.getLogger(__name__)

# Module-level cache for LLM classifications
_llm_classifications = {}
_classifications_loaded = False

# Parser constants
BROWSER_MAP = {
    'chrome': 'Google Chrome',
    'edge': 'Microsoft Edge',
    'firefox': 'Firefox',
    'safari': 'Safari',
    'opera': 'Opera'
}

GENERIC_APP_PATTERNS = [
    (r'excel', 'Excel'),
    (r'word', 'Word'),
    (r'outlook', 'Outlook'),
    (r'teams', 'Microsoft Teams'),
    (r'chrome', 'Google Chrome'),
    (r'edge', 'Microsoft Edge'),
    (r'firefox', 'Firefox'),
    (r'sap', 'SAP'),
]

WINDOW_APP_PATTERNS = [
    (r'Microsoft Teams', 'Microsoft Teams'),
    (r'Outlook', 'Outlook'),
    (r'Google Chrome', 'Google Chrome'),
    (r'Microsoft.*Edge', 'Microsoft Edge'),
    (r'Excel', 'Excel'),
    (r'Word', 'Word'),
    (r'PowerPoint', 'PowerPoint'),
    (r'OneDrive', 'OneDrive'),
    (r'SAP', 'SAP'),
    (r'Firefox', 'Firefox'),
    (r'ChatGPT', 'ChatGPT'),
    (r'OpenAI', 'OpenAI'),
    (r'File Explorer', 'File Explorer'),
    (r'7-Zip', '7-Zip'),
    (r'Adobe Reader', 'Adobe Acrobat'),
    (r'Adobe Acrobat', 'Adobe Acrobat'),
]

MISCLASSIFIED_PATTERNS = [
    # Email subjects (start with FW:, RE:, etc.)
    (r'^(fw|re|fwd):', 'Outlook'),
    # File paths or filenames with dates/numbers
    (r'^\d{8}_', 'File Explorer'),
    (r'^\d+\s*(december|january|february|march|april|may|june|july|august|september|october|november)', 'File Explorer'),
    # Standalone month names (not apps - trigger context inference)
    (r'^(january|february|march|april|may|june|july|august|september|october|november|december)$', None),
    # Document/spreadsheet names (trigger context inference)
    (r'.*\bsheet\s+(for|till|of)', None),  # "PO Sheet for...", "Expense Sheet for..."
    (r'.*\bexpense\s+sheet', None),  # "Ola Expense Sheet"
    (r'.*\binvoice\s+(for|of|submission)', None),  # "Invoice for Dec", "Submission of Invoice"
    (r'.*\bstatement\s+(for|till)', None),  # "Statement till December"
    (r'.*\b(po|purchase order)\s+sheet', None),  # "PO Sheet"
    (r'.*\btimesheet.*\bfor\b', None),  # "Timesheet for Nov"
    # Payment/invoice references
    (r'payment\s+made\s+on', 'Outlook'),
    (r'request\s+for\s+invoice', 'Outlook'),
    # System dialogs
    (r'save\s+print\s+output', 'Print Dialog'),
    (r'save\s+as', 'File Explorer'),
    (r'message\s*\(html\)', 'Outlook'),
    (r'email\s+account\s+setup', 'Outlook'),
]

NORMALIZATION_MAP = {
    # Microsoft Office
    "outlook.exe": "Outlook",
    "outlook": "Outlook",
    "ms-teams.exe": "Microsoft Teams",
    "microsoft teams": "Microsoft Teams",
    "teams": "Microsoft Teams",
    "excel.exe": "Excel",
    "excel": "Excel",
    "winword.exe": "Word",
    "word": "Word",
    "powerpnt.exe": "PowerPoint",
    "powerpoint": "PowerPoint",
    # Browsers
    "chrome.exe": "Google Chrome",
    "google chrome": "Google Chrome",
    "chrome": "Google Chrome",
    "msedge.exe": "Microsoft Edge",
    "microsoft edge": "Microsoft Edge",
    "edge": "Microsoft Edge",
    "firefox.exe": "Firefox",
    "firefox": "Firefox",
    # Cloud storage
    "onedrive.exe": "OneDrive",
    "onedrive": "OneDrive",
    "microsoft onedrive": "OneDrive",
    # ERP
    "sap": "SAP",
    "oracle": "Oracle",
    "dynamics": "Dynamics 365",
    "netsuite": "NetSuite",
    # AI tools
    "chatgpt": "ChatGPT",
    "openai": "OpenAI",
    # File system
    "file explorer": "File Explorer",
    "explorer.exe": "File Explorer",
    "explorer": "File Explorer",
    # Text editors
    "notepad.exe": "Notepad",
    "notepad": "Notepad",
    "notepad++": "Notepad++",
    # PDF viewers
    "adobe acrobat": "Adobe Acrobat",
    "acrobat reader": "Adobe Acrobat",
    "adobe acrobat reader": "Adobe Acrobat",
    "acrord32.exe": "Adobe Acrobat",
    # Archive tools
    "7zfm.exe": "7-Zip",
    "7-zip": "7-Zip",
    # System dialogs
    "print dialog": "Print Dialog",
    "save print output as": "Print Dialog",
    # Special
    "clipboard": "Clipboard",
    "data reconciliation": "Data Reconciliation",
    "xlmain": "Excel",
    "nuidialog": "System Dialog",
}


def load_llm_classifications(force_reload: bool = False) -> dict:
    """
    Load LLM-classified patterns from JSON file.

    Returns a dict mapping raw_pattern (lowercase) -> suggested_name
    Only includes entries where status == 'accepted'

    Args:
        force_reload: If True, reload from file even if already loaded
    """
    global _llm_classifications, _classifications_loaded

    if _classifications_loaded and not force_reload:
        return _llm_classifications

    _llm_classifications = {}

    # Look for JSON file in multiple locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'output', 'pattern_classifications.json'),
        os.path.join(os.path.dirname(__file__), '..', 'pattern_classifications.json'),
    ]

    for json_path in possible_paths:
        json_path = os.path.normpath(json_path)
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    classifications = json.load(f)

                # Build lookup dict (only accepted entries)
                for entry in classifications:
                    if entry.get('status') == 'accepted':
                        raw = entry.get('raw_pattern', '').lower().strip()
                        suggested = entry.get('llm_suggested_name')
                        if raw and suggested:
                            _llm_classifications[raw] = suggested

                logger.info(f"Loaded {len(_llm_classifications)} LLM classifications from {json_path}")
                break

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in {json_path}: {e}")
            except Exception as e:
                logger.warning(f"Error loading classifications from {json_path}: {e}")

    _classifications_loaded = True
    return _llm_classifications


def reload_classifications():
    """
    Reload classifications from JSON file.
    Call this after saving new classifications.
    """
    return load_llm_classifications(force_reload=True)


def _init_parse_result(action: str, action_type: str) -> Dict:
    return {
        "app": None,
        "context": None,
        "field": None,
        "value": None,
        "window": None,
        "raw_action": action,
        "event_type": action_type,
        "parse_status": "success",
        "parse_error": None
    }


def parse_action(action: str, action_type: str) -> Dict:
    """
    Parse action with resilient error handling.

    Error handling strategy:
    1. Try specific parser for the event type
    2. If specific parser fails → fall back to generic parser
    3. If generic parser fails → return minimal result with error logged
    4. NEVER crash, NEVER lose data

    Args:
        action: Raw action string from telemetry
        action_type: Type of action (field_input, browser_activity, etc.)

    Returns:
        Dictionary with extracted fields, always includes raw_action and event_type
    """
    result = _init_parse_result(action, action_type)

    # Route to appropriate parser
    parser_func = PARSER_MAP.get(action_type)

    if parser_func:
        # Try specific parser first
        try:
            result = parser_func(action, result)
        except Exception as e:
            # Specific parser failed → fall back to generic
            logger.warning(f"Parser {parser_func.__name__} failed for '{action[:50]}...': {e}")
            result["parse_error"] = f"{parser_func.__name__}: {str(e)}"
            try:
                result = parse_generic(action, result)
                result["parse_status"] = "fallback"
            except Exception as e2:
                # Generic also failed → return minimal result
                logger.error(f"Generic parser also failed: {e2}")
                result["parse_status"] = "error"
                result["parse_error"] = f"All parsers failed: {str(e2)}"
    else:
        # Unknown type → use generic parser
        try:
            result = parse_generic(action, result)
            result["parse_status"] = "unparsed"
        except Exception as e:
            # Generic failed → return minimal result
            logger.error(f"Generic parser failed for unknown type '{action_type}': {e}")
            result["parse_status"] = "error"
            result["parse_error"] = str(e)

    # Normalize app names (also wrapped for safety)
    try:
        result["app"] = normalize_app_name(result["app"], result, action_type)
    except Exception as e:
        logger.warning(f"App normalization failed: {e}")
        # Keep original app value

    return result


# =============================================================================
# SPECIFIC PARSERS
# =============================================================================

def parse_app_running(action: str, result: Dict) -> Dict:
    """
    Parse app_running actions like: "app_running - office OUTLOOK.EXE"
    """
    # Pattern: app_running - office APP.EXE
    match = re.search(r'app_running\s*-\s*office\s+(\S+)', action, re.IGNORECASE)
    if match:
        result["app"] = match.group(1)

    return result


def parse_browser_activity(action: str, result: Dict) -> Dict:
    """
    Parse browser_activity actions like:
    "web_browsing page_visit on chrome-ChatGPT https://chatgpt.com/"
    "web_browsing page_visit on edge-Microsoft Edge https://..."

    Extracts: browser, page title, URL, domain
    """
    # Pattern: web_browsing page_visit on {browser}-{title} {url}
    match = re.search(r'web_browsing\s+page_visit\s+on\s+(\w+)-(.+?)\s+(https?://\S+)', action, re.IGNORECASE)

    if match:
        browser = match.group(1).lower()
        page_title = match.group(2).strip()
        url = match.group(3).strip()

        # Normalize browser name
        result["app"] = BROWSER_MAP.get(browser, browser.title())
        result["context"] = page_title
        result["value"] = url

        # Extract domain from URL
        domain_match = re.search(r'https?://([^/]+)', url)
        if domain_match:
            result["field"] = domain_match.group(1)  # Store domain in field

        # Try to extract meaningful activity from page title
        result["window"] = f"{page_title} - {result['app']}"

    return result


def parse_field_input(action: str, result: Dict) -> Dict:
    """
    Parse field_input actions with multiple patterns.

    Examples:
    - "Chat | Trisha Banerjee | Microsoft Teams - Entered 'Hi' in field 'Type a message' within 'Chat | Trisha | Teams'"
    - "Inbox - invoice.payable@qbadvisory.com - Outlook - Entered 'Dear Team...' in field 'Message' within 'Inbox...'"
    - "Microsoft OneDrive - Entered 'email@domain.com' in field 'Enter your email' within 'Microsoft OneDrive'"
    """

    # Extract value: Entered 'VALUE' in field
    value_match = re.search(r"Entered\s+'([^']+)'", action)
    if value_match:
        result["value"] = value_match.group(1)

    # Extract field name: in field 'FIELD_NAME'
    field_match = re.search(r"in field\s+'([^']+)'", action)
    if field_match:
        result["field"] = field_match.group(1)

    # Extract window: within 'WINDOW'
    window_match = re.search(r"within\s+'([^']+)'", action)
    if window_match:
        result["window"] = window_match.group(1)
        # Also try to extract app from window
        result["app"] = extract_app_from_window(window_match.group(1))

    # Extract context (everything before " - Entered")
    context_match = re.search(r'^(.+?)\s*-\s*Entered', action)
    if context_match:
        context = context_match.group(1).strip()
        result["context"] = context

        # Try to extract app from context if not found yet
        if not result["app"]:
            result["app"] = extract_app_from_context(context)

    return result


def parse_active_window(action: str, result: Dict) -> Dict:
    """
    Parse active_window_activity actions like:
    "AcroRd32.exe: Security Warning"
    "explorer.exe: Anindya - File Explorer"
    "7zFM.exe: 7-Zip"

    Extracts: app (exe name), window/context (title)
    """
    # Pattern: {exe}: {window_title}
    match = re.match(r'^([^:]+\.exe):\s*(.+)$', action, re.IGNORECASE)
    if match:
        result["app"] = match.group(1)
        result["window"] = match.group(2).strip()
        result["context"] = result["window"]  # Use window as context
    else:
        # Fallback: try to extract any .exe
        exe_match = re.search(r'(\w+\.exe)', action, re.IGNORECASE)
        if exe_match:
            result["app"] = exe_match.group(1)
        result["window"] = action
        result["context"] = action

    return result


def parse_clipboard(action: str, result: Dict) -> Dict:
    """
    Parse clipboard actions like:
    "Copied short_text (76 chars): C:\\Users\\anindya..."
    "Copied text_paragraph (1035 chars): MEGHBELA CABLE..."

    Extracts: clipboard_type, char_count, content_preview
    """
    # Pattern: Copied {type} ({chars} chars): {content}
    match = re.match(r'^Copied\s+(\w+)\s+\((\d+)\s+chars?\):\s*(.*)$', action, re.IGNORECASE)
    if match:
        clipboard_type = match.group(1)  # short_text, text_paragraph, etc.
        char_count = match.group(2)
        content = match.group(3)

        result["app"] = "Clipboard"
        result["field"] = clipboard_type
        result["value"] = content[:100] if content else None  # Truncate for preview
        result["context"] = f"Clipboard ({char_count} chars)"
        result["window"] = f"Copied {clipboard_type}"
    else:
        # Fallback for other clipboard formats
        result["app"] = "Clipboard"
        result["value"] = action[:100] if action else None
        result["context"] = "Clipboard activity"

    return result


def parse_data_reconciliation(action: str, result: Dict) -> Dict:
    """
    Parse data_reconciliation actions like:
    "Data reconciliation between XLMAIN, NUIDialog"

    Extracts: apps involved in reconciliation
    """
    match = re.match(r'^Data reconciliation between\s+(.+),\s*(.+)$', action, re.IGNORECASE)
    if match:
        app1 = match.group(1).strip()
        app2 = match.group(2).strip()
        result["app"] = app1  # Primary app
        result["context"] = f"Reconciliation: {app1} <-> {app2}"
        result["field"] = "data_sync"
        result["window"] = f"{app1} - {app2}"
    else:
        # Fallback
        result["app"] = "Data Reconciliation"
        result["context"] = action
        result["field"] = "data_sync"

    return result


def parse_erp_activity(action: str, result: Dict) -> Dict:
    """
    Parse ERP activity actions like:
    "SAP: invoice"
    "ORACLE: data_entry"
    "DYNAMICS: general_navigation"
    "Unknown activity type: sap_interaction"

    Extracts: erp_system, action_type
    """
    # Pattern 1: {ERP}: {action}
    match = re.match(r'^(SAP|ORACLE|DYNAMICS|NETSUITE):\s*(.+)$', action, re.IGNORECASE)
    if match:
        result["app"] = match.group(1).upper()
        result["field"] = match.group(2).strip()
        result["context"] = f"{result['app']} - {result['field']}"
        result["window"] = result["context"]
        return result

    # Pattern 2: Unknown activity type: {type}
    unknown_match = re.match(r'^Unknown activity type:\s*(.+)$', action, re.IGNORECASE)
    if unknown_match:
        activity_type = unknown_match.group(1).strip()
        result["app"] = "ERP"
        result["field"] = activity_type
        result["context"] = f"ERP - {activity_type}"
        result["parse_status"] = "partial"  # Mark as partially parsed
        return result

    # Fallback
    result["app"] = "ERP"
    result["context"] = action
    result["field"] = "unknown"

    return result


def parse_generic(action: str, result: Dict) -> Dict:
    """
    Generic fallback parser for unknown event types.
    Extracts what it can and flags for review.

    This parser NEVER fails - it always returns something.
    """
    # Try to find any .exe reference
    exe_match = re.search(r'(\w+\.exe)', action, re.IGNORECASE)
    if exe_match:
        result["app"] = exe_match.group(1)

    # Try to find URL
    url_match = re.search(r'(https?://\S+)', action)
    if url_match:
        result["value"] = url_match.group(1)

    # Try to find email
    email_match = re.search(r'[\w.-]+@[\w.-]+\.\w+', action)
    if email_match and not result["value"]:
        result["value"] = email_match.group(0)

    # Use full action as context (truncated)
    result["context"] = action[:100] if len(action) > 100 else action

    # If no app found, try to extract from common patterns
    if not result["app"]:
        # Check for common app names in the action
        for pattern, app_name in GENERIC_APP_PATTERNS:
            if re.search(pattern, action, re.IGNORECASE):
                result["app"] = app_name
                break

    return result


# =============================================================================
# PARSER MAP
# =============================================================================

PARSER_MAP = {
    "business_app_usage": parse_app_running,
    "field_input": parse_field_input,
    "browser_activity": parse_browser_activity,
    "active_window_activity": parse_active_window,
    "clipboard": parse_clipboard,
    "data_reconcilation": parse_data_reconciliation,  # Note: typo in source data
    "data_reconciliation": parse_data_reconciliation,  # Correct spelling too
    "erp_activity": parse_erp_activity,
    "sap_interaction": parse_erp_activity,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_app_from_window(window: str) -> Optional[str]:
    """
    Extract app name from window title.

    Examples:
    - "Chat | Trisha Banerjee | Microsoft Teams" -> "Microsoft Teams"
    - "Inbox - invoice.payable@qbadvisory.com - Outlook" -> "Outlook"
    - "New tab - Google Chrome" -> "Google Chrome"
    """
    # Known app patterns
    for pattern, app_name in WINDOW_APP_PATTERNS:
        if re.search(pattern, window, re.IGNORECASE):
            return app_name

    # If no known app found, try to get last part after |
    if '|' in window:
        parts = window.split('|')
        return parts[-1].strip()

    # Try last part after -
    if ' - ' in window:
        parts = window.split(' - ')
        return parts[-1].strip()

    return window


def extract_app_from_context(context: str) -> Optional[str]:
    """
    Extract app name from context string.

    Examples:
    - "Chat | Trisha Banerjee | Microsoft Teams" -> "Microsoft Teams"
    - "Inbox - invoice.payable@qbadvisory.com - Outlook" -> "Outlook"
    """
    return extract_app_from_window(context)  # Same logic applies


def infer_app_from_context(original_app: str, parsed_result: dict, action_type: str) -> str:
    """
    Infer actual application from context when document name extracted as app.

    Uses clues from:
    - Window title (file extensions)
    - Event type
    - Action text

    Returns inferred app or "Unknown" if can't determine.
    """
    window = parsed_result.get('window', '')
    context = parsed_result.get('context', '')
    action = parsed_result.get('action', '')

    # Check for file extensions in window/context
    if re.search(r'\.(xlsx?|xlsm)', window + context, re.IGNORECASE):
        return 'Excel'
    if re.search(r'\.(docx?)', window + context, re.IGNORECASE):
        return 'Word'
    if re.search(r'\.(pdf)', window + context, re.IGNORECASE):
        return 'Adobe Acrobat'
    if re.search(r'\.(pptx?)', window + context, re.IGNORECASE):
        return 'PowerPoint'

    # Check action type context
    if action_type in ['field_input', 'email_activity']:
        if 'message' in action.lower() or 'mail' in action.lower():
            return 'Outlook'

    # Check for browser indicators
    if 'http' in action or 'www' in action:
        return 'Microsoft Edge'  # Default browser

    # Can't determine - return Unknown
    return 'Unknown'


def normalize_app_name(
    app: Optional[str],
    parsed_result: Optional[dict] = None,
    action_type: Optional[str] = None
) -> Optional[str]:
    """
    Normalize app names to consistent format.

    Examples:
    - "OUTLOOK.EXE" -> "Outlook"
    - "ms-teams.exe" -> "Microsoft Teams"
    - "EXCEL.EXE" -> "Excel"
    - "AcroRd32.exe" -> "Adobe Acrobat"
    """
    if not app:
        return None

    app_lower = app.lower().strip()

    # Detect misclassified patterns (these are NOT apps)
    for pattern, correct_app in MISCLASSIFIED_PATTERNS:
        if re.search(pattern, app_lower, re.IGNORECASE):
            if correct_app is None and parsed_result:
                # Pattern detected document name - infer actual app from context
                return infer_app_from_context(app, parsed_result, action_type or 'unknown')
            elif correct_app is None:
                # No context available, can't infer
                return 'Unknown'
            return correct_app

    # Check for exact match first
    if app_lower in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[app_lower]

    # Check for partial match
    for key, value in NORMALIZATION_MAP.items():
        if key in app_lower:
            return value

    # Check LLM classifications before returning raw value
    llm_classifications = load_llm_classifications()
    if app_lower in llm_classifications:
        return llm_classifications[app_lower]

    # Also check without .exe suffix for flexibility
    app_without_exe = app_lower.replace('.exe', '')
    if app_without_exe in llm_classifications:
        return llm_classifications[app_without_exe]

    # NEW: Try auto-classification for unknown apps
    try:
        result = classify_app(app_name=app, context=None)
        if result.get('success'):
            suggested_name = result.get('suggested_name', app.strip())
            if not result.get('from_cache'):
                logger.info(f"Auto-classified: {app} -> {suggested_name}")
            return suggested_name
        else:
            logger.debug(f"Classification failed: {result.get('error')}")
    except Exception as e:
        logger.warning(f"App classifier exception: {e}")

    # Return original with title case if no match
    return app.strip()


def extract_person_from_context(context: str) -> Optional[str]:
    """
    Extract person name from chat context.

    Example: "Chat | Trisha Banerjee | Microsoft Teams" -> "Trisha Banerjee"
    """
    if '|' in context:
        parts = [p.strip() for p in context.split('|')]
        if len(parts) >= 2:
            # Usually format is: Chat | Person Name | App
            # or: Type | Person Name | App
            if parts[0].lower() in ['chat', 'call', 'meeting']:
                return parts[1]

    return None


def get_session_type(app: str, context: str = None, field: str = None) -> str:
    """
    Determine the type of activity based on app and context.

    Returns: Communication, Email, File Management, Web Browsing, Document, ERP,
             Text Editing, Document Viewing, System Dialog, Cloud Storage,
             Clipboard, Data Sync, AI Tools, Other
    """
    if not app:
        return "Other"

    app_lower = app.lower()

    # Communication (chat, video calls)
    if any(x in app_lower for x in ['teams', 'slack', 'zoom', 'meet', 'skype']):
        return "Communication"

    # Email
    if 'outlook' in app_lower:
        return "Email"

    # Document/Spreadsheet (Microsoft Office)
    if any(x in app_lower for x in ['excel', 'word', 'powerpoint']):
        return "Document"

    # Web Browsing
    if any(x in app_lower for x in ['chrome', 'edge', 'firefox', 'browser', 'safari', 'opera']):
        return "Web Browsing"

    # File Management (File Explorer, file operations)
    if any(x in app_lower for x in ['file explorer', 'explorer', 'finder', '7-zip', '7zip']):
        return "File Management"

    # Text Editing
    if any(x in app_lower for x in ['notepad', 'sublime', 'vscode', 'vim', 'emacs']):
        return "Text Editing"

    # PDF/Document Viewing
    if any(x in app_lower for x in ['acrobat', 'pdf', 'reader', 'foxit']):
        return "Document Viewing"

    # System Dialogs (Print, Save As, etc.)
    if any(x in app_lower for x in ['print dialog', 'save as', 'open dialog', 'nuidialog', 'system dialog']):
        return "System Dialog"

    # ERP
    if any(x in app_lower for x in ['sap', 'd365', 'dynamics', 'oracle', 'netsuite', 'erp']):
        return "ERP"

    # Cloud Storage
    if any(x in app_lower for x in ['onedrive', 'dropbox', 'google drive', 'box']):
        return "Cloud Storage"

    # AI Tools
    if any(x in app_lower for x in ['chatgpt', 'openai', 'copilot', 'claude']):
        return "AI Tools"

    # Clipboard
    if 'clipboard' in app_lower:
        return "Clipboard"

    # Data Sync/Reconciliation
    if any(x in app_lower for x in ['reconciliation', 'data reconciliation', 'xlmain']):
        return "Data Sync"

    return "Other"


def classify_isolated_event(event: Dict) -> str:
    """
    Categorize isolated events for easier review in the dashboard.

    Categories:
    - Login/OTP: Authentication-related entries (codes, passwords, verification)
    - Search: Search queries and lookups
    - Page Visit: Browser page visits
    - Email: Email-related actions
    - Approval: Approval/submit/confirm actions (potentially critical)
    - Clipboard: Clipboard operations
    - ERP: ERP system interactions
    - File Operation: File management actions
    - Other: Everything else

    Args:
        event: Dictionary with keys like 'raw_action', 'field', 'value', 'app'

    Returns:
        Category string
    """
    action = str(event.get('raw_action', '')).lower()
    field = str(event.get('field', '')).lower() if event.get('field') else ''
    value = str(event.get('value', '')).lower() if event.get('value') else ''
    app = str(event.get('app', '')).lower() if event.get('app') else ''
    event_type = str(event.get('event_type', '')).lower() if event.get('event_type') else ''

    # Login/Authentication - codes, passwords, OTPs
    login_keywords = ['code', 'otp', 'verification', 'password', 'pin', 'authenticate', 'sign in', 'login']
    if any(kw in field for kw in login_keywords):
        return 'Login/OTP'

    # Approval actions - potentially critical business actions
    approval_keywords = ['approve', 'submit', 'confirm', 'send', 'accept', 'reject', 'complete', 'finalize']
    if any(kw in field for kw in approval_keywords) or any(kw in action for kw in approval_keywords):
        return 'Approval'

    # Search queries
    search_keywords = ['search', 'find', 'query', 'lookup', 'filter']
    if any(kw in field for kw in search_keywords):
        return 'Search'

    # Quick page visits (browser activity)
    if 'page_visit' in action or 'web_browsing' in action or event_type == 'browser_activity':
        return 'Page Visit'

    # Clipboard
    if 'clipboard' in app or event_type == 'clipboard':
        return 'Clipboard'

    # ERP
    if any(x in app for x in ['sap', 'oracle', 'dynamics', 'erp']) or event_type in ['erp_activity', 'sap_interaction']:
        return 'ERP'

    # Email-related
    email_keywords = ['email', 'inbox', 'outlook', 'mail', 'subject', 'to:', 'cc:', 'bcc:']
    if any(kw in action for kw in email_keywords) or 'outlook' in app:
        return 'Email'

    # Communication
    if any(x in app for x in ['teams', 'slack', 'chat']):
        return 'Communication'

    # File operations
    if event_type == 'active_window_activity' or 'file explorer' in app or 'explorer' in app:
        return 'File Operation'

    # Data sync
    if event_type == 'data_reconcilation' or 'reconciliation' in action:
        return 'Data Sync'

    return 'Other'


# =============================================================================
# TEST FUNCTION
# =============================================================================

if __name__ == "__main__":
    # Test cases for all event types
    test_actions = [
        # Existing types
        ("Chat | Trisha Banerjee | Microsoft Teams - Entered 'Hi there' in field 'Type a message' within 'Chat | Trisha Banerjee | Microsoft Teams'", "field_input"),
        ("app_running - office OUTLOOK.EXE", "business_app_usage"),
        ("web_browsing page_visit on chrome-ChatGPT https://chatgpt.com/", "browser_activity"),

        # New types
        ("AcroRd32.exe: Silvery_20B (10-11).pdf - Adobe Reader", "active_window_activity"),
        ("explorer.exe: Anindya - File Explorer", "active_window_activity"),
        ("Copied short_text (76 chars): C:\\Users\\anindya\\Desktop\\Silvery", "clipboard"),
        ("Data reconciliation between XLMAIN, NUIDialog", "data_reconcilation"),
        ("SAP: invoice", "erp_activity"),
        ("ORACLE: data_entry", "erp_activity"),
        ("Unknown activity type: sap_interaction", "sap_interaction"),

        # Unknown type (should use generic parser)
        ("Some random action we don't know about", "unknown_type"),
    ]

    print("Testing parser with all event types...")
    print("=" * 70)

    for action, action_type in test_actions:
        result = parse_action(action, action_type)
        print(f"\nType: {action_type}")
        print(f"Input: {action[:60]}...")
        print(f"Parsed:")
        print(f"  app: {result['app']}")
        print(f"  context: {result['context']}")
        print(f"  field: {result['field']}")
        print(f"  value: {result['value'][:30] if result['value'] else None}...")
        print(f"  parse_status: {result['parse_status']}")
        print(f"  Session Type: {get_session_type(result['app'], result['context'], result['field'])}")
