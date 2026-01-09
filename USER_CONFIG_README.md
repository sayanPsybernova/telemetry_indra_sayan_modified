# User Configuration Guide

## Quick Start

Your LM Studio URL has been configured! You should now see:
```
[OK] Loaded user_config.py - using your custom settings
```

When you import config or run any script.

## What Was Changed

### 1. Created `user_config.py`
Your personal configuration file that overrides defaults.

**Current settings:**
- `LLM_API_URL = "http://192.168.0.100:1234/v1/chat/completions"`

### 2. Updated `config.py`
Now loads `user_config.py` automatically at the end, so your settings override defaults.

### 3. Created `.gitignore`
Protects your `user_config.py` from being overwritten by git updates.

## How to Customize Settings

### Option 1: Edit user_config.py (Recommended)
```python
# Open user_config.py and add any settings you want to override:

LLM_API_URL = "http://192.168.0.100:1234/v1/chat/completions"
LLM_MODEL = "your-preferred-model"
LLM_TIMEOUT = 180
APP_CLASSIFIER_ENABLED = True
```

**Benefits:**
- ✅ Settings persist across code updates
- ✅ Not tracked by git
- ✅ Easy to manage

### Option 2: Edit config.py Directly
```python
# Change settings directly in config.py
# WARNING: May be overwritten when you update the code
```

## Available Settings You Can Override

From `config.py`, you can override any of these in `user_config.py`:

### LLM Settings
- `LLM_API_URL` - Your LM Studio URL
- `LLM_MODEL` - Model name
- `LLM_TIMEOUT` - Timeout in seconds
- `LLM_MAX_RETRIES` - Number of retries on failure

### App Classifier Settings
- `APP_CLASSIFIER_ENABLED` - Enable/disable auto-classification
- `APP_CLASSIFIER_AUTO_SAVE` - Auto-save classifications
- `APP_CLASSIFIER_LLM_PROVIDER` - "local" or "openrouter"

### Session Settings
- `TIME_GAP_THRESHOLD` - Session break threshold (seconds)
- `MIN_SESSION_ACTIONS` - Minimum actions per session

## Testing Your Configuration

### Test LLM Connection
```bash
python -c "from src.llm_sessionizer import LLMSessionizer; import config; print(LLMSessionizer.test_connection(config.LLM_API_URL, config.LLM_MODEL))"
```

### Test App Classifier
```bash
python test_classifier.py
```

### Test Parser Integration
```bash
python -c "from src.parser import normalize_app_name; print(normalize_app_name('UnknownApp.exe'))"
```

## Verification

Your TallyPrime.exe test was successful! ✓

Classification saved to: `output/pattern_classifications.json`
```json
{
  "raw_pattern": "TallyPrime.exe",
  "llm_category": "ERP",
  "llm_work_related": "Yes",
  "llm_suggested_name": "TallyPrime",
  "status": "accepted",
  "source": "app_classifier"
}
```

## Troubleshooting

### "Connection refused" errors
1. Check LM Studio is running
2. Verify URL in `user_config.py` matches LM Studio
3. Test connection: `python -c "import requests; print(requests.get('http://192.168.0.100:1234'))"`

### Settings not loading
1. Make sure `user_config.py` is in the root directory (same as `config.py`)
2. Check for syntax errors in `user_config.py`
3. Look for the `[OK] Loaded user_config.py` message

### Need to change systems
Just edit `user_config.py` and change the `LLM_API_URL` to your new system's IP address.

Example:
```python
# For system at 192.168.0.101
LLM_API_URL = "http://192.168.0.101:1234/v1/chat/completions"

# For system at 192.168.0.100
LLM_API_URL = "http://192.168.0.100:1234/v1/chat/completions"

# For localhost
LLM_API_URL = "http://localhost:1234/v1/chat/completions"
```

## What Happens Now

When you run your sessionizer:
1. Unknown apps will be auto-classified using your LM Studio
2. Classifications are cached in `output/pattern_classifications.json`
3. Future runs use the cache (instant, no LLM calls)
4. You'll see log messages like:
   ```
   Auto-classified: UnknownApp.exe -> Unknown App
   ```

## Files Created/Modified

- ✅ `user_config.py` - Your personal settings (ignored by git)
- ✅ `.gitignore` - Protects your settings
- ✅ `config.py` - Now loads user_config.py
- ✅ `test_classifier.py` - Quick test script
- ✅ `output/pattern_classifications.json` - Updated with TallyPrime
