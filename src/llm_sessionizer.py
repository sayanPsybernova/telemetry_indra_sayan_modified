"""
LLM-powered sessionization using local LLM API.

Features:
- OpenAI-compatible API calls
- Batch processing with progress tracking
- Robust JSON extraction from messy responses
- Retry logic with exponential backoff
- Streaming support for real-time display
- Debug logging of raw responses
"""

import requests
import json
import re
import time
import logging
from typing import List, Dict, Optional, Tuple, Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMSessionizer:
    """LLM-powered session analyzer."""

    def __init__(self,
                 api_url: str,
                 model: str,
                 prompt_template: str,
                 timeout: int = 120,
                 max_retries: int = 2,
                 provider: str = "local",
                 api_key: Optional[str] = None,
                 reasoning_enabled: bool = False,
                 extra_headers: Optional[Dict[str, str]] = None):
        """
        Initialize the LLM Sessionizer.

        Args:
            api_url: OpenAI-compatible API endpoint URL
            model: Model name/ID to use
            prompt_template: Template with {events} placeholder
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts on failure
            provider: LLM provider identifier ("local" or "openrouter")
            api_key: API key for providers that require it
            reasoning_enabled: Enable reasoning mode when supported
            extra_headers: Optional additional headers for requests
        """
        self.api_url = api_url
        self.model = model
        self.prompt_template = prompt_template
        self.timeout = timeout
        self.max_retries = max_retries
        self.provider = provider
        self.api_key = api_key
        self.reasoning_enabled = reasoning_enabled
        self.extra_headers = extra_headers or {}
        self.headers = self._build_headers()
        self.raw_responses = []  # For debugging

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.provider == "openrouter" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for key, value in self.extra_headers.items():
            if value:
                headers[key] = value
        return headers

    def _build_payload(self, prompt: str, stream: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4096
        }
        if stream:
            payload["stream"] = True
        if self.provider == "openrouter" and self.reasoning_enabled:
            payload["reasoning"] = {"enabled": True}
        return payload

    def format_events_for_prompt(self, events: List[Dict]) -> str:
        """
        Format events as readable text for the LLM.

        Args:
            events: List of event dictionaries

        Returns:
            Formatted string representation of events
        """
        lines = []
        for event in events:
            # Extract key fields
            event_id = event.get('id', event.get('index', 'N/A'))
            timestamp = event.get('timestamp', 'N/A')
            if hasattr(timestamp, 'isoformat'):
                timestamp = timestamp.isoformat()
            elif hasattr(timestamp, 'strftime'):
                timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')

            event_type = event.get('type', event.get('original_type', 'unknown'))
            app = event.get('app', 'Unknown')
            field = event.get('field', '')
            value = event.get('value', '')

            # Truncate long values
            if value and len(str(value)) > 50:
                value = str(value)[:50] + "..."

            # Format line
            line = f"ID: {event_id} | {timestamp} | {event_type} | {app}"
            if field:
                line += f" - {field}"
            if value:
                line += f" = '{value}'"

            lines.append(line)

        return "\n".join(lines)

    def _build_prompt(self, events: List[Dict]) -> str:
        formatted_events = self.format_events_for_prompt(events)
        return self.prompt_template.format(events=formatted_events)

    def call_llm(self, prompt: str) -> Tuple[str, Optional[str]]:
        """
        Call the LLM API (non-streaming).

        Args:
            prompt: The prompt to send

        Returns:
            Tuple of (response_text, error_message)
        """
        try:
            payload = self._build_payload(prompt)
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                try:
                    content = data['choices'][0]['message']['content']
                except (KeyError, IndexError, TypeError):
                    return "", f"Unexpected response format: {str(data)[:200]}"
                return content, None
            else:
                return "", f"HTTP {response.status_code}: {response.text[:200]}"

        except requests.exceptions.ConnectionError:
            return "", "Connection refused. Is LM Studio running?"
        except requests.exceptions.Timeout:
            return "", f"Request timed out after {self.timeout}s"
        except Exception as e:
            return "", str(e)

    def call_llm_streaming(self, prompt: str,
                          on_token: Optional[Callable[[str], None]] = None) -> Tuple[str, Optional[str]]:
        """
        Call LLM API with streaming support.

        Args:
            prompt: The prompt to send
            on_token: Callback function called with each token as it arrives

        Returns:
            Tuple of (full_response_text, error_message)
        """
        try:
            payload = self._build_payload(prompt, stream=True)
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                stream=True,
                timeout=self.timeout
            )

            if response.status_code != 200:
                return "", f"HTTP {response.status_code}: {response.text[:200]}"

            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith("data: "):
                        data = line_text[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            token = delta.get('content', '')
                            if token:
                                full_response += token
                                if on_token:
                                    on_token(token)
                        except json.JSONDecodeError:
                            pass

            return full_response, None

        except requests.exceptions.ConnectionError:
            return "", "Connection refused. Is LM Studio running?"
        except requests.exceptions.Timeout:
            return "", f"Request timed out after {self.timeout}s"
        except Exception as e:
            return "", str(e)

    def extract_json(self, response_text: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response using multiple strategies.

        Handles:
        - Clean JSON
        - JSON wrapped in markdown code blocks
        - JSON with leading/trailing text
        - Common formatting issues

        Args:
            response_text: Raw response from LLM

        Returns:
            Parsed JSON dict or None if extraction fails
        """
        if not response_text:
            return None

        # Strategy 1: Direct parse (clean response)
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code block
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON object by balanced braces
        brace_start = response_text.find('{')
        if brace_start != -1:
            depth = 0
            for i, char in enumerate(response_text[brace_start:], brace_start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response_text[brace_start:i+1])
                        except json.JSONDecodeError:
                            pass
                        break

        # Strategy 4: Try to fix common issues
        # Remove trailing commas, fix single quotes
        fixed = response_text
        # Find the JSON part first
        json_match = re.search(r'\{[\s\S]*\}', fixed)
        if json_match:
            fixed = json_match.group(0)
            # Fix trailing commas before } or ]
            fixed = re.sub(r',\s*}', '}', fixed)
            fixed = re.sub(r',\s*]', ']', fixed)
            # Try parsing
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        return None

    def process_batch(self, events: List[Dict]) -> Dict:
        """
        Process a single batch of events with retry logic.

        Args:
            events: List of event dictionaries

        Returns:
            Dict with success status, data or error, and attempt count
        """
        prompt = self._build_prompt(events)

        last_error = None
        response_text = None

        for attempt in range(self.max_retries + 1):
            response_text, error = self.call_llm(prompt)

            if error:
                last_error = error
                logger.warning(f"Attempt {attempt+1} failed: {error}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue

            # Log raw response for debugging
            self.raw_responses.append({
                'timestamp': datetime.now().isoformat(),
                'batch_size': len(events),
                'response': response_text[:1000] if response_text else None
            })

            # Try to extract JSON
            result = self.extract_json(response_text)
            if result:
                return {
                    'success': True,
                    'data': result,
                    'attempts': attempt + 1
                }

            last_error = "Failed to parse JSON from response"
            logger.warning(f"Attempt {attempt+1}: {last_error}")
            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        return {
            'success': False,
            'error': last_error,
            'attempts': self.max_retries + 1,
            'raw_response': response_text[:500] if response_text else None
        }

    def process_batch_streaming(self, events: List[Dict],
                                on_token: Optional[Callable[[str], None]] = None) -> Dict:
        """
        Process a single batch with streaming output.

        Args:
            events: List of event dictionaries
            on_token: Callback for each token

        Returns:
            Dict with success status and data
        """
        prompt = self._build_prompt(events)

        response_text, error = self.call_llm_streaming(prompt, on_token)

        if error:
            return {
                'success': False,
                'error': error,
                'attempts': 1,
                'raw_response': None
            }

        # Log raw response
        self.raw_responses.append({
            'timestamp': datetime.now().isoformat(),
            'batch_size': len(events),
            'response': response_text[:1000] if response_text else None
        })

        # Extract JSON
        result = self.extract_json(response_text)
        if result:
            return {
                'success': True,
                'data': result,
                'attempts': 1
            }

        return {
            'success': False,
            'error': "Failed to parse JSON from response",
            'attempts': 1,
            'raw_response': response_text[:500] if response_text else None
        }

    def process_all(self, events: List[Dict], batch_size: int,
                   progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict:
        """
        Process all events in batches.

        Args:
            events: All events to process
            batch_size: Number of events per batch
            progress_callback: Optional callback(current_batch, total_batches)

        Returns:
            Combined results with metadata
        """
        total_batches = (len(events) + batch_size - 1) // batch_size
        all_sessions = []
        all_isolated = []
        batch_results = []
        errors = []

        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            batch_num = i // batch_size + 1

            if progress_callback:
                progress_callback(batch_num, total_batches)

            result = self.process_batch(batch)
            batch_results.append(result)

            if result['success']:
                data = result['data']
                sessions = data.get('sessions', [])

                # Adjust session IDs to be globally unique
                for session in sessions:
                    session['batch_number'] = batch_num
                    session['session_id'] = f"{batch_num}_{session.get('session_id', 0)}"

                all_sessions.extend(sessions)

                isolated = data.get('isolated_events', {})
                if isinstance(isolated, list):
                    all_isolated.extend(isolated)
                elif isinstance(isolated, dict):
                    event_ids = isolated.get('event_ids', [])
                    if event_ids:
                        all_isolated.extend(event_ids)
            else:
                errors.append({
                    'batch': batch_num,
                    'error': result.get('error'),
                    'raw': result.get('raw_response')
                })

        return {
            'total_sessions': len(all_sessions),
            'sessions': all_sessions,
            'isolated_events': all_isolated,
            'isolated_count': len(all_isolated),
            'batches_processed': total_batches,
            'batch_results': batch_results,
            'error_count': len(errors),
            'errors': errors,
            'raw_responses': self.raw_responses,
            'timestamp': datetime.now().isoformat()
        }

    @staticmethod
    def test_connection(api_url: str, model: str, timeout: int = 10,
                        provider: str = "local",
                        api_key: Optional[str] = None,
                        reasoning_enabled: bool = False,
                        extra_headers: Optional[Dict[str, str]] = None) -> Dict:
        """
        Test LLM API connection with a simple prompt.

        Args:
            api_url: API endpoint URL
            model: Model name
            timeout: Timeout in seconds
            provider: LLM provider identifier ("local" or "openrouter")
            api_key: API key for providers that require it
            reasoning_enabled: Enable reasoning mode when supported
            extra_headers: Optional additional headers for requests

        Returns:
            Dict with success status, latency, and model info
        """
        test_prompt = 'Respond with exactly: {"status": "ok"}'

        try:
            if provider == "openrouter" and not api_key:
                return {"success": False, "error": "Missing OpenRouter API key"}

            headers = {"Content-Type": "application/json"}
            if provider == "openrouter" and api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            if extra_headers:
                for key, value in extra_headers.items():
                    if value:
                        headers[key] = value

            payload: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 50
            }
            if provider == "openrouter" and reasoning_enabled:
                payload["reasoning"] = {"enabled": True}

            start = time.time()
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            latency = time.time() - start

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "latency_ms": int(latency * 1000),
                    "model": data.get("model", model),
                    "message": "Connection successful!"
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}"
                }

        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Connection refused. Is LM Studio running?"}
        except requests.exceptions.Timeout:
            return {"success": False, "error": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def compare_with_rule_based(llm_sessions: List[Dict],
                                rule_sessions: List[Dict],
                                events_df) -> Dict:
        """
        Compare LLM sessionization vs rule-based sessionization.

        Args:
            llm_sessions: Sessions from LLM analysis
            rule_sessions: Sessions from rule-based analysis
            events_df: Original events DataFrame

        Returns:
            Comparison metrics and discrepancies
        """
        import pandas as pd

        # Build event-to-session mappings
        llm_mapping = {}  # event_id -> session_id
        for session in llm_sessions:
            for event_id in session.get('event_ids', []):
                llm_mapping[event_id] = session.get('session_id', 'unknown')

        rule_mapping = {}
        for session in rule_sessions:
            session_num = session.get('session_number', 0)
            for action in session.get('actions', []):
                # Try to get ID from action
                action_id = action.get('id', action.get('index'))
                if action_id is not None:
                    rule_mapping[action_id] = session_num

        # Calculate total events
        if isinstance(events_df, pd.DataFrame):
            total_events = len(events_df)
        else:
            total_events = len(events_df) if events_df else 0

        # Calculate agreement
        common_ids = set(llm_mapping.keys()) & set(rule_mapping.keys())
        matched = sum(1 for eid in common_ids if llm_mapping[eid] == rule_mapping[eid])
        agreement_rate = matched / len(common_ids) if common_ids else 0

        # Find discrepancies
        discrepancies = []
        for event_id in common_ids:
            llm_session = llm_mapping[event_id]
            rule_session = rule_mapping[event_id]
            if llm_session != rule_session:
                discrepancies.append({
                    'event_id': event_id,
                    'llm_session': llm_session,
                    'rule_session': rule_session
                })

        # Session statistics
        llm_sizes = [len(s.get('event_ids', [])) for s in llm_sessions]
        rule_sizes = [s.get('action_count', 0) for s in rule_sessions]

        return {
            'llm_session_count': len(llm_sessions),
            'rule_session_count': len(rule_sessions),
            'agreement_rate': agreement_rate,
            'total_events': total_events,
            'common_events': len(common_ids),
            'discrepancy_count': len(discrepancies),
            'discrepancies': discrepancies[:20],  # Limit for display
            'llm_avg_session_size': sum(llm_sizes) / len(llm_sizes) if llm_sizes else 0,
            'rule_avg_session_size': sum(rule_sizes) / len(rule_sizes) if rule_sizes else 0,
            'llm_total_events_mapped': len(llm_mapping),
            'rule_total_events_mapped': len(rule_mapping)
        }


# Prompt presets for the UI
PROMPT_PRESETS = {
    "Default (Structured)": """You are a work session analyzer. Given a batch of telemetry events, group them into logical work sessions.

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

Return ONLY the JSON, no explanations.""",

    "Minimal (Fast)": """Analyze events and return JSON with sessions and isolated_events.

Events:
{events}

Return JSON with format: {{"sessions": [...], "isolated_events": {{"event_ids": [], "reason": ""}}}}""",

    "Detailed (With Examples)": """You are analyzing work telemetry. Group related events into sessions.

Example input:
ID: 1 | 11:00:00 | field_input | Excel - Entered 'Sales' in cell A1
ID: 2 | 11:00:15 | field_input | Excel - Entered '500' in cell B1

Example output:
{{"sessions": [{{"session_id": 1, "event_ids": [1, 2], "start_time": "11:00:00", "end_time": "11:00:15", "primary_app": "Excel", "reasoning": "Consecutive Excel data entry", "intent": "Entering sales data"}}], "isolated_events": {{"event_ids": [], "reason": "All events grouped"}}}}

Now analyze:
{events}

Return only JSON:""",

    "Intent-Focused": """Analyze these work events and identify the user's intent for each work session.

EVENTS:
{events}

Group events that contribute to the same goal/task. Focus on:
1. What application is being used
2. What action is being performed
3. What the user is trying to accomplish

Return JSON:
{{
  "sessions": [
    {{
      "session_id": number,
      "event_ids": [IDs],
      "start_time": "timestamp",
      "end_time": "timestamp",
      "primary_app": "app",
      "reasoning": "detailed explanation",
      "intent": "user's goal/task"
    }}
  ],
  "isolated_events": {{"event_ids": [], "reason": ""}}
}}"""
}
