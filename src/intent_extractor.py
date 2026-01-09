"""
Intent extraction for individual sessions using an LLM.
"""

from __future__ import annotations

from typing import Dict, Optional, Any, List
import json
import os
import re
from datetime import datetime

import requests

import config


class IntentExtractor:
    """Extracts intent from a single session using a local or remote LLM."""

    def __init__(
        self,
        provider: str = config.INTENT_LLM_PROVIDER,
        api_url: str = config.INTENT_LLM_URL,
        model: str = config.INTENT_LLM_MODEL,
        output_file: str = config.INTENT_OUTPUT_FILE,
        timeout: int = config.INTENT_LLM_TIMEOUT,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.api_key = api_key if api_key is not None else self._default_api_key(provider)
        self.output_file = self._resolve_output_path(output_file)
        self.headers = self._build_headers()
        self._cache: Optional[Dict[str, Any]] = None

    def _default_api_key(self, provider: str) -> str:
        if provider == "openrouter":
            env_key = getattr(config, "OPENROUTER_API_KEY_ENV", "OPENROUTER_API_KEY")
            return getattr(config, "OPENROUTER_API_KEY", "") or os.getenv(env_key, "")
        if provider == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        if provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY", "")
        return ""

    def _resolve_output_path(self, output_file: str) -> str:
        if os.path.isabs(output_file):
            return output_file
        base_dir = getattr(config, "BASE_DIR", os.getcwd())
        return os.path.join(base_dir, output_file)

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.provider in {"openrouter", "openai", "anthropic"}:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _format_apps_breakdown(self, apps_breakdown: List[Dict[str, Any]]) -> str:
        if not apps_breakdown:
            return "N/A"
        parts = []
        for item in apps_breakdown:
            if not isinstance(item, dict):
                continue
            app = item.get("app", "Unknown")
            percentage = item.get("percentage")
            action_count = item.get("action_count")
            if percentage is None and action_count is None:
                parts.append(str(app))
            elif action_count is None:
                parts.append(f"{app} ({percentage}%)")
            else:
                parts.append(f"{app} ({percentage}%, {action_count} actions)")
        return ", ".join(parts) if parts else "N/A"

    def _format_list(self, values: List[Any]) -> str:
        if not values:
            return "N/A"
        cleaned = [str(v) for v in values if v]
        return ", ".join(cleaned) if cleaned else "N/A"




    def _format_action_lines(self, actions: List[Dict[str, Any]], max_actions: int) -> str:
        if not actions:
            return ""

        lines = []
        for action in actions[:max_actions]:
            timestamp = action.get("timestamp", "N/A")
            if hasattr(timestamp, "isoformat"):
                timestamp = timestamp.isoformat()
            app = action.get("app", "Unknown")
            field = action.get("field", "")
            value = action.get("value", "")
            value_text = str(value) if value is not None else ""
            if len(value_text) > 120:
                value_text = value_text[:120] + "..."

            parts = [str(timestamp), str(app)]
            parts.append(str(field) if field else "N/A")
            parts.append(value_text if value_text else "N/A")
            lines.append("- " + " | ".join(parts))

        return "\n".join(lines)


    def _build_prompt(
        self,
        session: Dict[str, Any],
        include_actions: bool = False,
        max_actions: int = 10
    ) -> str:
        summary = session.get("enhanced_summary") if isinstance(session, dict) else None
        if not isinstance(summary, dict):
            summary = {}

        apps_breakdown = summary.get("apps_breakdown", [])
        unique_contexts = summary.get("unique_contexts", [])
        app_sequence = summary.get("app_sequence", [])
        field_types = summary.get("field_types", [])

        duration = session.get("duration_formatted", "N/A")
        action_count = session.get("action_count", 0)

        actions_block = ""
        if include_actions:
            actions = session.get("actions", []) if isinstance(session, dict) else []
            action_lines = self._format_action_lines(actions, max_actions)
            if action_lines:
                actions_block = (
                    f"\nActions (top {max_actions}):\n"
                    f"{action_lines}\n"
                )

        return (
            "Analyze this work session and extract the user's intent.\n\n"
            "Session Data:\n"
            f"- Duration: {duration}\n"
            f"- Apps: {self._format_apps_breakdown(apps_breakdown)}\n"
            f"- Contexts: {self._format_list(unique_contexts)}\n"
            f"- App Flow: {self._format_list(app_sequence)}\n"
            f"- Fields: {self._format_list(field_types)}\n"
            f"- Actions: {action_count}\n\n"
            f"{actions_block}"
            "Return JSON only:\n"
            "{\n"
            '  "intent": "Brief description of what user was trying to do",\n'
            '  "task_type": "Category like \'Data Entry\', \'Research\', \'Communication\', \'Document Editing\'",\n'
            '  "key_entities": ["Entity1", "Entity2"],\n'
            '  "confidence": "high/medium/low"\n'
            "}"
        )

    def _call_llm(self, prompt: str, max_tokens: Optional[int] = None) -> tuple[str, Optional[str]]:
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            if max_tokens is not None:
                payload["max_tokens"] = int(max_tokens)
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    return "", f"Unexpected response format: {str(data)[:200]}"
                return content, None
            return "", f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.exceptions.ConnectionError:
            return "", "Connection refused. Is LM Studio running?"
        except requests.exceptions.Timeout:
            return "", f"Request timed out after {self.timeout}s"
        except Exception as exc:
            return "", str(exc)

    def _extract_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        if not response_text:
            return None

        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        brace_start = response_text.find("{")
        if brace_start != -1:
            depth = 0
            for i, char in enumerate(response_text[brace_start:], brace_start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response_text[brace_start : i + 1])
                        except json.JSONDecodeError:
                            break
        return None

    def _load_intents(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache

        if not os.path.exists(self.output_file):
            self._cache = {"intents": {}}
            return self._cache

        try:
            with open(self.output_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            data = {"intents": {}}

        if not isinstance(data, dict) or "intents" not in data:
            data = {"intents": {}}

        self._cache = data
        return self._cache

    def _write_intents(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        self._cache = data

    def get_cached_intent(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Check if intent already extracted for this session."""
        if not session_id:
            return None
        data = self._load_intents()
        intents = data.get("intents", {})
        return intents.get(session_id)

    def save_intent(self, session_id: str, session_number: Optional[int], intent: Dict[str, Any]) -> None:
        """Save intent data to the output file."""
        data = self._load_intents()
        intents = data.setdefault("intents", {})

        entities = intent.get("key_entities", [])
        if isinstance(entities, str):
            entities = [entities] if entities else []
        elif not isinstance(entities, list):
            entities = []

        entry = {
            "session_number": session_number,
            "extracted_at": datetime.now().isoformat(),
            "intent": intent.get("intent", "Extraction failed"),
            "task_type": intent.get("task_type"),
            "key_entities": entities,
            "confidence": intent.get("confidence"),
            "llm_provider": self.provider,
            "llm_model": self.model,
        }

        if "error" in intent:
            entry["error"] = intent["error"]
        if "raw_response" in intent:
            entry["raw_response"] = intent["raw_response"]

        intents[session_id] = entry
        self._write_intents(data)


    def extract_intent(
        self,
        session: Dict[str, Any],
        max_tokens: Optional[int] = None,
        include_actions: bool = False,
        max_actions: int = 10
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            session,
            include_actions=include_actions,
            max_actions=max_actions
        )
        response_text, error = self._call_llm(prompt, max_tokens=max_tokens)
        if error:
            return {"intent": "Extraction failed", "error": error}

        parsed = self._extract_json(response_text)
        if not isinstance(parsed, dict):
            return {"intent": "Extraction failed", "raw_response": response_text}

        return parsed
