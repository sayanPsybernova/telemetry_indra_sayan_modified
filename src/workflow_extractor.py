"""
Workflow extraction for related sessions using an LLM.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import json
import os
import re
import hashlib
from datetime import datetime

import requests

import config
from src.intent_extractor import IntentExtractor


class WorkflowExtractor:
    """Extracts workflow intent across related sessions."""

    def __init__(
        self,
        provider: str = config.INTENT_LLM_PROVIDER,
        api_url: str = config.INTENT_LLM_URL,
        model: str = config.INTENT_LLM_MODEL,
        output_file: str = config.WORKFLOW_OUTPUT_FILE,
        timeout: int = config.INTENT_LLM_TIMEOUT,
        intent_extractor: Optional[IntentExtractor] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.intent_extractor = intent_extractor or IntentExtractor()
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

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if hasattr(value, "to_pydatetime"):
            return value.to_pydatetime()
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return None

    def _format_span(self, seconds: Optional[int]) -> str:
        if seconds is None:
            return "N/A"
        if seconds < 60:
            return f"{int(seconds)}s"
        if seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

    def _gap_allows_chain(self, gap_info: Optional[Dict[str, Any]], threshold_seconds: int) -> bool:
        if not isinstance(gap_info, dict):
            return False
        if gap_info.get("is_same_day") is not True:
            return False
        seconds = gap_info.get("seconds")
        if seconds is None:
            return False
        try:
            return float(seconds) <= threshold_seconds
        except (TypeError, ValueError):
            return False

    def _calc_span_seconds(self, start: Optional[datetime], end: Optional[datetime]) -> Optional[int]:
        if not start or not end:
            return None
        return int((end - start).total_seconds())

    def find_related_sessions(
        self,
        anchor_session_id: str,
        sessions: List[Dict[str, Any]],
        gap_threshold_seconds: int,
        max_sessions: int,
        max_span_seconds: int,
    ) -> List[Dict[str, Any]]:
        if not anchor_session_id or not sessions:
            return []

        index_map = {
            s.get("session_id"): idx for idx, s in enumerate(sessions) if s.get("session_id")
        }
        anchor_idx = index_map.get(anchor_session_id)
        if anchor_idx is None:
            return []

        start_idx = anchor_idx
        end_idx = anchor_idx

        anchor_start = self._parse_datetime(sessions[anchor_idx].get("start_time"))
        anchor_end = self._parse_datetime(sessions[anchor_idx].get("end_time"))
        earliest_start = anchor_start
        latest_end = anchor_end

        while start_idx > 0:
            current = sessions[start_idx]
            gap_info = current.get("gap_from_previous")
            if not self._gap_allows_chain(gap_info, gap_threshold_seconds):
                break
            if (end_idx - start_idx + 1) >= max_sessions:
                break

            candidate = sessions[start_idx - 1]
            candidate_start = self._parse_datetime(candidate.get("start_time"))
            new_earliest = candidate_start or earliest_start
            span_seconds = self._calc_span_seconds(new_earliest, latest_end)
            if span_seconds is not None and span_seconds > max_span_seconds:
                break

            start_idx -= 1
            earliest_start = new_earliest

        while end_idx < len(sessions) - 1:
            next_session = sessions[end_idx + 1]
            gap_info = next_session.get("gap_from_previous")
            if not self._gap_allows_chain(gap_info, gap_threshold_seconds):
                break
            if (end_idx - start_idx + 1) >= max_sessions:
                break

            next_end = self._parse_datetime(next_session.get("end_time"))
            new_latest = next_end or latest_end
            span_seconds = self._calc_span_seconds(earliest_start, new_latest)
            if span_seconds is not None and span_seconds > max_span_seconds:
                break

            end_idx += 1
            latest_end = new_latest

        return sessions[start_idx : end_idx + 1]

    def build_workflow_id(self, session_ids: List[str]) -> str:
        normalized = sorted([sid for sid in session_ids if sid])
        joined = "|".join(normalized)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _load_workflows(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache

        if not os.path.exists(self.output_file):
            self._cache = {"workflows": {}}
            return self._cache

        try:
            with open(self.output_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            data = {"workflows": {}}

        if not isinstance(data, dict) or "workflows" not in data:
            data = {"workflows": {}}

        self._cache = data
        return self._cache

    def _write_workflows(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        self._cache = data

    def get_cached_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        if not workflow_id:
            return None
        data = self._load_workflows()
        return data.get("workflows", {}).get(workflow_id)

    def save_workflow(self, workflow_id: str, workflow_data: Dict[str, Any]) -> None:
        data = self._load_workflows()
        workflows = data.setdefault("workflows", {})
        workflows[workflow_id] = workflow_data
        self._write_workflows(data)

    def _format_apps_breakdown(self, apps_breakdown: List[Dict[str, Any]]) -> str:
        if not apps_breakdown:
            return "N/A"
        parts = []
        for item in apps_breakdown:
            if not isinstance(item, dict):
                continue
            app = item.get("app", "Unknown")
            percentage = item.get("percentage")
            if percentage is None:
                parts.append(str(app))
            else:
                parts.append(f"{app} ({percentage}%)")
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


    def _session_prompt_block(
        self,
        session: Dict[str, Any],
        include_actions: bool = False,
        max_actions: int = 10
    ) -> str:
        session_number = session.get("session_number", "N/A")
        duration = session.get("duration_formatted", "N/A")
        action_count = session.get("action_count", 0)
        session_id = session.get("session_id")

        actions_block = ""
        if include_actions:
            actions = session.get("actions", []) if isinstance(session, dict) else []
            action_lines = self._format_action_lines(actions, max_actions)
            if action_lines:
                actions_block = (
                    f"- Actions Detail (top {max_actions}):\n"
                    f"{action_lines}\n"
                )

        cached_intent = None
        if session_id:
            cached_intent = self.intent_extractor.get_cached_intent(session_id)

        if isinstance(cached_intent, dict) and not cached_intent.get("error"):
            intent_text = cached_intent.get("intent")
            if intent_text and intent_text != "Extraction failed":
                entities = cached_intent.get("key_entities", [])
                if isinstance(entities, list):
                    entities_text = ", ".join(str(e) for e in entities if e) or "N/A"
                elif isinstance(entities, str):
                    entities_text = entities
                else:
                    entities_text = "N/A"
                return (
                    f"Session {session_number}:\n"
                    f"- Duration: {duration}\n"
                    f"- Actions: {action_count}\n"
                    f"- Intent: {intent_text}\n"
                    f"- Task Type: {cached_intent.get('task_type', 'N/A')}\n"
                    f"- Key Entities: {entities_text}\n"
                    f"- Confidence: {cached_intent.get('confidence', 'N/A')}\n"
                    f"{actions_block}"
                )

        summary = session.get("enhanced_summary") if isinstance(session, dict) else None
        if not isinstance(summary, dict):
            summary = {}

        return (
            f"Session {session_number}:\n"
            f"- Duration: {duration}\n"
            f"- Actions: {action_count}\n"
            f"- Apps: {self._format_apps_breakdown(summary.get('apps_breakdown', []))}\n"
            f"- Contexts: {self._format_list(summary.get('unique_contexts', []))}\n"
            f"- App Flow: {self._format_list(summary.get('app_sequence', []))}\n"
            f"- Fields: {self._format_list(summary.get('field_types', []))}\n"
            f"{actions_block}"
        )


    def _build_prompt(
        self,
        related_sessions: List[Dict[str, Any]],
        include_actions: bool = False,
        max_actions: int = 10
    ) -> str:
        blocks = [self._session_prompt_block(s, include_actions, max_actions) for s in related_sessions]
        sessions_text = "\n".join(blocks)
        return (
            "Analyze these related work sessions and extract the overall workflow intent.\n\n"
            "Sessions:\n"
            f"{sessions_text}\n"
            "Return JSON only:\n"
            "{\n"
            '  "workflow_intent": "What user accomplished across all sessions",\n'
            '  "workflow_type": "Category like \'Invoice Processing\', \'Report Generation\'",\n'
            '  "key_steps": ["Step 1", "Step 2"],\n'
            '  "session_roles": ["Session 3: Data gathering", "Session 4: Validation"],\n'
            '  "key_entities": ["Entity1", "Entity2"],\n'
            '  "confidence": "high/medium/low"\n'
            "}"
        )

    def _call_llm(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, Optional[str]]:
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


    def extract_workflow(
        self,
        anchor_session_id: str,
        sessions: List[Dict[str, Any]],
        gap_threshold_seconds: int,
        max_sessions: int,
        max_span_seconds: int,
        force: bool = False,
        related_sessions: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        include_actions: bool = False,
        max_actions: int = 10
    ) -> Dict[str, Any]:
        if related_sessions is None:
            related_sessions = self.find_related_sessions(
                anchor_session_id,
                sessions,
                gap_threshold_seconds,
                max_sessions,
                max_span_seconds,
            )

        if len(related_sessions) < 2:
            return {
                "error": "No related sessions found. Use single-session intent instead."
            }

        session_ids = [s.get("session_id") for s in related_sessions if s.get("session_id")]
        workflow_id = self.build_workflow_id(session_ids)

        if not force:
            cached = self.get_cached_workflow(workflow_id)
            if cached:
                return cached

        prompt = self._build_prompt(
            related_sessions,
            include_actions=include_actions,
            max_actions=max_actions
        )
        response_text, error = self._call_llm(prompt, max_tokens=max_tokens)
        if error:
            llm_result: Dict[str, Any] = {"workflow_intent": "Extraction failed", "error": error}
        else:
            parsed = self._extract_json(response_text)
            if not isinstance(parsed, dict):
                llm_result = {"workflow_intent": "Extraction failed", "raw_response": response_text}
            else:
                llm_result = parsed

        start_times = [self._parse_datetime(s.get("start_time")) for s in related_sessions]
        end_times = [self._parse_datetime(s.get("end_time")) for s in related_sessions]
        start_times = [t for t in start_times if t]
        end_times = [t for t in end_times if t]
        earliest_start = min(start_times) if start_times else None
        latest_end = max(end_times) if end_times else None
        span_seconds = self._calc_span_seconds(earliest_start, latest_end)

        workflow_record = {
            "workflow_id": workflow_id,
            "anchor_session_id": anchor_session_id,
            "session_ids": session_ids,
            "session_numbers": [s.get("session_number") for s in related_sessions],
            "time_range": {
                "start": earliest_start.isoformat() if earliest_start else None,
                "end": latest_end.isoformat() if latest_end else None,
                "span_seconds": span_seconds,
                "span_formatted": self._format_span(span_seconds),
            },
            "extraction_settings": {
                "gap_threshold_seconds": gap_threshold_seconds,
                "gap_threshold_hours": gap_threshold_seconds / 3600,
                "max_sessions": max_sessions,
                "max_span_seconds": max_span_seconds,
                "max_span_hours": max_span_seconds / 3600,
                "method": "chain_gap_same_day",
            },
            "workflow_intent": llm_result.get("workflow_intent"),
            "workflow_type": llm_result.get("workflow_type"),
            "key_steps": llm_result.get("key_steps"),
            "session_roles": llm_result.get("session_roles"),
            "key_entities": llm_result.get("key_entities"),
            "confidence": llm_result.get("confidence"),
            "metadata": {
                "extracted_at": datetime.now().isoformat(),
                "llm_provider": self.provider,
                "llm_model": self.model,
            },
        }

        if "error" in llm_result:
            workflow_record["error"] = llm_result["error"]
        if "raw_response" in llm_result:
            workflow_record["raw_response"] = llm_result["raw_response"]

        self.save_workflow(workflow_id, workflow_record)
        return workflow_record
