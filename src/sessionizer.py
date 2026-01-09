"""
Sessionizer Agent - Groups raw telemetry events into meaningful work sessions.

This is the first agent in the pipeline that:
1. Takes raw telemetry events (14k+ rows)
2. Filters out noise (heartbeat/app_running events)
3. Groups remaining events into logical work sessions
4. Enriches sessions with metadata
"""
import pandas as pd
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from collections import Counter

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse_action, get_session_type, extract_person_from_context, classify_isolated_event, reload_classifications
from src.csv_utils import normalize_telemetry_columns
from config import (
    TIME_GAP_THRESHOLD, MICRO_SWITCH_THRESHOLD, MIN_SESSION_ACTIONS,
    EXCLUDE_EVENT_TYPES, ACTIVE_EVENT_TYPES, PASSIVE_EVENT_TYPES
)


class Sessionizer:
    """
    Groups raw telemetry events into meaningful work sessions.
    """

    def __init__(self, time_gap_threshold: int = TIME_GAP_THRESHOLD,
                 micro_switch_threshold: int = MICRO_SWITCH_THRESHOLD,
                 min_session_actions: int = MIN_SESSION_ACTIONS,
                 exclude_types: list = None,
                 active_types: list = None,
                 passive_types: list = None):
        """
        Initialize the Sessionizer.

        Args:
            time_gap_threshold: Seconds of gap to consider as new session (default: 240 = 4 min)
            micro_switch_threshold: Seconds for brief interruptions to merge (default: 20)
            min_session_actions: Minimum actions to form valid session (default: 2)
            exclude_types: List of event types to exclude (default: from config)
            active_types: Event types that define session boundaries (default: from config)
            passive_types: Event types that attach to sessions (default: from config)
        """
        self.time_gap_threshold = time_gap_threshold
        self.micro_switch_threshold = micro_switch_threshold
        self.min_session_actions = min_session_actions
        self.exclude_types = exclude_types if exclude_types is not None else EXCLUDE_EVENT_TYPES
        self.active_types = active_types if active_types is not None else ACTIVE_EVENT_TYPES
        self.passive_types = passive_types if passive_types is not None else PASSIVE_EVENT_TYPES

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load telemetry data from CSV file.
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)

        # Normalize columns to handle casing/spacing differences
        df = normalize_telemetry_columns(
            df,
            required_cols=["timestamp", "type", "action"],
            alias_map={
                "timestamp": ["activity_ts"],
                "type": ["activity_type"],
                "action": ["description"],
            },
        )

        # Parse timestamp - use mixed format to handle different formats
        # This handles ISO 8601, various date formats, and timezone-aware strings
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True, errors='coerce')

        # Check for parsing failures
        failed_timestamps = df['timestamp'].isna().sum()
        if failed_timestamps > 0:
            print(f"WARNING: {failed_timestamps} timestamps failed to parse and were set to NaT")
            # Remove rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])

        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"Loaded {len(df)} events")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Event types: {df['type'].value_counts().to_dict()}")

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data:
        1. Process ALL event types (optionally exclude some via config)
        2. Parse action field to extract structured data
        3. Add derived columns
        """
        print("\nPreprocessing data...")

        # Process ALL event types by default
        # Only exclude types explicitly listed in exclude_types
        if self.exclude_types:
            actions_df = df[~df['type'].isin(self.exclude_types)].copy()
            excluded_count = len(df) - len(actions_df)
            print(f"Excluded {excluded_count} events of types: {self.exclude_types}")
        else:
            actions_df = df.copy()

        # Report all event types being processed
        type_counts = actions_df['type'].value_counts().to_dict()
        print(f"Processing {len(actions_df)} events across {len(type_counts)} types:")
        for event_type, count in type_counts.items():
            print(f"  - {event_type}: {count}")

        # Force reload of LLM classifications cache to pick up any changes
        reload_classifications()
        print("Reloaded LLM classification cache")

        # Parse each action to extract structured fields
        parsed_data = []
        for idx, row in actions_df.iterrows():
            parsed = parse_action(row['action'], row['type'])
            parsed['source_id'] = row.get('id')
            parsed['timestamp'] = row['timestamp']
            parsed['original_type'] = row['type']
            parsed_data.append(parsed)

        parsed_df = pd.DataFrame(parsed_data)

        # Add session type
        parsed_df['session_type'] = parsed_df.apply(
            lambda x: get_session_type(x['app'], x['context'], x['field']),
            axis=1
        )

        # Categorize events as ACTIVE or PASSIVE
        def categorize_event(event_type):
            if event_type in self.active_types:
                return 'ACTIVE'
            elif event_type in self.passive_types:
                return 'PASSIVE'
            else:
                return 'ACTIVE'  # Treat unknown as ACTIVE for safety

        parsed_df['event_category'] = parsed_df['original_type'].apply(categorize_event)

        # Sort by timestamp
        parsed_df = parsed_df.sort_values('timestamp').reset_index(drop=True)

        print(f"Parsed {len(parsed_df)} events")
        print(f"Apps found: {parsed_df['app'].value_counts().head(10).to_dict()}")

        # Report category distribution
        category_counts = parsed_df['event_category'].value_counts()
        print(f"\nEvent categories:")
        for cat, count in category_counts.items():
            print(f"  {cat}: {count}")

        return parsed_df

    def should_start_new_session(self, current_session: List[Dict],
                                  new_event: Dict) -> Tuple[bool, str]:
        """
        Determine if a new event should start a new session.

        Simple rule: Start new session if idle time > threshold (15 minutes).
        App changes and context changes are ignored.

        Returns:
            Tuple of (should_start_new, reason)
        """
        if not current_session:
            return True, "first_event"

        last_event = current_session[-1]
        time_gap = (new_event['timestamp'] - last_event['timestamp']).total_seconds()

        if time_gap > self.time_gap_threshold:
            return True, f"idle_break_{int(time_gap)}s"

        return False, "same_session"

    def create_session(self, events: List[Dict], session_number: int) -> Dict:
        """
        Create a session object from a list of events.
        """
        if not events:
            return None

        # Get all apps used
        apps_used = [e['app'] for e in events if e['app']]
        app_counts = Counter(apps_used)

        # Primary app is the most frequent
        primary_app = app_counts.most_common(1)[0][0] if app_counts else "Unknown"

        # Get primary context
        contexts = [e['context'] for e in events if e['context']]
        context_counts = Counter(contexts)
        primary_context = context_counts.most_common(1)[0][0] if context_counts else None

        # Get session type from primary app
        session_type = get_session_type(primary_app, primary_context)

        # Calculate duration
        start_time = events[0]['timestamp']
        end_time = events[-1]['timestamp']
        duration_seconds = (end_time - start_time).total_seconds()

        # Extract all values entered (for summary)
        values = [e['value'] for e in events if e['value']]

        # Create summary
        summary = self._create_session_summary(primary_app, primary_context, values, events)

        return {
            "session_id": str(uuid.uuid4())[:8],
            "session_number": session_number,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": int(duration_seconds),
            "duration_formatted": self._format_duration(duration_seconds),
            "primary_app": primary_app,
            "primary_context": primary_context,
            "session_type": session_type,
            "action_count": len(events),
            "apps_used": list(set(apps_used)),
            "summary": summary,
            "actions": [
                {
                    "timestamp": e['timestamp'].isoformat(),
                    "app": e['app'],
                    "source_id": e.get('source_id'),
                    "field": e['field'],
                    "value": e['value'][:50] + "..." if e['value'] and len(e['value']) > 50 else e['value'],
                    "context": e['context']
                }
                for e in events
            ]
        }

    def _finalize_current_session(
        self,
        current_session_events: List[Dict],
        sessions: List[Dict],
        isolated_events: List[Dict],
        session_number: int,
        isolated_number: int
    ) -> Tuple[int, int]:
        """
        Finalize the current session into sessions or isolated events.
        """
        if not current_session_events:
            return session_number, isolated_number

        if len(current_session_events) >= self.min_session_actions:
            session = self.create_session(current_session_events, session_number)
            if session:
                sessions.append(session)
                session_number += 1
        else:
            for e in current_session_events:
                isolated_event = self._create_isolated_event(e, isolated_number)
                isolated_events.append(isolated_event)
                isolated_number += 1

        return session_number, isolated_number

    def _create_session_summary(self, app: str, context: str,
                                 values: List[str], events: List[Dict]) -> str:
        """
        Create a human-readable summary of the session.
        """
        if not app:
            return "Unknown activity"

        app_lower = app.lower()

        # Communication
        if 'teams' in app_lower:
            person = extract_person_from_context(context) if context else None
            if person:
                return f"Teams chat with {person}"
            return "Microsoft Teams activity"

        # Email
        if 'outlook' in app_lower:
            if context and 'inbox' in context.lower():
                return f"Email in {context.split('-')[0].strip() if '-' in context else 'Outlook'}"
            return "Outlook email activity"

        # Document work
        if 'excel' in app_lower:
            return "Excel spreadsheet work"
        if 'word' in app_lower:
            return "Word document editing"

        # Web browsing
        if any(x in app_lower for x in ['chrome', 'edge', 'firefox']):
            # Try to get what they were searching/doing
            search_values = [v for v in values if v and len(v) > 2]
            if search_values:
                return f"Web browsing - searched: {search_values[0][:30]}"
            return "Web browsing"

        # Cloud storage
        if 'onedrive' in app_lower:
            return "OneDrive file management"

        # Default
        return f"{app} activity"

    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in human-readable format.
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _parse_iso_datetime(self, value):
        """Parse ISO datetime string or passthrough datetime objects."""
        if isinstance(value, datetime):
            return value
        if not value:
            return None
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        return None

    def _format_gap(self, gap_seconds: int) -> str:
        """Format a gap duration for display."""
        if gap_seconds < 0:
            gap_seconds = 0

        if gap_seconds >= 86400:
            days = gap_seconds // 86400
            hours = (gap_seconds % 86400) // 3600
            return f"{days}d {hours}h"
        if gap_seconds >= 3600:
            hours = gap_seconds // 3600
            minutes = (gap_seconds % 3600) // 60
            return f"{hours}h {minutes}m"

        minutes = gap_seconds // 60
        seconds = gap_seconds % 60
        return f"{minutes}m {seconds}s"

    def _unique_in_order(self, values: List[str]) -> List[str]:
        """Return unique values preserving original order."""
        seen = set()
        unique = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique.append(value)
        return unique

    def _build_enhanced_summary(self, session: Dict) -> Dict:
        """Build a code-based session summary for UI display."""
        actions = session.get('actions') or []
        app_counts = Counter([a.get('app') for a in actions if a.get('app')])
        total_actions = len(actions)

        apps_breakdown = []
        for app, count in app_counts.most_common():
            percentage = int(round((count / total_actions) * 100)) if total_actions else 0
            apps_breakdown.append({
                "app": app,
                "percentage": percentage,
                "action_count": count
            })

        unique_contexts = self._unique_in_order([
            a.get('context') for a in actions if a.get('context')
        ])

        app_sequence = []
        last_app = None
        for action in actions:
            app = action.get('app')
            if not app:
                continue
            if app != last_app:
                app_sequence.append(app)
                last_app = app

        field_types = self._unique_in_order([
            a.get('field') for a in actions if a.get('field')
        ])

        passive_count = session.get('passive_event_count', 0) or 0

        return {
            "apps_breakdown": apps_breakdown,
            "unique_contexts": unique_contexts,
            "app_sequence": app_sequence,
            "field_types": field_types,
            "has_passive_events": passive_count > 0
        }

    def _add_gap_and_summary(self, sessions: List[Dict]) -> None:
        """Annotate sessions with gap_from_previous and enhanced_summary."""
        previous_end = None
        for idx, session in enumerate(sessions):
            start_time = self._parse_iso_datetime(session.get('start_time'))
            end_time = self._parse_iso_datetime(session.get('end_time'))

            if idx == 0 or not start_time or not previous_end:
                session['gap_from_previous'] = None
            else:
                gap_seconds = int((start_time - previous_end).total_seconds())
                if gap_seconds < 0:
                    gap_seconds = 0
                session['gap_from_previous'] = {
                    "seconds": gap_seconds,
                    "formatted": self._format_gap(gap_seconds),
                    "is_same_day": start_time.date() == previous_end.date()
                }

            session['enhanced_summary'] = self._build_enhanced_summary(session)
            previous_end = end_time if end_time else previous_end

    def sessionize(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """
        Main sessionization logic with active/passive awareness.

        Session boundaries are based on gaps between ACTIVE events only.
        PASSIVE events are attached to sessions they fall within.

        Returns:
            Tuple of (sessions, isolated_events)
            - sessions: List of multi-action sessions
            - isolated_events: List of single-action events for user review
        """
        print("\nSessionizing events (active-only boundaries)...")

        # Separate active and passive events
        active_df = df[df['event_category'] == 'ACTIVE'].copy()
        passive_df = df[df['event_category'] == 'PASSIVE'].copy()

        print(f"  Active events for boundary detection: {len(active_df)}")
        print(f"  Passive events to attach: {len(passive_df)}")

        sessions = []
        isolated_events = []
        current_session_events = []
        session_number = 1
        isolated_number = 1

        # Create sessions from ACTIVE events only
        for idx, row in active_df.iterrows():
            event = row.to_dict()

            should_start, reason = self.should_start_new_session(current_session_events, event)

            if should_start and current_session_events:
                session_number, isolated_number = self._finalize_current_session(
                    current_session_events,
                    sessions,
                    isolated_events,
                    session_number,
                    isolated_number
                )

                current_session_events = [event]
            else:
                current_session_events.append(event)

        # Don't forget the last session
        session_number, isolated_number = self._finalize_current_session(
            current_session_events,
            sessions,
            isolated_events,
            session_number,
            isolated_number
        )

        print(f"Created {len(sessions)} sessions from active events")

        # Attach passive events to sessions
        if sessions and len(passive_df) > 0:
            sessions = self._attach_passive_events(sessions, passive_df)
            total_passive = sum(s.get('passive_event_count', 0) for s in sessions)
            print(f"Attached {total_passive} passive events to sessions")

        print(f"Found {len(isolated_events)} isolated events (single-action, pending review)")

        return sessions, isolated_events

    def _attach_passive_events(self, sessions: List[Dict], passive_df: pd.DataFrame) -> List[Dict]:
        """
        Attach passive events to sessions based on timestamp.

        Rules:
        1. If passive event falls within session time range -> attach to that session
        2. If passive event is between sessions -> attach to nearest session
        3. Track attached passive events separately in session
        """
        for idx, row in passive_df.iterrows():
            event_time = row['timestamp']
            event_dict = row.to_dict()

            # Find which session this event belongs to
            attached = False

            for session in sessions:
                start = datetime.fromisoformat(session['start_time'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(session['end_time'].replace('Z', '+00:00'))

                # Make timestamps timezone-aware if needed
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=start.tzinfo)

                # Check if event falls within session
                if start <= event_time <= end:
                    if 'passive_events' not in session:
                        session['passive_events'] = []
                    session['passive_events'].append({
                        'timestamp': event_time.isoformat(),
                        'app': event_dict.get('app'),
                        'source_id': event_dict.get('source_id'),  # FIXED: Include source_id
                        'type': event_dict.get('original_type'),
                        'context': event_dict.get('context')
                    })
                    attached = True
                    break

            # If not attached, find nearest session
            if not attached:
                nearest_session = self._find_nearest_session(sessions, event_time)
                if nearest_session:
                    if 'passive_events' not in nearest_session:
                        nearest_session['passive_events'] = []
                    nearest_session['passive_events'].append({
                        'timestamp': event_time.isoformat(),
                        'app': event_dict.get('app'),
                        'source_id': event_dict.get('source_id'),  # FIXED: Include source_id
                        'type': event_dict.get('original_type'),
                        'context': event_dict.get('context'),
                        'attachment': 'nearest'  # Flag as attached to nearest
                    })

        # Update action counts to include passive
        for session in sessions:
            passive_count = len(session.get('passive_events', []))
            session['passive_event_count'] = passive_count
            session['total_event_count'] = session['action_count'] + passive_count

        return sessions

    def _find_nearest_session(self, sessions: List[Dict], event_time) -> Dict:
        """Find the session nearest to the given event time."""
        min_distance = float('inf')
        nearest = None

        for session in sessions:
            start = datetime.fromisoformat(session['start_time'].replace('Z', '+00:00'))
            end = datetime.fromisoformat(session['end_time'].replace('Z', '+00:00'))

            # Make timestamps timezone-aware if needed
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=start.tzinfo)

            # Distance to start or end of session
            dist_to_start = abs((event_time - start).total_seconds())
            dist_to_end = abs((event_time - end).total_seconds())
            dist = min(dist_to_start, dist_to_end)

            if dist < min_distance:
                min_distance = dist
                nearest = session

        return nearest

    def _create_isolated_event(self, event: Dict, isolated_number: int) -> Dict:
        """
        Create an isolated event object for user review.
        """
        return {
            "id": f"iso_{isolated_number:04d}",
            "source_id": event.get('source_id'),
            "timestamp": event['timestamp'].isoformat() if hasattr(event['timestamp'], 'isoformat') else str(event['timestamp']),
            "app": event.get('app'),
            "action": event.get('raw_action', ''),
            "field": event.get('field'),
            "value": event.get('value'),
            "context": event.get('context'),
            "type": event.get('original_type', 'unknown'),
            "category": classify_isolated_event(event),
            "status": "pending"  # pending | keep | drop
        }

    def get_statistics(self, sessions: List[Dict]) -> Dict:
        """
        Calculate statistics about the sessions.
        """
        if not sessions:
            return {}

        total_duration = sum(s['duration_seconds'] for s in sessions)
        avg_duration = total_duration / len(sessions)

        # App distribution
        app_counts = Counter(s['primary_app'] for s in sessions)

        # Session type distribution
        type_counts = Counter(s['session_type'] for s in sessions)

        # Time distribution
        hours = [datetime.fromisoformat(s['start_time']).hour for s in sessions]
        hour_counts = Counter(hours)

        # Calculate passive event statistics
        total_passive = sum(s.get('passive_event_count', 0) for s in sessions)
        total_events = sum(s.get('total_event_count', s['action_count']) for s in sessions)

        return {
            "total_sessions": len(sessions),
            "total_duration_seconds": total_duration,
            "total_duration_formatted": self._format_duration(total_duration),
            "average_duration_seconds": int(avg_duration),
            "average_duration_formatted": self._format_duration(avg_duration),
            "total_actions": sum(s['action_count'] for s in sessions),
            "total_passive_events": total_passive,  # NEW: Passive events count
            "total_events": total_events,  # NEW: Total events (active + passive)
            "app_distribution": dict(app_counts.most_common()),
            "session_type_distribution": dict(type_counts.most_common()),
            "hourly_distribution": dict(sorted(hour_counts.items())),
            "longest_session": max(sessions, key=lambda x: x['duration_seconds'])['summary'],
            "most_active_app": app_counts.most_common(1)[0][0] if app_counts else None
        }

    def process(self, filepath: str) -> Tuple[List[Dict], Dict, List[Dict]]:
        """
        Full processing pipeline: load, preprocess, sessionize, get stats.

        Returns:
            Tuple of (sessions list, statistics dict, isolated_events list)
        """
        # Load data
        df = self.load_data(filepath)

        # Preprocess
        parsed_df = self.preprocess(df)

        # Sessionize
        sessions, isolated_events = self.sessionize(parsed_df)

        # CRITICAL: Ensure sessions are in chronological order by start_time
        # Sort sessions by start_time and reassign session numbers
        if sessions:
            sessions = sorted(sessions, key=lambda s: s['start_time'])
            # Reassign session numbers to match chronological order
            for i, session in enumerate(sessions, start=1):
                session['session_number'] = i
            self._add_gap_and_summary(sessions)

        # Get statistics
        stats = self.get_statistics(sessions)

        # Add isolated events count to stats
        stats['isolated_events_count'] = len(isolated_events)

        return sessions, stats, isolated_events


# Test function
if __name__ == "__main__":
    from config import DATA_FILE

    sessionizer = Sessionizer()
    sessions, stats, isolated_events = sessionizer.process(DATA_FILE)

    print("\n" + "="*60)
    print("SESSIONIZATION COMPLETE")
    print("="*60)
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nFirst 3 sessions:")
    for session in sessions[:3]:
        print(f"\n  Session #{session['session_number']}: {session['summary']}")
        print(f"    Duration: {session['duration_formatted']}")
        print(f"    App: {session['primary_app']}")
        print(f"    Actions: {session['action_count']}")

    print(f"\nIsolated Events (first 5):")
    for event in isolated_events[:5]:
        print(f"\n  {event['id']}: {event['category']}")
        print(f"    App: {event['app']}")
        print(f"    Field: {event['field']}")
        print(f"    Value: {event['value'][:30] if event['value'] else 'N/A'}...")
