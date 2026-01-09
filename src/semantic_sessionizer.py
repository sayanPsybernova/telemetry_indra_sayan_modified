"""
Semantic Sessionizer - Uses embeddings for session boundary detection.

Complements the rule-based Sessionizer by using cosine similarity of
event embeddings to determine session boundaries.
"""
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sessionizer import Sessionizer
from src.embeddings import cosine_similarity, EmbeddingCache
from src.parser import get_session_type, extract_person_from_context, classify_isolated_event
from config import (
    ACTIVE_EVENT_TYPES, PASSIVE_EVENT_TYPES, MIN_SESSION_ACTIONS,
    SEMANTIC_HARD_BREAK_SEC, SEMANTIC_SOFT_BREAK_SEC, SEMANTIC_MICRO_SWITCH_SEC,
    SEMANTIC_SIM_SAME, SEMANTIC_SIM_SPLIT
)

logger = logging.getLogger(__name__)


class SemanticSessionizer:
    """
    Groups telemetry events into sessions using semantic similarity.

    Uses embeddings to compare consecutive events and determine boundaries
    based on similarity thresholds combined with time gaps.
    """

    def __init__(self,
                 hard_break_sec: int = SEMANTIC_HARD_BREAK_SEC,
                 soft_break_sec: int = SEMANTIC_SOFT_BREAK_SEC,
                 micro_switch_sec: int = SEMANTIC_MICRO_SWITCH_SEC,
                 sim_same: float = SEMANTIC_SIM_SAME,
                 sim_split: float = SEMANTIC_SIM_SPLIT,
                 min_session_actions: int = MIN_SESSION_ACTIONS):
        """
        Initialize the Semantic Sessionizer.

        Args:
            hard_break_sec: Time gap that always creates new session (default: 300)
            soft_break_sec: Time gap threshold for low-similarity breaks (default: 60)
            micro_switch_sec: Brief interruption threshold (default: 20)
            sim_same: Similarity threshold to stay in same session (default: 0.58)
            sim_split: Similarity threshold below which to split (default: 0.40)
            min_session_actions: Minimum actions for valid session (default: 2)
        """
        self.hard_break_sec = hard_break_sec
        self.soft_break_sec = soft_break_sec
        self.micro_switch_sec = micro_switch_sec
        self.sim_same = sim_same
        self.sim_split = sim_split
        self.min_session_actions = min_session_actions

        # Reuse Sessionizer for data loading and preprocessing
        self._base_sessionizer = Sessionizer(
            min_session_actions=min_session_actions,
            active_types=ACTIVE_EVENT_TYPES,
            passive_types=PASSIVE_EVENT_TYPES
        )

        # Embedding cache
        self._cache = EmbeddingCache()

    def process(self, filepath: str) -> Tuple[List[Dict], Dict, List[Dict]]:
        """
        Full processing pipeline: load, preprocess, sessionize with embeddings.

        Args:
            filepath: Path to input CSV file

        Returns:
            Tuple of (sessions list, statistics dict, isolated_events list)
        """
        # Reuse base sessionizer for loading and preprocessing
        df = self._base_sessionizer.load_data(filepath)
        parsed_df = self._base_sessionizer.preprocess(df)

        # Sessionize using semantic boundaries
        sessions, isolated_events = self._sessionize_semantic(parsed_df)

        # Get statistics
        stats = self._get_statistics(sessions)
        stats['isolated_events_count'] = len(isolated_events)

        # Log cache stats
        cache_stats = self._cache.stats
        print(f"\nEmbedding cache: {cache_stats['size']} cached, {cache_stats['hit_rate']} hit rate")

        return sessions, stats, isolated_events

    def _sessionize_semantic(self, df) -> Tuple[List[Dict], List[Dict]]:
        """
        Main sessionization logic using semantic similarity.

        Session boundaries are based on gaps between ACTIVE events only.
        PASSIVE events are attached to sessions they fall within.

        Returns:
            Tuple of (sessions, isolated_events)
        """
        print("\nSessionizing events (semantic boundaries)...")

        # Separate active and passive events
        active_df = df[df['event_category'] == 'ACTIVE'].copy()
        passive_df = df[df['event_category'] == 'PASSIVE'].copy()

        print(f"  Active events for boundary detection: {len(active_df)}")
        print(f"  Passive events to attach: {len(passive_df)}")

        # BATCH EMBED ALL TEXTS UPFRONT (fast path)
        print("\nPreparing embeddings (batch processing)...")
        all_texts = []
        for idx, row in active_df.iterrows():
            event = row.to_dict()
            text = self._build_embedding_text(event)
            all_texts.append(text)

        # Remove duplicates while preserving order for stats
        unique_texts = list(dict.fromkeys(all_texts))
        print(f"  Unique texts to embed: {len(unique_texts)} (from {len(all_texts)} events)")

        # Batch embed all at once
        from src.embeddings import get_embeddings_batch
        embeddings_map = get_embeddings_batch(unique_texts, show_progress=True)

        # Preload cache
        self._cache.preload(embeddings_map)

        print("\nBuilding sessions...")
        sessions = []
        isolated_events = []
        current_session_events = []
        current_session_start_reason = "first_event"
        current_session_start_gap = 0.0
        current_session_start_sim = None

        session_number = 1
        isolated_number = 1

        prev_event = None
        prev_embedding = None

        # Process ACTIVE events - embeddings already cached, so this is fast
        for idx, row in tqdm(active_df.iterrows(), total=len(active_df), desc="Processing events"):
            event = row.to_dict()

            # Get embedding from cache (instant lookup)
            embedding_text = self._build_embedding_text(event)
            curr_embedding = self._cache.get(embedding_text)

            if prev_event is None:
                # First event - start first session
                current_session_events = [event]
                current_session_start_reason = "first_event"
                current_session_start_gap = 0.0
                current_session_start_sim = None
                prev_event = event
                prev_embedding = curr_embedding
                continue

            # Determine if we should start a new session
            is_new_session, reason, time_gap, similarity = self._determine_boundary(
                prev_event, event, prev_embedding, curr_embedding
            )

            if is_new_session:
                # Save current session if it has enough actions
                session_number, isolated_number = self._finalize_current_session(
                    current_session_events,
                    current_session_start_reason,
                    current_session_start_gap,
                    current_session_start_sim,
                    sessions,
                    isolated_events,
                    session_number,
                    isolated_number
                )

                # Start new session
                current_session_events = [event]
                current_session_start_reason = reason
                current_session_start_gap = time_gap
                current_session_start_sim = similarity
            else:
                current_session_events.append(event)

            prev_event = event
            prev_embedding = curr_embedding

        # Don't forget the last session
        session_number, isolated_number = self._finalize_current_session(
            current_session_events,
            current_session_start_reason,
            current_session_start_gap,
            current_session_start_sim,
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

    def _build_embedding_text(self, event: Dict) -> str:
        """
        Build text for embedding from event fields.

        Concatenates app + context + field + value + event_type, handling None values.
        """
        parts = []

        if event.get('app'):
            parts.append(str(event['app']))
        if event.get('context'):
            parts.append(str(event['context']))
        if event.get('field'):
            parts.append(str(event['field']))
        if event.get('value'):
            # Truncate long values
            value = str(event['value'])[:200]
            parts.append(value)
        if event.get('original_type'):
            parts.append(str(event['original_type']))

        return " ".join(parts) if parts else "unknown"

    def _finalize_current_session(
        self,
        current_session_events: List[Dict],
        start_reason: str,
        start_time_gap: float,
        start_similarity: Optional[float],
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
            session = self._create_session(
                current_session_events,
                session_number,
                start_reason,
                start_time_gap,
                start_similarity
            )
            if session:
                sessions.append(session)
                session_number += 1
        else:
            for e in current_session_events:
                isolated_event = self._create_isolated_event(e, isolated_number)
                isolated_events.append(isolated_event)
                isolated_number += 1

        return session_number, isolated_number

    def _determine_boundary(self, prev: Dict, curr: Dict,
                            prev_emb: Optional[List[float]],
                            curr_emb: Optional[List[float]]) -> Tuple[bool, str, float, Optional[float]]:
        """
        Apply boundary rules in order.

        Returns:
            Tuple of (is_new_session, reason, time_gap_sec, similarity)
        """
        # Calculate time gap
        time_gap = (curr['timestamp'] - prev['timestamp']).total_seconds()

        # Check app change
        app_changed = prev.get('app') != curr.get('app')

        # Rule 1: Hard break - always creates new session
        if time_gap > self.hard_break_sec:
            return True, "hard_break", time_gap, None

        # Rule 2: Micro-switch - app changed but very brief gap, stay in session
        if app_changed and time_gap < self.micro_switch_sec:
            return False, "micro_switch", time_gap, None

        # Rule 3: Compute similarity and apply thresholds
        # If embedding failed for either event, treat as benefit_of_doubt
        if prev_emb is None or curr_emb is None:
            return False, "benefit_of_doubt", time_gap, None

        similarity = cosine_similarity(prev_emb, curr_emb)

        # High similarity - stay in same session
        if similarity >= self.sim_same:
            return False, "high_similarity", time_gap, similarity

        # Low similarity with soft break gap - split
        if similarity < self.sim_split and time_gap > self.soft_break_sec:
            return True, "low_similarity_gap", time_gap, similarity

        # Benefit of doubt - stay in same session
        return False, "benefit_of_doubt", time_gap, similarity

    def _create_session(self, events: List[Dict], session_number: int,
                        start_reason: str, start_time_gap: float,
                        start_similarity: Optional[float]) -> Dict:
        """
        Create a session object from a list of events.

        Includes diagnostic fields for boundary analysis.
        """
        if not events:
            return None

        # Get all apps used
        apps_used = [e['app'] for e in events if e.get('app')]
        app_counts = Counter(apps_used)

        # Primary app is the most frequent
        primary_app = app_counts.most_common(1)[0][0] if app_counts else "Unknown"

        # Get primary context
        contexts = [e['context'] for e in events if e.get('context')]
        context_counts = Counter(contexts)
        primary_context = context_counts.most_common(1)[0][0] if context_counts else None

        # Get session type from primary app
        session_type = get_session_type(primary_app, primary_context)

        # Calculate duration
        start_time = events[0]['timestamp']
        end_time = events[-1]['timestamp']
        duration_seconds = (end_time - start_time).total_seconds()

        # Extract values for summary
        values = [e['value'] for e in events if e.get('value')]

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
                    "app": e.get('app'),
                    "source_id": e.get('source_id'),
                    "field": e.get('field'),
                    "value": e['value'][:50] + "..." if e.get('value') and len(e['value']) > 50 else e.get('value'),
                    "context": e.get('context')
                }
                for e in events
            ],
            # Diagnostic fields
            "start_reason": start_reason,
            "start_time_gap_sec": round(start_time_gap, 2),
            "start_similarity": round(start_similarity, 4) if start_similarity is not None else None
        }

    def _create_session_summary(self, app: str, context: str,
                                values: List[str], events: List[Dict]) -> str:
        """Create a human-readable summary of the session."""
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
            search_values = [v for v in values if v and len(v) > 2]
            if search_values:
                return f"Web browsing - searched: {search_values[0][:30]}"
            return "Web browsing"

        # Cloud storage
        if 'onedrive' in app_lower:
            return "OneDrive file management"

        return f"{app} activity"

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
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

    def _create_isolated_event(self, event: Dict, isolated_number: int) -> Dict:
        """Create an isolated event object for user review."""
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
            "status": "pending"
        }

    def _attach_passive_events(self, sessions: List[Dict], passive_df) -> List[Dict]:
        """
        Attach passive events to sessions based on timestamp.

        Rules:
        1. If passive event falls within session time range -> attach to that session
        2. If passive event is between sessions -> attach to nearest session
        """
        for idx, row in passive_df.iterrows():
            event_time = row['timestamp']
            event_dict = row.to_dict()

            attached = False

            for session in sessions:
                start = datetime.fromisoformat(session['start_time'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(session['end_time'].replace('Z', '+00:00'))

                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=start.tzinfo)

                if start <= event_time <= end:
                    if 'passive_events' not in session:
                        session['passive_events'] = []
                    session['passive_events'].append({
                        'timestamp': event_time.isoformat(),
                        'app': event_dict.get('app'),
                        'type': event_dict.get('original_type'),
                        'context': event_dict.get('context')
                    })
                    attached = True
                    break

            if not attached:
                nearest_session = self._find_nearest_session(sessions, event_time)
                if nearest_session:
                    if 'passive_events' not in nearest_session:
                        nearest_session['passive_events'] = []
                    nearest_session['passive_events'].append({
                        'timestamp': event_time.isoformat(),
                        'app': event_dict.get('app'),
                        'type': event_dict.get('original_type'),
                        'context': event_dict.get('context'),
                        'attachment': 'nearest'
                    })

        # Update counts
        for session in sessions:
            passive_count = len(session.get('passive_events', []))
            session['passive_event_count'] = passive_count
            session['total_event_count'] = session['action_count'] + passive_count

        return sessions

    def _find_nearest_session(self, sessions: List[Dict], event_time) -> Optional[Dict]:
        """Find the session nearest to the given event time."""
        min_distance = float('inf')
        nearest = None

        for session in sessions:
            start = datetime.fromisoformat(session['start_time'].replace('Z', '+00:00'))
            end = datetime.fromisoformat(session['end_time'].replace('Z', '+00:00'))

            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=start.tzinfo)

            dist_to_start = abs((event_time - start).total_seconds())
            dist_to_end = abs((event_time - end).total_seconds())
            dist = min(dist_to_start, dist_to_end)

            if dist < min_distance:
                min_distance = dist
                nearest = session

        return nearest

    def _get_statistics(self, sessions: List[Dict]) -> Dict:
        """Calculate statistics about the sessions."""
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

        # Boundary reason distribution (semantic-specific)
        reason_counts = Counter(s.get('start_reason', 'unknown') for s in sessions)

        return {
            "total_sessions": len(sessions),
            "total_duration_seconds": total_duration,
            "total_duration_formatted": self._format_duration(total_duration),
            "average_duration_seconds": int(avg_duration),
            "average_duration_formatted": self._format_duration(avg_duration),
            "total_actions": sum(s['action_count'] for s in sessions),
            "app_distribution": dict(app_counts.most_common()),
            "session_type_distribution": dict(type_counts.most_common()),
            "hourly_distribution": dict(sorted(hour_counts.items())),
            "longest_session": max(sessions, key=lambda x: x['duration_seconds'])['summary'],
            "most_active_app": app_counts.most_common(1)[0][0] if app_counts else None,
            "boundary_reason_distribution": dict(reason_counts.most_common())
        }


# Test function
if __name__ == "__main__":
    from config import DATA_FILE

    sessionizer = SemanticSessionizer()
    sessions, stats, isolated_events = sessionizer.process(DATA_FILE)

    print("\n" + "=" * 60)
    print("SEMANTIC SESSIONIZATION COMPLETE")
    print("=" * 60)
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nFirst 3 sessions:")
    for session in sessions[:3]:
        print(f"\n  Session #{session['session_number']}: {session['summary']}")
        print(f"    Duration: {session['duration_formatted']}")
        print(f"    App: {session['primary_app']}")
        print(f"    Actions: {session['action_count']}")
        print(f"    Start reason: {session['start_reason']}")
        print(f"    Start gap: {session['start_time_gap_sec']}s")
        print(f"    Start similarity: {session['start_similarity']}")
