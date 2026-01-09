"""
Event Whitelist Module - Manages user decisions on isolated events.

This module handles:
1. Loading/saving user decisions (keep/drop) for isolated events
2. Managing promoted actions (events user decided to keep)
3. Persisting decisions to JSON file
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime
import os

# Default paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
DECISIONS_FILE = OUTPUT_DIR / "event_decisions.json"
SESSIONS_FILE = OUTPUT_DIR / "sessions.json"


def load_decisions() -> Dict:
    """
    Load user decisions from file.

    Returns:
        Dictionary with structure:
        {
            "kept_ids": ["iso_0001", ...],
            "dropped_ids": ["iso_0002", ...],
            "last_updated": "2025-12-23T14:30:00"
        }
    """
    if DECISIONS_FILE.exists():
        with open(DECISIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "kept_ids": [],
        "dropped_ids": [],
        "last_updated": None
    }


def save_decisions(decisions: Dict) -> None:
    """Save user decisions to file."""
    decisions["last_updated"] = datetime.now().isoformat()
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(DECISIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(decisions, f, indent=2)

def _update_decisions(update_fn) -> None:
    decisions = load_decisions()
    update_fn(decisions)
    save_decisions(decisions)


def get_kept_event_ids() -> Set[str]:
    """Get set of event IDs user has marked to keep."""
    decisions = load_decisions()
    return set(decisions.get("kept_ids", []))


def get_dropped_event_ids() -> Set[str]:
    """Get set of event IDs user has marked to drop."""
    decisions = load_decisions()
    return set(decisions.get("dropped_ids", []))


def mark_event_keep(event_id: str) -> None:
    """Mark an isolated event to be kept (promoted)."""
    def apply(decisions: Dict) -> None:
        # Remove from dropped if present
        if event_id in decisions["dropped_ids"]:
            decisions["dropped_ids"].remove(event_id)
        # Add to kept if not already there
        if event_id not in decisions["kept_ids"]:
            decisions["kept_ids"].append(event_id)

    _update_decisions(apply)


def mark_event_drop(event_id: str) -> None:
    """Mark an isolated event to be dropped."""
    def apply(decisions: Dict) -> None:
        # Remove from kept if present
        if event_id in decisions["kept_ids"]:
            decisions["kept_ids"].remove(event_id)
        # Add to dropped if not already there
        if event_id not in decisions["dropped_ids"]:
            decisions["dropped_ids"].append(event_id)

    _update_decisions(apply)


def mark_event_pending(event_id: str) -> None:
    """Reset an event to pending status (remove from both kept and dropped)."""
    def apply(decisions: Dict) -> None:
        if event_id in decisions["kept_ids"]:
            decisions["kept_ids"].remove(event_id)
        if event_id in decisions["dropped_ids"]:
            decisions["dropped_ids"].remove(event_id)

    _update_decisions(apply)


def bulk_mark_keep(event_ids: List[str]) -> None:
    """Mark multiple events to keep."""
    def apply(decisions: Dict) -> None:
        for event_id in event_ids:
            if event_id in decisions["dropped_ids"]:
                decisions["dropped_ids"].remove(event_id)
            if event_id not in decisions["kept_ids"]:
                decisions["kept_ids"].append(event_id)

    _update_decisions(apply)


def bulk_mark_drop(event_ids: List[str]) -> None:
    """Mark multiple events to drop."""
    def apply(decisions: Dict) -> None:
        for event_id in event_ids:
            if event_id in decisions["kept_ids"]:
                decisions["kept_ids"].remove(event_id)
            if event_id not in decisions["dropped_ids"]:
                decisions["dropped_ids"].append(event_id)

    _update_decisions(apply)


def apply_decisions_to_sessions_file() -> Dict:
    """
    Apply user decisions to the sessions.json file.

    Moves kept events from isolated_events to promoted_actions.
    Removes dropped events from isolated_events.

    Returns:
        Summary of changes made.
    """
    if not SESSIONS_FILE.exists():
        return {"error": "Sessions file not found"}

    # Load current sessions data
    with open(SESSIONS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load decisions
    decisions = load_decisions()
    kept_ids = set(decisions.get("kept_ids", []))
    dropped_ids = set(decisions.get("dropped_ids", []))

    # Get current isolated events
    isolated_events = data.get("isolated_events", [])
    promoted_actions = data.get("promoted_actions", [])

    # Track changes
    new_promoted = []
    remaining_isolated = []
    dropped_count = 0

    for event in isolated_events:
        event_id = event.get("id")

        if event_id in kept_ids:
            # Move to promoted actions
            event["status"] = "kept"
            event["promoted_at"] = datetime.now().isoformat()
            new_promoted.append(event)
        elif event_id in dropped_ids:
            # Mark as dropped (will be excluded)
            dropped_count += 1
        else:
            # Still pending
            remaining_isolated.append(event)

    # Update data
    data["promoted_actions"] = promoted_actions + new_promoted
    data["isolated_events"] = remaining_isolated

    # Update statistics
    if "statistics" in data:
        data["statistics"]["isolated_events_count"] = len(remaining_isolated)
        data["statistics"]["promoted_actions_count"] = len(data["promoted_actions"])

    # Save updated data
    with open(SESSIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {
        "promoted": len(new_promoted),
        "dropped": dropped_count,
        "remaining_pending": len(remaining_isolated),
        "total_promoted": len(data["promoted_actions"])
    }


def get_event_status(event_id: str) -> str:
    """
    Get the current status of an event.

    Returns: "keep", "drop", or "pending"
    """
    decisions = load_decisions()

    if event_id in decisions.get("kept_ids", []):
        return "keep"
    elif event_id in decisions.get("dropped_ids", []):
        return "drop"
    else:
        return "pending"


def clear_all_decisions() -> None:
    """Clear all user decisions (reset everything to pending)."""
    save_decisions({
        "kept_ids": [],
        "dropped_ids": [],
        "last_updated": datetime.now().isoformat()
    })


# Test function
if __name__ == "__main__":
    print("Event Whitelist Module Test")
    print("=" * 40)

    # Test basic operations
    print("\n1. Loading decisions...")
    decisions = load_decisions()
    print(f"   Kept: {len(decisions.get('kept_ids', []))}")
    print(f"   Dropped: {len(decisions.get('dropped_ids', []))}")

    print("\n2. Testing mark operations...")
    mark_event_keep("test_001")
    mark_event_drop("test_002")

    print(f"   test_001 status: {get_event_status('test_001')}")
    print(f"   test_002 status: {get_event_status('test_002')}")
    print(f"   test_003 status: {get_event_status('test_003')}")

    print("\n3. Cleaning up test data...")
    mark_event_pending("test_001")
    mark_event_pending("test_002")

    print("   Test complete!")
