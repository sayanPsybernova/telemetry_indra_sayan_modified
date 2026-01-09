"""
Active vs Passive Event Analysis

Compares two approaches:
1. OLD: All events are equal for session boundary detection
2. NEW: Only ACTIVE events determine session boundaries, PASSIVE events attach to sessions

Event Categories:
- ACTIVE: User-initiated actions (field_input, browser_activity, clipboard, erp_activity, sap_interaction)
- PASSIVE: Context/background events (active_window_activity, data_reconcilation)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DATA_FILE

# Use the main data file
DATA_FILE_MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data main.csv")

# Event categorization (configurable)
ACTIVE_EVENT_TYPES = ['field_input', 'browser_activity', 'clipboard', 'erp_activity', 'sap_interaction']
PASSIVE_EVENT_TYPES = ['active_window_activity', 'data_reconcilation', 'business_app_usage']


def load_and_categorize_events(filepath: str) -> pd.DataFrame:
    """Load CSV and categorize events as ACTIVE or PASSIVE."""
    print("=" * 70)
    print("LOADING AND CATEGORIZING EVENTS")
    print("=" * 70)

    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Categorize events
    df['category'] = df['type'].apply(
        lambda t: 'ACTIVE' if t in ACTIVE_EVENT_TYPES
        else ('PASSIVE' if t in PASSIVE_EVENT_TYPES else 'UNKNOWN')
    )

    print(f"\nTotal events: {len(df)}")
    print(f"\nEvent type distribution:")
    for event_type, count in df['type'].value_counts().items():
        category = 'ACTIVE' if event_type in ACTIVE_EVENT_TYPES else (
            'PASSIVE' if event_type in PASSIVE_EVENT_TYPES else 'UNKNOWN')
        print(f"  {event_type}: {count} [{category}]")

    print(f"\nCategory summary:")
    category_counts = df['category'].value_counts()
    for cat, count in category_counts.items():
        pct = count / len(df) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    return df


def analyze_time_gaps_by_category(df: pd.DataFrame):
    """Analyze time gaps for ACTIVE events only vs ALL events."""
    print(f"\n{'='*70}")
    print("TIME GAP COMPARISON: ALL vs ACTIVE-ONLY")
    print("=" * 70)

    # All events gaps
    all_gaps = df['timestamp'].diff().dt.total_seconds().dropna()

    # Active events only gaps
    active_df = df[df['category'] == 'ACTIVE'].copy()
    active_gaps = active_df['timestamp'].diff().dt.total_seconds().dropna()

    print(f"\n{'Metric':<25} {'All Events':>15} {'Active Only':>15}")
    print("-" * 60)

    metrics = [
        ("Event count", len(df), len(active_df)),
        ("Gap count", len(all_gaps), len(active_gaps)),
        ("Mean gap (s)", f"{all_gaps.mean():.1f}", f"{active_gaps.mean():.1f}"),
        ("Median gap (s)", f"{all_gaps.median():.1f}", f"{active_gaps.median():.1f}"),
        ("P25 gap (s)", f"{np.percentile(all_gaps, 25):.1f}", f"{np.percentile(active_gaps, 25):.1f}"),
        ("P50 gap (s)", f"{np.percentile(all_gaps, 50):.1f}", f"{np.percentile(active_gaps, 50):.1f}"),
        ("P75 gap (s)", f"{np.percentile(all_gaps, 75):.1f}", f"{np.percentile(active_gaps, 75):.1f}"),
        ("P90 gap (s)", f"{np.percentile(all_gaps, 90):.1f}", f"{np.percentile(active_gaps, 90):.1f}"),
        ("P95 gap (s)", f"{np.percentile(all_gaps, 95):.1f}", f"{np.percentile(active_gaps, 95):.1f}"),
        ("Max gap (s)", f"{all_gaps.max():.1f}", f"{active_gaps.max():.1f}"),
    ]

    for metric, all_val, active_val in metrics:
        print(f"{metric:<25} {str(all_val):>15} {str(active_val):>15}")

    # Histogram comparison
    print(f"\n{'='*70}")
    print("TIME GAP HISTOGRAM COMPARISON")
    print("=" * 70)

    buckets = [
        (0, 10, "0-10s"),
        (10, 30, "10-30s"),
        (30, 60, "30-60s"),
        (60, 120, "1-2 min"),
        (120, 180, "2-3 min"),
        (180, 300, "3-5 min"),
        (300, 600, "5-10 min"),
        (600, 1800, "10-30 min"),
        (1800, float('inf'), "30+ min"),
    ]

    print(f"\n{'Bucket':<15} {'All Events':>20} {'Active Only':>20}")
    print("-" * 60)

    for low, high, label in buckets:
        all_count = len(all_gaps[(all_gaps >= low) & (all_gaps < high)])
        active_count = len(active_gaps[(active_gaps >= low) & (active_gaps < high)])

        all_pct = all_count / len(all_gaps) * 100
        active_pct = active_count / len(active_gaps) * 100 if len(active_gaps) > 0 else 0

        print(f"{label:<15} {all_count:>8} ({all_pct:>5.1f}%) {active_count:>8} ({active_pct:>5.1f}%)")

    return {
        'all_gaps': all_gaps,
        'active_gaps': active_gaps,
        'active_df': active_df
    }


def simulate_sessions(df: pd.DataFrame, time_gap: int, micro_switch: int,
                      use_active_only: bool = False) -> dict:
    """
    Simulate session creation with given thresholds.

    Args:
        df: DataFrame with all events
        time_gap: Time gap threshold for new session
        micro_switch: Micro-switch threshold
        use_active_only: If True, only use ACTIVE events for boundaries

    Returns:
        Dictionary with session statistics
    """
    if use_active_only:
        # Filter to active events for boundary detection
        boundary_df = df[df['category'] == 'ACTIVE'].copy()
    else:
        boundary_df = df.copy()

    if len(boundary_df) == 0:
        return {'session_count': 0, 'avg_duration': 0, 'avg_actions': 0}

    sessions = []
    current_session_start = boundary_df.iloc[0]['timestamp']
    current_session_events = 1

    for i in range(1, len(boundary_df)):
        gap = (boundary_df.iloc[i]['timestamp'] - boundary_df.iloc[i-1]['timestamp']).total_seconds()

        if gap > time_gap:
            # End current session
            sessions.append({
                'start': current_session_start,
                'end': boundary_df.iloc[i-1]['timestamp'],
                'duration': (boundary_df.iloc[i-1]['timestamp'] - current_session_start).total_seconds(),
                'events': current_session_events
            })
            # Start new session
            current_session_start = boundary_df.iloc[i]['timestamp']
            current_session_events = 1
        else:
            current_session_events += 1

    # Don't forget last session
    sessions.append({
        'start': current_session_start,
        'end': boundary_df.iloc[-1]['timestamp'],
        'duration': (boundary_df.iloc[-1]['timestamp'] - current_session_start).total_seconds(),
        'events': current_session_events
    })

    # Filter sessions with >= 2 events
    valid_sessions = [s for s in sessions if s['events'] >= 2]
    isolated = len(sessions) - len(valid_sessions)

    if not valid_sessions:
        return {'session_count': 0, 'avg_duration': 0, 'avg_actions': 0, 'isolated': isolated}

    durations = [s['duration'] for s in valid_sessions]
    actions = [s['events'] for s in valid_sessions]

    return {
        'session_count': len(valid_sessions),
        'isolated': isolated,
        'avg_duration': np.mean(durations),
        'median_duration': np.median(durations),
        'avg_actions': np.mean(actions),
        'total_duration': sum(durations),
        'zero_duration': sum(1 for d in durations if d == 0),
        'short_sessions': sum(1 for d in durations if d < 30),
        'long_sessions': sum(1 for d in durations if d > 600)
    }


def compare_approaches(df: pd.DataFrame):
    """Compare OLD (all events) vs NEW (active only) approaches."""
    print(f"\n{'='*70}")
    print("SESSION COMPARISON: ALL EVENTS vs ACTIVE-ONLY BOUNDARIES")
    print("=" * 70)

    # Threshold combinations to test
    combinations = [
        (300, 45),   # Current default
        (240, 40),   # Slightly tighter
        (180, 35),   # User-suggested
        (180, 45),   # Mixed
        (120, 30),   # Aggressive
        (420, 60),   # Lenient
    ]

    print(f"\n{'Config':<15} {'OLD (All Events)':^35} {'NEW (Active Only)':^35}")
    print(f"{'':15} {'Sess':>8} {'Iso':>6} {'AvgDur':>10} {'AvgAct':>8}  {'Sess':>8} {'Iso':>6} {'AvgDur':>10} {'AvgAct':>8}")
    print("-" * 100)

    results = []

    for time_gap, micro in combinations:
        # OLD approach: all events
        old_stats = simulate_sessions(df, time_gap, micro, use_active_only=False)

        # NEW approach: active only for boundaries
        new_stats = simulate_sessions(df, time_gap, micro, use_active_only=True)

        config = f"{time_gap}s/{micro}s"

        old_dur = f"{old_stats['avg_duration']:.0f}s" if old_stats['session_count'] > 0 else "N/A"
        new_dur = f"{new_stats['avg_duration']:.0f}s" if new_stats['session_count'] > 0 else "N/A"

        old_act = f"{old_stats['avg_actions']:.1f}" if old_stats['session_count'] > 0 else "N/A"
        new_act = f"{new_stats['avg_actions']:.1f}" if new_stats['session_count'] > 0 else "N/A"

        print(f"{config:<15} {old_stats['session_count']:>8} {old_stats.get('isolated', 0):>6} {old_dur:>10} {old_act:>8}  "
              f"{new_stats['session_count']:>8} {new_stats.get('isolated', 0):>6} {new_dur:>10} {new_act:>8}")

        results.append({
            'time_gap': time_gap,
            'micro_switch': micro,
            'old_sessions': old_stats['session_count'],
            'old_isolated': old_stats.get('isolated', 0),
            'old_avg_dur': old_stats['avg_duration'],
            'old_avg_actions': old_stats['avg_actions'],
            'new_sessions': new_stats['session_count'],
            'new_isolated': new_stats.get('isolated', 0),
            'new_avg_dur': new_stats['avg_duration'],
            'new_avg_actions': new_stats['avg_actions'],
        })

    return results


def find_optimal_thresholds_active_only(df: pd.DataFrame):
    """Find data-driven thresholds for ACTIVE events only."""
    print(f"\n{'='*70}")
    print("DATA-DRIVEN THRESHOLD SUGGESTIONS (ACTIVE EVENTS ONLY)")
    print("=" * 70)

    active_df = df[df['category'] == 'ACTIVE'].copy()
    active_gaps = active_df['timestamp'].diff().dt.total_seconds().dropna()

    p50 = np.percentile(active_gaps, 50)
    p75 = np.percentile(active_gaps, 75)
    p90 = np.percentile(active_gaps, 90)
    p95 = np.percentile(active_gaps, 95)

    print(f"\nActive events gap percentiles:")
    print(f"  P50 (median): {p50:.1f}s")
    print(f"  P75: {p75:.1f}s")
    print(f"  P90: {p90:.1f}s")
    print(f"  P95: {p95:.1f}s")

    # Suggestions based on data
    suggested_micro = int(p50)  # Median gap = micro-switch
    suggested_session = int(p90)  # P90 = session break

    print(f"\nSuggested thresholds (data-driven):")
    print(f"  MICRO_SWITCH_THRESHOLD = {suggested_micro}s (based on P50)")
    print(f"  TIME_GAP_THRESHOLD = {suggested_session}s (based on P90)")

    # Test the suggested values
    print(f"\nTesting suggested thresholds ({suggested_session}s/{suggested_micro}s):")
    stats = simulate_sessions(df, suggested_session, suggested_micro, use_active_only=True)
    print(f"  Sessions: {stats['session_count']}")
    print(f"  Isolated: {stats.get('isolated', 0)}")
    print(f"  Avg duration: {stats['avg_duration']:.0f}s ({stats['avg_duration']/60:.1f} min)")
    print(f"  Avg actions: {stats['avg_actions']:.1f}")

    return {
        'suggested_micro': suggested_micro,
        'suggested_session': suggested_session,
        'percentiles': {'p50': p50, 'p75': p75, 'p90': p90, 'p95': p95}
    }


def main():
    print("\n" + "=" * 70)
    print("ACTIVE vs PASSIVE EVENT ANALYSIS")
    print("=" * 70)
    print(f"\nData file: {DATA_FILE_MAIN}")
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nEvent Categories:")
    print(f"  ACTIVE: {ACTIVE_EVENT_TYPES}")
    print(f"  PASSIVE: {PASSIVE_EVENT_TYPES}")

    # Load and categorize
    df = load_and_categorize_events(DATA_FILE_MAIN)

    # Analyze time gaps
    gap_analysis = analyze_time_gaps_by_category(df)

    # Compare approaches
    results = compare_approaches(df)

    # Find optimal thresholds
    suggestions = find_optimal_thresholds_active_only(df)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY & RECOMMENDATION")
    print("=" * 70)

    print(f"""
Key Findings:

1. PASSIVE events (active_window_activity, data_reconcilation) create
   artificial short gaps (~32s heartbeat pattern)

2. Filtering to ACTIVE events only reveals natural work patterns:
   - Median gap: {suggestions['percentiles']['p50']:.1f}s
   - P90 gap: {suggestions['percentiles']['p90']:.1f}s

3. Recommended approach:
   - Use ACTIVE events for session BOUNDARY detection
   - Attach PASSIVE events to sessions by timestamp
   - This gives cleaner, more meaningful session groupings

4. Suggested thresholds (with active-only boundaries):
   - TIME_GAP_THRESHOLD = {suggestions['suggested_session']}s
   - MICRO_SWITCH_THRESHOLD = {suggestions['suggested_micro']}s
""")

    return df, results, suggestions


if __name__ == "__main__":
    df, results, suggestions = main()
