"""
Threshold Analysis Script
Analyzes how different threshold combinations affect session creation.

This script:
1. First analyzes time gap distribution to find natural breakpoints
2. Tests multiple threshold combinations
3. Compares session counts and characteristics
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.sessionizer import Sessionizer
from config import DATA_FILE

# Use the main data file instead
DATA_FILE_MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data main.csv")


def analyze_time_gaps(filepath: str) -> dict:
    """Analyze time gap distribution between consecutive events."""
    print("=" * 70)
    print("TIME GAP DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Load and preprocess data
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Process ALL event types (no filtering - matches updated sessionizer)
    actions_df = df.copy()

    print(f"\nTotal events: {len(df)}")
    print(f"Event types: {df['type'].value_counts().to_dict()}")

    # Calculate time gaps
    actions_df['time_gap'] = actions_df['timestamp'].diff().dt.total_seconds()
    gaps = actions_df['time_gap'].dropna()

    print(f"\n{'='*70}")
    print("PERCENTILE BREAKDOWN")
    print("=" * 70)

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = {}
    for p in percentiles:
        val = np.percentile(gaps, p)
        percentile_values[p] = val
        print(f"  P{p:2d}: {val:8.1f}s ({val/60:.1f} min)")

    print(f"\n  Mean: {gaps.mean():8.1f}s ({gaps.mean()/60:.1f} min)")
    print(f"  Median: {gaps.median():8.1f}s ({gaps.median()/60:.1f} min)")
    print(f"  Max: {gaps.max():8.1f}s ({gaps.max()/60:.1f} min)")

    # Histogram buckets
    print(f"\n{'='*70}")
    print("TIME GAP HISTOGRAM")
    print("=" * 70)

    buckets = [
        (0, 10, "0-10s (rapid)"),
        (10, 30, "10-30s (quick)"),
        (30, 60, "30-60s (normal)"),
        (60, 120, "1-2 min"),
        (120, 180, "2-3 min"),
        (180, 300, "3-5 min"),
        (300, 600, "5-10 min"),
        (600, 1800, "10-30 min"),
        (1800, float('inf'), "30+ min (break)")
    ]

    total = len(gaps)
    cumulative = 0

    for low, high, label in buckets:
        count = len(gaps[(gaps >= low) & (gaps < high)])
        pct = count / total * 100
        cumulative += pct
        bar = "#" * int(pct / 2)
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Natural breakpoints analysis
    print(f"\n{'='*70}")
    print("NATURAL BREAKPOINT ANALYSIS")
    print("=" * 70)

    # Find gaps where there's a significant jump
    sorted_gaps = np.sort(gaps)
    gap_diffs = np.diff(sorted_gaps)

    # Find the largest jumps (potential natural breakpoints)
    top_jump_indices = np.argsort(gap_diffs)[-10:]
    print("\n  Largest gap jumps (potential natural breakpoints):")
    for idx in reversed(top_jump_indices[-5:]):
        print(f"    Gap jumps from {sorted_gaps[idx]:.1f}s to {sorted_gaps[idx+1]:.1f}s")

    # Suggested thresholds based on data
    print(f"\n{'='*70}")
    print("DATA-DRIVEN THRESHOLD SUGGESTIONS")
    print("=" * 70)

    micro_switch_suggestion = percentile_values[50]  # Median
    session_break_suggestion = percentile_values[90]  # P90

    print(f"\n  Current thresholds:")
    print(f"    MICRO_SWITCH_THRESHOLD = 45s")
    print(f"    TIME_GAP_THRESHOLD = 300s")

    print(f"\n  Suggested (data-driven):")
    print(f"    MICRO_SWITCH_THRESHOLD = {micro_switch_suggestion:.0f}s (P50 - median gap)")
    print(f"    TIME_GAP_THRESHOLD = {session_break_suggestion:.0f}s (P90 - 90% of gaps are smaller)")

    return {
        'gaps': gaps,
        'percentiles': percentile_values,
        'mean': gaps.mean(),
        'median': gaps.median(),
        'suggested_micro': micro_switch_suggestion,
        'suggested_session': session_break_suggestion
    }


def test_threshold_combinations(filepath: str, combinations: list) -> pd.DataFrame:
    """Test multiple threshold combinations and compare results."""
    print(f"\n{'='*70}")
    print("THRESHOLD COMPARISON ANALYSIS")
    print("=" * 70)

    results = []

    for time_gap, micro_switch in combinations:
        print(f"\n  Testing: TIME_GAP={time_gap}s, MICRO_SWITCH={micro_switch}s...")

        # Create sessionizer with these thresholds
        sessionizer = Sessionizer(
            time_gap_threshold=time_gap,
            micro_switch_threshold=micro_switch,
            min_session_actions=2  # Keep constant
        )

        # Suppress output during processing
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            sessions, stats, isolated = sessionizer.process(filepath)

        # Calculate additional metrics
        durations = [s['duration_seconds'] for s in sessions]
        action_counts = [s['action_count'] for s in sessions]

        results.append({
            'time_gap': time_gap,
            'micro_switch': micro_switch,
            'session_count': len(sessions),
            'isolated_count': len(isolated),
            'total_duration_min': sum(durations) / 60,
            'avg_duration_sec': np.mean(durations) if durations else 0,
            'median_duration_sec': np.median(durations) if durations else 0,
            'avg_actions': np.mean(action_counts) if action_counts else 0,
            'zero_duration_count': sum(1 for d in durations if d == 0),
            'short_sessions': sum(1 for d in durations if d < 30),
            'long_sessions': sum(1 for d in durations if d > 600)
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print comparison table
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Time Gap':>10} {'Micro':>8} {'Sessions':>10} {'Isolated':>10} {'Avg Dur':>10} {'Avg Acts':>10}")
    print("-" * 70)

    for _, row in df.iterrows():
        print(f"{row['time_gap']:>10}s {row['micro_switch']:>8}s {row['session_count']:>10} "
              f"{row['isolated_count']:>10} {row['avg_duration_sec']:>9.1f}s {row['avg_actions']:>10.1f}")

    return df


def main():
    print("\n" + "=" * 70)
    print("SESSIONIZER THRESHOLD ANALYSIS")
    print("=" * 70)
    print(f"\nData file: {DATA_FILE_MAIN}")
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Analyze time gap distribution
    gap_analysis = analyze_time_gaps(DATA_FILE_MAIN)

    # Step 2: Define threshold combinations to test
    # Include current values, suggested values, and variations
    combinations = [
        # Current defaults
        (300, 45),

        # User-suggested based on earlier analysis
        (180, 35),

        # Data-driven suggestions
        (int(gap_analysis['suggested_session']), int(gap_analysis['suggested_micro'])),

        # Variations for comparison
        (120, 30),   # More aggressive (shorter sessions)
        (180, 45),   # Medium session break, current micro
        (240, 40),   # Medium
        (300, 30),   # Current session break, tighter micro
        (420, 60),   # More lenient (longer sessions)
        (600, 60),   # Very lenient
    ]

    # Remove duplicates while preserving order
    seen = set()
    unique_combinations = []
    for combo in combinations:
        if combo not in seen:
            seen.add(combo)
            unique_combinations.append(combo)

    # Step 3: Test all combinations
    results_df = test_threshold_combinations(DATA_FILE_MAIN, unique_combinations)

    # Step 4: Summary and recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find best configurations
    current = results_df[results_df['time_gap'] == 300].iloc[0]
    suggested = results_df[
        (results_df['time_gap'] == 180) &
        (results_df['micro_switch'] == 35)
    ]

    print(f"\n  Current (300s/45s):")
    print(f"    Sessions: {current['session_count']}")
    print(f"    Isolated: {current['isolated_count']}")
    print(f"    Avg duration: {current['avg_duration_sec']:.1f}s")

    if not suggested.empty:
        suggested = suggested.iloc[0]
        print(f"\n  User-suggested (180s/35s):")
        print(f"    Sessions: {suggested['session_count']}")
        print(f"    Isolated: {suggested['isolated_count']}")
        print(f"    Avg duration: {suggested['avg_duration_sec']:.1f}s")

        session_diff = suggested['session_count'] - current['session_count']
        isolated_diff = suggested['isolated_count'] - current['isolated_count']
        print(f"\n  Change from current:")
        print(f"    Sessions: {'+' if session_diff > 0 else ''}{session_diff}")
        print(f"    Isolated: {'+' if isolated_diff > 0 else ''}{isolated_diff}")

    # Save results to CSV for further analysis
    output_path = os.path.join(os.path.dirname(DATA_FILE_MAIN), "output", "threshold_analysis.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    results = main()
