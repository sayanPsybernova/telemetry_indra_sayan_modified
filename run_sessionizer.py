"""
Sessionizer Agent - Entry Point

This script runs the Sessionizer to process raw telemetry data
and output grouped sessions.

Usage:
    python run_sessionizer.py
    python run_sessionizer.py --input path/to/data.csv --output path/to/sessions.json
"""
import json
import argparse
import os
from datetime import datetime

from src.sessionizer import Sessionizer
from config import DATA_FILE, OUTPUT_FILE, OUTPUT_DIR, TIME_GAP_THRESHOLD, MIN_SESSION_ACTIONS


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sessionizer Agent - Group telemetry events into sessions')
    parser.add_argument('--input', '-i', default=DATA_FILE, help='Input CSV file path')
    parser.add_argument('--output', '-o', default=OUTPUT_FILE, help='Output JSON file path')
    parser.add_argument('--time-gap', type=int, default=None, help='Time gap threshold in seconds (default: from config)')
    parser.add_argument('--min-actions', type=int, default=MIN_SESSION_ACTIONS, help='Minimum actions per session (default: from config)')

    args = parser.parse_args()

    # Use config default if not specified
    time_gap = args.time_gap if args.time_gap is not None else TIME_GAP_THRESHOLD

    print("="*60)
    print("SESSIONIZER AGENT")
    print("="*60)
    print(f"\nInput file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Time gap threshold: {time_gap} seconds")
    print(f"Minimum actions per session: {args.min_actions}")

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"\nERROR: Input file not found: {args.input}")
        return 1

    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize sessionizer
    sessionizer = Sessionizer(
        time_gap_threshold=time_gap,
        min_session_actions=args.min_actions
    )

    # Process data
    sessions, stats, isolated_events = sessionizer.process(args.input)

    # Prepare output
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "input_file": args.input,
            "parameters": {
                "time_gap_threshold": time_gap,
                "min_actions": args.min_actions
            }
        },
        "statistics": stats,
        "sessions": sessions,
        "isolated_events": isolated_events,  # Events pending user review
        "promoted_actions": []  # User-approved single actions (populated by dashboard)
    }

    # Save to JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)

    # Print summary
    print(f"\n[SUMMARY]")
    print(f"   Total events processed: {stats.get('total_actions', 0)}")
    print(f"   Sessions created: {stats.get('total_sessions', 0)}")
    print(f"   Isolated events (pending review): {stats.get('isolated_events_count', 0)}")
    print(f"   Total duration: {stats.get('total_duration_formatted', 'N/A')}")
    print(f"   Average session duration: {stats.get('average_duration_formatted', 'N/A')}")

    print(f"\n[APP DISTRIBUTION]")
    for app, count in list(stats.get('app_distribution', {}).items())[:5]:
        pct = (count / stats['total_sessions'] * 100) if stats['total_sessions'] > 0 else 0
        print(f"   {app}: {count} sessions ({pct:.1f}%)")

    print(f"\n[SESSION TYPES]")
    for stype, count in stats.get('session_type_distribution', {}).items():
        pct = (count / stats['total_sessions'] * 100) if stats['total_sessions'] > 0 else 0
        print(f"   {stype}: {count} sessions ({pct:.1f}%)")

    print(f"\n[OUTPUT] Saved to: {args.output}")

    # Print first few sessions as preview
    print(f"\n[PREVIEW] First 5 sessions:")
    print("-"*60)
    for session in sessions[:5]:
        print(f"\n  Session #{session['session_number']}: {session['summary']}")
        print(f"    Time: {session['start_time'][:19]} to {session['end_time'][:19]}")
        print(f"    Duration: {session['duration_formatted']}")
        print(f"    App: {session['primary_app']}")
        print(f"    Actions: {session['action_count']}")
        if session['actions']:
            print(f"    First action: {session['actions'][0].get('field', 'N/A')} = {session['actions'][0].get('value', 'N/A')[:30] if session['actions'][0].get('value') else 'N/A'}")

    # Print isolated events summary
    if isolated_events:
        print(f"\n[ISOLATED EVENTS] {len(isolated_events)} events pending review:")
        print("-"*60)
        # Group by category
        from collections import Counter
        category_counts = Counter(e['category'] for e in isolated_events)
        for category, count in category_counts.most_common():
            print(f"   {category}: {count}")

        print(f"\n  First 3 isolated events:")
        for event in isolated_events[:3]:
            print(f"\n  {event['id']} [{event['category']}]")
            print(f"    App: {event['app']}")
            print(f"    Field: {event['field']}")
            value_preview = event['value'][:30] if event['value'] else 'N/A'
            print(f"    Value: {value_preview}")

    return 0


if __name__ == "__main__":
    exit(main())
