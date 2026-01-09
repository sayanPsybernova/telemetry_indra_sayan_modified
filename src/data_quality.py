"""
Data Quality Module - Analyzes telemetry data quality and parsing accuracy.

Provides metrics on:
1. Event type distribution (processed vs dropped)
2. App detection accuracy
3. Parse success/failure rates
4. Unmatched patterns
5. Session type coverage
"""
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Optional
import re

from src.parser import parse_action, normalize_app_name
from config import ACTIVE_EVENT_TYPES, PASSIVE_EVENT_TYPES, EXCLUDE_EVENT_TYPES
from src.csv_utils import normalize_telemetry_columns


class DataQualityAnalyzer:
    """
    Analyzes raw telemetry data and processed sessions for quality metrics.
    """

    def __init__(self):
        self.raw_df = None
        self.processed_df = None
        self.sessions = None

    def _load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV and normalize required column names.
        """
        df = pd.read_csv(filepath)
        return normalize_telemetry_columns(
            df,
            required_cols=["timestamp", "type", "action"],
            alias_map={
                "timestamp": ["activity_ts"],
                "type": ["activity_type"],
                "action": ["description"],
            },
        )

    def analyze_raw_data(self, filepath: str) -> Dict:
        """
        Analyze raw CSV data before processing.

        Returns metrics on event types, timestamps, and data completeness.
        """
        df = self._load_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        self.raw_df = df

        # Event type distribution
        type_counts = df['type'].value_counts().to_dict()
        total_events = len(df)

        # Calculate what gets processed vs dropped
        processed_types = set(ACTIVE_EVENT_TYPES + PASSIVE_EVENT_TYPES)
        excluded_types = set(EXCLUDE_EVENT_TYPES)
        is_excluded = df['type'].isin(excluded_types) if excluded_types else pd.Series(False, index=df.index)
        is_unrecognized = (~df['type'].isin(processed_types)) & (~is_excluded)

        parse_failed_mask = pd.Series(False, index=df.index)
        if total_events > 0:
            for idx, row in df.iterrows():
                if is_excluded.loc[idx]:
                    continue
                parsed = parse_action(row['action'], row['type'])
                if parsed.get('parse_status') == 'error':
                    parse_failed_mask.loc[idx] = True

        dropped_mask = is_excluded | is_unrecognized | parse_failed_mask
        dropped_count = int(dropped_mask.sum())
        processed_count = total_events - dropped_count

        # Date range
        date_range = {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat(),
            'days': (df['timestamp'].max() - df['timestamp'].min()).days
        }

        # Check for missing data
        missing_data = {
            'timestamp': df['timestamp'].isna().sum(),
            'action': df['action'].isna().sum(),
            'type': df['type'].isna().sum()
        }

        return {
            'total_events': total_events,
            'event_type_distribution': type_counts,
            'processed_events': processed_count,
            'processed_percentage': round(processed_count / total_events * 100, 1),
            'dropped_events': dropped_count,
            'dropped_percentage': round(dropped_count / total_events * 100, 1),
            'excluded_events': int(is_excluded.sum()),
            'unrecognized_events': int(is_unrecognized.sum()),
            'parse_failed_events': int(parse_failed_mask.sum()),
            'date_range': date_range,
            'missing_data': missing_data
        }

    def analyze_parsing(self, filepath: str) -> Dict:
        """
        Analyze parsing success rate for each event type.

        Returns detailed metrics on what was successfully parsed.
        """
        if self.raw_df is None:
            df = self._load_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        else:
            df = self.raw_df

        # Parse all processable events
        processable = df[df['type'].isin(['field_input', 'browser_activity'])]

        parse_results = {
            'total_parsed': 0,
            'successful_app_extraction': 0,
            'successful_context_extraction': 0,
            'successful_value_extraction': 0,
            'failed_parsing': [],
            'by_type': {}
        }

        # Track unique values
        unique_apps = set()
        unique_contexts = set()
        unrecognized_patterns = []

        for event_type in ['field_input', 'browser_activity']:
            type_df = processable[processable['type'] == event_type]
            type_results = {
                'total': len(type_df),
                'app_success': 0,
                'context_success': 0,
                'value_success': 0
            }

            for idx, row in type_df.iterrows():
                parsed = parse_action(row['action'], row['type'])
                parse_results['total_parsed'] += 1

                if parsed['app']:
                    parse_results['successful_app_extraction'] += 1
                    type_results['app_success'] += 1
                    unique_apps.add(parsed['app'])
                else:
                    unrecognized_patterns.append({
                        'type': event_type,
                        'action': row['action'][:100]
                    })

                if parsed['context']:
                    parse_results['successful_context_extraction'] += 1
                    type_results['context_success'] += 1
                    unique_contexts.add(parsed['context'][:50] if parsed['context'] else None)

                if parsed['value']:
                    parse_results['successful_value_extraction'] += 1
                    type_results['value_success'] += 1

            parse_results['by_type'][event_type] = type_results

        # Calculate success rates
        total = parse_results['total_parsed']
        if total > 0:
            parse_results['app_success_rate'] = round(
                parse_results['successful_app_extraction'] / total * 100, 1
            )
            parse_results['context_success_rate'] = round(
                parse_results['successful_context_extraction'] / total * 100, 1
            )
            parse_results['value_success_rate'] = round(
                parse_results['successful_value_extraction'] / total * 100, 1
            )

        parse_results['unique_apps'] = list(unique_apps)
        parse_results['unique_apps_count'] = len(unique_apps)
        parse_results['unrecognized_patterns'] = unrecognized_patterns[:20]  # Limit to 20

        return parse_results

    def analyze_sessions(self, sessions: List[Dict]) -> Dict:
        """
        Analyze session quality metrics.
        """
        self.sessions = sessions

        if not sessions:
            return {'error': 'No sessions to analyze'}

        # Session type distribution
        type_counts = Counter(s['session_type'] for s in sessions)
        other_count = type_counts.get('Other', 0)
        other_percentage = round(other_count / len(sessions) * 100, 1) if sessions else 0

        # App distribution
        app_counts = Counter(s['primary_app'] for s in sessions)

        # Duration analysis
        durations = [s['duration_seconds'] for s in sessions]
        action_counts = [s['action_count'] for s in sessions]

        # Identify potential issues
        issues = []

        # Check for short sessions
        zero_duration = sum(1 for d in durations if d == 0)
        if zero_duration > len(sessions) * 0.1:
            issues.append(f"{zero_duration} sessions ({round(zero_duration/len(sessions)*100, 1)}%) have 0 duration")

        # Check for high "Other" rate
        if other_percentage > 30:
            issues.append(f"{other_percentage}% of sessions classified as 'Other' - rules may need expansion")

        # Check for potential misclassified apps
        suspicious_apps = []
        for app in app_counts.keys():
            if app and (
                len(app) > 50 or  # Long strings are likely misclassified
                re.match(r'^\d', app) or  # Starts with number
                'FW:' in app or 'RE:' in app or  # Email subjects
                '/' in app or '\\' in app  # File paths
            ):
                suspicious_apps.append(app)

        if suspicious_apps:
            issues.append(f"{len(suspicious_apps)} potentially misclassified apps detected")

        return {
            'total_sessions': len(sessions),
            'session_type_distribution': dict(type_counts),
            'other_percentage': other_percentage,
            'app_distribution': dict(app_counts.most_common(15)),
            'suspicious_apps': suspicious_apps,
            'duration_stats': {
                'min': min(durations),
                'max': max(durations),
                'avg': round(sum(durations) / len(durations), 1),
                'zero_duration_count': zero_duration
            },
            'action_stats': {
                'min': min(action_counts),
                'max': max(action_counts),
                'avg': round(sum(action_counts) / len(action_counts), 1),
                'total': sum(action_counts)
            },
            'issues': issues
        }

    def get_full_report(self, csv_path: str, sessions: List[Dict]) -> Dict:
        """
        Generate comprehensive data quality report.
        """
        raw_analysis = self.analyze_raw_data(csv_path)
        parsing_analysis = self.analyze_parsing(csv_path)
        session_analysis = self.analyze_sessions(sessions)

        # Overall quality score (simple heuristic)
        quality_factors = [
            raw_analysis['processed_percentage'] / 100,  # Data coverage
            parsing_analysis.get('app_success_rate', 0) / 100,  # Parse success
            1 - (session_analysis.get('other_percentage', 100) / 100),  # Classification coverage
        ]
        quality_score = round(sum(quality_factors) / len(quality_factors) * 100, 1)

        return {
            'quality_score': quality_score,
            'raw_data': raw_analysis,
            'parsing': parsing_analysis,
            'sessions': session_analysis,
            'recommendations': self._generate_recommendations(
                raw_analysis, parsing_analysis, session_analysis
            )
        }

    def _generate_recommendations(self, raw: Dict, parsing: Dict, sessions: Dict) -> List[str]:
        """
        Generate actionable recommendations based on analysis.
        """
        recommendations = []

        # Data coverage recommendations
        if raw['dropped_percentage'] > 50:
            recommendations.append(
                f"Consider processing more event types - currently dropping {raw['dropped_percentage']}% of data"
            )

        # Parsing recommendations
        if parsing.get('app_success_rate', 100) < 90:
            recommendations.append(
                f"App extraction success rate is {parsing.get('app_success_rate')}% - review unrecognized patterns"
            )

        # Classification recommendations
        if sessions.get('other_percentage', 0) > 25:
            recommendations.append(
                f"High 'Other' classification ({sessions['other_percentage']}%) - add more session type rules"
            )

        # Suspicious apps
        if sessions.get('suspicious_apps'):
            recommendations.append(
                f"Review {len(sessions['suspicious_apps'])} potentially misclassified apps"
            )

        if not recommendations:
            recommendations.append("Data quality looks good! No major issues detected.")

        return recommendations


# Standalone function for quick analysis
def analyze_data_quality(csv_path: str, sessions: List[Dict]) -> Dict:
    """
    Quick function to get full data quality report.
    """
    analyzer = DataQualityAnalyzer()
    return analyzer.get_full_report(csv_path, sessions)


if __name__ == "__main__":
    import json
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATA_FILE, OUTPUT_FILE

    # Load sessions
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        sessions = data['sessions']

    # Run analysis
    analyzer = DataQualityAnalyzer()
    report = analyzer.get_full_report(DATA_FILE, sessions)

    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    print(f"\n[QUALITY SCORE] {report['quality_score']}%")

    print(f"\n[RAW DATA]")
    print(f"  Total events: {report['raw_data']['total_events']:,}")
    print(f"  Processed: {report['raw_data']['processed_events']:,} ({report['raw_data']['processed_percentage']}%)")
    print(f"  Dropped: {report['raw_data']['dropped_events']:,} ({report['raw_data']['dropped_percentage']}%)")

    print(f"\n[PARSING]")
    print(f"  App extraction: {report['parsing'].get('app_success_rate', 'N/A')}%")
    print(f"  Unique apps found: {report['parsing']['unique_apps_count']}")

    print(f"\n[SESSIONS]")
    print(f"  Total: {report['sessions']['total_sessions']}")
    print(f"  'Other' type: {report['sessions']['other_percentage']}%")

    print(f"\n[RECOMMENDATIONS]")
    for rec in report['recommendations']:
        print(f"  - {rec}")

    if report['sessions'].get('suspicious_apps'):
        print(f"\n[SUSPICIOUS APPS]")
        for app in report['sessions']['suspicious_apps'][:5]:
            print(f"  - {app[:60]}...")
