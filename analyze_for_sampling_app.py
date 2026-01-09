"""
Exploratory Data Analysis App for Telemetry Data

This Streamlit app analyzes telemetry data to understand patterns
before building the semantic sessionizer.

Run with: streamlit run analyze_for_sampling_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
from collections import Counter
import re
import sys
import os
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.parser import parse_action, normalize_app_name, reload_classifications
from src.llm_sessionizer import LLMSessionizer
from config import (
    LLM_API_URL, LLM_MODEL, LLM_TIMEOUT,
    OPENROUTER_API_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL,
    OPENROUTER_FREE_MODELS, OPENROUTER_REASONING_DEFAULT
)

# Page config
st.set_page_config(
    page_title="Telemetry Data Explorer",
    page_icon="",
    layout="wide"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Section 11: Sample Pairs
if 'sample_pairs' not in st.session_state:
    st.session_state.sample_pairs = []
if 'pairs_per_category' not in st.session_state:
    st.session_state.pairs_per_category = 25

# Section 12: Unknown Pattern Classifier
if 'unknown_patterns' not in st.session_state:
    st.session_state.unknown_patterns = []
if 'pattern_classifications' not in st.session_state:
    st.session_state.pattern_classifications = {}
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = "Local (LM Studio)"
if 'openrouter_api_key' not in st.session_state:
    st.session_state.openrouter_api_key = OPENROUTER_API_KEY
if 'llm_connected' not in st.session_state:
    st.session_state.llm_connected = False
if 'parser_coverage' not in st.session_state:
    st.session_state.parser_coverage = None
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = []

# Section 13: Pair Labeling
if 'current_pair_idx' not in st.session_state:
    st.session_state.current_pair_idx = 0
if 'labeling_mode' not in st.session_state:
    st.session_state.labeling_mode = 'manual'
if 'llm_suggestion' not in st.session_state:
    st.session_state.llm_suggestion = None
if 'bulk_suggestions' not in st.session_state:
    st.session_state.bulk_suggestions = []
if 'bulk_suggestion_filter' not in st.session_state:
    st.session_state.bulk_suggestion_filter = 'all'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to standard schema.

    Handles two schemas:
    - Schema A: id, activity_ts, description, activity_type
    - Schema B: id, timestamp, action, type

    Returns DataFrame with columns: id, timestamp, action, type
    """
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    # Column mapping for Schema A -> Standard
    column_map = {
        'activity_ts': 'timestamp',
        'description': 'action',
        'activity_type': 'type'
    }

    # Apply renames
    for old_name, new_name in column_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    # Validate required columns exist
    required = ['timestamp', 'action', 'type']
    missing = [col for col in required if col not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.info(f"Found columns: {list(df.columns)}")
        return None

    return df


def load_file(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel file into DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


@st.cache_data
def extract_apps(df: pd.DataFrame) -> pd.Series:
    """Extract and normalize app names from action column."""
    apps = []
    for idx, row in df.iterrows():
        try:
            parsed = parse_action(row['action'], row['type'])
            app = parsed.get('app')
            if app:
                app = normalize_app_name(app)
            apps.append(app if app else 'Unknown')
        except Exception:
            apps.append('Unknown')
    return pd.Series(apps, index=df.index)


@st.cache_data
def extract_contexts(df: pd.DataFrame) -> pd.Series:
    """Extract context/window from action column."""
    contexts = []
    for idx, row in df.iterrows():
        try:
            parsed = parse_action(row['action'], row['type'])
            context = parsed.get('context') or parsed.get('window')
            contexts.append(context if context else '')
        except Exception:
            contexts.append('')
    return pd.Series(contexts, index=df.index)


def calculate_time_gaps(df: pd.DataFrame) -> pd.Series:
    """Calculate time gaps between consecutive events in seconds."""
    return df['timestamp'].diff().dt.total_seconds()


def bucket_time_gap(gap_seconds):
    """Categorize time gap into buckets."""
    if pd.isna(gap_seconds):
        return 'N/A'
    elif gap_seconds < 1:
        return '<1s'
    elif gap_seconds < 10:
        return '1-10s'
    elif gap_seconds < 30:
        return '10-30s'
    elif gap_seconds < 60:
        return '30-60s'
    elif gap_seconds < 300:
        return '60-300s'
    else:
        return '>300s'


def analyze_transitions(df: pd.DataFrame, apps: pd.Series) -> dict:
    """Analyze transition patterns between consecutive events."""
    transitions = []
    same_app_count = 0
    diff_app_count = 0

    for i in range(1, len(df)):
        app_a = apps.iloc[i-1]
        app_b = apps.iloc[i]
        type_a = df.iloc[i-1]['type']
        type_b = df.iloc[i]['type']

        transition = f"{app_a} ({type_a}) → {app_b} ({type_b})"
        transitions.append(transition)

        if app_a == app_b:
            same_app_count += 1
        else:
            diff_app_count += 1

    transition_counts = Counter(transitions)

    return {
        'transitions': transition_counts.most_common(30),
        'same_app_count': same_app_count,
        'diff_app_count': diff_app_count,
        'total': len(transitions)
    }


def analyze_same_timestamp_batches(df: pd.DataFrame) -> dict:
    """Analyze events with identical timestamps (batch captures)."""
    # Group by exact timestamp
    timestamp_groups = df.groupby('timestamp').size()

    # Filter to batches (more than 1 event at same timestamp)
    batches = timestamp_groups[timestamp_groups > 1]

    if len(batches) == 0:
        return {
            'total_batches': 0,
            'avg_batch_size': 0,
            'max_batch_size': 0,
            'total_events_in_batches': 0,
            'top_batches': []
        }

    # Get top 10 largest batches
    top_batches = batches.nlargest(10)
    top_batches_list = [
        {'timestamp': str(ts), 'count': int(count)}
        for ts, count in top_batches.items()
    ]

    return {
        'total_batches': len(batches),
        'avg_batch_size': float(batches.mean()),
        'max_batch_size': int(batches.max()),
        'total_events_in_batches': int(batches.sum()),
        'top_batches': top_batches_list
    }


def extract_entities(df: pd.DataFrame) -> dict:
    """Extract file names and URLs from actions."""
    files = []
    urls = []
    domains = []

    file_pattern = re.compile(r'[\w\-\.]+\.(xlsx|xls|pdf|docx|doc|csv|txt|pptx|ppt|zip|rar)', re.IGNORECASE)
    url_pattern = re.compile(r'https?://[^\s\'"<>]+')
    domain_pattern = re.compile(r'https?://([^/\s]+)')

    for action in df['action'].dropna():
        # Find files
        file_matches = file_pattern.findall(action)
        # Get full filename not just extension
        full_file_matches = re.findall(r'[\w\-\.]+\.(?:xlsx|xls|pdf|docx|doc|csv|txt|pptx|ppt|zip|rar)', action, re.IGNORECASE)
        files.extend(full_file_matches)

        # Find URLs
        url_matches = url_pattern.findall(action)
        urls.extend(url_matches)

        # Extract domains
        for url in url_matches:
            domain_match = domain_pattern.match(url)
            if domain_match:
                domains.append(domain_match.group(1))

    return {
        'files': Counter(files).most_common(20),
        'urls': Counter(urls).most_common(20),
        'domains': Counter(domains).most_common(20)
    }


def calculate_sampling_stats(df: pd.DataFrame, apps: pd.Series, contexts: pd.Series, gaps: pd.Series) -> dict:
    """Calculate statistics for sampling candidates."""
    stats = {
        'app_change_pairs': 0,
        'gap_30_300_pairs': 0,
        'same_app_context_change': 0,
        'dense_pairs': 0  # gap < 10s
    }

    for i in range(1, len(df)):
        gap = gaps.iloc[i]
        app_a = apps.iloc[i-1]
        app_b = apps.iloc[i]
        ctx_a = contexts.iloc[i-1]
        ctx_b = contexts.iloc[i]

        # App change
        if app_a != app_b:
            stats['app_change_pairs'] += 1

        # Time gap 30-300s
        if pd.notna(gap) and 30 <= gap <= 300:
            stats['gap_30_300_pairs'] += 1

        # Same app but different context
        if app_a == app_b and ctx_a != ctx_b and ctx_a and ctx_b:
            stats['same_app_context_change'] += 1

        # Dense activity (gap < 10s)
        if pd.notna(gap) and gap < 10:
            stats['dense_pairs'] += 1

    return stats


# =============================================================================
# SECTION 11-13 HELPER FUNCTIONS
# =============================================================================

def create_pair(df: pd.DataFrame, idx: int, category: str, apps: pd.Series, gaps: pd.Series) -> dict:
    """Create a pair object for labeling."""
    action_a = str(df.iloc[idx-1]['action']) if pd.notna(df.iloc[idx-1]['action']) else ''
    action_b = str(df.iloc[idx]['action']) if pd.notna(df.iloc[idx]['action']) else ''

    return {
        'pair_id': f'{category}_{idx}',
        'category': category,
        'event_a_idx': idx - 1,
        'event_b_idx': idx,
        'event_a_timestamp': str(df.iloc[idx-1]['timestamp']),
        'event_b_timestamp': str(df.iloc[idx]['timestamp']),
        'event_a_action': action_a[:500],
        'event_b_action': action_b[:500],
        'event_a_app': apps.iloc[idx-1],
        'event_b_app': apps.iloc[idx],
        'event_a_type': df.iloc[idx-1]['type'],
        'event_b_type': df.iloc[idx]['type'],
        'time_gap': float(gaps.iloc[idx]) if pd.notna(gaps.iloc[idx]) else 0.0,
        'app_changed': apps.iloc[idx-1] != apps.iloc[idx],
        'label': None,
        'labeled_by': None
    }


def generate_sample_pairs(df: pd.DataFrame, apps: pd.Series, contexts: pd.Series,
                          gaps: pd.Series, n_per_category: int = 25) -> list:
    """Generate sample pairs from 5 categories for labeling."""
    pairs = []

    # Category A: App Change
    app_change_indices = [i for i in range(1, len(df))
                          if apps.iloc[i-1] != apps.iloc[i]]
    sampled_a = random.sample(app_change_indices,
                              min(n_per_category, len(app_change_indices)))

    # Category B: Medium Gap (30-300s)
    medium_gap_indices = [i for i in range(1, len(df))
                          if pd.notna(gaps.iloc[i]) and 30 <= gaps.iloc[i] <= 300]
    sampled_b = random.sample(medium_gap_indices,
                              min(n_per_category, len(medium_gap_indices)))

    # Category C: Same App, Different Context
    context_change_indices = [i for i in range(1, len(df))
                              if apps.iloc[i-1] == apps.iloc[i]
                              and contexts.iloc[i-1] != contexts.iloc[i]
                              and contexts.iloc[i-1] and contexts.iloc[i]]
    sampled_c = random.sample(context_change_indices,
                              min(n_per_category, len(context_change_indices)))

    # Category D: Same-Timestamp Batch (<1s gap)
    batch_indices = [i for i in range(1, len(df))
                     if pd.notna(gaps.iloc[i]) and gaps.iloc[i] < 1]
    sampled_d = random.sample(batch_indices,
                              min(n_per_category, len(batch_indices)))

    # Category E: Long Gap (>300s, likely different sessions)
    long_gap_indices = [i for i in range(1, len(df))
                        if pd.notna(gaps.iloc[i]) and gaps.iloc[i] > 300]
    sampled_e = random.sample(long_gap_indices,
                              min(n_per_category, len(long_gap_indices)))

    # Build pair objects
    for idx in sampled_a:
        pairs.append(create_pair(df, idx, 'A_app_change', apps, gaps))
    for idx in sampled_b:
        pairs.append(create_pair(df, idx, 'B_medium_gap', apps, gaps))
    for idx in sampled_c:
        pairs.append(create_pair(df, idx, 'C_context_change', apps, gaps))
    for idx in sampled_d:
        pairs.append(create_pair(df, idx, 'D_batch', apps, gaps))
    for idx in sampled_e:
        pairs.append(create_pair(df, idx, 'E_long_gap', apps, gaps))

    return pairs


def find_unknown_patterns(df: pd.DataFrame, apps: pd.Series) -> list:
    """Find events with Unknown or .exe app names for classification.

    Deduplicates by raw_app - each unique app appears only once,
    with the first sample action preserved for context and a count
    of how many events use this app.
    """
    unknowns = {}  # Keyed by raw_app for deduplication

    for i, app in enumerate(apps):
        if app in ['Unknown', None, ''] or (isinstance(app, str) and app.endswith('.exe')):
            action = str(df.iloc[i]['action'])[:150] if pd.notna(df.iloc[i]['action']) else ''
            event_type = df.iloc[i]['type']

            if app not in unknowns:
                # First occurrence - create entry with sample action
                unknowns[app] = {
                    'index': i,
                    'raw_app': app,
                    'pattern': action,  # Keep first sample action for context
                    'type': event_type,
                    'count': 1,  # Track occurrence count
                    'classification': None,
                    'suggested_name': None,
                    'is_work_related': None
                }
            else:
                # Already seen - just increment count
                unknowns[app]['count'] += 1

    # Sort by count descending (most frequent apps first)
    return sorted(unknowns.values(), key=lambda x: x['count'], reverse=True)


def analyze_parser_coverage(df: pd.DataFrame) -> dict:
    """
    Analyze what patterns the parser handles vs what's unknown.

    Returns dict with:
    - known_patterns: list of {raw, normalized, count}
    - unknown_patterns: list of {raw, type, sample_action, count}
    - metrics: {total, known_count, unknown_count, coverage_pct}
    """
    raw_apps = []
    normalized_apps = []

    for idx, row in df.iterrows():
        action = row['action']
        event_type = row['type']

        try:
            parsed = parse_action(action, event_type)
            raw_app = parsed.get('app')  # Before normalization
            normalized = normalize_app_name(raw_app) if raw_app else 'Unknown'
        except:
            raw_app = None
            normalized = 'Unknown'

        raw_apps.append(raw_app if raw_app else 'Unknown')
        normalized_apps.append(normalized if normalized else 'Unknown')

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({
        'raw': raw_apps,
        'normalized': normalized_apps,
        'action': df['action'].astype(str).str[:150],
        'type': df['type']
    })

    # Known patterns: where normalization changed something meaningful
    def is_known(row):
        raw = str(row['raw']).lower() if row['raw'] else ''
        norm = str(row['normalized']).lower() if row['normalized'] else ''

        # It's unknown if:
        # 1. normalized is 'unknown' or empty
        # 2. normalized still ends with .exe (parser didn't convert it)
        if norm in ['unknown', '']:
            return False
        if norm.endswith('.exe'):
            return False
        return True

    analysis_df['is_known'] = analysis_df.apply(is_known, axis=1)

    # Group known patterns
    known_df = analysis_df[analysis_df['is_known']]
    known_grouped = known_df.groupby(['raw', 'normalized']).size().reset_index(name='count')
    known_grouped = known_grouped.sort_values('count', ascending=False)
    known_patterns = known_grouped.to_dict('records')

    # Group unknown patterns
    unknown_df = analysis_df[~analysis_df['is_known']]
    unknown_patterns = []
    for raw_val in unknown_df['raw'].unique():
        subset = unknown_df[unknown_df['raw'] == raw_val]
        unknown_patterns.append({
            'raw': raw_val,
            'type': subset['type'].mode().iloc[0] if len(subset) > 0 else 'unknown',
            'sample_action': subset['action'].iloc[0] if len(subset) > 0 else '',
            'count': len(subset)
        })
    unknown_patterns = sorted(unknown_patterns, key=lambda x: x['count'], reverse=True)

    # Metrics
    total_events = len(df)
    known_events = len(known_df)
    unknown_events = len(unknown_df)
    coverage_pct = (known_events / total_events * 100) if total_events > 0 else 0

    return {
        'known_patterns': known_patterns,
        'unknown_patterns': unknown_patterns,
        'metrics': {
            'total_unique': len(analysis_df['raw'].unique()),
            'known_unique': len(known_grouped),
            'unknown_unique': len(unknown_patterns),
            'total_events': total_events,
            'known_events': known_events,
            'unknown_events': unknown_events,
            'coverage_pct': coverage_pct
        }
    }


def get_llm_instance(provider: str, api_url: str = None, model: str = None,
                     api_key: str = None) -> LLMSessionizer:
    """Create LLM instance based on provider settings."""
    if provider == "OpenRouter (Online)":
        return LLMSessionizer(
            api_url=OPENROUTER_API_URL,
            model=model or OPENROUTER_MODEL,
            prompt_template="",
            timeout=LLM_TIMEOUT,
            provider="openrouter",
            api_key=api_key or OPENROUTER_API_KEY
        )
    else:
        return LLMSessionizer(
            api_url=api_url or LLM_API_URL,
            model=model or LLM_MODEL,
            prompt_template="",
            timeout=LLM_TIMEOUT,
            provider="local"
        )


# LLM Prompts
CLASSIFY_PATTERN_PROMPT = """Classify this application from user telemetry.

APPLICATION TO CLASSIFY: {raw_app}
Event Type: {event_type}
Sample Action (for context only): {pattern}

IMPORTANT: You are classifying the APPLICATION "{raw_app}", NOT whatever is mentioned in the sample action text.

Respond with JSON only:
{{
    "app_category": "Browser|Office|Communication|FileManager|System|ERP|Other",
    "is_work_related": "Yes|No|Maybe",
    "suggested_name": "clean normalized name for {raw_app}"
}}

For suggested_name: Provide a clean, human-readable name for the application "{raw_app}".
- If it's a known app (e.g., "chrome.exe" -> "Chrome"), use the common name.
- If unknown, derive from the executable name (e.g., "ShellExperienceHost.exe" -> "Shell Experience Host").
- Do NOT use names from the sample action text unless they match the actual application."""

PAIR_LABELING_PROMPT = """Should these two consecutive telemetry events be in the SAME work session or DIFFERENT sessions?

Event A ({event_a_timestamp}):
- App: {event_a_app}
- Type: {event_a_type}
- Action: {event_a_action}

Event B ({event_b_timestamp}):
- App: {event_b_app}
- Type: {event_b_type}
- Action: {event_b_action}

Time gap: {time_gap:.1f} seconds

Consider:
- Are they working on the same task/goal?
- Is the time gap reasonable for continuous work?
- Does the app/context change suggest a task switch?

Answer with JSON only:
{{
    "decision": "SAME" or "DIFFERENT",
    "confidence": "high" or "medium" or "low",
    "reason": "brief explanation"
}}"""


# =============================================================================
# MAIN APP
# =============================================================================

st.title("Telemetry Data Explorer")
st.markdown("Exploratory analysis of telemetry data for semantic sessionizer design")

# =============================================================================
# SECTION 1: FILE UPLOAD
# =============================================================================

st.header("1. File Upload")

uploaded_file = st.file_uploader(
    "Upload your telemetry data (CSV or Excel)",
    type=['csv', 'xlsx', 'xls'],
    help="Expected columns: timestamp, action, type (or activity_ts, description, activity_type)"
)

if uploaded_file is None:
    st.info("Please upload a file to begin analysis.")
    st.stop()

# Load and normalize
raw_df = load_file(uploaded_file)
if raw_df is None:
    st.stop()

st.success(f"Loaded {len(raw_df):,} rows from {uploaded_file.name}")

# Show raw columns
with st.expander("Raw Column Info"):
    st.write(f"**Original columns:** {list(raw_df.columns)}")

df = normalize_columns(raw_df)
if df is None:
    st.stop()

# Parse timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
df = df.sort_values('timestamp').reset_index(drop=True)

st.write("**Preview (first 10 rows):**")
st.dataframe(df.head(10), use_container_width=True)

# =============================================================================
# SECTION 2: BASIC STATS
# =============================================================================

st.header("2. Basic Stats")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Events", f"{len(df):,}")

with col2:
    date_range_days = (df['timestamp'].max() - df['timestamp'].min()).days
    st.metric("Date Range", f"{date_range_days} days")

with col3:
    st.metric("Start Date", df['timestamp'].min().strftime('%Y-%m-%d'))

with col4:
    st.metric("End Date", df['timestamp'].max().strftime('%Y-%m-%d'))

# Events per day
events_per_day = df.groupby(df['timestamp'].dt.date).size().reset_index()
events_per_day.columns = ['Date', 'Events']

fig_daily = px.line(
    events_per_day, x='Date', y='Events',
    title='Events per Day',
    markers=True
)
fig_daily.update_layout(height=300)
st.plotly_chart(fig_daily, use_container_width=True)

# =============================================================================
# SECTION 3: ACTIVITY TYPE DISTRIBUTION
# =============================================================================

st.header("3. Activity Type Distribution")

type_counts = df['type'].value_counts().reset_index()
type_counts.columns = ['Activity Type', 'Count']
type_counts['Percentage'] = (type_counts['Count'] / len(df) * 100).round(2)

col_chart, col_table = st.columns([2, 1])

with col_chart:
    fig_types = px.bar(
        type_counts, x='Activity Type', y='Count',
        color='Activity Type',
        title='Activity Type Distribution'
    )
    fig_types.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_types, use_container_width=True)

with col_table:
    st.write("**Activity Type Counts:**")
    st.dataframe(type_counts, use_container_width=True, hide_index=True)

# =============================================================================
# SECTION 4: APP DISTRIBUTION
# =============================================================================

st.header("4. App Distribution")

with st.spinner("Extracting app names from actions..."):
    apps = extract_apps(df)
    df['_app'] = apps

app_counts = apps.value_counts().reset_index()
app_counts.columns = ['App', 'Count']
app_counts['Percentage'] = (app_counts['Count'] / len(df) * 100).round(2)

col_chart2, col_table2 = st.columns([2, 1])

with col_chart2:
    # Top 20 apps
    fig_apps = px.bar(
        app_counts.head(20), x='App', y='Count',
        color='App',
        title='Top 20 Apps'
    )
    fig_apps.update_layout(showlegend=False, height=400)
    fig_apps.update_xaxes(tickangle=45)
    st.plotly_chart(fig_apps, use_container_width=True)

with col_table2:
    st.write("**App Counts (Top 20):**")
    st.dataframe(app_counts.head(20), use_container_width=True, hide_index=True)

# =============================================================================
# SECTION 5: TIME GAP ANALYSIS
# =============================================================================

st.header("5. Time Gap Analysis")

gaps = calculate_time_gaps(df)
df['_gap'] = gaps
valid_gaps = gaps.dropna()

# Statistics
col_stats1, col_stats2 = st.columns(2)

with col_stats1:
    st.subheader("Statistics")
    stats_data = {
        'Metric': ['Min', 'Max', 'Mean', 'Median', 'Std Dev'],
        'Value (seconds)': [
            f"{valid_gaps.min():.2f}",
            f"{valid_gaps.max():.2f}",
            f"{valid_gaps.mean():.2f}",
            f"{valid_gaps.median():.2f}",
            f"{valid_gaps.std():.2f}"
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

with col_stats2:
    st.subheader("Percentiles")
    percentiles = [25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(valid_gaps, percentiles)
    percentile_data = {
        'Percentile': [f'P{p}' for p in percentiles],
        'Value (seconds)': [f"{v:.2f}" for v in percentile_values]
    }
    st.dataframe(pd.DataFrame(percentile_data), use_container_width=True, hide_index=True)

# Histogram with buckets
st.subheader("Time Gap Distribution")

gap_buckets = gaps.apply(bucket_time_gap)
bucket_order = ['<1s', '1-10s', '10-30s', '30-60s', '60-300s', '>300s']
bucket_counts = gap_buckets.value_counts().reindex(bucket_order, fill_value=0).reset_index()
bucket_counts.columns = ['Bucket', 'Count']
bucket_counts['Percentage'] = (bucket_counts['Count'] / len(valid_gaps) * 100).round(2)

fig_gaps = px.bar(
    bucket_counts, x='Bucket', y='Count',
    color='Bucket',
    title='Time Gap Distribution by Bucket',
    text='Percentage'
)
fig_gaps.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig_gaps.update_layout(showlegend=False, height=400)
st.plotly_chart(fig_gaps, use_container_width=True)

# =============================================================================
# SECTION 5.5: SAME-TIMESTAMP BATCHES
# =============================================================================

st.header("5.5 Same-Timestamp Batches")
st.markdown("Events captured at exactly the same timestamp (batch captures that should always be same session)")

batch_analysis = analyze_same_timestamp_batches(df)

col_b1, col_b2, col_b3, col_b4 = st.columns(4)

with col_b1:
    st.metric("Total Batches", f"{batch_analysis['total_batches']:,}")

with col_b2:
    st.metric("Avg Batch Size", f"{batch_analysis['avg_batch_size']:.1f}")

with col_b3:
    st.metric("Max Batch Size", f"{batch_analysis['max_batch_size']}")

with col_b4:
    st.metric("Events in Batches", f"{batch_analysis['total_events_in_batches']:,}")

if batch_analysis['top_batches']:
    st.write("**Top 10 Largest Batches:**")
    batch_df = pd.DataFrame(batch_analysis['top_batches'])
    batch_df.columns = ['Timestamp', 'Event Count']
    st.dataframe(batch_df, use_container_width=True, hide_index=True)
else:
    st.info("No same-timestamp batches found in this dataset.")

# =============================================================================
# SECTION 6: TRANSITION PATTERNS
# =============================================================================

st.header("6. Transition Patterns")

with st.spinner("Analyzing transition patterns..."):
    transition_analysis = analyze_transitions(df, apps)

col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
    st.metric("Same App Transitions", f"{transition_analysis['same_app_count']:,}")

with col_t2:
    st.metric("Different App Transitions", f"{transition_analysis['diff_app_count']:,}")

with col_t3:
    same_pct = transition_analysis['same_app_count'] / transition_analysis['total'] * 100 if transition_analysis['total'] > 0 else 0
    st.metric("Same App %", f"{same_pct:.1f}%")

st.write("**Top 30 Transition Patterns:**")
transition_df = pd.DataFrame(transition_analysis['transitions'], columns=['Transition', 'Count'])
st.dataframe(transition_df, use_container_width=True, hide_index=True)

# =============================================================================
# SECTION 7: CONTEXT/WINDOW ANALYSIS
# =============================================================================

st.header("7. Context/Window Analysis")

with st.spinner("Extracting contexts..."):
    contexts = extract_contexts(df)
    df['_context'] = contexts

# For same-app consecutive events, calculate context change percentage
same_app_pairs = 0
context_change_count = 0

for i in range(1, len(df)):
    if apps.iloc[i-1] == apps.iloc[i]:
        same_app_pairs += 1
        ctx_a = contexts.iloc[i-1]
        ctx_b = contexts.iloc[i]
        if ctx_a != ctx_b and ctx_a and ctx_b:
            context_change_count += 1

context_change_pct = context_change_count / same_app_pairs * 100 if same_app_pairs > 0 else 0

col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    st.metric("Same-App Consecutive Pairs", f"{same_app_pairs:,}")

with col_c2:
    st.metric("Context Changes", f"{context_change_count:,}")

with col_c3:
    st.metric("Context Change Rate", f"{context_change_pct:.1f}%")

# Show unique contexts sample
unique_contexts = contexts[contexts != ''].unique()
st.write(f"**Unique Contexts Found:** {len(unique_contexts):,}")

with st.expander("Sample Contexts (first 20)"):
    for ctx in unique_contexts[:20]:
        st.write(f"- {ctx[:100]}{'...' if len(ctx) > 100 else ''}")

# =============================================================================
# SECTION 8: ENTITY EXTRACTION
# =============================================================================

st.header("8. Entity Extraction")

with st.spinner("Extracting entities..."):
    entities = extract_entities(df)

col_e1, col_e2 = st.columns(2)

with col_e1:
    st.subheader("Files Found")
    if entities['files']:
        files_df = pd.DataFrame(entities['files'], columns=['File', 'Count'])
        st.dataframe(files_df, use_container_width=True, hide_index=True)
    else:
        st.info("No file references found")

with col_e2:
    st.subheader("Domains Found")
    if entities['domains']:
        domains_df = pd.DataFrame(entities['domains'], columns=['Domain', 'Count'])
        st.dataframe(domains_df, use_container_width=True, hide_index=True)
    else:
        st.info("No domains found")

# URLs in expander (can be long)
with st.expander("Full URLs Found"):
    if entities['urls']:
        urls_df = pd.DataFrame(entities['urls'], columns=['URL', 'Count'])
        st.dataframe(urls_df, use_container_width=True, hide_index=True)
    else:
        st.info("No URLs found")

# =============================================================================
# SECTION 9: SAMPLING CANDIDATES
# =============================================================================

st.header("9. Sampling Candidates")
st.markdown("Key transition points for semantic boundary labeling")

with st.spinner("Calculating sampling statistics..."):
    sampling_stats = calculate_sampling_stats(df, apps, contexts, gaps)

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.metric(
        "App Change Pairs",
        f"{sampling_stats['app_change_pairs']:,}",
        help="Consecutive events where app changed"
    )

with col_s2:
    st.metric(
        "Gap 30-300s Pairs",
        f"{sampling_stats['gap_30_300_pairs']:,}",
        help="Pairs with time gap between 30 seconds and 5 minutes"
    )

with col_s3:
    st.metric(
        "Same App, Diff Context",
        f"{sampling_stats['same_app_context_change']:,}",
        help="Same app but context changed"
    )

with col_s4:
    st.metric(
        "Dense Pairs (<10s)",
        f"{sampling_stats['dense_pairs']:,}",
        help="Rapid consecutive events (likely same session)"
    )

# =============================================================================
# SECTION 10: EXPORT
# =============================================================================

st.header("10. Export Analysis")

# Compile all analysis into a dictionary
analysis_export = {
    'metadata': {
        'file_name': uploaded_file.name,
        'total_events': len(df),
        'date_range': {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat(),
            'days': date_range_days
        },
        'analyzed_at': datetime.now().isoformat()
    },
    'activity_types': type_counts.to_dict('records'),
    'app_distribution': app_counts.head(30).to_dict('records'),
    'time_gap_stats': {
        'min': float(valid_gaps.min()),
        'max': float(valid_gaps.max()),
        'mean': float(valid_gaps.mean()),
        'median': float(valid_gaps.median()),
        'percentiles': {f'p{p}': float(v) for p, v in zip(percentiles, percentile_values)}
    },
    'time_gap_buckets': bucket_counts.to_dict('records'),
    'same_timestamp_batches': batch_analysis,
    'transitions': {
        'same_app_count': transition_analysis['same_app_count'],
        'diff_app_count': transition_analysis['diff_app_count'],
        'top_30': [{'transition': t, 'count': c} for t, c in transition_analysis['transitions']]
    },
    'context_analysis': {
        'same_app_pairs': same_app_pairs,
        'context_changes': context_change_count,
        'context_change_rate': context_change_pct,
        'unique_contexts_count': len(unique_contexts)
    },
    'entities': {
        'files': [{'name': f, 'count': c} for f, c in entities['files']],
        'domains': [{'domain': d, 'count': c} for d, c in entities['domains']]
    },
    'sampling_candidates': sampling_stats
}

# Export button
export_json = json.dumps(analysis_export, indent=2, default=str)

st.download_button(
    label="Download Analysis JSON",
    data=export_json,
    file_name=f"telemetry_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json"
)

st.success("Analysis complete! Use the button above to download the full report.")

# =============================================================================
# SECTION 11: SAMPLE EVENT PAIRS GENERATOR
# =============================================================================

st.header("11. Sample Event Pairs Generator")
st.markdown("Generate sample pairs from different categories for session boundary labeling")

col_gen1, col_gen2 = st.columns([1, 2])

with col_gen1:
    n_per_category = st.slider(
        "Pairs per category",
        min_value=5,
        max_value=50,
        value=st.session_state.pairs_per_category,
        help="Number of pairs to sample from each category"
    )
    st.session_state.pairs_per_category = n_per_category

    if st.button("Generate Sample Pairs", type="primary"):
        with st.spinner("Generating sample pairs..."):
            st.session_state.sample_pairs = generate_sample_pairs(
                df, apps, contexts, gaps, n_per_category
            )
        st.success(f"Generated {len(st.session_state.sample_pairs)} pairs!")

with col_gen2:
    st.markdown("""
    **Categories:**
    - **A: App Change** - Consecutive events where application changed
    - **B: Medium Gap** - Time gap between 30-300 seconds
    - **C: Context Change** - Same app but different window/context
    - **D: Batch** - Events with <1s gap (likely same session)
    - **E: Long Gap** - Time gap >300 seconds (likely different sessions)
    """)

# Display generated pairs
if st.session_state.sample_pairs:
    pairs = st.session_state.sample_pairs

    # Summary by category
    category_counts = Counter(p['category'] for p in pairs)
    st.write("**Generated Pairs by Category:**")
    cat_df = pd.DataFrame([
        {'Category': cat, 'Count': count}
        for cat, count in sorted(category_counts.items())
    ])
    st.dataframe(cat_df, use_container_width=True, hide_index=True)

    # Pairs table
    st.subheader("Sample Pairs Preview")

    # Create display dataframe
    display_pairs = []
    for i, p in enumerate(pairs):
        display_pairs.append({
            '#': i + 1,
            'Category': p['category'],
            'App A': p['event_a_app'][:20],
            'App B': p['event_b_app'][:20],
            'Gap (s)': f"{p['time_gap']:.1f}",
            'App Changed': 'Yes' if p['app_changed'] else 'No',
            'Label': p['label'] or '-'
        })

    pairs_df = pd.DataFrame(display_pairs)
    st.dataframe(pairs_df, use_container_width=True, hide_index=True, height=300)

    # Expandable details
    with st.expander("View Pair Details"):
        pair_idx = st.number_input(
            "Pair number",
            min_value=1,
            max_value=len(pairs),
            value=1
        ) - 1

        selected_pair = pairs[pair_idx]
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("**Event A:**")
            st.write(f"Time: {selected_pair['event_a_timestamp']}")
            st.write(f"App: {selected_pair['event_a_app']}")
            st.write(f"Type: {selected_pair['event_a_type']}")
            st.text_area("Action A", selected_pair['event_a_action'], height=150, disabled=True, key="detail_action_a")

        with col_d2:
            st.markdown("**Event B:**")
            st.write(f"Time: {selected_pair['event_b_timestamp']}")
            st.write(f"App: {selected_pair['event_b_app']}")
            st.write(f"Type: {selected_pair['event_b_type']}")
            st.text_area("Action B", selected_pair['event_b_action'], height=150, disabled=True, key="detail_action_b")

    # Export pairs
    if st.button("Download Sample Pairs CSV"):
        pairs_export_df = pd.DataFrame(pairs)
        csv = pairs_export_df.to_csv(index=False)
        st.download_button(
            label="Click to Download",
            data=csv,
            file_name=f"sample_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_pairs_csv"
        )

# =============================================================================
# SECTION 12: UNKNOWN PATTERN CLASSIFIER (LLM)
# =============================================================================

st.header("12. Unknown Pattern Classifier (LLM)")
st.markdown("Analyze parser coverage and classify unknown patterns")

# -----------------------------------------------------------------------------
# 12.1 Parser Coverage Analysis
# -----------------------------------------------------------------------------
st.subheader("12.1 Parser Coverage Analysis")
st.markdown("What the parser already handles vs what needs classification")

# Analyze coverage
if st.button("Analyze Parser Coverage", key="analyze_coverage"):
    with st.spinner("Analyzing parser coverage..."):
        # Reload classifications from JSON to pick up newly saved patterns
        reload_classifications()
        st.session_state.parser_coverage = analyze_parser_coverage(df)

if st.session_state.parser_coverage:
    coverage = st.session_state.parser_coverage
    metrics = coverage['metrics']

    # Metrics row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Unique Patterns", f"{metrics['total_unique']:,}")
    with col_m2:
        st.metric("Already Normalized", f"{metrics['known_unique']:,}",
                  help="Patterns the parser successfully converts")
    with col_m3:
        st.metric("Unknown/Raw", f"{metrics['unknown_unique']:,}",
                  help="Patterns still ending with .exe or Unknown")
    with col_m4:
        st.metric("Coverage %", f"{metrics['coverage_pct']:.1f}%",
                  help="Percentage of events with known apps")

    # Coverage progress bar with color
    coverage_pct = metrics['coverage_pct']
    if coverage_pct >= 90:
        st.progress(coverage_pct / 100, text=f"Coverage: {coverage_pct:.1f}%")
        st.success("Excellent coverage!")
    elif coverage_pct >= 70:
        st.progress(coverage_pct / 100, text=f"Coverage: {coverage_pct:.1f}%")
        st.warning("Good coverage, but some patterns need classification")
    else:
        st.progress(coverage_pct / 100, text=f"Coverage: {coverage_pct:.1f}%")
        st.error("Low coverage - many patterns need classification")

    # Two side-by-side tables
    col_known, col_unknown = st.columns(2)

    with col_known:
        st.markdown("**Known Patterns (Parser Handles)**")
        if coverage['known_patterns']:
            known_display_df = pd.DataFrame(coverage['known_patterns'])
            known_display_df.columns = ['Raw Pattern', 'Normalized Name', 'Count']
            st.dataframe(known_display_df, use_container_width=True, hide_index=True, height=300)
        else:
            st.info("No known patterns found")

    with col_unknown:
        st.markdown("**Unknown Patterns (Needs Classification)**")
        if coverage['unknown_patterns']:
            unknown_display_df = pd.DataFrame(coverage['unknown_patterns'])
            unknown_display_df.columns = ['Raw Pattern', 'Type', 'Sample Action', 'Count']
            unknown_display_df['Sample Action'] = unknown_display_df['Sample Action'].str[:60] + '...'
            st.dataframe(unknown_display_df, use_container_width=True, hide_index=True, height=300)
        else:
            st.success("All patterns are known!")

st.divider()

# -----------------------------------------------------------------------------
# 12.2 LLM Classification
# -----------------------------------------------------------------------------
st.subheader("12.2 LLM Classification")

# LLM Configuration
with st.expander("LLM Configuration", expanded=True):
    llm_provider = st.radio(
        "Provider",
        ["Local (LM Studio)", "OpenRouter (Online)"],
        horizontal=True,
        index=0 if st.session_state.llm_provider == "Local (LM Studio)" else 1,
        key="llm_provider_radio"
    )
    st.session_state.llm_provider = llm_provider

    if llm_provider == "OpenRouter (Online)":
        col_or1, col_or2 = st.columns(2)
        with col_or1:
            openrouter_api_key = st.text_input(
                "OpenRouter API Key",
                value=st.session_state.openrouter_api_key,
                type="password",
                key="or_api_key"
            )
            st.session_state.openrouter_api_key = openrouter_api_key
        with col_or2:
            model_options = OPENROUTER_FREE_MODELS + ["Custom..."]
            selected_model = st.selectbox(
                "Model",
                model_options,
                index=0,
                key="or_model_select"
            )
            if selected_model == "Custom...":
                selected_model = st.text_input("Custom Model ID", value=OPENROUTER_MODEL, key="or_custom_model")

        api_url = OPENROUTER_API_URL
        api_key = openrouter_api_key
    else:
        col_local1, col_local2 = st.columns(2)
        with col_local1:
            api_url = st.text_input("API URL", value=LLM_API_URL, key="local_api_url")
        with col_local2:
            selected_model = st.text_input("Model", value=LLM_MODEL, key="local_model")
        api_key = None

    # Test connection button
    col_test1, col_test2 = st.columns([1, 3])
    with col_test1:
        if st.button("Test Connection", key="test_llm_conn"):
            with st.spinner("Testing connection..."):
                try:
                    result = LLMSessionizer.test_connection(
                        api_url=api_url,
                        model=selected_model,
                        timeout=10,
                        provider="openrouter" if llm_provider == "OpenRouter (Online)" else "local",
                        api_key=api_key
                    )
                    if result.get('success'):
                        st.session_state.llm_connected = True
                        st.success(f"Connected! Response: {result.get('response', '')[:100]}")
                    else:
                        st.session_state.llm_connected = False
                        st.error(f"Connection failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.session_state.llm_connected = False
                    st.error(f"Connection error: {e}")

    with col_test2:
        if st.session_state.llm_connected:
            st.success("LLM Connected")
        else:
            st.warning("LLM not connected - test connection first")

# Find unknown patterns
st.subheader("Unknown Patterns Discovery")

if st.button("Find Unknown Patterns", key="find_unknown"):
    with st.spinner("Scanning for unknown patterns..."):
        st.session_state.unknown_patterns = find_unknown_patterns(df, apps)
    st.success(f"Found {len(st.session_state.unknown_patterns)} unique unknown patterns")

if st.session_state.unknown_patterns:
    unknown_patterns = st.session_state.unknown_patterns

    st.write(f"**Found {len(unknown_patterns)} unique unknown patterns**")

    # Display patterns table
    patterns_display = []
    total_events_covered = sum(p.get('count', 1) for p in unknown_patterns)
    for i, p in enumerate(unknown_patterns):
        classification = st.session_state.pattern_classifications.get(p['raw_app'], {})
        patterns_display.append({
            '#': i + 1,
            'Raw App': p['raw_app'][:30] if p['raw_app'] else 'None',
            'Count': p.get('count', 1),  # Show event count
            'Type': p['type'],
            'Sample Action': p['pattern'][:60] + ('...' if len(p['pattern']) > 60 else ''),
            'Category': classification.get('app_category', '-'),
            'Work Related': classification.get('is_work_related', '-'),
            'Suggested Name': classification.get('suggested_name', '-')
        })

    st.write(f"These {len(unknown_patterns)} apps cover **{total_events_covered:,}** events total")

    patterns_df = pd.DataFrame(patterns_display)
    st.dataframe(patterns_df, use_container_width=True, hide_index=True, height=300)

    # Classify with LLM
    st.subheader("LLM Classification")

    col_class1, col_class2 = st.columns(2)
    with col_class1:
        num_to_classify = st.slider(
            "Patterns to classify",
            min_value=1,
            max_value=min(50, len(unknown_patterns)),
            value=min(10, len(unknown_patterns)),
            key="num_classify"
        )

    with col_class2:
        if st.button("Classify with LLM", disabled=not st.session_state.llm_connected, key="classify_btn"):
            llm = get_llm_instance(llm_provider, api_url, selected_model, api_key)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, pattern in enumerate(unknown_patterns[:num_to_classify]):
                status_text.text(f"Classifying pattern {i+1}/{num_to_classify}...")
                progress_bar.progress((i + 1) / num_to_classify)

                prompt = CLASSIFY_PATTERN_PROMPT.format(
                    raw_app=pattern['raw_app'] or 'Unknown',
                    pattern=pattern['pattern'][:300],
                    event_type=pattern['type']
                )

                try:
                    response, error = llm.call_llm(prompt)
                    if error:
                        st.warning(f"Pattern {i+1} error: {error}")
                        continue

                    # Parse JSON response
                    try:
                        # Find JSON in response
                        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                        if json_match:
                            classification = json.loads(json_match.group())
                            # Use raw_app as key for deduplication consistency
                            st.session_state.pattern_classifications[pattern['raw_app']] = classification
                            # Track classification result
                            st.session_state.classification_results.append({
                                'raw_pattern': pattern['raw_app'],
                                'action_sample': pattern['pattern'][:100],
                                'event_count': pattern.get('count', 1),  # Include event count
                                'llm_category': classification.get('app_category'),
                                'llm_work_related': classification.get('is_work_related'),
                                'llm_suggested_name': classification.get('suggested_name'),
                                'status': 'pending'
                            })
                    except json.JSONDecodeError:
                        st.warning(f"Pattern {i+1}: Could not parse LLM response")

                except Exception as e:
                    st.error(f"Pattern {i+1} error: {e}")

            status_text.text("Classification complete!")
            st.rerun()

    # Save classifications
    if st.session_state.classification_results:
        st.subheader("Save Classifications")
        if st.button("Save to JSON", key="save_classifications"):
            output_path = os.path.join(os.path.dirname(__file__), "output", "pattern_classifications.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(st.session_state.classification_results, f, indent=2)
            # Reload parser cache so new classifications are immediately available
            loaded_count = len(reload_classifications())
            st.success(f"Saved {len(st.session_state.classification_results)} classifications. Parser now has {loaded_count} patterns loaded.")

st.divider()

# -----------------------------------------------------------------------------
# 12.3 Classification Results Summary
# -----------------------------------------------------------------------------
st.subheader("12.3 Classification Results Summary")

if st.session_state.classification_results:
    results = st.session_state.classification_results

    # Before/After comparison
    if st.session_state.parser_coverage:
        metrics = st.session_state.parser_coverage['metrics']
        classified_count = len([r for r in results if r['status'] == 'accepted'])
        total_classified = len(results)

        st.markdown("**Before/After Comparison**")
        comparison_data = {
            'Metric': ['Known Patterns', 'Unknown Patterns', 'Coverage'],
            'Before': [
                metrics['known_unique'],
                metrics['unknown_unique'],
                f"{metrics['coverage_pct']:.1f}%"
            ],
            'After (if all accepted)': [
                metrics['known_unique'] + total_classified,
                max(0, metrics['unknown_unique'] - total_classified),
                f"{min(100, (metrics['known_events'] + total_classified) / metrics['total_events'] * 100):.1f}%"
            ]
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    # Newly classified table with Accept/Reject
    st.markdown("**Newly Classified Patterns**")

    for i, result in enumerate(results):
        status_icon = {'pending': '-', 'accepted': '✅', 'rejected': '❌'}.get(result['status'], '?')
        event_count = result.get('event_count', 1)
        with st.expander(f"{status_icon} {result['raw_pattern']} → {result['llm_suggested_name']} ({event_count} events)", expanded=False):
            col_info, col_accept, col_reject = st.columns([3, 1, 1])
            with col_info:
                st.write(f"**Events:** {event_count}")
                st.write(f"**Category:** {result['llm_category']}")
                st.write(f"**Work Related:** {result['llm_work_related']}")
                st.write(f"**Sample:** {result['action_sample']}")
            with col_accept:
                if st.button("Accept", key=f"accept_{i}", type="primary", disabled=result['status'] == 'accepted'):
                    st.session_state.classification_results[i]['status'] = 'accepted'
                    st.rerun()
            with col_reject:
                if st.button("Reject", key=f"reject_{i}", disabled=result['status'] == 'rejected'):
                    st.session_state.classification_results[i]['status'] = 'rejected'
                    st.rerun()

    # Summary counts
    accepted = len([r for r in results if r['status'] == 'accepted'])
    rejected = len([r for r in results if r['status'] == 'rejected'])
    pending = len([r for r in results if r['status'] == 'pending'])
    st.write(f"**Summary:** ✅ {accepted} accepted | ❌ {rejected} rejected | {pending} pending")

    # Bulk accept/reject buttons
    col_bulk_a, col_bulk_r, col_bulk_c = st.columns(3)
    with col_bulk_a:
        if st.button("Accept All Pending", key="accept_all_pending"):
            for r in st.session_state.classification_results:
                if r['status'] == 'pending':
                    r['status'] = 'accepted'
            st.rerun()
    with col_bulk_r:
        if st.button("Reject All Pending", key="reject_all_pending"):
            for r in st.session_state.classification_results:
                if r['status'] == 'pending':
                    r['status'] = 'rejected'
            st.rerun()
    with col_bulk_c:
        if st.button("Clear All Results", key="clear_results"):
            st.session_state.classification_results = []
            st.rerun()
else:
    st.info("No patterns have been classified yet. Use the LLM Classification above to classify unknown patterns.")

# =============================================================================
# SECTION 13: PAIR LABELING INTERFACE
# =============================================================================

st.header("13. Pair Labeling Interface")
st.markdown("Label event pairs as SAME or DIFFERENT session")

# Check if we have pairs to label
if not st.session_state.sample_pairs:
    st.warning("No sample pairs generated. Go to Section 11 to generate pairs first.")
else:
    pairs = st.session_state.sample_pairs

    # Progress
    labeled_count = sum(1 for p in pairs if p.get('label'))
    progress_pct = labeled_count / len(pairs)
    st.progress(progress_pct)
    st.write(f"**Labeled {labeled_count} of {len(pairs)} pairs ({progress_pct*100:.1f}%)**")

    # Bulk actions
    st.subheader("Bulk Actions")
    col_bulk1, col_bulk2, col_bulk3 = st.columns(3)

    with col_bulk1:
        if st.button("Auto-label Dense (<1s) as SAME", key="bulk_dense"):
            count = 0
            for pair in pairs:
                if pair['time_gap'] < 1 and not pair.get('label'):
                    pair['label'] = 'SAME'
                    pair['labeled_by'] = 'auto_dense'
                    count += 1
            st.success(f"Labeled {count} dense pairs as SAME")
            st.rerun()

    with col_bulk2:
        if st.button("Auto-label Long Gap (>300s) as DIFFERENT", key="bulk_gap"):
            count = 0
            for pair in pairs:
                if pair['time_gap'] > 300 and not pair.get('label'):
                    pair['label'] = 'DIFFERENT'
                    pair['labeled_by'] = 'auto_gap'
                    count += 1
            st.success(f"Labeled {count} long-gap pairs as DIFFERENT")
            st.rerun()

    with col_bulk3:
        if st.button("Clear All Labels", key="clear_labels"):
            for pair in pairs:
                pair['label'] = None
                pair['labeled_by'] = None
            st.session_state.current_pair_idx = 0
            st.session_state.bulk_suggestions = []
            st.success("All labels cleared")
            st.rerun()

    # Bulk LLM Suggestions
    st.subheader("Bulk LLM Suggestions")
    unlabeled_pairs = [(i, p) for i, p in enumerate(pairs) if not p.get('label')]
    st.write(f"**{len(unlabeled_pairs)} unlabeled pairs** available for LLM suggestions")

    col_blk1, col_blk2 = st.columns(2)
    with col_blk1:
        if st.button("Get Bulk LLM Suggestions", key="bulk_llm_suggest",
                     disabled=len(unlabeled_pairs) == 0 or not st.session_state.llm_connected):
            # Use same LLM config as single suggestion (api_url, selected_model, api_key from Section 12)
            llm = get_llm_instance(llm_provider, api_url, selected_model, api_key)

            progress_bar = st.progress(0)
            status_text = st.empty()
            suggestions = []

            for idx, (pair_idx, pair) in enumerate(unlabeled_pairs):
                status_text.text(f"Processing pair {idx+1} of {len(unlabeled_pairs)}...")
                progress_bar.progress((idx + 1) / len(unlabeled_pairs))

                prompt = PAIR_LABELING_PROMPT.format(**pair)

                try:
                    response, error = llm.call_llm(prompt)
                    if error:
                        suggestions.append({
                            'pair_idx': pair_idx,
                            'pair_id': pair.get('pair_id', f'pair_{pair_idx}'),
                            'app_a': pair['event_a_app'][:20],
                            'app_b': pair['event_b_app'][:20],
                            'time_gap': pair['time_gap'],
                            'decision': None,
                            'confidence': None,
                            'reason': None,
                            'status': 'pending',
                            'error': str(error)
                        })
                    else:
                        # Parse JSON response
                        try:
                            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                            if json_match:
                                result = json.loads(json_match.group())
                                suggestions.append({
                                    'pair_idx': pair_idx,
                                    'pair_id': pair.get('pair_id', f'pair_{pair_idx}'),
                                    'app_a': pair['event_a_app'][:20],
                                    'app_b': pair['event_b_app'][:20],
                                    'time_gap': pair['time_gap'],
                                    'decision': result.get('decision'),
                                    'confidence': result.get('confidence', 'medium'),
                                    'reason': result.get('reason', ''),
                                    'status': 'pending',
                                    'error': None
                                })
                            else:
                                suggestions.append({
                                    'pair_idx': pair_idx,
                                    'pair_id': pair.get('pair_id', f'pair_{pair_idx}'),
                                    'app_a': pair['event_a_app'][:20],
                                    'app_b': pair['event_b_app'][:20],
                                    'time_gap': pair['time_gap'],
                                    'decision': None,
                                    'confidence': None,
                                    'reason': None,
                                    'status': 'pending',
                                    'error': 'Could not parse JSON'
                                })
                        except json.JSONDecodeError:
                            suggestions.append({
                                'pair_idx': pair_idx,
                                'pair_id': pair.get('pair_id', f'pair_{pair_idx}'),
                                'app_a': pair['event_a_app'][:20],
                                'app_b': pair['event_b_app'][:20],
                                'time_gap': pair['time_gap'],
                                'decision': None,
                                'confidence': None,
                                'reason': None,
                                'status': 'pending',
                                'error': 'JSON parse error'
                            })
                except Exception as e:
                    suggestions.append({
                        'pair_idx': pair_idx,
                        'pair_id': pair.get('pair_id', f'pair_{pair_idx}'),
                        'app_a': pair['event_a_app'][:20],
                        'app_b': pair['event_b_app'][:20],
                        'time_gap': pair['time_gap'],
                        'decision': None,
                        'confidence': None,
                        'reason': None,
                        'status': 'pending',
                        'error': str(e)
                    })

                # Small delay to avoid overwhelming the LLM
                import time
                time.sleep(0.5)

            st.session_state.bulk_suggestions = suggestions
            status_text.text("Bulk suggestions complete!")
            st.rerun()

    with col_blk2:
        if st.button("Clear Bulk Suggestions", key="clear_bulk_suggestions",
                     disabled=len(st.session_state.bulk_suggestions) == 0):
            st.session_state.bulk_suggestions = []
            st.rerun()

    # Display bulk suggestions review interface
    if st.session_state.bulk_suggestions:
        st.subheader("Review Bulk Suggestions")

        suggestions = st.session_state.bulk_suggestions

        # Summary stats
        valid_suggestions = [s for s in suggestions if s['decision']]
        same_count = sum(1 for s in valid_suggestions if s['decision'] == 'SAME')
        diff_count = sum(1 for s in valid_suggestions if s['decision'] == 'DIFFERENT')
        error_count = sum(1 for s in suggestions if s['error'])
        pending_count = sum(1 for s in suggestions if s['status'] == 'pending' and s['decision'])

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("SAME", same_count)
        with col_stat2:
            st.metric("DIFFERENT", diff_count)
        with col_stat3:
            st.metric("Errors", error_count)
        with col_stat4:
            st.metric("Pending Review", pending_count)

        # Filter controls
        filter_col1, filter_col2 = st.columns([1, 3])
        with filter_col1:
            filter_option = st.selectbox(
                "Filter by",
                ['all', 'SAME only', 'DIFFERENT only', 'high confidence', 'errors'],
                key="bulk_filter"
            )

        # Bulk action buttons
        col_act1, col_act2, col_act3 = st.columns(3)
        with col_act1:
            high_conf_pending = [s for s in suggestions if s['confidence'] == 'high' and s['status'] == 'pending' and s['decision']]
            if st.button(f"Accept All High Confidence ({len(high_conf_pending)})", key="accept_high_conf"):
                for s in suggestions:
                    if s['confidence'] == 'high' and s['status'] == 'pending' and s['decision']:
                        s['status'] = 'accepted'
                        pairs[s['pair_idx']]['label'] = s['decision']
                        pairs[s['pair_idx']]['labeled_by'] = 'llm_bulk'
                st.rerun()

        with col_act2:
            all_pending = [s for s in suggestions if s['status'] == 'pending' and s['decision']]
            if st.button(f"Accept All Pending ({len(all_pending)})", key="accept_all_bulk"):
                for s in suggestions:
                    if s['status'] == 'pending' and s['decision']:
                        s['status'] = 'accepted'
                        pairs[s['pair_idx']]['label'] = s['decision']
                        pairs[s['pair_idx']]['labeled_by'] = 'llm_bulk'
                st.rerun()

        with col_act3:
            if st.button("Reject All Pending", key="reject_all_bulk"):
                for s in suggestions:
                    if s['status'] == 'pending':
                        s['status'] = 'rejected'
                st.rerun()

        # Filter suggestions
        filtered_suggestions = suggestions
        if filter_option == 'SAME only':
            filtered_suggestions = [s for s in suggestions if s['decision'] == 'SAME']
        elif filter_option == 'DIFFERENT only':
            filtered_suggestions = [s for s in suggestions if s['decision'] == 'DIFFERENT']
        elif filter_option == 'high confidence':
            filtered_suggestions = [s for s in suggestions if s['confidence'] == 'high']
        elif filter_option == 'errors':
            filtered_suggestions = [s for s in suggestions if s['error']]

        # Sort by confidence (high first)
        confidence_order = {'high': 0, 'medium': 1, 'low': 2, None: 3}
        filtered_suggestions = sorted(filtered_suggestions, key=lambda x: confidence_order.get(x['confidence'], 3))

        # Display table
        st.write(f"**Showing {len(filtered_suggestions)} suggestions**")

        for i, s in enumerate(filtered_suggestions):
            # Color based on confidence
            if s['confidence'] == 'high':
                conf_color = "background-color: #d4edda"  # green
            elif s['confidence'] == 'medium':
                conf_color = "background-color: #fff3cd"  # yellow
            else:
                conf_color = "background-color: #f8d7da"  # red

            status_icon = {'pending': '-', 'accepted': '✅', 'rejected': '❌'}.get(s['status'], '?')
            decision_display = s['decision'] if s['decision'] else 'ERROR'

            with st.expander(
                f"{status_icon} Pair {s['pair_idx']+1}: {s['app_a']} -> {s['app_b']} | {s['time_gap']:.1f}s | {decision_display} ({s['confidence'] or 'N/A'})",
                expanded=False
            ):
                col_info, col_actions = st.columns([3, 1])
                with col_info:
                    if s['error']:
                        st.error(f"Error: {s['error']}")
                    else:
                        st.write(f"**Decision:** {s['decision']}")
                        st.write(f"**Confidence:** {s['confidence']}")
                        st.write(f"**Reason:** {s['reason']}")

                with col_actions:
                    if s['status'] == 'pending' and s['decision']:
                        if st.button("Accept", key=f"accept_bulk_{s['pair_idx']}"):
                            s['status'] = 'accepted'
                            pairs[s['pair_idx']]['label'] = s['decision']
                            pairs[s['pair_idx']]['labeled_by'] = 'llm_bulk'
                            st.rerun()
                        if st.button("Reject", key=f"reject_bulk_{s['pair_idx']}"):
                            s['status'] = 'rejected'
                            st.rerun()

    st.divider()

    # Current pair display
    st.subheader("Manual Labeling")

    # Navigation
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])

    with col_nav1:
        if st.button("← Previous", key="prev_pair", disabled=st.session_state.current_pair_idx == 0):
            st.session_state.current_pair_idx -= 1
            st.session_state.llm_suggestion = None
            st.rerun()

    with col_nav2:
        # Jump to pair
        new_idx = st.number_input(
            "Go to pair #",
            min_value=1,
            max_value=len(pairs),
            value=st.session_state.current_pair_idx + 1,
            key="jump_pair"
        )
        if new_idx - 1 != st.session_state.current_pair_idx:
            st.session_state.current_pair_idx = new_idx - 1
            st.session_state.llm_suggestion = None
            st.rerun()

    with col_nav3:
        if st.button("Next →", key="next_pair", disabled=st.session_state.current_pair_idx >= len(pairs) - 1):
            st.session_state.current_pair_idx += 1
            st.session_state.llm_suggestion = None
            st.rerun()

    # Current pair
    current_idx = st.session_state.current_pair_idx
    pair = pairs[current_idx]

    # Info bar
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Time Gap", f"{pair['time_gap']:.1f}s")
    with col_info2:
        st.info(f"Category: {pair['category']}")
    with col_info3:
        current_label = pair.get('label', 'Not labeled')
        label_color = {'SAME': '[SAME]', 'DIFFERENT': '[DIFF]', 'UNSURE': '[?]'}.get(current_label, '[-]')
        st.write(f"**Current Label:** {label_color} {current_label}")

    # Event display
    col_evt1, col_evt2 = st.columns(2)

    with col_evt1:
        st.markdown("### Event A")
        st.write(f"**Time:** {pair['event_a_timestamp']}")
        st.write(f"**App:** {pair['event_a_app']}")
        st.write(f"**Type:** {pair['event_a_type']}")
        st.text_area("Action", pair['event_a_action'], height=150, disabled=True, key=f"evt_a_{current_idx}")

    with col_evt2:
        st.markdown("### Event B")
        st.write(f"**Time:** {pair['event_b_timestamp']}")
        st.write(f"**App:** {pair['event_b_app']}")
        st.write(f"**Type:** {pair['event_b_type']}")
        st.text_area("Action", pair['event_b_action'], height=150, disabled=True, key=f"evt_b_{current_idx}")

    # Label buttons
    st.subheader("Label this pair:")
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

    with col_btn1:
        if st.button("✅ SAME SESSION", type="primary", key="label_same"):
            pairs[current_idx]['label'] = 'SAME'
            pairs[current_idx]['labeled_by'] = 'human'
            if current_idx < len(pairs) - 1:
                st.session_state.current_pair_idx += 1
            st.session_state.llm_suggestion = None
            st.rerun()

    with col_btn2:
        if st.button("❌ DIFFERENT SESSION", key="label_diff"):
            pairs[current_idx]['label'] = 'DIFFERENT'
            pairs[current_idx]['labeled_by'] = 'human'
            if current_idx < len(pairs) - 1:
                st.session_state.current_pair_idx += 1
            st.session_state.llm_suggestion = None
            st.rerun()

    with col_btn3:
        if st.button("? UNSURE", key="label_unsure"):
            pairs[current_idx]['label'] = 'UNSURE'
            pairs[current_idx]['labeled_by'] = 'human'
            if current_idx < len(pairs) - 1:
                st.session_state.current_pair_idx += 1
            st.session_state.llm_suggestion = None
            st.rerun()

    with col_btn4:
        if st.button("Get LLM Suggestion", disabled=not st.session_state.llm_connected, key="get_llm_suggest"):
            llm = get_llm_instance(llm_provider, api_url, selected_model, api_key)
            prompt = PAIR_LABELING_PROMPT.format(**pair)

            with st.spinner("Getting LLM suggestion..."):
                try:
                    response, error = llm.call_llm(prompt)
                    if error:
                        st.error(f"LLM Error: {error}")
                    else:
                        # Parse response
                        try:
                            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                            if json_match:
                                st.session_state.llm_suggestion = json.loads(json_match.group())
                            else:
                                st.session_state.llm_suggestion = {'raw': response}
                        except json.JSONDecodeError:
                            st.session_state.llm_suggestion = {'raw': response}
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # Display LLM suggestion
    if st.session_state.llm_suggestion:
        suggestion = st.session_state.llm_suggestion
        st.divider()
        st.subheader("LLM Suggestion")

        if 'decision' in suggestion:
            col_sug1, col_sug2 = st.columns(2)
            with col_sug1:
                decision_icon = '[SAME]' if suggestion['decision'] == 'SAME' else '[DIFF]'
                st.write(f"**Decision:** {decision_icon} {suggestion['decision']}")
                st.write(f"**Confidence:** {suggestion.get('confidence', 'N/A')}")
            with col_sug2:
                st.write(f"**Reason:** {suggestion.get('reason', 'N/A')}")

            # Accept suggestion button
            if st.button(f"Accept: {suggestion['decision']}", key="accept_suggestion"):
                pairs[current_idx]['label'] = suggestion['decision']
                pairs[current_idx]['labeled_by'] = 'llm_assisted'
                if current_idx < len(pairs) - 1:
                    st.session_state.current_pair_idx += 1
                st.session_state.llm_suggestion = None
                st.rerun()
        else:
            st.text_area("Raw LLM Response", suggestion.get('raw', str(suggestion)), height=100, disabled=True)

    # Export labeled pairs
    st.divider()
    st.subheader("Export Labeled Pairs")

    labeled_pairs = [p for p in pairs if p.get('label')]
    st.write(f"**{len(labeled_pairs)} pairs labeled**")

    if labeled_pairs:
        labeled_df = pd.DataFrame(pairs)
        csv = labeled_df.to_csv(index=False)
        st.download_button(
            label="Download Labeled Pairs CSV",
            data=csv,
            file_name=f"labeled_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_labeled_csv"
        )

        # Summary stats
        label_counts = Counter(p['label'] for p in labeled_pairs)
        st.write("**Label Distribution:**")
        for label, count in sorted(label_counts.items()):
            pct = count / len(labeled_pairs) * 100
            st.write(f"- {label}: {count} ({pct:.1f}%)")
