"""
Sessionizer Dashboard - Visualize telemetry sessions

Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter
import os
import re
import sys
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_FILE, OUTPUT_FILE, TIME_GAP_THRESHOLD, MIN_SESSION_ACTIONS,
    ACTIVE_EVENT_TYPES, PASSIVE_EVENT_TYPES, EXCLUDE_EVENT_TYPES,
    LLM_API_URL, LLM_MODEL, LLM_TIMEOUT, LLM_MAX_RETRIES,
    LLM_BATCH_SIZE, LLM_BATCH_SIZE_MIN, LLM_BATCH_SIZE_MAX, LLM_PROMPT_TEMPLATE,
    OPENROUTER_API_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL,
    OPENROUTER_FREE_MODELS, OPENROUTER_REASONING_DEFAULT
)
import config  # For APP_CLASSIFIER_* settings
from src.parser import NORMALIZATION_MAP  # For known apps count
from src.data_quality import DataQualityAnalyzer
from src.sessionizer import Sessionizer
from src.event_whitelist import (
    load_decisions, save_decisions, mark_event_keep, mark_event_drop,
    mark_event_pending, bulk_mark_keep, bulk_mark_drop, apply_decisions_to_sessions_file,
    get_event_status, clear_all_decisions
)
from src.llm_sessionizer import LLMSessionizer, PROMPT_PRESETS
from src.intent_extractor import IntentExtractor
from src.workflow_extractor import WorkflowExtractor

# Page config
st.set_page_config(
    page_title="Sessionizer Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

# Load data functions
@st.cache_data
def load_sessions(filepath):
    """Load sessions from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

@st.cache_data
def get_quality_report(csv_path, sessions):
    """Get data quality report"""
    analyzer = DataQualityAnalyzer()
    return analyzer.get_full_report(csv_path, sessions)

@st.cache_data
def run_sessionizer_with_params(csv_path, time_gap, min_actions):
    """Run sessionizer with custom parameters"""
    sessionizer = Sessionizer(
        time_gap_threshold=time_gap,
        min_session_actions=min_actions
    )
    sessions, stats, _ = sessionizer.process(csv_path)
    return sessions, stats

@st.cache_data
def process_uploaded_csv(file_content, filename, time_gap, min_actions):
    """Process uploaded CSV file through sessionizer"""
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp:
        tmp.write(file_content)
        temp_path = tmp.name

    try:
        # Run sessionizer
        sessionizer = Sessionizer(
            time_gap_threshold=time_gap,
            min_session_actions=min_actions
        )
        sessions, stats, isolated_events = sessionizer.process(temp_path)

        # Create metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "input_file": filename,
            "parameters": {
                "time_gap_threshold": time_gap,
                "min_actions": min_actions
            }
        }

        return sessions, stats, metadata, temp_path, isolated_events
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e

def _is_uploaded_source() -> bool:
    return st.session_state.get("data_source", "default") == "uploaded"

def _get_decisions_store() -> dict:
    if _is_uploaded_source():
        if "uploaded_decisions" not in st.session_state:
            st.session_state.uploaded_decisions = {
                "kept_ids": [],
                "dropped_ids": [],
                "last_updated": None
            }
        return st.session_state.uploaded_decisions
    return load_decisions()

def _save_uploaded_decisions(decisions: dict) -> None:
    decisions["last_updated"] = datetime.now().isoformat()
    st.session_state.uploaded_decisions = decisions

def _set_uploaded_decision(event_id: str, status: str) -> None:
    decisions = _get_decisions_store()
    if status == "keep":
        if event_id in decisions["dropped_ids"]:
            decisions["dropped_ids"].remove(event_id)
        if event_id not in decisions["kept_ids"]:
            decisions["kept_ids"].append(event_id)
    elif status == "drop":
        if event_id in decisions["kept_ids"]:
            decisions["kept_ids"].remove(event_id)
        if event_id not in decisions["dropped_ids"]:
            decisions["dropped_ids"].append(event_id)
    elif status == "pending":
        if event_id in decisions["kept_ids"]:
            decisions["kept_ids"].remove(event_id)
        if event_id in decisions["dropped_ids"]:
            decisions["dropped_ids"].remove(event_id)
    _save_uploaded_decisions(decisions)

def _set_uploaded_decisions(event_ids: list, status: str) -> None:
    decisions = _get_decisions_store()
    for event_id in event_ids:
        if status == "keep":
            if event_id in decisions["dropped_ids"]:
                decisions["dropped_ids"].remove(event_id)
            if event_id not in decisions["kept_ids"]:
                decisions["kept_ids"].append(event_id)
        elif status == "drop":
            if event_id in decisions["kept_ids"]:
                decisions["kept_ids"].remove(event_id)
            if event_id not in decisions["dropped_ids"]:
                decisions["dropped_ids"].append(event_id)
        elif status == "pending":
            if event_id in decisions["kept_ids"]:
                decisions["kept_ids"].remove(event_id)
            if event_id in decisions["dropped_ids"]:
                decisions["dropped_ids"].remove(event_id)
    _save_uploaded_decisions(decisions)

def _mark_event_keep(event_id: str) -> None:
    if not _is_uploaded_source():
        mark_event_keep(event_id)
        return
    _set_uploaded_decision(event_id, "keep")

def _mark_event_drop(event_id: str) -> None:
    if not _is_uploaded_source():
        mark_event_drop(event_id)
        return
    _set_uploaded_decision(event_id, "drop")

def _mark_event_pending(event_id: str) -> None:
    if not _is_uploaded_source():
        mark_event_pending(event_id)
        return
    _set_uploaded_decision(event_id, "pending")

def _bulk_mark_keep(event_ids: list) -> None:
    if not _is_uploaded_source():
        bulk_mark_keep(event_ids)
        return
    _set_uploaded_decisions(event_ids, "keep")

def _bulk_mark_drop(event_ids: list) -> None:
    if not _is_uploaded_source():
        bulk_mark_drop(event_ids)
        return
    _set_uploaded_decisions(event_ids, "drop")

def _clear_all_decisions() -> None:
    if not _is_uploaded_source():
        clear_all_decisions()
        return
    st.session_state.uploaded_decisions = {
        "kept_ids": [],
        "dropped_ids": [],
        "last_updated": datetime.now().isoformat()
    }

APP_COLOR_MAP = {
    "excel": "#2ca02c",
    "outlook": "#1f77b4",
    "browser": "#ff7f0e",
    "teams": "#9467bd",
    "file_explorer": "#bcbd22",
    "other": "#7f7f7f"
}


def _normalize_app_key(app_name: str) -> str:
    if not app_name:
        return "other"
    name = str(app_name).lower()
    if "excel" in name:
        return "excel"
    if "outlook" in name:
        return "outlook"
    if "chrome" in name or "edge" in name:
        return "browser"
    if "teams" in name:
        return "teams"
    if "file explorer" in name or "explorer" in name:
        return "file_explorer"
    return "other"


def _get_app_color(app_name: str) -> str:
    return APP_COLOR_MAP.get(_normalize_app_key(app_name), APP_COLOR_MAP["other"])


def _build_color_map(apps: list) -> dict:
    return {app: _get_app_color(app) for app in apps if app}


def _parse_session_roles(workflow_data: dict) -> dict:
    roles_map = {}
    if not isinstance(workflow_data, dict):
        return roles_map
    roles = workflow_data.get("session_roles")
    if isinstance(roles, dict):
        for key, value in roles.items():
            try:
                roles_map[int(key)] = str(value)
            except (TypeError, ValueError):
                roles_map[str(key)] = str(value)
        return roles_map
    if isinstance(roles, list):
        for item in roles:
            if not item:
                continue
            text = str(item)
            match = re.match(r"Session\s*(\d+)\s*[:\-]\s*(.*)", text, re.IGNORECASE)
            if match:
                roles_map[int(match.group(1))] = match.group(2).strip()
    return roles_map


def build_workflow_viz_data(workflow_data, related_sessions):
    """
    Returns dict with:
    - timeline_data: list of {session_number, app, start_time, end_time, duration, intent}
    - app_breakdown: list of {app, action_count, percentage}
    - flow_data: list of {session_number, app, duration, gap_from_previous}
    """
    intent_extractor = IntentExtractor()
    roles_map = _parse_session_roles(workflow_data)

    timeline_data = []
    app_counts = Counter()
    flow_data = []

    for session in related_sessions:
        session_number = session.get("session_number")
        app = session.get("primary_app") or "Unknown"
        start_time = session.get("start_time")
        end_time = session.get("end_time")
        duration = session.get("duration_formatted", "N/A")
        session_id = session.get("session_id")

        intent_text = "N/A"
        if session_id:
            cached_intent = intent_extractor.get_cached_intent(session_id)
            if isinstance(cached_intent, dict) and cached_intent.get("intent"):
                intent_text = cached_intent.get("intent")

        timeline_data.append({
            "session_number": session_number,
            "app": app,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "intent": intent_text
        })

        actions = session.get("actions")
        if isinstance(actions, list) and actions:
            for action in actions:
                action_app = action.get("app") or app
                if action_app:
                    app_counts[action_app] += 1
        else:
            count = session.get("action_count", 0) or 0
            if app:
                app_counts[app] += int(count)

        gap_info = session.get("gap_from_previous")
        gap_formatted = None
        if isinstance(gap_info, dict):
            gap_formatted = gap_info.get("formatted")

        flow_data.append({
            "session_number": session_number,
            "app": app,
            "duration": duration,
            "gap_from_previous": gap_formatted,
            "intent": intent_text,
            "session_role": roles_map.get(session_number) or roles_map.get(str(session_number))
        })

    total_actions = sum(app_counts.values()) if app_counts else 0
    app_breakdown = []
    for app_name, count in app_counts.most_common():
        percentage = int(round((count / total_actions) * 100)) if total_actions else 0
        app_breakdown.append({
            "app": app_name,
            "action_count": count,
            "percentage": percentage
        })

    return {
        "timeline_data": timeline_data,
        "app_breakdown": app_breakdown,
        "flow_data": flow_data
    }

def create_timeline_events(sessions, isolated_events, promoted_actions):
    """
    Combine all events into a single chronological timeline.

    Args:
        sessions: List of session dictionaries
        isolated_events: List of isolated event dictionaries
        promoted_actions: List of promoted action dictionaries

    Returns:
        List of timeline events sorted by timestamp
    """
    timeline = []

    # Add sessions
    for s in sessions:
        # Convert timestamp to string if it's a Timestamp object
        start_ts = s['start_time'] if isinstance(s['start_time'], str) else s['start_time'].isoformat()
        end_ts = s['end_time'] if isinstance(s['end_time'], str) else s['end_time'].isoformat()
        timeline.append({
            'timestamp': start_ts,
            'end_time': end_ts,
            'type': 'SESSION',
            'id': f"Session {s['session_number']}",
            'summary': s['summary'],
            'app': s['primary_app'],
            'action_count': s['action_count'],
            'duration': s['duration_formatted'],
            'session_type': s.get('session_type', 'Other'),
            'details': s  # Full session data for expansion
        })

    # Add isolated events
    for e in isolated_events:
        timeline.append({
            'timestamp': e['timestamp'],
            'end_time': e['timestamp'],
            'type': 'ISOLATED',
            'id': e['id'],
            'summary': f"{e.get('category', 'Other')}: {e.get('field', 'N/A')}",
            'app': e.get('app') or 'Unknown',
            'action_count': 1,
            'duration': '0s',
            'session_type': e.get('category', 'Other'),
            'details': e
        })

    # Add promoted actions
    for p in promoted_actions:
        timeline.append({
            'timestamp': p['timestamp'],
            'end_time': p['timestamp'],
            'type': 'PROMOTED',
            'id': p['id'],
            'summary': f"[KEPT] {p.get('category', 'Other')}: {p.get('field', 'N/A')}",
            'app': p.get('app') or 'Unknown',
            'action_count': 1,
            'duration': '0s',
            'session_type': p.get('category', 'Other'),
            'details': p
        })

    # Sort by timestamp (all timestamps are now strings in ISO format, which sorts correctly)
    timeline.sort(key=lambda x: x['timestamp'])

    return timeline


def _apply_decisions() -> dict:
    if not _is_uploaded_source():
        return apply_decisions_to_sessions_file()

    decisions = _get_decisions_store()
    kept_ids = set(decisions.get("kept_ids", []))
    dropped_ids = set(decisions.get("dropped_ids", []))

    isolated_events = st.session_state.get("uploaded_isolated_events", [])
    promoted_actions = st.session_state.get("uploaded_promoted_actions", [])

    new_promoted = []
    remaining_isolated = []
    dropped_count = 0

    for event in isolated_events:
        event_id = event.get("id")
        if event_id in kept_ids:
            event["status"] = "kept"
            event["promoted_at"] = datetime.now().isoformat()
            new_promoted.append(event)
        elif event_id in dropped_ids:
            dropped_count += 1
        else:
            remaining_isolated.append(event)

    promoted_actions = promoted_actions + new_promoted
    st.session_state.uploaded_promoted_actions = promoted_actions
    st.session_state.uploaded_isolated_events = remaining_isolated

    if "uploaded_stats" in st.session_state:
        st.session_state.uploaded_stats["isolated_events_count"] = len(remaining_isolated)
        st.session_state.uploaded_stats["promoted_actions_count"] = len(promoted_actions)

    return {
        "promoted": len(new_promoted),
        "dropped": dropped_count,
        "remaining_pending": len(remaining_isolated),
        "total_promoted": len(promoted_actions)
    }

# ============================================
# FILE UPLOAD HANDLING (Must be before sidebar)
# ============================================

# Initialize session state for data source
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'default'
if 'uploaded_csv_path' not in st.session_state:
    st.session_state.uploaded_csv_path = None

# Get default file paths
DEFAULT_SESSIONS_FILE = OUTPUT_FILE
DEFAULT_CSV_FILE = DATA_FILE

# ============================================
# SIDEBAR - Data Source Selection
# ============================================
with st.sidebar:
    st.header("Sessionizer Dashboard")
    st.markdown("---")

    # Data Source Section
    st.subheader("Data Source")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload raw telemetry CSV with columns: timestamp, action, type"
    )

    # Parameters for processing uploaded file
    if uploaded_file is not None:
        st.write("**Processing Parameters:**")

        # Time Gap slider (always shown, no approach selector)
        upload_time_gap = st.slider(
            "Time Gap (s)",
            min_value=60,
            max_value=1800,  # Increased to 30 minutes
            value=TIME_GAP_THRESHOLD,  # Default: 900s (15 min)
            step=30,
            key="upload_time_gap",
            help="Idle time gap that triggers a new session. Default: 900s (15 min)"
        )

        upload_min_actions = st.slider(
            "Min Actions",
            min_value=1,
            max_value=10,
            value=MIN_SESSION_ACTIONS,
            step=1,
            key="upload_min_actions"
        )

        if st.button("Process Uploaded File", type="primary"):
            with st.spinner("Processing file..."):
                try:
                    file_content = uploaded_file.getvalue()
                    sessions, stats, metadata, temp_path, isolated_events = process_uploaded_csv(
                        file_content,
                        uploaded_file.name,
                        upload_time_gap,
                        upload_min_actions
                    )
                    st.session_state.data_source = 'uploaded'
                    st.session_state.uploaded_sessions = sessions
                    st.session_state.uploaded_stats = stats
                    st.session_state.uploaded_metadata = metadata
                    st.session_state.uploaded_csv_path = temp_path
                    st.session_state.uploaded_isolated_events = isolated_events
                    st.session_state.uploaded_promoted_actions = []
                    st.success(f"Processed {len(sessions)} sessions, {len(isolated_events)} isolated events!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing file: {e}")

    # Use sample data button
    if st.session_state.data_source == 'uploaded':
        if st.button("Use Sample Data Instead"):
            st.session_state.data_source = 'default'
            # Clean up temp file
            if st.session_state.uploaded_csv_path and os.path.exists(st.session_state.uploaded_csv_path):
                os.unlink(st.session_state.uploaded_csv_path)
            st.session_state.uploaded_csv_path = None
            st.rerun()

    st.markdown("---")

# ============================================
# LOAD DATA BASED ON SOURCE
# ============================================
if st.session_state.data_source == 'uploaded' and 'uploaded_sessions' in st.session_state:
    # Use uploaded data
    sessions = st.session_state.uploaded_sessions
    stats = st.session_state.uploaded_stats
    metadata = st.session_state.uploaded_metadata
    isolated_events = st.session_state.get('uploaded_isolated_events', [])
    promoted_actions = st.session_state.get('uploaded_promoted_actions', [])
    CSV_FILE = st.session_state.uploaded_csv_path
    data_source_label = f"Uploaded: {metadata['input_file']}"
else:
    # Use default data
    if not os.path.exists(DEFAULT_SESSIONS_FILE):
        st.error(f"Sessions file not found: {DEFAULT_SESSIONS_FILE}")
        st.info("Please run the sessionizer first: python run_sessionizer.py")
        st.info("Or upload a CSV file using the sidebar.")
        st.stop()

    data = load_sessions(DEFAULT_SESSIONS_FILE)
    sessions = data['sessions']
    stats = data['statistics']
    metadata = data['metadata']
    isolated_events = data.get('isolated_events', [])
    promoted_actions = data.get('promoted_actions', [])
    CSV_FILE = DEFAULT_CSV_FILE
    data_source_label = f"Sample: {os.path.basename(metadata['input_file'])}"

# Convert to DataFrame for easier manipulation
sessions_df = pd.DataFrame(sessions)
# Handle ISO8601 strings with mixed precision (with/without milliseconds)
sessions_df['start_time'] = pd.to_datetime(sessions_df['start_time'], format='mixed', utc=True, errors='coerce')
sessions_df['end_time'] = pd.to_datetime(sessions_df['end_time'], format='mixed', utc=True, errors='coerce')

# CRITICAL FIX: Sort sessions by start_time to ensure chronological order
sessions_df = sessions_df.sort_values('start_time').reset_index(drop=True)
# Update session numbers to match chronological order
sessions_df['session_number'] = range(1, len(sessions_df) + 1)
# Update the sessions list to match the sorted order
sessions = sessions_df.to_dict('records')

# ============================================
# SIDEBAR (continued) - Data Info
# ============================================
with st.sidebar:
    # Data source indicator
    if st.session_state.data_source == 'uploaded':
        st.success(f"Using: {data_source_label}")
    else:
        st.info(f"Using: {data_source_label}")

    st.subheader("Data Info")
    st.write(f"**Generated:** {metadata['generated_at'][:19]}")
    st.write(f"**Sessions:** {len(sessions)}")

    st.subheader("Parameters")
    st.write(f"**Approach:** Time-Based")
    if metadata['parameters'].get('time_gap_threshold'):
        st.write(f"**Time Gap:** {metadata['parameters']['time_gap_threshold']}s")
    st.write(f"**Min Actions:** {metadata['parameters']['min_actions']}")

    # App Classifier Info
    st.subheader("App Classifier")

    # Load cache
    cache_file = os.path.join(os.path.dirname(OUTPUT_FILE), 'pattern_classifications.json')
    auto_count = 0
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                classifications = json.load(f)
                auto_count = len([c for c in classifications if c.get('source') == 'app_classifier'])
        except:
            pass

    known_count = len(NORMALIZATION_MAP)

    st.write(f"**Known:** {known_count}")
    st.write(f"**Auto-classified:** {auto_count}")
    try:
        status_icon = '✅' if config.APP_CLASSIFIER_ENABLED else '❌'
        status_text = 'Enabled' if config.APP_CLASSIFIER_ENABLED else 'Disabled'
        st.write(f"**Status:** {status_icon} {status_text}")
    except AttributeError:
        st.write(f"**Status:** ❓ Unknown")

    

    with st.expander("LLM Settings", expanded=False):
        intent_no_cap = st.toggle(
            "Intent: No cap (omit max_tokens)",
            value=True,
            key="intent_no_cap"
        )
        intent_max_tokens = st.slider(
            "Intent max tokens",
            min_value=512,
            max_value=4096,
            value=1024,
            step=128,
            disabled=intent_no_cap,
            key="intent_max_tokens"
        )
        workflow_no_cap = st.toggle(
            "Workflow: No cap (omit max_tokens)",
            value=True,
            key="workflow_no_cap"
        )
        workflow_max_tokens = st.slider(
            "Workflow max tokens",
            min_value=1024,
            max_value=8192,
            value=2048,
            step=256,
            disabled=workflow_no_cap,
            key="workflow_max_tokens"
        )
        include_actions = st.toggle(
            "Include top N actions in prompt",
            value=True,
            key="llm_include_actions"
        )
        max_actions = st.slider(
            "Max actions to include",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            disabled=not include_actions,
            key="llm_max_actions"
        )
    st.markdown("---")
    st.markdown("*Built with Streamlit*")

# ============================================
# TABS
# ============================================
tab_overview, tab_explorer, tab_quality, tab_tuning, tab_isolated, tab_timeline, tab_llm, tab_apps = st.tabs([
    "Overview",
    "Session Explorer",
    "Data Quality",
    "Parameter Tuning",
    "Isolated Actions",
    "Timeline View",
    "LLM Sessionization",
    "App Classifier"
])

# ============================================
# TAB 1: OVERVIEW
# ============================================
with tab_overview:
    st.title("Sessions Overview")

    # KEY METRICS
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="Total Sessions", value=stats['total_sessions'])

    with col2:
        st.metric(label="Total Actions", value=f"{stats['total_actions']:,}")

    with col3:
        st.metric(label="Total Duration", value=stats['total_duration_formatted'])

    with col4:
        st.metric(label="Avg Duration", value=stats['average_duration_formatted'])

    with col5:
        st.metric(label="Most Active App", value=stats['most_active_app'])

    # Show passive events stats if available
    if stats.get('total_passive_events', 0) > 0:
        st.markdown("---")
        col_p1, col_p2, col_p3 = st.columns(3)

        with col_p1:
            st.metric(label="Active Events", value=f"{stats['total_actions']:,}")
        with col_p2:
            st.metric(label="Passive Events", value=f"{stats.get('total_passive_events', 0):,}")
        with col_p3:
            st.metric(label="Total Events", value=f"{stats.get('total_events', stats['total_actions']):,}")

    # Show Isolated Events and Promoted Actions summary
    if isolated_events or promoted_actions:
        st.markdown("---")

        col_iso, col_promo = st.columns(2)

        with col_iso:
            if isolated_events:
                st.warning(f"**Isolated Actions:** {len(isolated_events)} events pending review")
                st.write("Go to 'Isolated Actions' tab to review and decide which to keep.")

        with col_promo:
            if promoted_actions:
                st.success(f"**Promoted Actions:** {len(promoted_actions)} user-approved single actions")
                with st.expander("View Promoted Actions"):
                    for action in promoted_actions[:5]:
                        st.write(f"- [{action.get('category', 'Other')}] {action.get('app', 'N/A')}: {action.get('field', 'N/A')}")
                    if len(promoted_actions) > 5:
                        st.write(f"... and {len(promoted_actions) - 5} more")

    st.markdown("---")

    # CHARTS
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("App Distribution")
        app_data = pd.DataFrame([
            {"App": app, "Sessions": count}
            for app, count in stats['app_distribution'].items()
        ])
        fig_apps = px.pie(
            app_data, values='Sessions', names='App', hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_apps.update_traces(textposition='inside', textinfo='percent+label')
        fig_apps.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_apps, width='stretch')

    with col_right:
        st.subheader("Session Types")
        type_data = pd.DataFrame([
            {"Type": stype, "Sessions": count}
            for stype, count in stats['session_type_distribution'].items()
        ])
        fig_types = px.bar(
            type_data, x='Type', y='Sessions', color='Type',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_types.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_types, width='stretch')

    # HOURLY ACTIVITY
    st.subheader("Activity by Hour of Day")
    hourly_data = pd.DataFrame([
        {"Hour": f"{int(hour):02d}:00", "Sessions": count}
        for hour, count in sorted(stats['hourly_distribution'].items(), key=lambda x: int(x[0]))
    ])
    fig_hourly = px.bar(
        hourly_data, x='Hour', y='Sessions', color='Sessions',
        color_continuous_scale='Blues'
    )
    fig_hourly.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_hourly, width='stretch')

    # SESSION TIMELINE
    st.subheader("Session Timeline (First 30)")
    timeline_data = sessions_df[['session_number', 'start_time', 'end_time', 'primary_app', 'summary', 'duration_formatted']].copy()
    fig_timeline = px.timeline(
        timeline_data.head(30),
        x_start='start_time', x_end='end_time', y='session_number',
        color='primary_app', hover_data=['summary', 'duration_formatted'],
        labels={'session_number': 'Session #', 'primary_app': 'App'}
    )
    fig_timeline.update_yaxes(autorange="reversed")
    fig_timeline.update_layout(height=500)
    st.plotly_chart(fig_timeline, width='stretch')

    # APP CLASSIFICATION STATUS
    st.subheader("App Classification Status")

    # Load classification cache
    cache_file = os.path.join(os.path.dirname(OUTPUT_FILE), 'pattern_classifications.json')
    auto_classified = []
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                auto_classified = json.load(f)
        except Exception as e:
            st.warning(f"Could not load classification cache: {e}")

    # Count apps
    known_apps = len(NORMALIZATION_MAP)
    auto_classified_count = len([c for c in auto_classified if c.get('source') == 'app_classifier'])

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        try:
            status_text = "Enabled" if config.APP_CLASSIFIER_ENABLED else "Disabled"
            delta_text = "Active" if config.APP_CLASSIFIER_ENABLED else None
        except AttributeError:
            status_text = "Unknown"
            delta_text = None
        st.metric("Classifier Status", status_text, delta=delta_text)

    with col2:
        st.metric("Known Apps", known_apps, help="Apps in NORMALIZATION_MAP")

    with col3:
        st.metric("Auto-Classified", auto_classified_count, help="Apps classified by LLM")

    with col4:
        try:
            provider = config.APP_CLASSIFIER_LLM_PROVIDER
            provider_url = config.LLM_API_URL if provider == 'local' else 'OpenRouter'
            st.metric("LLM Provider", provider.title(), help=f"Using {provider_url}")
        except AttributeError:
            st.metric("LLM Provider", "Unknown")

    # Info expander
    with st.expander("View LLM Configuration", expanded=False):
        try:
            provider = config.APP_CLASSIFIER_LLM_PROVIDER
            if provider == 'local':
                url = config.LLM_API_URL
                model = config.LLM_MODEL
            else:
                url = config.OPENROUTER_API_URL
                model = config.OPENROUTER_MODEL
            auto_save = config.APP_CLASSIFIER_AUTO_SAVE

            st.code(f"""LLM Provider: {provider}
LLM URL: {url}
LLM Model: {model}
Auto-save: {auto_save}""", language="yaml")
        except AttributeError as e:
            st.warning(f"Could not load full configuration: {e}")

# ============================================
# TAB 2: SESSION EXPLORER
# ============================================
with tab_explorer:
    st.title("Session Explorer")

    # Filters
    col_filter1, col_filter2, col_filter3 = st.columns(3)

    with col_filter1:
        app_filter = st.selectbox(
            "Filter by App",
            options=["All"] + list(stats['app_distribution'].keys()),
            key="explorer_app"
        )

    with col_filter2:
        type_filter = st.selectbox(
            "Filter by Session Type",
            options=["All"] + list(stats['session_type_distribution'].keys()),
            key="explorer_type"
        )

    with col_filter3:
        sort_by = st.selectbox(
            "Sort by",
            options=["Session Number", "Duration (Longest)", "Duration (Shortest)", "Actions (Most)", "Actions (Least)"],
            key="explorer_sort"
        )

    workflow_gap_default = getattr(config, "WORKFLOW_GAP_THRESHOLD_HOURS", 4)
    workflow_gap_hours = st.slider(
        "Workflow Grouping Threshold (hours)",
        min_value=1,
        max_value=8,
        value=int(workflow_gap_default),
        step=1,
        key="workflow_gap_threshold"
    )
    workflow_gap_seconds = int(workflow_gap_hours * 3600)
    workflow_max_sessions = getattr(config, "WORKFLOW_MAX_SESSIONS", 10)
    workflow_max_span_hours = getattr(config, "WORKFLOW_MAX_SPAN_HOURS", 8)
    workflow_max_span_seconds = int(workflow_max_span_hours * 3600)
    intent_token_limit = None if intent_no_cap else intent_max_tokens
    workflow_token_limit = None if workflow_no_cap else workflow_max_tokens


    # Apply filters
    filtered_df = sessions_df.copy()

    if app_filter != "All":
        filtered_df = filtered_df[filtered_df['primary_app'] == app_filter]

    if type_filter != "All":
        filtered_df = filtered_df[filtered_df['session_type'] == type_filter]

    # Apply sorting
    if sort_by == "Duration (Longest)":
        filtered_df = filtered_df.sort_values('duration_seconds', ascending=False)
    elif sort_by == "Duration (Shortest)":
        filtered_df = filtered_df.sort_values('duration_seconds', ascending=True)
    elif sort_by == "Actions (Most)":
        filtered_df = filtered_df.sort_values('action_count', ascending=False)
    elif sort_by == "Actions (Least)":
        filtered_df = filtered_df.sort_values('action_count', ascending=True)

    st.write(f"Showing {len(filtered_df)} sessions")

    if not filtered_df.empty:
        page_size = st.slider(
            "Sessions per page",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="explorer_page_size"
        )
        max_page = max(1, (len(filtered_df) + page_size - 1) // page_size)
        if "explorer_page" not in st.session_state:
            st.session_state.explorer_page = 1
        if st.session_state.explorer_page > max_page:
            st.session_state.explorer_page = max_page
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=max_page,
            value=st.session_state.explorer_page,
            step=1,
            key="explorer_page"
        )
        start = (page - 1) * page_size
        end = start + page_size
        display_df = filtered_df.iloc[start:end]
        st.caption(f"Displaying sessions {start + 1}-{min(end, len(filtered_df))} of {len(filtered_df)}")
    else:
        display_df = filtered_df

    # Display sessions
    intent_extractor = IntentExtractor()
    workflow_extractor = WorkflowExtractor(intent_extractor=intent_extractor)
    for idx, row in display_df.iterrows():
        with st.expander(f"Session #{row['session_number']}: {row['summary']}"):
            gap_info = row.get('gap_from_previous', None)
            if isinstance(gap_info, float) and pd.isna(gap_info):
                gap_info = None

            gap_text = "Gap unavailable"
            if isinstance(gap_info, dict):
                formatted_gap = gap_info.get('formatted')
                is_same_day = gap_info.get('is_same_day', True)
                if formatted_gap:
                    if is_same_day:
                        gap_text = f"{formatted_gap} after previous session"
                    else:
                        gap_text = f"Next day - {formatted_gap} after previous"
            elif row['session_number'] == 1:
                gap_text = "First session"

            st.caption(gap_text)

            col_info1, col_info2, col_info3 = st.columns(3)

            with col_info1:
                st.write(f"**App:** {row['primary_app']}")
                st.write(f"**Type:** {row['session_type']}")

            with col_info2:
                st.write(f"**Duration:** {row['duration_formatted']}")
                passive_count = row.get('passive_event_count', 0)
                # Handle NaN values from DataFrame
                if pd.isna(passive_count):
                    passive_count = 0
                else:
                    passive_count = int(passive_count)

                if passive_count > 0:
                    st.write(f"**Actions:** {row['action_count']} | **Passive:** {passive_count}")
                else:
                    st.write(f"**Actions:** {row['action_count']}")

            with col_info3:
                st.write(f"**Start:** {row['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**End:** {row['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")

            if row['primary_context']:
                st.write(f"**Context:** {row['primary_context']}")

            enhanced_summary = row.get('enhanced_summary', None)
            if isinstance(enhanced_summary, float) and pd.isna(enhanced_summary):
                enhanced_summary = None

            if not isinstance(enhanced_summary, dict):
                enhanced_summary = {}

            apps_breakdown = enhanced_summary.get('apps_breakdown', [])
            if apps_breakdown:
                apps_text = ", ".join(
                    f"{item.get('app', 'Unknown')} ({item.get('percentage', 0)}%)"
                    for item in apps_breakdown
                )
            else:
                apps_text = "N/A"

            contexts = enhanced_summary.get('unique_contexts', [])
            contexts_text = ", ".join(contexts) if contexts else "N/A"

            app_sequence = enhanced_summary.get('app_sequence', [])
            app_flow_text = " -> ".join(app_sequence) if app_sequence else "N/A"

            fields = enhanced_summary.get('field_types', [])
            fields_text = ", ".join(fields) if fields else "N/A"

            with st.expander("Session Summary", expanded=False):
                st.write(f"**Apps:** {apps_text}")
                st.write(f"**Contexts:** {contexts_text}")
                st.write(f"**App Flow:** {app_flow_text}")
                st.write(f"**Fields:** {fields_text}")

            session_id = row.get('session_id')
            session_number = row.get('session_number')
            if isinstance(session_number, float) and pd.isna(session_number):
                session_number = None
            if session_number is not None:
                session_number = int(session_number)

            session_dict = row.to_dict()
            intent_data = None
            if session_id:
                intent_data = intent_extractor.get_cached_intent(session_id)
            else:
                st.caption("Intent extraction unavailable (missing session_id).")

            if not intent_data and session_id:
                if st.button("Extract Intent", key=f"intent_extract_{session_id}"):
                    with st.spinner("Extracting intent..."):
                        intent = intent_extractor.extract_intent(
                            session_dict,
                            max_tokens=intent_token_limit,
                            include_actions=include_actions,
                            max_actions=max_actions
                        )
                        intent_extractor.save_intent(session_id, session_number, intent)
                    intent_data = intent_extractor.get_cached_intent(session_id)

            if intent_data and session_id:
                with st.expander("📋 Intent Analysis", expanded=False):
                    if st.button("Re-extract", key=f"intent_reextract_{session_id}"):
                        with st.spinner("Extracting intent..."):
                            intent = intent_extractor.extract_intent(
                                session_dict,
                                max_tokens=intent_token_limit,
                                include_actions=include_actions,
                                max_actions=max_actions
                            )
                            intent_extractor.save_intent(session_id, session_number, intent)
                        intent_data = intent_extractor.get_cached_intent(session_id)

                    intent_text = intent_data.get("intent", "N/A")
                    task_type = intent_data.get("task_type", "N/A")
                    entities = intent_data.get("key_entities", [])
                    if isinstance(entities, list):
                        entities_text = ", ".join(str(e) for e in entities if e) or "N/A"
                    elif isinstance(entities, str):
                        entities_text = entities
                    else:
                        entities_text = "N/A"
                    confidence = intent_data.get("confidence", "N/A")
                    if isinstance(confidence, str):
                        confidence_text = confidence.capitalize()
                    else:
                        confidence_text = str(confidence)

                    st.write(f"**Intent:** {intent_text}")
                    st.write(f"**Task Type:** {task_type}")
                    st.write(f"**Key Entities:** {entities_text}")
                    st.write(f"**Confidence:** {confidence_text}")

                    error = intent_data.get("error")
                    raw_response = intent_data.get("raw_response")
                    if error:
                        st.warning(f"Extraction error: {error}")
                    if raw_response:
                        with st.expander("Raw LLM Response", expanded=False):
                            st.code(raw_response, language="text")

            related_sessions = []
            if session_id:
                related_sessions = workflow_extractor.find_related_sessions(
                    session_id,
                    sessions,
                    workflow_gap_seconds,
                    workflow_max_sessions,
                    workflow_max_span_seconds
                )

            with st.expander("Workflow Analysis", expanded=False):
                if not session_id:
                    st.write("Workflow analysis unavailable (missing session_id).")
                elif len(related_sessions) < 2:
                    st.write("No related sessions found (isolated session).")
                else:
                    related_numbers = [
                        s.get("session_number") for s in related_sessions if s.get("session_number") is not None
                    ]
                    related_numbers_text = ", ".join(f"#{int(n)}" for n in related_numbers)
                    st.write(
                        f"Related Sessions: {related_numbers_text} (within {workflow_gap_hours}h, same day)"
                    )

                    workflow_id = workflow_extractor.build_workflow_id(
                        [s.get("session_id") for s in related_sessions]
                    )
                    workflow_data = workflow_extractor.get_cached_workflow(workflow_id)

                    if not workflow_data:
                        if st.button("Extract Workflow", key=f"workflow_extract_{session_id}"):
                            with st.spinner("Extracting workflow..."):
                                workflow_data = workflow_extractor.extract_workflow(
                                    session_id,
                                    sessions,
                                    workflow_gap_seconds,
                                    workflow_max_sessions,
                                    workflow_max_span_seconds,
                                    related_sessions=related_sessions,
                                    max_tokens=workflow_token_limit,
                                    include_actions=include_actions,
                                    max_actions=max_actions
                                )

                    if workflow_data:
                        if st.button("Re-extract", key=f"workflow_reextract_{session_id}"):
                            with st.spinner("Extracting workflow..."):
                                workflow_data = workflow_extractor.extract_workflow(
                                    session_id,
                                    sessions,
                                    workflow_gap_seconds,
                                    workflow_max_sessions,
                                    workflow_max_span_seconds,
                                    force=True,
                                    related_sessions=related_sessions,
                                    max_tokens=workflow_token_limit,
                                    include_actions=include_actions,
                                    max_actions=max_actions
                                )

                        workflow_intent = workflow_data.get("workflow_intent", "N/A")
                        workflow_type = workflow_data.get("workflow_type", "N/A")
                        key_entities = workflow_data.get("key_entities", [])
                        if isinstance(key_entities, list):
                            entities_text = ", ".join(str(e) for e in key_entities if e) or "N/A"
                        elif isinstance(key_entities, str):
                            entities_text = key_entities
                        else:
                            entities_text = "N/A"
                        confidence = workflow_data.get("confidence", "N/A")
                        if isinstance(confidence, str):
                            confidence_text = confidence.capitalize()
                        else:
                            confidence_text = str(confidence)

                        st.write(f"**Workflow Intent:** {workflow_intent}")
                        st.write(f"**Workflow Type:** {workflow_type}")

                        key_steps = workflow_data.get("key_steps")
                        key_steps = workflow_data.get("key_steps")
                        if isinstance(key_steps, list) and key_steps:
                            steps_text = "\n".join(
                                f"{idx + 1}. {step}" for idx, step in enumerate(key_steps)
                            )
                            st.write("**Key Steps:**")
                            st.markdown(steps_text)
                        elif key_steps:
                            st.write(f"**Key Steps:** {key_steps}")
                        else:
                            st.write("**Key Steps:** N/A")

                        session_roles = workflow_data.get("session_roles")
                        if isinstance(session_roles, list) and session_roles:
                            roles_text = "\n".join(str(role) for role in session_roles)
                            st.write("**Session Roles:**")
                            st.markdown(roles_text)
                        elif isinstance(session_roles, dict) and session_roles:
                            roles_text = "\n".join(
                                f"Session {k}: {v}" for k, v in session_roles.items()
                            )
                            st.write("**Session Roles:**")
                            st.markdown(roles_text)
                        elif session_roles:
                            st.write(f"**Session Roles:** {session_roles}")
                        else:
                            st.write("**Session Roles:** N/A")

                        st.write(f"**Key Entities:** {entities_text}")
                        st.write(f"**Confidence:** {confidence_text}")

                        error = workflow_data.get("error")
                        raw_response = workflow_data.get("raw_response")
                        if error:
                            st.warning(f"Extraction error: {error}")
                        if raw_response:
                            with st.expander("Raw LLM Response", expanded=False):
                                st.code(raw_response, language="text")

                        viz_data = build_workflow_viz_data(workflow_data, related_sessions)
                        timeline_data = viz_data.get("timeline_data", [])
                        app_breakdown = viz_data.get("app_breakdown", [])
                        flow_data = viz_data.get("flow_data", [])

                        if timeline_data or app_breakdown or flow_data:
                            st.subheader("Workflow Visualizations")
                            viz_tabs = st.tabs(["Timeline", "App Breakdown", "Session Flow"])

                            apps_for_colors = []
                            if timeline_data:
                                apps_for_colors.extend([d.get("app") for d in timeline_data if d.get("app")])
                            if app_breakdown:
                                apps_for_colors.extend([d.get("app") for d in app_breakdown if d.get("app")])
                            if flow_data:
                                apps_for_colors.extend([d.get("app") for d in flow_data if d.get("app")])
                            color_map = _build_color_map(list(dict.fromkeys(apps_for_colors)))

                            with viz_tabs[0]:
                                timeline_df = pd.DataFrame(timeline_data)
                                if not timeline_df.empty:
                                    timeline_df["start_time"] = pd.to_datetime(
                                        timeline_df["start_time"], errors="coerce", utc=True
                                    )
                                    timeline_df["end_time"] = pd.to_datetime(
                                        timeline_df["end_time"], errors="coerce", utc=True
                                    )
                                    timeline_df["intent"] = timeline_df["intent"].fillna("N/A")

                                    def _label_row(row):
                                        session_num = row.get("session_number")
                                        if isinstance(session_num, float) and pd.isna(session_num):
                                            session_num = None
                                        if session_num is None:
                                            return f"Session - {row.get('app', 'Unknown')}"
                                        return f"Session {int(session_num)} - {row.get('app', 'Unknown')}"

                                    timeline_df["label"] = timeline_df.apply(_label_row, axis=1)

                                    fig_timeline = px.timeline(
                                        timeline_df,
                                        x_start="start_time",
                                        x_end="end_time",
                                        y="label",
                                        color="app",
                                        color_discrete_map=color_map,
                                        custom_data=["session_number", "app", "duration", "intent"]
                                    )
                                    fig_timeline.update_yaxes(autorange="reversed")
                                    fig_timeline.update_layout(
                                        height=450,
                                        margin=dict(l=20, r=20, t=30, b=20)
                                    )
                                    fig_timeline.update_traces(
                                        hovertemplate=(
                                            "Session %{customdata[0]}<br>"
                                            "App: %{customdata[1]}<br>"
                                            "Duration: %{customdata[2]}<br>"
                                            "Intent: %{customdata[3]}<extra></extra>"
                                        )
                                    )
                                    chart_key_base = session_id or session_number or idx
                                    st.plotly_chart(
                                        fig_timeline,
                                        width="stretch",
                                        key=f"workflow_timeline_{chart_key_base}"
                                    )
                                else:
                                    st.write("No timeline data available.")

                            with viz_tabs[1]:
                                app_df = pd.DataFrame(app_breakdown)
                                if not app_df.empty:
                                    fig_pie = px.pie(
                                        app_df,
                                        values="action_count",
                                        names="app",
                                        hole=0.4,
                                        color="app",
                                        color_discrete_map=color_map
                                    )
                                    fig_pie.update_traces(
                                        textposition="inside",
                                        textinfo="percent+label",
                                        hovertemplate=(
                                            "App: %{label}<br>"
                                            "Actions: %{value}<br>"
                                            "Percent: %{percent}<extra></extra>"
                                        )
                                    )
                                    fig_pie.update_layout(
                                        height=450,
                                        margin=dict(l=20, r=20, t=30, b=20)
                                    )
                                    chart_key_base = session_id or session_number or idx
                                    st.plotly_chart(
                                        fig_pie,
                                        width="stretch",
                                        key=f"workflow_app_breakdown_{chart_key_base}"
                                    )
                                else:
                                    st.write("No app breakdown available.")

                            with viz_tabs[2]:
                                flow_df = pd.DataFrame(flow_data)
                                if not flow_df.empty:
                                    x_positions = list(range(len(flow_df)))
                                    y_positions = [0] * len(flow_df)
                                    labels = []
                                    customdata = []
                                    colors = []

                                    for _, flow_row in flow_df.iterrows():
                                        session_num = flow_row.get("session_number")
                                        if isinstance(session_num, float) and pd.isna(session_num):
                                            session_num = None
                                        session_label = int(session_num) if session_num is not None else "N/A"
                                        app = flow_row.get("app", "Unknown")
                                        duration = flow_row.get("duration", "N/A")
                                        labels.append(f"Session {session_label}\n{app}\n{duration}")
                                        intent_text = flow_row.get("intent") or "N/A"
                                        role_text = flow_row.get("session_role") or "N/A"
                                        customdata.append([session_label, app, duration, intent_text, role_text])
                                        colors.append(_get_app_color(app))

                                    fig_flow = go.Figure()
                                    if len(x_positions) > 1:
                                        fig_flow.add_trace(
                                            go.Scatter(
                                                x=x_positions,
                                                y=y_positions,
                                                mode="lines",
                                                line=dict(color="#cccccc", width=2),
                                                hoverinfo="skip",
                                                showlegend=False
                                            )
                                        )

                                    fig_flow.add_trace(
                                        go.Scatter(
                                            x=x_positions,
                                            y=y_positions,
                                            mode="markers+text",
                                            marker=dict(size=22, color=colors, symbol="square"),
                                            text=labels,
                                            textposition="bottom center",
                                            customdata=customdata,
                                            hovertemplate=(
                                                "Session %{customdata[0]}<br>"
                                                "App: %{customdata[1]}<br>"
                                                "Duration: %{customdata[2]}<br>"
                                                "Intent: %{customdata[3]}<br>"
                                                "Role: %{customdata[4]}<extra></extra>"
                                            ),
                                            showlegend=False
                                        )
                                    )

                                    for idx in range(1, len(flow_df)):
                                        gap_text = flow_df.iloc[idx].get("gap_from_previous")
                                        if gap_text:
                                            fig_flow.add_annotation(
                                                x=(x_positions[idx - 1] + x_positions[idx]) / 2,
                                                y=0.18,
                                                text=f"Gap: {gap_text}",
                                                showarrow=False,
                                                font=dict(size=12, color="#555555")
                                            )

                                    fig_flow.update_layout(
                                        height=450,
                                        margin=dict(l=20, r=20, t=30, b=20),
                                        xaxis=dict(visible=False),
                                        yaxis=dict(visible=False)
                                    )
                                    chart_key_base = session_id or session_number or idx
                                    st.plotly_chart(
                                        fig_flow,
                                        width="stretch",
                                        key=f"workflow_flow_{chart_key_base}"
                                    )
                                else:
                                    st.write("No flow data available.")
            if row['actions']:
                st.write("**Actions:**")
                actions_df = pd.DataFrame(row['actions'])
                if 'source_id' not in actions_df.columns:
                    actions_df['source_id'] = None
                actions_df = actions_df[['timestamp', 'app', 'source_id', 'field', 'value']]
                actions_df.columns = ['Time', 'App', 'Source ID', 'Field', 'Value']
                st.dataframe(actions_df, width='stretch', hide_index=True)

            # Display passive events if they exist
            passive_events = row.get('passive_events')
            if passive_events is not None and isinstance(passive_events, list) and len(passive_events) > 0:
                passive_count = len(passive_events)
                with st.expander(f"Passive Events ({passive_count}) - Background/Heartbeat events", expanded=False):
                    passive_df = pd.DataFrame(passive_events)
                    if 'source_id' not in passive_df.columns:
                        passive_df['source_id'] = None
                    # Select and reorder columns
                    passive_df = passive_df[['timestamp', 'app', 'source_id', 'type', 'context']]
                    passive_df.columns = ['Time', 'App', 'Source ID', 'Type', 'Context']
                    st.dataframe(passive_df, width='stretch', hide_index=True)
                    # Show attachment info if present
                    nearest_count = sum(1 for e in passive_events if e.get('attachment') == 'nearest')
                    if nearest_count > 0:
                        st.caption(f"ℹ️ {nearest_count} passive event(s) attached to nearest session (outside time range)")

# ============================================
# TAB 3: DATA QUALITY
# ============================================
with tab_quality:
    st.title("Data Quality Report")

    # Get quality report
    if os.path.exists(CSV_FILE):
        with st.spinner("Analyzing data quality..."):
            quality_report = get_quality_report(CSV_FILE, sessions)

        # Quality Score
        score = quality_report['quality_score']
        score_color = "green" if score >= 80 else "orange" if score >= 60 else "red"

        col_score, col_info = st.columns([1, 3])
        with col_score:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;">
                <h1 style="color: white; margin: 0;">{score}%</h1>
                <p style="color: white; margin: 0;">Quality Score</p>
            </div>
            """, unsafe_allow_html=True)

        with col_info:
            st.subheader("Recommendations")
            for rec in quality_report['recommendations']:
                st.write(f"- {rec}")

        st.markdown("---")

        # Raw Data Analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Raw Data Coverage")
            raw = quality_report['raw_data']

            # Event type pie chart
            processed_types = set(ACTIVE_EVENT_TYPES + PASSIVE_EVENT_TYPES)
            excluded_types = set(EXCLUDE_EVENT_TYPES)
            event_data = pd.DataFrame([
                {
                    "Type": etype,
                    "Count": count,
                    "Status": "Processed" if etype in processed_types and etype not in excluded_types else "Dropped"
                }
                for etype, count in raw['event_type_distribution'].items()
            ])

            fig_events = px.pie(
                event_data, values='Count', names='Type', color='Status',
                color_discrete_map={'Processed': '#28a745', 'Dropped': '#dc3545'},
                hole=0.4
            )
            fig_events.update_traces(textposition='inside', textinfo='percent+label')
            fig_events.update_layout(height=300)
            st.plotly_chart(fig_events, width='stretch')

            st.metric("Total Events", f"{raw['total_events']:,}")
            st.metric("Processed", f"{raw['processed_events']:,} ({raw['processed_percentage']}%)")
            st.metric("Dropped", f"{raw['dropped_events']:,} ({raw['dropped_percentage']}%)")

        with col2:
            st.subheader("Parsing Success Rates")
            parsing = quality_report['parsing']

            # Success rate bars
            rates = pd.DataFrame([
                {"Field": "App Extraction", "Rate": parsing.get('app_success_rate', 0)},
                {"Field": "Context Extraction", "Rate": parsing.get('context_success_rate', 0)},
                {"Field": "Value Extraction", "Rate": parsing.get('value_success_rate', 0)},
            ])

            fig_rates = px.bar(
                rates, x='Field', y='Rate', color='Rate',
                color_continuous_scale=['red', 'orange', 'green'],
                range_color=[0, 100]
            )
            fig_rates.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_rates, width='stretch')

            st.metric("Unique Apps Found", parsing['unique_apps_count'])
            st.metric("Total Parsed", f"{parsing['total_parsed']:,}")

        st.markdown("---")

        # Session Analysis
        st.subheader("Session Classification")
        session_quality = quality_report['sessions']

        col3, col4 = st.columns(2)

        with col3:
            # Session type distribution with "Other" highlighted
            type_df = pd.DataFrame([
                {"Type": t, "Count": c, "Category": "Classified" if t != "Other" else "Unclassified"}
                for t, c in session_quality['session_type_distribution'].items()
            ])

            fig_types = px.bar(
                type_df, x='Type', y='Count', color='Category',
                color_discrete_map={'Classified': '#28a745', 'Unclassified': '#ffc107'}
            )
            fig_types.update_layout(height=300)
            st.plotly_chart(fig_types, width='stretch')

            st.metric("'Other' Sessions", f"{session_quality['other_percentage']}%",
                     delta=f"{session_quality['other_percentage'] - 25:.1f}% from target 25%" if session_quality['other_percentage'] > 25 else "Good!")

        with col4:
            # Duration and action stats
            st.write("**Duration Statistics:**")
            duration = session_quality['duration_stats']
            st.write(f"- Min: {duration['min']}s")
            st.write(f"- Max: {duration['max']}s")
            st.write(f"- Avg: {duration['avg']}s")
            st.write(f"- Zero duration: {duration['zero_duration_count']}")

            st.write("**Action Statistics:**")
            actions = session_quality['action_stats']
            st.write(f"- Min: {actions['min']}")
            st.write(f"- Max: {actions['max']}")
            st.write(f"- Avg: {actions['avg']}")
            st.write(f"- Total: {actions['total']:,}")

        # Issues and suspicious apps
        if session_quality.get('issues'):
            st.subheader("Issues Detected")
            for issue in session_quality['issues']:
                st.warning(issue)

        if session_quality.get('suspicious_apps'):
            st.subheader("Potentially Misclassified Apps")
            for app in session_quality['suspicious_apps']:
                st.code(app[:80])

        # Unique apps list
        with st.expander("All Detected Apps"):
            apps_list = parsing.get('unique_apps', [])
            cols = st.columns(3)
            for i, app in enumerate(sorted(apps_list)):
                cols[i % 3].write(f"- {app}")

    else:
        st.error(f"CSV file not found: {CSV_FILE}")
        st.info("Cannot generate quality report without raw data file.")

# ============================================
# TAB 4: PARAMETER TUNING
# ============================================
with tab_tuning:
    st.title("Parameter Tuning")
    st.write("Experiment with different sessionization parameters to see how they affect the results.")

    # Current parameters display
    st.subheader("Current Parameters")
    col_current1, col_current2, col_current3 = st.columns(3)
    with col_current1:
        time_gap_value = metadata['parameters'].get('time_gap_threshold', 900)
        st.info(f"**Time Gap:** {time_gap_value}s ({time_gap_value/60:.0f} min)")
    with col_current2:
        st.info(f"**Min Actions:** {metadata['parameters']['min_actions']}")
    with col_current3:
        st.info(f"**Sessions:** {stats['total_sessions']}")

    st.markdown("---")

    # Parameter sliders
    st.subheader("Adjust Parameters")

    col_param1, col_param2 = st.columns(2)

    with col_param1:
        new_time_gap = st.slider(
            "Time Gap Threshold (seconds)",
            min_value=60,
            max_value=1800,  # Increased to 30 minutes
            value=metadata['parameters'].get('time_gap_threshold', 900),
            step=30,
            help="Idle time gap that triggers a new session. Default: 900s (15 min)"
        )

    with col_param2:
        new_min_actions = st.slider(
            "Minimum Actions per Session",
            min_value=1,
            max_value=10,
            value=metadata['parameters']['min_actions'],
            step=1,
            help="Sessions with fewer actions are discarded. Default: 2"
        )

    # Check if parameters changed
    params_changed = (
        new_time_gap != metadata['parameters']['time_gap_threshold'] or
        new_min_actions != metadata['parameters']['min_actions']
    )

    if params_changed:
        st.warning("Parameters have changed from the current session data.")

        if st.button("Re-run Sessionizer with New Parameters", type="primary"):
            if os.path.exists(CSV_FILE):
                with st.spinner("Running sessionizer with new parameters..."):
                    try:
                        new_sessions, new_stats = run_sessionizer_with_params(
                            CSV_FILE, new_time_gap, new_min_actions
                        )

                        # Show comparison
                        st.success("Sessionization complete!")

                        st.subheader("Results Comparison")

                        # Comparison metrics
                        col_old, col_new, col_diff = st.columns(3)

                        with col_old:
                            st.write("**Current**")
                            st.metric("Sessions", stats['total_sessions'])
                            st.metric("Total Actions", stats['total_actions'])
                            st.metric("Avg Duration", stats['average_duration_formatted'])

                        with col_new:
                            st.write("**New Parameters**")
                            st.metric("Sessions", new_stats['total_sessions'])
                            st.metric("Total Actions", new_stats['total_actions'])
                            st.metric("Avg Duration", new_stats['average_duration_formatted'])

                        with col_diff:
                            st.write("**Difference**")
                            session_diff = new_stats['total_sessions'] - stats['total_sessions']
                            action_diff = new_stats['total_actions'] - stats['total_actions']
                            st.metric("Sessions", f"{session_diff:+d}")
                            st.metric("Total Actions", f"{action_diff:+d}")

                        # App distribution comparison
                        st.subheader("App Distribution Comparison")
                        col_chart1, col_chart2 = st.columns(2)

                        with col_chart1:
                            st.write("**Current**")
                            current_app_df = pd.DataFrame([
                                {"App": app, "Sessions": count}
                                for app, count in list(stats['app_distribution'].items())[:8]
                            ])
                            fig1 = px.bar(current_app_df, x='App', y='Sessions', color='App')
                            fig1.update_layout(showlegend=False, height=300)
                            st.plotly_chart(fig1, width='stretch')

                        with col_chart2:
                            st.write("**New Parameters**")
                            new_app_df = pd.DataFrame([
                                {"App": app, "Sessions": count}
                                for app, count in list(new_stats['app_distribution'].items())[:8]
                            ])
                            fig2 = px.bar(new_app_df, x='App', y='Sessions', color='App')
                            fig2.update_layout(showlegend=False, height=300)
                            st.plotly_chart(fig2, width='stretch')

                        # Session type comparison
                        st.subheader("Session Type Comparison")
                        col_type1, col_type2 = st.columns(2)

                        with col_type1:
                            st.write("**Current**")
                            for stype, count in stats['session_type_distribution'].items():
                                pct = count / stats['total_sessions'] * 100
                                st.write(f"- {stype}: {count} ({pct:.1f}%)")

                        with col_type2:
                            st.write("**New Parameters**")
                            for stype, count in new_stats['session_type_distribution'].items():
                                pct = count / new_stats['total_sessions'] * 100
                                st.write(f"- {stype}: {count} ({pct:.1f}%)")

                        # Option to save
                        st.markdown("---")
                        st.info("To save these results permanently, run the sessionizer from command line with the new parameters:")
                        st.code(f"python run_sessionizer.py --time-gap {new_time_gap} --min-actions {new_min_actions}")

                    except Exception as e:
                        st.error(f"Error running sessionizer: {e}")
            else:
                st.error(f"CSV file not found: {CSV_FILE}")

    else:
        st.info("Adjust the sliders above to experiment with different parameters.")

    # Parameter explanation
    st.markdown("---")
    st.subheader("Parameter Guide")

    with st.expander("Understanding the Parameters"):
        st.markdown("""
        ### Time Gap Threshold

        **What it does:** Determines when to start a new session based on idle time only.

        Sessions are created purely based on idle time between events:
        - Any gap **greater than** the threshold starts a new session
        - Any gap **less than** the threshold continues the same session
        - App changes and context changes are **ignored**

        | Value | Effect |
        |-------|--------|
        | 300s (5 min) | More sessions - breaks after 5 minutes of inactivity |
        | 900s (15 min) | Balanced - default threshold for typical work sessions |
        | 1800s (30 min) | Fewer sessions - only breaks after extended idle time |

        **Recommendation:** Start with 900s (15 min default), adjust based on your workflow patterns.

        **Note:** The simplified logic means sessions may contain multiple apps and contexts as long as idle time stays below the threshold.

        ---

        ### Minimum Actions per Session

        **What it does:** Filters out sessions with too few actions (noise reduction).

        | Value | Effect |
        |-------|--------|
        | 1 | Keep all sessions including single-click ones |
        | 2 | Remove accidental clicks, keep intentional work (recommended) |
        | 5+ | Only keep substantial work sessions |

        **Recommendation:** Use 2 (default) for most cases.
        """)

# ============================================
# TAB 5: ISOLATED ACTIONS
# ============================================
with tab_isolated:
    st.title("Isolated Actions Review")
    st.write("These single-action events were not grouped into sessions. Review and decide which to keep.")

    # Initialize session state for decisions if not exists
    if 'event_decisions' not in st.session_state:
        st.session_state.event_decisions = {}

    # Check if we have isolated events
    if not isolated_events:
        st.info("No isolated events found. All events were grouped into sessions.")
        if promoted_actions:
            st.success(f"You have {len(promoted_actions)} promoted actions.")
    else:
        # Summary metrics
        col_iso1, col_iso2, col_iso3, col_iso4 = st.columns(4)

        # Count by status
        decisions = _get_decisions_store()
        kept_ids = set(decisions.get('kept_ids', []))
        dropped_ids = set(decisions.get('dropped_ids', []))

        pending_count = sum(1 for e in isolated_events if e['id'] not in kept_ids and e['id'] not in dropped_ids)
        kept_count = sum(1 for e in isolated_events if e['id'] in kept_ids)
        dropped_count = sum(1 for e in isolated_events if e['id'] in dropped_ids)

        with col_iso1:
            st.metric("Total Isolated", len(isolated_events))
        with col_iso2:
            st.metric("Pending Review", pending_count)
        with col_iso3:
            st.metric("Marked Keep", kept_count)
        with col_iso4:
            st.metric("Marked Drop", dropped_count)

        st.markdown("---")

        # Category filter
        categories = ['All'] + sorted(list(set(e.get('category', 'Other') for e in isolated_events)))

        col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 2])

        with col_filter1:
            selected_category = st.selectbox(
                "Filter by Category",
                options=categories,
                key="isolated_category_filter"
            )

        with col_filter2:
            status_filter = st.selectbox(
                "Filter by Status",
                options=["All", "Pending", "Keep", "Drop"],
                key="isolated_status_filter"
            )

        with col_filter3:
            sort_option = st.selectbox(
                "Sort by",
                options=["Timestamp (Newest)", "Timestamp (Oldest)", "Category", "App"],
                key="isolated_sort"
            )

        # Filter events
        filtered_events = isolated_events.copy()

        if selected_category != 'All':
            filtered_events = [e for e in filtered_events if e.get('category') == selected_category]

        if status_filter != 'All':
            if status_filter == 'Pending':
                filtered_events = [e for e in filtered_events if e['id'] not in kept_ids and e['id'] not in dropped_ids]
            elif status_filter == 'Keep':
                filtered_events = [e for e in filtered_events if e['id'] in kept_ids]
            elif status_filter == 'Drop':
                filtered_events = [e for e in filtered_events if e['id'] in dropped_ids]

        # Sort events
        if sort_option == "Timestamp (Newest)":
            filtered_events = sorted(filtered_events, key=lambda x: x.get('timestamp', ''), reverse=True)
        elif sort_option == "Timestamp (Oldest)":
            filtered_events = sorted(filtered_events, key=lambda x: x.get('timestamp', ''))
        elif sort_option == "Category":
            filtered_events = sorted(filtered_events, key=lambda x: x.get('category', 'Other'))
        elif sort_option == "App":
            filtered_events = sorted(filtered_events, key=lambda x: x.get('app', '') or '')

        st.write(f"Showing {len(filtered_events)} of {len(isolated_events)} isolated events")

        # Bulk actions
        col_bulk1, col_bulk2, col_bulk3, col_bulk4 = st.columns(4)

        with col_bulk1:
            if st.button("Mark Filtered as Keep", key="bulk_keep"):
                event_ids = [e['id'] for e in filtered_events]
                _bulk_mark_keep(event_ids)
                st.success(f"Marked {len(event_ids)} events to keep")
                st.rerun()

        with col_bulk2:
            if st.button("Mark Filtered as Drop", key="bulk_drop"):
                event_ids = [e['id'] for e in filtered_events]
                _bulk_mark_drop(event_ids)
                st.success(f"Marked {len(event_ids)} events to drop")
                st.rerun()

        with col_bulk3:
            if st.button("Apply All Decisions", type="primary", key="apply_decisions"):
                result = _apply_decisions()
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success(f"Applied: {result['promoted']} promoted, {result['dropped']} dropped, {result['remaining_pending']} pending")
                    st.rerun()

        with col_bulk4:
            if st.button("Reset All Decisions", key="reset_decisions"):
                _clear_all_decisions()
                st.success("All decisions cleared")
                st.rerun()

        st.markdown("---")

        if filtered_events:
            page_size = st.slider(
                "Isolated events per page",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                key="isolated_page_size"
            )
            max_page = max(1, (len(filtered_events) + page_size - 1) // page_size)
            if "isolated_page" not in st.session_state:
                st.session_state.isolated_page = 1
            if st.session_state.isolated_page > max_page:
                st.session_state.isolated_page = max_page
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=max_page,
                value=st.session_state.isolated_page,
                step=1,
                key="isolated_page"
            )
            start = (page - 1) * page_size
            end = start + page_size
            display_events = filtered_events[start:end]
            st.caption(f"Displaying events {start + 1}-{min(end, len(filtered_events))} of {len(filtered_events)}")
        else:
            display_events = filtered_events

        # Display events
        for i, event in enumerate(display_events):
            event_id = event.get('id', f'event_{i}')

            # Determine current status
            if event_id in kept_ids:
                status_indicator = "[KEEP]"
                status_color = "green"
            elif event_id in dropped_ids:
                status_indicator = "[DROP]"
                status_color = "red"
            else:
                status_indicator = "[PENDING]"
                status_color = "orange"

            # Create expander title
            action_preview = event.get('action', '')[:60] if event.get('action') else 'N/A'
            category = event.get('category', 'Other')
            expander_title = f"{status_indicator} {category}: {action_preview}..."

            with st.expander(expander_title):
                col_info, col_action = st.columns([3, 1])

                with col_info:
                    st.write(f"**ID:** {event_id}")
                    if event.get('source_id') is not None:
                        st.write(f"**Source ID:** {event.get('source_id')}")
                    st.write(f"**Timestamp:** {event.get('timestamp', 'N/A')}")
                    st.write(f"**App:** {event.get('app', 'N/A')}")
                    st.write(f"**Category:** {category}")
                    st.write(f"**Field:** {event.get('field', 'N/A')}")

                    value = event.get('value')
                    if value:
                        st.write(f"**Value:** {value[:100]}{'...' if len(value) > 100 else ''}")
                    else:
                        st.write("**Value:** N/A")

                    if event.get('context'):
                        st.write(f"**Context:** {event.get('context')[:100]}")

                    st.write(f"**Action:** {event.get('action', 'N/A')}")

                with col_action:
                    st.write("**Decision:**")

                    if st.button("Keep", key=f"keep_{event_id}", type="primary" if event_id not in kept_ids else "secondary"):
                        _mark_event_keep(event_id)
                        st.rerun()

                    if st.button("Drop", key=f"drop_{event_id}", type="secondary"):
                        _mark_event_drop(event_id)
                        st.rerun()

                    if event_id in kept_ids or event_id in dropped_ids:
                        if st.button("Reset to Pending", key=f"pending_{event_id}"):
                            _mark_event_pending(event_id)
                            st.rerun()

        # Category summary at the bottom
        st.markdown("---")
        st.subheader("Category Summary")

        from collections import Counter
        category_counts = Counter(e.get('category', 'Other') for e in isolated_events)

        cat_df = pd.DataFrame([
            {"Category": cat, "Count": count}
            for cat, count in category_counts.most_common()
        ])

        if not cat_df.empty:
            fig_cat = px.bar(
                cat_df, x='Category', y='Count', color='Category',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_cat.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_cat, width='stretch')

    # Show promoted actions section
    if promoted_actions:
        st.markdown("---")
        st.subheader("Promoted Actions")
        st.write("These are single-action events you have decided to keep.")

        page_size = st.slider(
            "Promoted actions per page",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="promoted_page_size"
        )
        max_page = max(1, (len(promoted_actions) + page_size - 1) // page_size)
        if "promoted_page" not in st.session_state:
            st.session_state.promoted_page = 1
        if st.session_state.promoted_page > max_page:
            st.session_state.promoted_page = max_page
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=max_page,
            value=st.session_state.promoted_page,
            step=1,
            key="promoted_page"
        )
        start = (page - 1) * page_size
        end = start + page_size
        display_promoted = promoted_actions[start:end]
        st.caption(f"Displaying promoted actions {start + 1}-{min(end, len(promoted_actions))} of {len(promoted_actions)}")

        for i, action in enumerate(display_promoted):
            with st.expander(f"[PROMOTED] {action.get('category', 'Other')}: {action.get('action', '')[:60]}..."):
                st.write(f"**Timestamp:** {action.get('timestamp', 'N/A')}")
                st.write(f"**App:** {action.get('app', 'N/A')}")
                st.write(f"**Field:** {action.get('field', 'N/A')}")
                st.write(f"**Value:** {action.get('value', 'N/A')}")
                st.write(f"**Promoted at:** {action.get('promoted_at', 'N/A')}")

# ============================================
# TAB 6: TIMELINE VIEW
# ============================================
with tab_timeline:
    st.title("Timeline View")
    st.write("Chronological view of all work activity - sessions and isolated events interleaved by timestamp.")

    # Create unified timeline
    timeline = create_timeline_events(sessions, isolated_events, promoted_actions)

    if not timeline:
        st.info("No events to display.")
    else:
        # Summary metrics
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)

        session_count = sum(1 for e in timeline if e['type'] == 'SESSION')
        isolated_count = sum(1 for e in timeline if e['type'] == 'ISOLATED')
        promoted_count = sum(1 for e in timeline if e['type'] == 'PROMOTED')

        with col_t1:
            st.metric("Total Events", len(timeline))
        with col_t2:
            st.metric("Sessions", session_count)
        with col_t3:
            st.metric("Isolated", isolated_count)
        with col_t4:
            st.metric("Promoted", promoted_count)

        st.markdown("---")

        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            type_filter = st.multiselect(
                "Event Types",
                options=["SESSION", "ISOLATED", "PROMOTED"],
                default=["SESSION", "ISOLATED", "PROMOTED"],
                key="timeline_type_filter"
            )

        with col_f2:
            # Date filter - extract unique dates from timestamps
            dates = sorted(set(e['timestamp'][:10] for e in timeline))
            selected_date = st.selectbox(
                "Date",
                options=["All"] + dates,
                key="timeline_date_filter"
            )

        with col_f3:
            # App filter
            apps = sorted(set(e['app'] for e in timeline if e['app']))
            app_filter = st.selectbox(
                "App",
                options=["All"] + apps,
                key="timeline_app_filter"
            )

        # Apply filters
        filtered_timeline = timeline.copy()

        # Filter by type
        if type_filter:
            filtered_timeline = [e for e in filtered_timeline if e['type'] in type_filter]

        # Filter by date
        if selected_date != "All":
            filtered_timeline = [e for e in filtered_timeline if e['timestamp'].startswith(selected_date)]

        # Filter by app
        if app_filter != "All":
            filtered_timeline = [e for e in filtered_timeline if e['app'] == app_filter]

        st.write(f"Showing {len(filtered_timeline)} of {len(timeline)} events")

        if filtered_timeline:
            page_size = st.slider(
                "Timeline events per page",
                min_value=25,
                max_value=500,
                value=100,
                step=25,
                key="timeline_page_size"
            )
            max_page = max(1, (len(filtered_timeline) + page_size - 1) // page_size)
            if "timeline_page" not in st.session_state:
                st.session_state.timeline_page = 1
            if st.session_state.timeline_page > max_page:
                st.session_state.timeline_page = max_page
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=max_page,
                value=st.session_state.timeline_page,
                step=1,
                key="timeline_page"
            )
            start = (page - 1) * page_size
            end = start + page_size
            display_timeline = filtered_timeline[start:end]
            st.caption(f"Displaying events {start + 1}-{min(end, len(filtered_timeline))} of {len(filtered_timeline)}")
        else:
            display_timeline = filtered_timeline

        st.markdown("---")

        # Display timeline events
        for event in display_timeline:
            # Determine icon and color based on type
            if event['type'] == 'SESSION':
                icon = "[S]"
                badge_color = "blue"
            elif event['type'] == 'PROMOTED':
                icon = "[P]"
                badge_color = "green"
            else:  # ISOLATED
                icon = "[I]"
                badge_color = "orange"

            # Extract time portion (HH:MM:SS) from timestamp
            time_str = event['timestamp'][11:19] if len(event['timestamp']) > 19 else event['timestamp']

            # Build compact title
            app_display = event['app'] if event['app'] else 'Unknown'
            summary_display = event['summary'][:50] if event['summary'] else 'N/A'

            if event['type'] == 'SESSION':
                compact_title = f"{time_str} {icon} {app_display}: {summary_display} ({event['action_count']} actions, {event['duration']})"
            else:
                compact_title = f"{time_str} {icon} {app_display}: {summary_display}"

            # Expandable expander for each event
            with st.expander(compact_title):
                col_detail, col_actions = st.columns([3, 1])

                with col_detail:
                    st.write(f"**ID:** {event['id']}")
                    st.write(f"**Type:** {event['type']}")
                    st.write(f"**Timestamp:** {event['timestamp']}")
                    st.write(f"**App:** {event['app']}")

                    if event['type'] == 'SESSION':
                        st.write(f"**Duration:** {event['duration']}")
                        st.write(f"**Actions:** {event['action_count']}")
                        st.write(f"**Session Type:** {event['session_type']}")

                        # Show individual actions within the session
                        if event['details'].get('actions'):
                            st.write("**Action Details:**")
                            for action in event['details']['actions']:
                                action_time = action.get('timestamp', '')
                                action_time_short = action_time[11:19] if len(action_time) > 19 else action_time
                                field = action.get('field', 'N/A')
                                value = action.get('value', 'N/A')
                                if value and len(value) > 50:
                                    value = value[:50] + '...'
                                st.write(f"  - {action_time_short}: {field} = {value}")
                    else:
                        # Isolated or Promoted event details
                        st.write(f"**Category:** {event['details'].get('category', 'Other')}")
                        st.write(f"**Field:** {event['details'].get('field', 'N/A')}")

                        value = event['details'].get('value')
                        if value:
                            display_value = value[:100] + '...' if len(value) > 100 else value
                            st.write(f"**Value:** {display_value}")
                        else:
                            st.write("**Value:** N/A")

                        if event['details'].get('context'):
                            context = event['details']['context']
                            display_context = context[:100] + '...' if len(context) > 100 else context
                            st.write(f"**Context:** {display_context}")

                        if event['details'].get('action'):
                            st.write(f"**Action:** {event['details']['action']}")

                with col_actions:
                    st.write(f"**Type:** {event['type']}")
                    if event['type'] == 'SESSION':
                        st.write(f"{event['action_count']} actions")
                        st.write(f"{event['duration']}")

        # Timeline visualization
        st.markdown("---")
        st.subheader("Timeline Visualization")

        # Prepare data for timeline chart
        if display_timeline:
            timeline_chart_data = []
            for i, event in enumerate(display_timeline):
                timeline_chart_data.append({
                    'Index': i,
                    'Time': event['timestamp'][:19],
                    'Type': event['type'],
                    'App': event['app'],
                    'Summary': event['summary'][:30] if event['summary'] else 'N/A'
                })

            timeline_df = pd.DataFrame(timeline_chart_data)
            timeline_df['Time'] = pd.to_datetime(timeline_df['Time'])

            # Color map for event types
            color_map = {
                'SESSION': '#3366cc',
                'ISOLATED': '#ff9900',
                'PROMOTED': '#109618'
            }

            fig_timeline = px.scatter(
                timeline_df,
                x='Time',
                y='Type',
                color='Type',
                hover_data=['App', 'Summary'],
                color_discrete_map=color_map,
                title='Events over Time'
            )

            fig_timeline.update_traces(marker=dict(size=10))
            fig_timeline.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig_timeline, width='stretch')

            # Events per hour chart
            timeline_df['Hour'] = timeline_df['Time'].dt.hour
            hourly_counts = timeline_df.groupby(['Hour', 'Type']).size().reset_index(name='Count')

            fig_hourly = px.bar(
                hourly_counts,
                x='Hour',
                y='Count',
                color='Type',
                color_discrete_map=color_map,
                title='Events by Hour of Day',
                barmode='stack'
            )
            fig_hourly.update_layout(height=300)
            st.plotly_chart(fig_hourly, width='stretch')

# ============================================
# TAB 7: LLM SESSIONIZATION
# ============================================
with tab_llm:
    st.title("LLM-Powered Sessionization")
    st.markdown("Use a local LLM or OpenRouter (online) to group events into sessions with reasoning and intent detection.")

    # Initialize session state for LLM
    if 'llm_results' not in st.session_state:
        st.session_state.llm_results = None
    if 'custom_prompt' not in st.session_state:
        st.session_state.custom_prompt = LLM_PROMPT_TEMPLATE
    if 'llm_streaming_text' not in st.session_state:
        st.session_state.llm_streaming_text = ""
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = "Local (LM Studio)"
    if 'openrouter_api_key' not in st.session_state:
        st.session_state.openrouter_api_key = OPENROUTER_API_KEY
    if 'openrouter_model' not in st.session_state:
        st.session_state.openrouter_model = OPENROUTER_MODEL
    if 'openrouter_reasoning' not in st.session_state:
        st.session_state.openrouter_reasoning = OPENROUTER_REASONING_DEFAULT
    if 'openrouter_model_choice' not in st.session_state:
        if st.session_state.openrouter_model in OPENROUTER_FREE_MODELS:
            st.session_state.openrouter_model_choice = st.session_state.openrouter_model
        else:
            st.session_state.openrouter_model_choice = "Custom model id..."

    # --- Configuration Section ---
    with st.expander("LLM Configuration", expanded=True):
        llm_provider = st.radio(
            "Provider",
            ["Local (LM Studio)", "OpenRouter (Online)"],
            horizontal=True,
            key="llm_provider"
        )

        col_cfg1, col_cfg2 = st.columns(2)

        with col_cfg1:
            if llm_provider == "OpenRouter (Online)":
                openrouter_api_key = st.text_input(
                    "OpenRouter API Key",
                    type="password",
                    key="openrouter_api_key"
                )
                st.caption(f"Endpoint: {OPENROUTER_API_URL}")
                openrouter_reasoning = st.checkbox(
                    "Enable reasoning (if supported by the model)",
                    key="openrouter_reasoning"
                )

                model_options = OPENROUTER_FREE_MODELS + ["Custom model id..."]
                openrouter_model_choice = st.selectbox(
                    "Model",
                    options=model_options,
                    key="openrouter_model_choice"
                )
                if openrouter_model_choice == "Custom model id...":
                    openrouter_model = st.text_input(
                        "Custom model id",
                        value=st.session_state.openrouter_model,
                        key="openrouter_model_custom"
                    )
                else:
                    openrouter_model = openrouter_model_choice
                st.session_state.openrouter_model = openrouter_model
                st.caption("Tip: OpenRouter free models include :free in the model id.")
            else:
                llm_api_url = st.text_input("API URL", value=LLM_API_URL, key="llm_api_url")
                llm_model = st.text_input("Model", value=LLM_MODEL, key="llm_model")

        with col_cfg2:
            llm_batch_size = st.slider(
                "Batch Size",
                min_value=LLM_BATCH_SIZE_MIN,
                max_value=LLM_BATCH_SIZE_MAX,
                value=LLM_BATCH_SIZE,
                help="Number of events to send per LLM request"
            )
            llm_timeout = st.number_input(
                "Timeout (seconds)",
                min_value=30,
                max_value=600,
                value=LLM_TIMEOUT,
                help="Request timeout in seconds"
            )

        if llm_provider == "OpenRouter (Online)":
            effective_api_url = OPENROUTER_API_URL
            effective_model = openrouter_model
            effective_provider = "openrouter"
            effective_api_key = openrouter_api_key
            effective_reasoning = openrouter_reasoning
        else:
            effective_api_url = llm_api_url
            effective_model = llm_model
            effective_provider = "local"
            effective_api_key = None
            effective_reasoning = False

        # Connection Test
        st.subheader("Connection Test")
        col_test1, col_test2 = st.columns([1, 3])

        with col_test1:
            if st.button("Test Connection", type="secondary", key="test_llm_connection"):
                with st.spinner("Testing connection..."):
                    result = LLMSessionizer.test_connection(
                        effective_api_url,
                        effective_model,
                        timeout=10,
                        provider=effective_provider,
                        api_key=effective_api_key,
                        reasoning_enabled=effective_reasoning
                    )

                if result["success"]:
                    st.success(f"Connection successful!")
                    st.info(f"Latency: {result['latency_ms']}ms | Model: {result['model']}")
                else:
                    st.error(f"Connection failed: {result['error']}")

        with col_test2:
            st.caption("Tests connection to the selected LLM provider with a simple prompt")

    # --- Prompt Editor Section ---
    with st.expander("Prompt Template", expanded=False):
        st.subheader("Edit Prompt Template")

        # Preset selector
        col_preset1, col_preset2 = st.columns([2, 1])

        with col_preset1:
            preset_names = list(PROMPT_PRESETS.keys())
            selected_preset = st.selectbox("Load Preset", options=preset_names, key="llm_preset_select")

        with col_preset2:
            if st.button("Apply Preset", key="apply_preset"):
                st.session_state.custom_prompt = PROMPT_PRESETS[selected_preset]
                st.rerun()

        # Prompt editor
        prompt_template = st.text_area(
            "Prompt Template",
            value=st.session_state.custom_prompt,
            height=300,
            help="Use {events} placeholder for the event data",
            key="llm_prompt_editor"
        )

        # Update session state
        st.session_state.custom_prompt = prompt_template

        # Validation and controls
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

        with col_ctrl1:
            if st.button("Reset to Default", key="reset_prompt"):
                st.session_state.custom_prompt = LLM_PROMPT_TEMPLATE
                st.rerun()

        with col_ctrl2:
            # Validate prompt has required placeholder
            if '{events}' not in prompt_template:
                st.warning("Missing {events} placeholder!")
            else:
                st.success("Prompt valid")

        with col_ctrl3:
            token_estimate = len(prompt_template.split())
            st.caption(f"~{token_estimate} words in template")

        # Preview with sample data
        show_preview = st.checkbox(
            "Show prompt preview with sample data",
            value=False,
            key="llm_prompt_preview"
        )
        if show_preview:
            sample_events = """ID: 1001 | 2025-12-18 11:46:41 | field_input | Excel - Title = 'Invoice'
ID: 1002 | 2025-12-18 11:46:55 | field_input | Excel - Amount = '500'
ID: 1003 | 2025-12-18 11:47:10 | browser_activity | Chrome - Visited google.com"""

            try:
                preview = prompt_template.format(events=sample_events)
                st.code(preview, language="text")
                st.caption(f"Preview token estimate: ~{len(preview.split())} words")
            except KeyError as e:
                st.error(f"Invalid placeholder in prompt: {e}")

    # --- Data Source Section ---
    st.subheader("Data Source")

    llm_data_source = st.radio(
        "Select data source:",
        ["Use loaded sessions (extract events)", "Upload new CSV"],
        horizontal=True,
        key="llm_data_source"
    )

    events_for_llm = None

    if llm_data_source == "Upload new CSV":
        llm_uploaded_file = st.file_uploader("Upload CSV for LLM analysis", type=['csv'], key="llm_csv_upload")

        if llm_uploaded_file:
            try:
                llm_df = pd.read_csv(llm_uploaded_file)
                st.success(f"Loaded {len(llm_df)} events from uploaded file")

                # Preview
                with st.expander("Preview Uploaded Data"):
                    st.dataframe(llm_df.head(10))

                events_for_llm = llm_df.to_dict('records')
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
    else:
        # Extract events from current sessions
        all_events = []
        for session in sessions:
            for action in session.get('actions', []):
                action_copy = action.copy()
                action_copy['session_number'] = session['session_number']
                all_events.append(action_copy)

        # Add isolated events
        for event in isolated_events:
            event_copy = event.copy()
            event_copy['session_number'] = None
            all_events.append(event_copy)

        if all_events:
            # Sort by timestamp
            all_events.sort(key=lambda x: x.get('timestamp', ''))
            events_for_llm = all_events
            st.info(f"Using {len(all_events)} events from loaded sessions and isolated events")

            with st.expander("Preview Events"):
                preview_df = pd.DataFrame(all_events[:20])
                st.dataframe(preview_df)
        else:
            st.warning("No events available. Load data first or upload a CSV.")

    # --- Processing Options ---
    st.subheader("Processing Options")

    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        use_streaming = st.checkbox("Enable streaming (show tokens as they arrive)", value=False, key="llm_streaming")

    with col_opt2:
        show_raw_response = st.checkbox("Show raw LLM responses in debug", value=True, key="llm_show_raw")

    # --- Run LLM Sessionization ---
    st.markdown("---")

    if st.button("Run LLM Sessionization", type="primary", key="run_llm_sessionization"):
        if events_for_llm is None or len(events_for_llm) == 0:
            st.error("No events available. Please load data or upload a CSV.")
        elif '{events}' not in st.session_state.custom_prompt:
            st.error("Invalid prompt: Missing {events} placeholder.")
        elif effective_provider == "openrouter" and not effective_api_key:
            st.error("OpenRouter API key is required.")
        else:
            # Initialize sessionizer
            sessionizer = LLMSessionizer(
                api_url=effective_api_url,
                model=effective_model,
                prompt_template=st.session_state.custom_prompt,
                timeout=llm_timeout,
                max_retries=LLM_MAX_RETRIES,
                provider=effective_provider,
                api_key=effective_api_key,
                reasoning_enabled=effective_reasoning
            )

            total_events = len(events_for_llm)
            total_batches = (total_events + llm_batch_size - 1) // llm_batch_size

            st.info(f"Processing {total_events} events in {total_batches} batches...")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            streaming_container = st.empty() if use_streaming else None

            all_sessions = []
            all_isolated = []
            batch_results = []
            errors = []

            for i in range(0, total_events, llm_batch_size):
                batch = events_for_llm[i:i + llm_batch_size]
                batch_num = i // llm_batch_size + 1

                status_text.text(f"Processing batch {batch_num}/{total_batches}...")

                if use_streaming:
                    # Streaming mode - use list as mutable container
                    streaming_state = {"text": ""}

                    def on_token(token):
                        streaming_state["text"] += token
                        streaming_container.code(streaming_state["text"][-2000:], language="json")

                    result = sessionizer.process_batch_streaming(batch, on_token=on_token)
                else:
                    # Non-streaming mode
                    result = sessionizer.process_batch(batch)

                batch_results.append({
                    'batch': batch_num,
                    'success': result['success'],
                    'attempts': result.get('attempts', 1),
                    'error': result.get('error'),
                    'raw_response': result.get('raw_response')
                })

                if result['success']:
                    data = result['data']
                    batch_sessions = data.get('sessions', [])

                    # Add batch info to sessions
                    for session in batch_sessions:
                        session['batch_number'] = batch_num
                        session['session_id'] = f"{batch_num}_{session.get('session_id', 0)}"

                    all_sessions.extend(batch_sessions)

                    isolated = data.get('isolated_events', {})
                    if isinstance(isolated, list):
                        all_isolated.extend(isolated)
                    elif isinstance(isolated, dict):
                        event_ids = isolated.get('event_ids', [])
                        if event_ids:
                            all_isolated.extend(event_ids)
                else:
                    errors.append({
                        'batch': batch_num,
                        'error': result.get('error'),
                        'raw': result.get('raw_response')
                    })

                progress_bar.progress(batch_num / total_batches)

            progress_bar.progress(1.0)
            status_text.text("Processing complete!")

            # Store results
            st.session_state.llm_results = {
                'total_sessions': len(all_sessions),
                'sessions': all_sessions,
                'isolated_events': all_isolated,
                'isolated_count': len(all_isolated),
                'batches_processed': total_batches,
                'batch_results': batch_results,
                'error_count': len(errors),
                'errors': errors,
                'raw_responses': sessionizer.raw_responses if show_raw_response else [],
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'provider': effective_provider,
                    'api_url': effective_api_url,
                    'model': effective_model,
                    'reasoning_enabled': effective_reasoning,
                    'batch_size': llm_batch_size,
                    'timeout': llm_timeout
                }
            }

            st.success(f"LLM Sessionization complete! Found {len(all_sessions)} sessions.")

    # --- Results Display ---
    if st.session_state.llm_results:
        st.markdown("---")
        st.subheader("LLM Sessionization Results")

        results = st.session_state.llm_results

        # Summary metrics
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)

        with col_r1:
            st.metric("Sessions Found", results['total_sessions'])
        with col_r2:
            st.metric("Batches Processed", results['batches_processed'])
        with col_r3:
            st.metric("Isolated Events", results['isolated_count'])
        with col_r4:
            st.metric("Errors", results['error_count'])

        # Result tabs
        result_tabs = st.tabs(["Sessions", "Isolated Events", "Batch Details", "Debug Log", "Comparison"])

        # --- Sessions Tab ---
        with result_tabs[0]:
            st.subheader(f"LLM-Detected Sessions ({results['total_sessions']})")

            if results['sessions']:
                for session in results['sessions']:
                    session_id = session.get('session_id', 'N/A')
                    intent = session.get('intent', 'No intent detected')[:60]
                    event_count = len(session.get('event_ids', []))

                    with st.expander(f"Session {session_id}: {intent}... ({event_count} events)"):
                        col_s1, col_s2 = st.columns(2)

                        with col_s1:
                            st.write(f"**Primary App:** {session.get('primary_app', 'N/A')}")
                            st.write(f"**Start Time:** {session.get('start_time', 'N/A')}")
                            st.write(f"**End Time:** {session.get('end_time', 'N/A')}")
                            st.write(f"**Batch:** {session.get('batch_number', 'N/A')}")

                        with col_s2:
                            st.write(f"**Event IDs:** {session.get('event_ids', [])}")

                        st.write("**Reasoning:**")
                        st.info(session.get('reasoning', 'No reasoning provided'))

                        st.write("**Intent:**")
                        st.success(session.get('intent', 'No intent detected'))
            else:
                st.info("No sessions detected by LLM.")

        # --- Isolated Events Tab ---
        with result_tabs[1]:
            st.subheader(f"Isolated Events ({results['isolated_count']})")

            if results['isolated_events']:
                st.write("Event IDs that the LLM couldn't group into sessions:")
                st.json(results['isolated_events'])
            else:
                st.success("All events were successfully grouped into sessions!")

        # --- Batch Details Tab ---
        with result_tabs[2]:
            st.subheader("Batch Processing Details")

            for batch in results['batch_results']:
                status_icon = "✅" if batch['success'] else "❌"
                batch_num = batch['batch']

                with st.expander(f"{status_icon} Batch {batch_num} - Attempts: {batch['attempts']}"):
                    if batch['success']:
                        st.success("Successfully processed")
                    else:
                        st.error(f"Error: {batch.get('error', 'Unknown error')}")

                    if batch.get('raw_response'):
                        st.write("**Raw Response (truncated):**")
                        st.code(batch['raw_response'][:500], language="json")

        # --- Debug Log Tab ---
        with result_tabs[3]:
            st.subheader("Debug Log")

            if results.get('raw_responses'):
                st.write(f"**Total raw responses logged:** {len(results['raw_responses'])}")

                for i, response in enumerate(results['raw_responses']):
                    with st.expander(f"Response {i+1} - Batch size: {response.get('batch_size', 'N/A')}"):
                        st.write(f"**Timestamp:** {response.get('timestamp', 'N/A')}")
                        st.code(response.get('response', 'No response'), language="json")
            else:
                st.info("No debug logs available. Enable 'Show raw LLM responses in debug' option.")

            # Configuration used
            st.write("**Configuration Used:**")
            st.json(results.get('config', {}))

        # --- Comparison Tab ---
        with result_tabs[4]:
            st.subheader("LLM vs Rule-Based Comparison")

            if results['sessions'] and sessions:
                # Convert events to DataFrame for comparison
                events_df = pd.DataFrame(events_for_llm) if events_for_llm else pd.DataFrame()

                comparison = LLMSessionizer.compare_with_rule_based(
                    results['sessions'],
                    sessions,
                    events_df
                )

                # Comparison metrics
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)

                with col_c1:
                    st.metric("LLM Sessions", comparison['llm_session_count'])
                with col_c2:
                    st.metric("Rule-Based Sessions", comparison['rule_session_count'])
                with col_c3:
                    st.metric("Agreement Rate", f"{comparison['agreement_rate']:.1%}")
                with col_c4:
                    st.metric("Discrepancies", comparison['discrepancy_count'])

                st.markdown("---")

                # Side-by-side comparison
                col_llm, col_rule = st.columns(2)

                with col_llm:
                    st.markdown("**LLM Approach**")
                    st.metric("Avg Session Size", f"{comparison['llm_avg_session_size']:.1f} events")
                    st.metric("Events Mapped", comparison['llm_total_events_mapped'])

                    st.write("**First 3 LLM Sessions:**")
                    for s in results['sessions'][:3]:
                        st.info(f"Session {s.get('session_id')}: {s.get('intent', 'N/A')[:40]}...")

                with col_rule:
                    st.markdown("**Rule-Based Approach**")
                    st.metric("Avg Session Size", f"{comparison['rule_avg_session_size']:.1f} events")
                    st.metric("Events Mapped", comparison['rule_total_events_mapped'])

                    st.write("**First 3 Rule-Based Sessions:**")
                    for s in sessions[:3]:
                        st.info(f"Session {s.get('session_number')}: {s.get('summary', 'N/A')[:40]}...")

                # Discrepancy explorer
                if comparison['discrepancies']:
                    st.markdown("---")
                    st.subheader("Discrepancies")
                    st.write("Events where LLM and rule-based approaches disagreed:")

                    discrepancy_df = pd.DataFrame(comparison['discrepancies'])
                    st.dataframe(discrepancy_df, width='stretch')
                else:
                    st.success("No discrepancies found between the two approaches!")

                # Visual comparison chart
                st.markdown("---")
                st.subheader("Session Size Comparison")

                # Prepare data for chart
                llm_sizes = [len(s.get('event_ids', [])) for s in results['sessions'][:15]]
                rule_sizes = [s.get('action_count', 0) for s in sessions[:15]]

                comparison_data = []
                for i, size in enumerate(llm_sizes):
                    comparison_data.append({'Session': f'LLM-{i+1}', 'Size': size, 'Approach': 'LLM'})
                for i, size in enumerate(rule_sizes):
                    comparison_data.append({'Session': f'Rule-{i+1}', 'Size': size, 'Approach': 'Rule-Based'})

                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    fig_comp = px.bar(
                        comp_df,
                        x='Session',
                        y='Size',
                        color='Approach',
                        barmode='group',
                        title='Session Sizes: LLM vs Rule-Based (First 15)',
                        color_discrete_map={'LLM': '#3366cc', 'Rule-Based': '#109618'}
                    )
                    fig_comp.update_layout(height=400)
                    st.plotly_chart(fig_comp, width='stretch')

            else:
                st.warning("Need both LLM results and rule-based sessions for comparison. Run LLM sessionization first.")

        # --- Export Results ---
        st.markdown("---")
        st.subheader("Export Results")

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            export_json = json.dumps(results, indent=2, default=str)
            st.download_button(
                "Download Results (JSON)",
                data=export_json,
                file_name=f"llm_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with col_exp2:
            # Create summary for CSV export
            if results['sessions']:
                sessions_export = []
                for s in results['sessions']:
                    sessions_export.append({
                        'session_id': s.get('session_id'),
                        'primary_app': s.get('primary_app'),
                        'start_time': s.get('start_time'),
                        'end_time': s.get('end_time'),
                        'event_count': len(s.get('event_ids', [])),
                        'intent': s.get('intent'),
                        'reasoning': s.get('reasoning')
                    })

                sessions_csv_df = pd.DataFrame(sessions_export)
                csv_data = sessions_csv_df.to_csv(index=False)

                st.download_button(
                    "Download Sessions (CSV)",
                    data=csv_data,
                    file_name=f"llm_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# ============================================
# TAB 8: APP CLASSIFIER
# ============================================
with tab_apps:
    st.title("🤖 App Classifier")
    st.markdown("Auto-classification of unknown apps using LLM")

    # Load cache
    cache_file = os.path.join(os.path.dirname(OUTPUT_FILE), 'pattern_classifications.json')
    classifications = []
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                classifications = json.load(f)
        except Exception as e:
            st.error(f"Could not load classification cache: {e}")

    # Extract apps from current sessions
    all_apps = {}
    for session in sessions:
        app = session.get('primary_app')
        if app:
            all_apps[app] = all_apps.get(app, 0) + 1

    # SECTION 1: Summary Stats
    st.subheader("Classification Summary")

    auto_classified = [c for c in classifications if c.get('source') == 'app_classifier']
    known_apps = len(NORMALIZATION_MAP)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Apps in Data", len(all_apps))
    with col2:
        st.metric("Hardcoded (NORMALIZATION_MAP)", known_apps)
    with col3:
        st.metric("Auto-Classified (LLM)", len(auto_classified))
    with col4:
        # Count unknown apps (not in either)
        unknown = []
        for app in all_apps.keys():
            app_lower = app.lower()
            in_normalization = app_lower in [v.lower() for v in NORMALIZATION_MAP.values()]
            in_cache = any(
                c.get('raw_pattern', '').lower() == app_lower or
                c.get('llm_suggested_name', '').lower() == app_lower
                for c in classifications
            )
            if not in_normalization and not in_cache:
                unknown.append(app)
        st.metric("Unknown (Not Classified)", len(unknown))

    st.divider()

    # SECTION 2: Recently Auto-Classified
    st.subheader("Recently Auto-Classified Apps")

    if not auto_classified:
        st.info("No auto-classified apps yet. Apps will be classified automatically when sessions are processed.")
    else:
        # Sort by classified_at descending
        sorted_classified = sorted(
            [c for c in auto_classified if c.get('classified_at')],
            key=lambda x: x.get('classified_at', ''),
            reverse=True
        )

        # Show last 20
        for entry in sorted_classified[:20]:
            with st.expander(f"{entry.get('raw_pattern')} → {entry.get('llm_suggested_name')}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Raw Pattern:** `{entry.get('raw_pattern')}`")
                    st.write(f"**Classified Name:** {entry.get('llm_suggested_name')}")
                    st.write(f"**Category:** {entry.get('llm_category', 'Unknown')}")
                with col2:
                    st.write(f"**Work-Related:** {entry.get('llm_work_related', 'Unknown')}")
                    st.write(f"**Event Count:** {entry.get('event_count', 0)}")
                    st.write(f"**Classified At:** {entry.get('classified_at', 'N/A')[:19]}")

    st.divider()

    # SECTION 3: Apps in Current Data
    st.subheader("Apps in Current Session Data")

    # Build table
    app_data = []
    for app, count in sorted(all_apps.items(), key=lambda x: x[1], reverse=True):
        app_lower = app.lower()

        # Determine source
        source = "Unknown"
        if app_lower in [v.lower() for v in NORMALIZATION_MAP.values()]:
            source = "NORMALIZATION_MAP"
        else:
            for c in classifications:
                if c.get('raw_pattern', '').lower() == app_lower:
                    if c.get('source') == 'app_classifier':
                        source = "Auto-Classified"
                    else:
                        source = "Manual/LLM Cache"
                    break

        app_data.append({
            'App': app,
            'Count': count,
            'Source': source
        })

    df_apps = pd.DataFrame(app_data)

    # Filter
    filter_source = st.selectbox(
        "Filter by source:",
        ["All", "NORMALIZATION_MAP", "Auto-Classified", "Manual/LLM Cache", "Unknown"]
    )

    if filter_source != "All":
        df_filtered = df_apps[df_apps['Source'] == filter_source]
    else:
        df_filtered = df_apps

    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True
    )

    # Unknown apps warning
    if len(unknown) > 0:
        st.warning(f"⚠️ {len(unknown)} unknown apps found. Enable APP_CLASSIFIER to auto-classify them.")
        with st.expander("View Unknown Apps"):
            for app in sorted(unknown):
                st.write(f"- `{app}` (used in {all_apps[app]} sessions)")

    st.divider()

    # SECTION 4: Charts
    st.subheader("Classification Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Source distribution
        source_counts = df_apps['Source'].value_counts()
        fig_source = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title="Apps by Classification Source"
        )
        st.plotly_chart(fig_source, width='stretch')

    with col2:
        # Category distribution (for auto-classified only)
        if auto_classified:
            categories = {}
            for c in auto_classified:
                cat = c.get('llm_category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1

            fig_cat = px.bar(
                x=list(categories.keys()),
                y=list(categories.values()),
                title="Auto-Classified Apps by Category",
                labels={'x': 'Category', 'y': 'Count'}
            )
            st.plotly_chart(fig_cat, width='stretch')
        else:
            st.info("No auto-classified apps to show category breakdown.")
