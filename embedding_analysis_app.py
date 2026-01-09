"""
Embedding Analysis App
Test embedding quality and find optimal similarity thresholds using labeled pairs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from typing import List, Dict, Tuple, Optional

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Embedding Analysis",
    page_icon="🔍",
    layout="wide"
)

st.title("Embedding Analysis Tool")
st.markdown("Test embedding quality and find optimal similarity thresholds using labeled pairs.")

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'ollama_url' not in st.session_state:
    st.session_state.ollama_url = "http://localhost:11434"
if 'model_name' not in st.session_state:
    st.session_state.model_name = "embeddinggemma:300m"
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = {}
if 'pairs_df' not in st.session_state:
    st.session_state.pairs_df = None
if 'similarity_scores' not in st.session_state:
    st.session_state.similarity_scores = []
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5
if 'connection_ok' not in st.session_state:
    st.session_state.connection_ok = False
if 'embeddings_generated' not in st.session_state:
    st.session_state.embeddings_generated = False

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def test_ollama_connection(api_url: str, model: str) -> Tuple[bool, str, List[str]]:
    """Test connection to Ollama and check if model exists."""
    try:
        # Test basic connection
        response = requests.get(f"{api_url}/api/tags", timeout=10)
        if response.status_code != 200:
            return False, f"Ollama returned status {response.status_code}", []

        # Get available models
        data = response.json()
        available_models = [m.get('name', '') for m in data.get('models', [])]

        # Check if requested model exists
        model_found = any(model in m for m in available_models)

        if model_found:
            return True, f"Connected! Model '{model}' is available.", available_models
        else:
            return False, f"Model '{model}' not found.", available_models

    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Is it running?", []
    except requests.exceptions.Timeout:
        return False, "Connection timed out.", []
    except Exception as e:
        return False, f"Error: {str(e)}", []


def get_embedding(text: str, api_url: str, model: str) -> Optional[List[float]]:
    """Get embedding for a single text from Ollama."""
    try:
        response = requests.post(
            f"{api_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("embedding")
        return None
    except Exception:
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def find_optimal_threshold(scores: List[float], labels: List[bool]) -> Tuple[float, float]:
    """Find threshold that maximizes accuracy."""
    best_acc = 0.0
    best_threshold = 0.5

    for t in np.arange(0.0, 1.01, 0.01):
        predictions = [s >= t for s in scores]
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        acc = correct / len(labels) if labels else 0
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    return best_threshold, best_acc


def calculate_metrics(scores: List[float], labels: List[bool], threshold: float) -> Dict:
    """Calculate classification metrics at a given threshold."""
    predictions = [s >= threshold for s in scores]

    tp = sum(1 for p, l in zip(predictions, labels) if p and l)  # Predicted SAME, actually SAME
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)  # Predicted DIFF, actually DIFF
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)  # Predicted SAME, actually DIFF
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)  # Predicted DIFF, actually SAME

    total = len(labels)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1
    }


def truncate_text(text: str, max_len: int = 50) -> str:
    """Truncate text to max length with ellipsis."""
    if pd.isna(text):
        return ""
    text = str(text)
    return text[:max_len] + "..." if len(text) > max_len else text


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

st.header("1. Configuration")

col1, col2 = st.columns(2)

with col1:
    ollama_url = st.text_input(
        "Ollama URL",
        value=st.session_state.ollama_url,
        help="URL where Ollama is running"
    )
    st.session_state.ollama_url = ollama_url

with col2:
    model_name = st.text_input(
        "Embedding Model",
        value=st.session_state.model_name,
        help="Name of the embedding model to use"
    )
    st.session_state.model_name = model_name

if st.button("Test Connection"):
    with st.spinner("Testing connection..."):
        success, message, available_models = test_ollama_connection(ollama_url, model_name)
        st.session_state.connection_ok = success

        if success:
            st.success(message)
        else:
            st.error(message)
            if available_models:
                st.info(f"Available models: {', '.join(available_models)}")
            else:
                st.info(f"Try running: `ollama pull {model_name}`")

st.divider()

# =============================================================================
# SECTION 2: DATA LOADING
# =============================================================================

st.header("2. Data Loading")

uploaded_file = st.file_uploader(
    "Upload labeled_pairs.csv",
    type=['csv'],
    help="CSV file exported from analyze_for_sampling_app.py"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_cols = ['event_a_action', 'event_b_action', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info(f"Available columns: {list(df.columns)}")
        else:
            st.session_state.pairs_df = df

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pairs", len(df))
            with col2:
                same_count = len(df[df['label'] == 'SAME'])
                st.metric("SAME Pairs", same_count)
            with col3:
                diff_count = len(df[df['label'] == 'DIFFERENT'])
                st.metric("DIFFERENT Pairs", diff_count)

            # Preview table
            st.subheader("Preview (first 10 rows)")
            preview_cols = ['event_a_action', 'event_b_action', 'label']
            if 'time_gap' in df.columns:
                preview_cols.append('time_gap')
            if 'labeled_by' in df.columns:
                preview_cols.append('labeled_by')

            preview_df = df[preview_cols].head(10).copy()
            preview_df['event_a_action'] = preview_df['event_a_action'].apply(lambda x: truncate_text(x, 60))
            preview_df['event_b_action'] = preview_df['event_b_action'].apply(lambda x: truncate_text(x, 60))

            st.dataframe(preview_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

st.divider()

# =============================================================================
# SECTION 3: EMBEDDING GENERATION
# =============================================================================

st.header("3. Embedding Generation")

if st.session_state.pairs_df is None:
    st.info("Please upload a labeled pairs CSV file first.")
else:
    df = st.session_state.pairs_df

    # Get unique texts
    all_texts = set(df['event_a_action'].dropna().astype(str).tolist() +
                    df['event_b_action'].dropna().astype(str).tolist())

    # Count already embedded
    already_embedded = sum(1 for t in all_texts if t in st.session_state.embeddings)

    st.write(f"Unique texts to embed: **{len(all_texts)}**")
    if already_embedded > 0:
        st.write(f"Already embedded: **{already_embedded}** (cached)")

    if st.button("Generate Embeddings"):
        if not st.session_state.connection_ok:
            st.warning("Please test the Ollama connection first.")
        else:
            texts_to_embed = [t for t in all_texts if t not in st.session_state.embeddings]

            if not texts_to_embed:
                st.success("All texts already embedded!")
                st.session_state.embeddings_generated = True
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                failed_texts = []

                for i, text in enumerate(texts_to_embed):
                    status_text.text(f"Processing text {i+1} of {len(texts_to_embed)}...")
                    progress_bar.progress((i + 1) / len(texts_to_embed))

                    embedding = get_embedding(text, st.session_state.ollama_url, st.session_state.model_name)

                    if embedding:
                        st.session_state.embeddings[text] = embedding
                    else:
                        failed_texts.append(text[:50])

                elapsed_time = time.time() - start_time
                status_text.empty()
                progress_bar.empty()

                st.session_state.embeddings_generated = True

                # Show results
                st.success(f"Embedding generation complete!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Texts Embedded", len(st.session_state.embeddings))
                with col2:
                    if st.session_state.embeddings:
                        sample_emb = list(st.session_state.embeddings.values())[0]
                        st.metric("Embedding Dimension", len(sample_emb))
                with col3:
                    st.metric("Time Taken", f"{elapsed_time:.1f}s")

                if failed_texts:
                    with st.expander(f"Failed embeddings ({len(failed_texts)})"):
                        for t in failed_texts[:10]:
                            st.write(f"- {t}...")

    # Show embedding status
    if st.session_state.embeddings_generated and st.session_state.embeddings:
        st.info(f"Embeddings ready: {len(st.session_state.embeddings)} texts cached")

st.divider()

# =============================================================================
# SECTION 4: SIMILARITY ANALYSIS
# =============================================================================

st.header("4. Similarity Analysis")

if not st.session_state.embeddings_generated or not st.session_state.embeddings:
    st.info("Please generate embeddings first.")
elif st.session_state.pairs_df is None:
    st.info("Please upload a labeled pairs CSV file first.")
else:
    df = st.session_state.pairs_df

    # Calculate similarities
    similarities = []
    for _, row in df.iterrows():
        text_a = str(row['event_a_action']) if pd.notna(row['event_a_action']) else ""
        text_b = str(row['event_b_action']) if pd.notna(row['event_b_action']) else ""

        emb_a = st.session_state.embeddings.get(text_a)
        emb_b = st.session_state.embeddings.get(text_b)

        if emb_a and emb_b:
            sim = cosine_similarity(emb_a, emb_b)
        else:
            sim = None
        similarities.append(sim)

    st.session_state.similarity_scores = similarities

    # Create results dataframe with label indicator
    results_df = pd.DataFrame({
        'Pair #': range(1, len(df) + 1),
        'Event A': df['event_a_action'].apply(lambda x: truncate_text(x, 50)),
        'Event B': df['event_b_action'].apply(lambda x: truncate_text(x, 50)),
        'Label': df['label'].apply(lambda x: f"✓ {x}" if x == 'SAME' else f"✗ {x}"),
        'Similarity': [f"{s:.4f}" if s is not None else "N/A" for s in similarities]
    })

    # Add column config for better display
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            'Pair #': st.column_config.NumberColumn('Pair #', width='small'),
            'Event A': st.column_config.TextColumn('Event A', width='large'),
            'Event B': st.column_config.TextColumn('Event B', width='large'),
            'Label': st.column_config.TextColumn('Label', width='small'),
            'Similarity': st.column_config.TextColumn('Similarity', width='small')
        }
    )

st.divider()

# =============================================================================
# SECTION 5: DISTRIBUTION VISUALIZATION
# =============================================================================

st.header("5. Distribution Visualization")

if not st.session_state.similarity_scores or st.session_state.pairs_df is None:
    st.info("Please complete similarity analysis first.")
else:
    df = st.session_state.pairs_df
    scores = st.session_state.similarity_scores

    # Prepare data for histogram
    valid_data = []
    for i, score in enumerate(scores):
        if score is not None:
            label = df.iloc[i]['label']
            valid_data.append({'Similarity': score, 'Label': label})

    if valid_data:
        hist_df = pd.DataFrame(valid_data)

        # Create overlaid histogram
        fig = px.histogram(
            hist_df,
            x='Similarity',
            color='Label',
            barmode='overlay',
            nbins=30,
            color_discrete_map={'SAME': '#3498db', 'DIFFERENT': '#e74c3c'},
            opacity=0.7,
            title='Similarity Score Distribution by Label'
        )

        # Add threshold line
        fig.add_vline(
            x=st.session_state.threshold,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Threshold: {st.session_state.threshold:.2f}"
        )

        fig.update_layout(
            xaxis_title='Cosine Similarity',
            yaxis_title='Count',
            legend_title='Label',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid similarity scores to visualize.")

st.divider()

# =============================================================================
# SECTION 6: THRESHOLD DISCOVERY
# =============================================================================

st.header("6. Threshold Discovery")

if not st.session_state.similarity_scores or st.session_state.pairs_df is None:
    st.info("Please complete similarity analysis first.")
else:
    df = st.session_state.pairs_df
    scores = st.session_state.similarity_scores

    # Convert labels to boolean (SAME = True)
    labels = [df.iloc[i]['label'] == 'SAME' for i in range(len(df))]

    # Filter out None scores
    valid_pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    valid_scores = [p[0] for p in valid_pairs]
    valid_labels = [p[1] for p in valid_pairs]

    if valid_scores:
        # Threshold slider
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.threshold,
            step=0.01,
            help="Pairs with similarity >= threshold are classified as SAME"
        )
        st.session_state.threshold = threshold

        # Calculate metrics at current threshold
        metrics = calculate_metrics(valid_scores, valid_labels, threshold)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("True Positives", metrics['tp'], help="SAME correctly identified")
        with col2:
            st.metric("True Negatives", metrics['tn'], help="DIFFERENT correctly identified")
        with col3:
            st.metric("False Positives", metrics['fp'], help="DIFFERENT misclassified as SAME")
        with col4:
            st.metric("False Negatives", metrics['fn'], help="SAME misclassified as DIFFERENT")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.1%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.1%}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.1%}")

        # Find optimal threshold button
        st.markdown("---")
        if st.button("Find Optimal Threshold"):
            opt_threshold, opt_accuracy = find_optimal_threshold(valid_scores, valid_labels)
            st.session_state.threshold = opt_threshold

            st.success(f"**Optimal Threshold: {opt_threshold:.2f}** (Accuracy: {opt_accuracy:.1%})")
            st.rerun()

st.divider()

# =============================================================================
# SECTION 7: RESULTS SUMMARY
# =============================================================================

st.header("7. Results Summary")

if not st.session_state.similarity_scores or st.session_state.pairs_df is None:
    st.info("Please complete the analysis first.")
else:
    df = st.session_state.pairs_df
    scores = st.session_state.similarity_scores

    # Convert labels to boolean
    labels = [df.iloc[i]['label'] == 'SAME' for i in range(len(df))]

    # Filter valid scores
    valid_pairs = [(s, l, i) for i, (s, l) in enumerate(zip(scores, labels)) if s is not None]
    valid_scores = [p[0] for p in valid_pairs]
    valid_labels = [p[1] for p in valid_pairs]

    if valid_scores:
        # Calculate summary stats
        same_scores = [s for s, l in zip(valid_scores, valid_labels) if l]
        diff_scores = [s for s, l in zip(valid_scores, valid_labels) if not l]

        opt_threshold, opt_accuracy = find_optimal_threshold(valid_scores, valid_labels)

        # Summary metrics
        st.subheader("Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_same = np.mean(same_scores) if same_scores else 0
            st.metric("Avg Similarity (SAME)", f"{avg_same:.4f}")
        with col2:
            avg_diff = np.mean(diff_scores) if diff_scores else 0
            st.metric("Avg Similarity (DIFFERENT)", f"{avg_diff:.4f}")
        with col3:
            st.metric("Optimal Threshold", f"{opt_threshold:.2f}")
        with col4:
            st.metric("Best Accuracy", f"{opt_accuracy:.1%}")

        # Separation gap
        if same_scores and diff_scores:
            gap = avg_same - avg_diff
            st.info(f"**Separation Gap:** {gap:.4f} (higher is better for classification)")

        # Misclassified pairs
        st.subheader("Misclassified Pairs")

        threshold = st.session_state.threshold

        # SAME pairs with low similarity
        low_sim_same = [(i, s) for s, l, i in valid_pairs if l and s < threshold]

        # DIFFERENT pairs with high similarity
        high_sim_diff = [(i, s) for s, l, i in valid_pairs if not l and s >= threshold]

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**SAME pairs with low similarity ({len(low_sim_same)}):**")
            if low_sim_same:
                for idx, sim in low_sim_same[:10]:
                    row = df.iloc[idx]
                    with st.expander(f"Pair {idx+1} (sim: {sim:.4f})"):
                        st.write(f"**Event A:** {row['event_a_action'][:200]}")
                        st.write(f"**Event B:** {row['event_b_action'][:200]}")
            else:
                st.write("None - good embedding quality!")

        with col2:
            st.write(f"**DIFFERENT pairs with high similarity ({len(high_sim_diff)}):**")
            if high_sim_diff:
                for idx, sim in high_sim_diff[:10]:
                    row = df.iloc[idx]
                    with st.expander(f"Pair {idx+1} (sim: {sim:.4f})"):
                        st.write(f"**Event A:** {row['event_a_action'][:200]}")
                        st.write(f"**Event B:** {row['event_b_action'][:200]}")
            else:
                st.write("None - good embedding quality!")

        # Export button
        st.subheader("Export Results")

        export_df = df.copy()
        export_df['similarity'] = scores
        export_df['predicted_label'] = ['SAME' if s is not None and s >= threshold else 'DIFFERENT'
                                         for s in scores]
        export_df['correct'] = export_df['label'] == export_df['predicted_label']

        csv = export_df.to_csv(index=False)

        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name=f"embedding_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        st.caption("Exported CSV includes: original data + similarity scores + predicted labels + correctness")
