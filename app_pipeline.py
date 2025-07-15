#!/usr/bin/env python3
"""
Streamlit UI for Entity Resolution Pipeline
Provides web interface for the entity resolution pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
from typing import List, Dict, Any, Optional

# Import pipeline components
from main import EntityResolutionPipeline

# Page config
st.set_page_config(
    page_title="Entity Resolution Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stage-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">🔍 Entity Resolution Pipeline</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Pipeline Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload CSV File", 
        type=['csv'],
        help="Upload a CSV file containing records to deduplicate"
    )
    
    # Sample data option
    if st.sidebar.button("📊 Use Sample Data"):
        if 'sample_data' not in st.session_state:
            # Create sample data
            try:
                sample_df = pd.read_csv('data/sample_data.csv')
                st.session_state.sample_data = sample_df
                st.session_state.data_source = "Sample Data"
                st.sidebar.success("✅ Sample data loaded!")
            except FileNotFoundError:
                st.sidebar.error("❌ Sample data not found. Please run: python main.py --create-sample")
                return
    
    # Pipeline parameters
    st.sidebar.subheader("🎛️ Parameters")
    
    model_name = st.sidebar.selectbox(
        "🤖 Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "sentence-transformers/paraphrase-MiniLM-L6-v2"],
        index=0,
        help="Sentence transformer model for generating embeddings"
    )
    
    # Similarity method selection
    use_levenshtein = st.sidebar.checkbox(
        "📏 Use Levenshtein Distance",
        value=True,
        help="Use Levenshtein distance for attribute-level similarity (recommended)"
    )
    
    if use_levenshtein:
        similarity_threshold = 0.7  # Default when using Levenshtein
        st.sidebar.info("🎯 Using per-attribute thresholds (configured below)")
    else:
        similarity_threshold = st.sidebar.slider(
            "🎯 Global Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Global threshold for considering records as duplicates"
        )
    
    use_apd = st.sidebar.checkbox(
        "🔄 Enable APD Partitioning",
        value=False,
        help="Use Adaptive Partitioning for Distributed processing"
    )
    
    num_blocking_cols = st.sidebar.slider(
        "📊 Blocking Columns",
        min_value=1,
        max_value=20,
        value=12,
        help="Number of top features to use for blocking key generation"
    )
    
    # Process data
    if uploaded_file is not None or 'sample_data' in st.session_state:
        
        # Load data
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.session_state.data_source = uploaded_file.name
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                return
        else:
            data = st.session_state.sample_data
            st.session_state.data = data
        
        # Data overview
        st.markdown('<div class="stage-header">📊 Data Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📁 Source", st.session_state.data_source)
        with col2:
            st.metric("📈 Records", len(data))
        with col3:
            st.metric("📋 Columns", len(data.columns))
        with col4:
            st.metric("💾 Size", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Show data preview
        with st.expander("🔍 Data Preview", expanded=False):
            st.dataframe(data.head(10), use_container_width=True)
        
        # Display selected blocking key attributes based on data type
        st.markdown('<div class="stage-header">🔑 Selected Blocking Key Attributes</div>', unsafe_allow_html=True)
        
        # Show info about the blocking column limit
        st.info(f"🎯 **Blocking Limit**: Showing top {num_blocking_cols} attributes (configurable in sidebar)")
        
        # Detect data type and show appropriate blocking attributes
        column_names = data.columns.tolist()
        
        # Check if it's Grainger data (has _grainger columns)
        grainger_columns = [col for col in column_names if '_grainger' in col.lower()]
        
        if grainger_columns:
            # Grainger data
            st.success("📊 **Grainger Data Detected** - Displaying optimal blocking attributes for Grainger dataset")
            
            grainger_blocking_attrs = [
                '_grainger.City',
                '_grainger.SubIndustry', 
                '_grainger.AAD_NAICS_DESC_RP_BASKET',
                '_grainger.ZIP',
                '_grainger.Scores.MRO_POTENTIAL',
                '_grainger.Industry',
                '_grainger.CUSTOMER_LIFECYCLE',
                '_grainger.Address',
                '_grainger.AAD_RP_BASKET',
                '_grainger.AccountCompanyName',
                '_grainger.State',
                '_grainger.R12_TOTAL_SALES'
            ]
            
            # Check which attributes are actually available
            available_grainger_attrs = []
            missing_grainger_attrs = []
            
            for attr in grainger_blocking_attrs:
                # Check both exact match and variations
                found = False
                # Extract the key part after the dot (e.g., "City" from "_grainger.City")
                key_part = attr.split('.')[-1] if '.' in attr else attr.replace('_grainger', '')
                
                for col in column_names:
                    # Check if the key part matches (case insensitive)
                    if (key_part.lower() in col.lower() and '_grainger' in col.lower()) or attr.lower() == col.lower():
                        available_grainger_attrs.append(col)
                        found = True
                        break
                if not found:
                    missing_grainger_attrs.append(attr)
            
            # Limit to num_blocking_cols
            limited_grainger_attrs = available_grainger_attrs[:num_blocking_cols]
            
            if limited_grainger_attrs:
                st.success(f"✅ **Top {len(limited_grainger_attrs)} Blocking Attributes** (from {len(available_grainger_attrs)} available):")
                
                # Display in a nice formatted way
                cols = st.columns(3)
                for i, attr in enumerate(limited_grainger_attrs):
                    with cols[i % 3]:
                        st.write(f"• `{attr}`")
                
                # Show remaining attributes in expander if there are more
                if len(available_grainger_attrs) > num_blocking_cols:
                    remaining_attrs = available_grainger_attrs[num_blocking_cols:]
                    with st.expander(f"📋 Remaining Available Attributes ({len(remaining_attrs)} more)", expanded=False):
                        st.info(f"**Additional Attributes** (not used for blocking due to limit of {num_blocking_cols}):")
                        for attr in remaining_attrs:
                            st.write(f"• `{attr}`")
            
            if missing_grainger_attrs:
                with st.expander("⚠️ Missing Blocking Attributes", expanded=False):
                    st.warning(f"**Missing Attributes** ({len(missing_grainger_attrs)} not found):")
                    for attr in missing_grainger_attrs:
                        st.write(f"• `{attr}`")
        else:
            # Sample data
            st.success("📊 **Sample Data Detected** - Displaying optimal blocking attributes for research papers")
            
            sample_blocking_attrs = [
                'title',
                'author', 
                'venue',
                'year',
                'language',
                'country',
                'institution',
                'publisher',
                'issue',
                'keywords',
                'abstract',
                'funding'
            ]
            
            # Check which attributes are actually available
            available_sample_attrs = []
            missing_sample_attrs = []
            
            for attr in sample_blocking_attrs:
                # Check both exact match and variations (authors vs author, etc.)
                found = False
                for col in column_names:
                    if attr.lower() in col.lower() or col.lower() in attr.lower():
                        available_sample_attrs.append(col)
                        found = True
                        break
                if not found:
                    missing_sample_attrs.append(attr)
            
            # Limit to num_blocking_cols
            limited_sample_attrs = available_sample_attrs[:num_blocking_cols]
            
            if limited_sample_attrs:
                st.success(f"✅ **Top {len(limited_sample_attrs)} Blocking Attributes** (from {len(available_sample_attrs)} available):")
                
                # Display in a nice formatted way
                cols = st.columns(3)
                for i, attr in enumerate(limited_sample_attrs):
                    with cols[i % 3]:
                        st.write(f"• `{attr}`")
                
                # Show remaining attributes in expander if there are more
                if len(available_sample_attrs) > num_blocking_cols:
                    remaining_attrs = available_sample_attrs[num_blocking_cols:]
                    with st.expander(f"📋 Remaining Available Attributes ({len(remaining_attrs)} more)", expanded=False):
                        st.info(f"**Additional Attributes** (not used for blocking due to limit of {num_blocking_cols}):")
                        for attr in remaining_attrs:
                            st.write(f"• `{attr}`")
            
            if missing_sample_attrs:
                with st.expander("⚠️ Missing Blocking Attributes", expanded=False):
                    st.warning(f"**Missing Attributes** ({len(missing_sample_attrs)} not found):")
                    for attr in missing_sample_attrs:
                        st.write(f"• `{attr}`")
        
        # Feature selection
        st.markdown('<div class="stage-header">🎯 Feature Selection</div>', unsafe_allow_html=True)
        
        # Auto-detect features (exclude ID columns)
        auto_features = [col for col in data.columns if not col.lower().endswith('id')]
        
        selected_features = st.multiselect(
            "📋 Select Features for Entity Resolution",
            options=data.columns.tolist(),
            default=auto_features,
            help="Choose which columns to use for finding duplicates"
        )
        
        if not selected_features:
            st.warning("⚠️ Please select at least one feature to proceed.")
            return
        
        # Per-attribute threshold configuration (only for Levenshtein)
        attribute_thresholds = {}
        if use_levenshtein:
            st.markdown('<div class="stage-header">🎯 Per-Attribute Threshold Configuration</div>', unsafe_allow_html=True)
            
            st.info("🔧 Configure similarity thresholds for each attribute. Higher values = stricter matching.")
            
            # Create threshold sliders for each selected feature
            threshold_cols = st.columns(min(3, len(selected_features)))
            
            for i, feature in enumerate(selected_features):
                with threshold_cols[i % len(threshold_cols)]:
                    threshold = st.slider(
                        f"📏 {feature}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,  # Default threshold
                        step=0.05,
                        key=f"threshold_{feature}",
                        help=f"Similarity threshold for {feature} attribute"
                    )
                    attribute_thresholds[feature] = threshold
            
            # Show threshold summary
            with st.expander("📊 Threshold Summary", expanded=False):
                threshold_df = pd.DataFrame([
                    {"Attribute": attr, "Threshold": f"{thresh:.2f}"}
                    for attr, thresh in attribute_thresholds.items()
                ])
                st.dataframe(threshold_df, use_container_width=True)
        
        # Run pipeline button
        if st.button("🚀 Run Entity Resolution Pipeline", type="primary"):
            run_pipeline(data, selected_features, model_name, similarity_threshold, use_apd, num_blocking_cols, 
                        use_levenshtein, attribute_thresholds)

def run_pipeline(data: pd.DataFrame, selected_features: List[str], model_name: str, 
                similarity_threshold: float, use_apd: bool, num_blocking_cols: int,
                use_levenshtein: bool = True, attribute_thresholds: Optional[Dict[str, float]] = None):
    """Run the entity resolution pipeline with progress tracking"""
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Stage tracking
    stage_container = st.container()
    
    try:
        # Initialize pipeline
        status_text.text("🔧 Initializing pipeline...")
        pipeline = EntityResolutionPipeline(
            selected_features=selected_features,
            num_blocking_cols=num_blocking_cols,
            model_name=model_name,
            use_apd=use_apd,
            similarity_threshold=similarity_threshold,
            attribute_thresholds=attribute_thresholds,
            use_levenshtein=use_levenshtein
        )
        progress_bar.progress(10)
        
        # Stage 1: Load data
        status_text.text("📁 Stage 1: Loading data...")
        with stage_container:
            st.markdown('<div class="stage-header">📁 Stage 1: Data Loading</div>', unsafe_allow_html=True)
            
        # Add row_id to data
        data_with_id = data.copy()
        data_with_id['row_id'] = range(len(data_with_id))
        pipeline.raw_data = data_with_id
        progress_bar.progress(20)
        
        # Stage 2: Vectorization
        status_text.text("🔄 Stage 2: Creating embeddings...")
        with stage_container:
            st.markdown('<div class="stage-header">🔄 Stage 2: Blocking Key Vectorization</div>', unsafe_allow_html=True)
            stage2_start = time.time()
            
        vectorized_data = pipeline.run_blocking_vectorization()
        stage2_time = time.time() - stage2_start
        
        with stage_container:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Time", f"{stage2_time:.2f}s")
            with col2:
                st.metric("📊 Records", len(vectorized_data))
            with col3:
                st.metric("🧮 Dimensions", len(vectorized_data['features'].iloc[0]) if len(vectorized_data) > 0 else 0)
        
        progress_bar.progress(40)
        
        # Stage 3: APD (optional)
        if use_apd:
            status_text.text("🔄 Stage 3: APD Partitioning...")
            with stage_container:
                st.markdown('<div class="stage-header">🔄 Stage 3: APD Partitioning</div>', unsafe_allow_html=True)
                stage3_start = time.time()
                
            partitioned_data = pipeline.run_apd_partitioning()
            stage3_time = time.time() - stage3_start
            
            with stage_container:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("⏱️ Time", f"{stage3_time:.2f}s")
                with col2:
                    if 'partition_id' in partitioned_data.columns:
                        st.metric("🗂️ Partitions", partitioned_data['partition_id'].nunique())
        
        progress_bar.progress(60)
        
        # Stage 4: Similarity Search
        status_text.text("🔍 Stage 4: Finding similar records...")
        with stage_container:
            st.markdown('<div class="stage-header">🔍 Stage 4: Similarity Search</div>', unsafe_allow_html=True)
            stage4_start = time.time()
            
        similarity_results = pipeline.run_hnsw_similarity()
        stage4_time = time.time() - stage4_start
        
        with stage_container:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Time", f"{stage4_time:.2f}s")
            with col2:
                total_neighbors = sum(len(row['neighbors']) for _, row in similarity_results.iterrows())
                st.metric("🔗 Neighbor Pairs", total_neighbors)
            with col3:
                st.metric("📊 Records Processed", len(similarity_results))
        
        progress_bar.progress(80)
        
        # Stage 5: Candidate Pairs
        status_text.text("🎯 Stage 5: Generating candidate pairs...")
        with stage_container:
            st.markdown('<div class="stage-header">🎯 Stage 5: Candidate Pair Generation</div>', unsafe_allow_html=True)
            stage5_start = time.time()
            
        candidate_pairs = pipeline.run_candidate_pair_generation()
        stage5_time = time.time() - stage5_start
        
        with stage_container:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Time", f"{stage5_time:.2f}s")
            with col2:
                st.metric("👥 Candidate Pairs", len(candidate_pairs))
            with col3:
                if 'distance' in candidate_pairs.columns:
                    avg_distance = candidate_pairs['distance'].mean()
                    st.metric("📏 Avg Distance", f"{avg_distance:.3f}")
                else:
                    st.metric("📏 Distance Info", "Not Available")
        
        progress_bar.progress(90)
        
        # Stage 6: Clustering
        status_text.text("🎯 Stage 6: Entity clustering...")
        with stage_container:
            st.markdown('<div class="stage-header">🎯 Stage 6: Entity Clustering</div>', unsafe_allow_html=True)
            stage6_start = time.time()
            
        final_results = pipeline.run_entity_clustering()
        stage6_time = time.time() - stage6_start
        
        progress_bar.progress(100)
        status_text.text("✅ Pipeline completed successfully!")
        
        # Results summary
        with stage_container:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⏱️ Time", f"{stage6_time:.2f}s")
            with col2:
                cluster_counts = final_results['cluster_id'].value_counts()
                multi_record_clusters = cluster_counts[cluster_counts > 1]
                st.metric("🎯 Multi-Record Clusters", len(multi_record_clusters))
            with col3:
                st.metric("📊 Total Clusters", final_results['cluster_id'].nunique())
            with col4:
                if len(multi_record_clusters) > 0:
                    st.metric("📈 Largest Cluster", cluster_counts.max())
                else:
                    st.metric("📈 Largest Cluster", 1)
        
        # Store results
        st.session_state.results = final_results
        st.session_state.candidate_pairs = candidate_pairs
        st.session_state.pipeline_completed = True
        
        # Show results
        show_results(final_results, candidate_pairs)
        
    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        st.exception(e)

def show_results(results: pd.DataFrame, candidate_pairs: pd.DataFrame):
    """Display pipeline results with visualizations"""
    
    st.markdown('<div class="stage-header">🎉 Results</div>', unsafe_allow_html=True)
    
    # Cluster statistics
    cluster_stats = results['cluster_id'].value_counts().sort_values(ascending=False)
    multi_record_clusters = cluster_stats[cluster_stats > 1]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Total Records", 
            len(results),
            help="Total number of input records"
        )
    
    with col2:
        st.metric(
            "🎯 Clusters Found", 
            len(cluster_stats),
            help="Total number of clusters (including singletons)"
        )
    
    with col3:
        st.metric(
            "👥 Multi-Record Clusters", 
            len(multi_record_clusters),
            help="Clusters containing multiple records (duplicates found)"
        )
    
    with col4:
        records_in_clusters = multi_record_clusters.sum() if len(multi_record_clusters) > 0 else 0
        st.metric(
            "🔗 Records in Clusters", 
            records_in_clusters,
            help="Total records involved in multi-record clusters"
        )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster size distribution
        st.subheader("📊 Cluster Size Distribution")
        if len(multi_record_clusters) > 0:
            fig = px.histogram(
                x=multi_record_clusters.values,
                nbins=min(20, len(multi_record_clusters)),
                title="Distribution of Multi-Record Cluster Sizes",
                labels={"x": "Cluster Size", "y": "Number of Clusters"}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No multi-record clusters found. All records are unique.")
    
    with col2:
        # Distance distribution (if available)
        st.subheader("📏 Distance Distribution")
        if 'distance' in candidate_pairs.columns and len(candidate_pairs) > 0:
            fig = px.histogram(
                candidate_pairs, 
                x='distance',
                nbins=30,
                title="Distribution of Similarity Distances",
                labels={"distance": "Distance", "count": "Number of Pairs"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Distance information not available.")
    
    # Detailed results
    if len(multi_record_clusters) > 0:
        st.subheader("🔍 Detected Duplicate Groups")
        
        # Show top clusters
        top_clusters = multi_record_clusters.head(10)
        
        for cluster_id, cluster_size in top_clusters.items():
            with st.expander(f"🎯 Cluster {cluster_id} ({cluster_size} records)", expanded=False):
                cluster_records = results[results['cluster_id'] == cluster_id]
                st.dataframe(cluster_records.drop(['row_id'], axis=1, errors='ignore'), use_container_width=True)
    
    # Export results
    st.subheader("💾 Export Results")
    
    # Download clustered results
    csv_buffer = io.StringIO()
    results.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download Clustered Results",
        data=csv_buffer.getvalue(),
        file_name="entity_resolution_results.csv",
        mime="text/csv",
        help="Download the complete results with cluster assignments"
    )
    
    # Show raw results table
    with st.expander("📋 Full Results Table", expanded=False):
        st.dataframe(results, use_container_width=True)

if __name__ == "__main__":
    main() 