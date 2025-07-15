#!/usr/bin/env python3
"""
Entity Resolution Pipeline Implementation
Based on the Adobe Entity Resolution Scala architecture

Pipeline stages:
1. Data Loading
2. Blocking Key Vectorization (embeddings)
3. APD Partitioning (optional)
4. HNSW Similarity Search
5. Candidate Pair Generation
6. Entity Resolution/Clustering
"""

import pandas as pd
import numpy as np
import networkx as nx
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import json

# Import our components
from components.blocking_key_vectorization import BlockingKeyVectorizationComponent
from components.apd_component import APDComponent, create_apd_component
from components.simple_similarity import SimpleSimilarity, SimpleSimilarityModel
from components.candidate_pair_generation import CandidatePairGenerationComponent
from components.levenshtein_similarity import LevenshteinSimilarityComponent

class EntityResolutionPipeline:
    """
    Main Entity Resolution Pipeline
    Implements the complete workflow from data loading to entity clustering
    """
    
    def __init__(
        self,
        selected_features: Optional[List[str]] = None,
        num_blocking_cols: int = 12,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 512,
        use_apd: bool = False,
        hnsw_params: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.8,
        min_cluster_size: int = 2,
        attribute_thresholds: Optional[Dict[str, float]] = None,
        use_levenshtein: bool = True
    ):
        """
        Initialize the Entity Resolution Pipeline
        
        Args:
            selected_features: List of features to use (if None, uses all available)
            num_blocking_cols: Number of top features for blocking key
            model_name: Sentence transformer model name
            batch_size: Batch size for embedding generation
            use_apd: Whether to use APD partitioning
            hnsw_params: HNSW parameters dict
            similarity_threshold: Threshold for pair matching
            min_cluster_size: Minimum cluster size for output
        """
        self.selected_features = selected_features
        self.num_blocking_cols = num_blocking_cols
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_apd = use_apd
        self.similarity_threshold = similarity_threshold
        print(f"🎯 Similarity threshold set to: {similarity_threshold}")
        print(f"   (Lower threshold = more strict matching)")
        self.min_cluster_size = min_cluster_size
        self.attribute_thresholds = attribute_thresholds or {}
        self.use_levenshtein = use_levenshtein
        
        # Print attribute-specific thresholds if provided
        if self.attribute_thresholds:
            print(f"🎯 Per-attribute thresholds:")
            for attr, threshold in self.attribute_thresholds.items():
                print(f"   {attr}: {threshold:.3f}")
        else:
            print(f"🎯 Using default thresholds for all attributes")
        
        # Set default HNSW parameters
        self.hnsw_params = hnsw_params or {
            'M': 16,
            'ef_construction': 100,
            'ef': 30,
            'k': 20
        }
        
        # Initialize components
        self.blocking_component = BlockingKeyVectorizationComponent()
        self.apd_component = None
        self.hnsw_similarity = None
        self.pair_generator = CandidatePairGenerationComponent()
        self.levenshtein_similarity = LevenshteinSimilarityComponent(self.attribute_thresholds)
        
        # Pipeline state
        self.raw_data = None
        self.vectorized_data = None
        self.partitioned_data = None
        self.hnsw_results = None
        self.candidate_pairs = None
        self.final_clusters = None
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            data_path: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        print(f"📁 Loading data from: {data_path}")
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Auto-detect features if not specified
        if self.selected_features is None:
            # Exclude ID columns and use all others
            self.selected_features = [col for col in df.columns if not col.lower().endswith('id')]
            print(f"🔍 Auto-detected features: {self.selected_features}")
        
        # Add row_id to the raw data
        df_with_row_id = df.copy()
        df_with_row_id['row_id'] = range(len(df_with_row_id))
        
        self.raw_data = df_with_row_id
        return df_with_row_id
    
    def run_blocking_vectorization(self) -> pd.DataFrame:
        """
        Run blocking key vectorization stage
        
        Returns:
            DataFrame with blocking keys and embeddings
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print(f"\n🚀 Stage 1: Blocking Key Vectorization")
        print("=" * 50)
        
        start_time = time.time()
        
        if self.selected_features is None:
            raise ValueError("Selected features cannot be None")
            
        self.vectorized_data = self.blocking_component.create_blocking_keys_and_embeddings(
            df=self.raw_data,
            selected_features=self.selected_features,
            num_blocking_cols=self.num_blocking_cols,
            model_name=self.model_name,
            batch_size=self.batch_size
        )
        
        end_time = time.time()
        print(f"⏱️  Vectorization completed in {end_time - start_time:.2f} seconds")
        
        return self.vectorized_data
    
    def run_apd_partitioning(self) -> pd.DataFrame:
        """
        Run APD partitioning stage (optional)
        
        Returns:
            DataFrame with partition assignments
        """
        if self.vectorized_data is None:
            raise ValueError("No vectorized data. Run blocking vectorization first.")
        
        if not self.use_apd:
            print("⏭️  Skipping APD partitioning (use_apd=False)")
            self.partitioned_data = self.vectorized_data
            return self.partitioned_data
        
        print(f"\n🚀 Stage 2: APD Partitioning")
        print("=" * 50)
        
        start_time = time.time()
        
        # Create APD component
        self.apd_component = create_apd_component()
        
        # Learn segmentation tree
        segmentation_tree = self.apd_component.learn_segmentation_tree(self.vectorized_data)
        
        # Partition data
        self.partitioned_data = self.apd_component.partition_data(
            self.vectorized_data, segmentation_tree
        )
        
        end_time = time.time()
        print(f"⏱️  APD partitioning completed in {end_time - start_time:.2f} seconds")
        
        return self.partitioned_data
    
    def run_hnsw_similarity(self) -> pd.DataFrame:
        """
        Run HNSW similarity search stage
        
        Returns:
            DataFrame with neighbor information
        """
        data_to_use = self.partitioned_data if self.partitioned_data is not None else self.vectorized_data
        
        if data_to_use is None:
            raise ValueError("No data available. Run previous stages first.")
        
        print(f"\n🚀 Stage 3: HNSW Similarity Search")
        print("=" * 50)
        
        start_time = time.time()
        
        # Create similarity component
        self.hnsw_similarity = SimpleSimilarity(
            identifier_col="row_id",
            features_col="normFeatures",
            k=self.hnsw_params.get('k', 30)
        )
        
        # Fit and transform
        model = self.hnsw_similarity.fit(data_to_use)
        self.hnsw_results = model.transform(data_to_use)
        
        end_time = time.time()
        print(f"⏱️  HNSW similarity search completed in {end_time - start_time:.2f} seconds")
        
        return self.hnsw_results
    
    def run_candidate_pair_generation(self) -> pd.DataFrame:
        """
        Run candidate pair generation stage
        
        Returns:
            DataFrame with candidate pairs
        """
        if self.hnsw_results is None:
            raise ValueError("No HNSW results. Run HNSW similarity search first.")
        
        print(f"\n🚀 Stage 4: Candidate Pair Generation")
        print("=" * 50)
        
        start_time = time.time()
        
        if self.vectorized_data is None:
            raise ValueError("Vectorized data is None")
        if self.selected_features is None:
            raise ValueError("Selected features is None")
        
        # Get the actual column names from the vectorized data
        # Exclude special columns like 'blockingKey', 'features', 'normFeatures', 'row_id'
        excluded_cols = {'blockingKey', 'features', 'normFeatures', 'row_id'}
        available_attributes = [col for col in self.vectorized_data.columns if col not in excluded_cols]
        
        # Find the intersection of selected features and available attributes
        attributes_to_use = [attr for attr in self.selected_features if attr in available_attributes]
        
        if not attributes_to_use:
            print(f"⚠️ Warning: No matching attributes found between selected features and vectorized data!")
            print(f"   Selected features: {self.selected_features}")
            print(f"   Available attributes: {available_attributes}")
            # Use available attributes as fallback
            attributes_to_use = available_attributes
        
        print(f"🔍 Using attributes for candidate pairs: {attributes_to_use}")
            
        # Generate candidate pairs
        self.candidate_pairs = self.pair_generator.extract_candidate_pairs_from_hnsw(
            knn_df=self.hnsw_results,
            partitioned_vectors=self.vectorized_data,
            blocking_attributes=attributes_to_use  # Use the verified attributes
        )
        
        # Filter by distance threshold
        self.candidate_pairs = self.pair_generator.filter_by_distance_threshold(
            self.candidate_pairs, self.similarity_threshold
        )
        
        # Apply Levenshtein similarity computation if enabled
        if self.use_levenshtein and len(self.candidate_pairs) > 0:
            print(f"🔄 Computing Levenshtein similarity for attributes...")
            
            # Use the same attributes we used for candidate pair generation
            clean_attributes = attributes_to_use
            
            # Compute Levenshtein similarities
            self.candidate_pairs = self.levenshtein_similarity.compute_similarity_scores(
                self.candidate_pairs, clean_attributes
            )
            
            # Apply per-attribute thresholds
            self.candidate_pairs = self.levenshtein_similarity.apply_attribute_thresholds(
                self.candidate_pairs, clean_attributes
            )
            
            # Get similarity statistics
            stats = self.levenshtein_similarity.get_similarity_statistics(
                self.candidate_pairs, clean_attributes
            )
            
            print(f"📊 Attribute similarity statistics:")
            for attr, attr_stats in stats.items():
                print(f"   {attr}: mean={attr_stats['mean']:.3f}, min={attr_stats['min']:.3f}, max={attr_stats['max']:.3f}")
        
        # Additional strict filtering for clustering (prevent huge clusters)
        if not self.use_levenshtein:  # Only apply if not using Levenshtein (which has its own filtering)
            clustering_threshold = min(self.similarity_threshold * 0.7, 0.3) if self.similarity_threshold else 0.3
            print(f"🎯 Applying clustering threshold: {clustering_threshold:.3f} (stricter than similarity threshold)")
            
            if 'distance' in self.candidate_pairs.columns:
                initial_pairs = len(self.candidate_pairs)
                self.candidate_pairs = self.candidate_pairs[self.candidate_pairs['distance'] <= clustering_threshold]
                final_pairs = len(self.candidate_pairs)
                print(f"🔍 Clustering distance filter: {initial_pairs} → {final_pairs} pairs")
            else:
                print("⚠️ Warning: No distance column for clustering threshold")
        
        end_time = time.time()
        print(f"⏱️  Candidate pair generation completed in {end_time - start_time:.2f} seconds")
        
        return self.candidate_pairs
    
    def run_entity_clustering(self) -> pd.DataFrame:
        """
        Run entity clustering stage using connected components
        
        Returns:
            DataFrame with cluster assignments
        """
        if self.candidate_pairs is None:
            raise ValueError("No candidate pairs. Run candidate pair generation first.")
        
        print(f"\n🚀 Stage 5: Entity Clustering")
        print("=" * 50)
        
        start_time = time.time()
        
        # Create graph from candidate pairs
        G = nx.Graph()
        
        # Add edges for similar pairs
        for _, row in self.candidate_pairs.iterrows():
            src_id = row['src_id']
            dst_id = row['dst_id']
            G.add_edge(src_id, dst_id)
        
        # Find connected components
        clusters = []
        cluster_id = 0
        
        for component in nx.connected_components(G):
            if len(component) >= self.min_cluster_size:
                for node_id in component:
                    clusters.append({
                        'row_id': node_id,
                        'cluster_id': cluster_id
                    })
                cluster_id += 1
        
        # Create cluster DataFrame
        cluster_df = pd.DataFrame(clusters)
        
        if self.raw_data is None:
            raise ValueError("Raw data is None")
            
        # Merge with original data
        if len(cluster_df) > 0:
            self.final_clusters = self.raw_data.merge(
                cluster_df, on='row_id', how='left'
            )
            # Fill missing cluster IDs with unique values
            max_cluster_id = cluster_df['cluster_id'].max() if len(cluster_df) > 0 else -1
            missing_mask = self.final_clusters['cluster_id'].isna()
            missing_count = missing_mask.sum()
            
            # Assign unique cluster IDs to unmatched records
            unique_cluster_ids = list(range(max_cluster_id + 1, max_cluster_id + 1 + missing_count))
            self.final_clusters.loc[missing_mask, 'cluster_id'] = unique_cluster_ids
        else:
            # No clusters found, assign unique cluster ID to each record
            self.final_clusters = self.raw_data.copy()
            self.final_clusters['cluster_id'] = range(len(self.final_clusters))
        
        end_time = time.time()
        print(f"⏱️  Entity clustering completed in {end_time - start_time:.2f} seconds")
        
        # Print cluster statistics
        cluster_stats = self.final_clusters['cluster_id'].value_counts()
        multi_record_clusters = cluster_stats[cluster_stats > 1]
        
        print(f"\n📊 Clustering Results:")
        print(f"   Total records: {len(self.final_clusters)}")
        print(f"   Total clusters: {len(cluster_stats)}")
        print(f"   Multi-record clusters: {len(multi_record_clusters)}")
        print(f"   Records in multi-record clusters: {multi_record_clusters.sum()}")
        print(f"   Largest cluster size: {cluster_stats.max()}")
        
        return self.final_clusters
    
    def run_full_pipeline(self, data_path: str) -> pd.DataFrame:
        """
        Run the complete entity resolution pipeline
        
        Args:
            data_path: Path to input CSV file
            
        Returns:
            DataFrame with cluster assignments
        """
        print("🚀 Entity Resolution Pipeline")
        print("=" * 60)
        
        pipeline_start_time = time.time()
        
        try:
            # Stage 1: Load data
            self.load_data(data_path)
            
            # Stage 2: Blocking key vectorization
            self.run_blocking_vectorization()
            
            # Stage 3: APD partitioning (optional)
            self.run_apd_partitioning()
            
            # Stage 4: HNSW similarity search
            self.run_hnsw_similarity()
            
            # Stage 5: Candidate pair generation
            self.run_candidate_pair_generation()
            
            # Stage 6: Entity clustering
            self.run_entity_clustering()
            
            pipeline_end_time = time.time()
            total_time = pipeline_end_time - pipeline_start_time
            
            print(f"\n🎉 Pipeline Completed Successfully!")
            print(f"⏱️  Total processing time: {total_time:.2f} seconds")
            print(f"📈 Throughput: {len(self.raw_data) / total_time:.1f} records/second")
            
            return self.final_clusters
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            raise
    
    def save_results(self, output_path: str) -> None:
        """
        Save clustering results to CSV
        
        Args:
            output_path: Path for output CSV file
        """
        if self.final_clusters is None:
            raise ValueError("No results to save. Run the pipeline first.")
        
        print(f"💾 Saving results to: {output_path}")
        self.final_clusters.to_csv(output_path, index=False)
        print(f"✅ Results saved successfully")

def create_sample_data(output_path: str = "data/sample_data.csv") -> None:
    """Create sample data for testing"""
    
    # Create data directory
    Path(output_path).parent.mkdir(exist_ok=True)
    
    # Copy sample data from parent directory
    parent_sample = Path("../sample_data.csv")
    if parent_sample.exists():
        import shutil
        shutil.copy(parent_sample, output_path)
        print(f"✅ Sample data copied to {output_path}")
    else:
        print(f"⚠️  Parent sample data not found, creating minimal sample...")
        # Create minimal sample data
        sample_data = {
            'id': [f'id_{i}' for i in range(20)],
            'title': ['Deep Learning'] * 5 + ['Machine Learning'] * 5 + ['Neural Networks'] * 5 + ['AI Research'] * 5,
            'authors': ['John Smith'] * 5 + ['Jane Doe'] * 5 + ['Bob Johnson'] * 5 + ['Alice Brown'] * 5,
            'venue': ['NIPS'] * 10 + ['ICML'] * 10,
            'year': [2020] * 20
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        print(f"✅ Minimal sample data created at {output_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument("--data", type=str, default="data/sample_data.csv", 
                       help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="data/results.csv",
                       help="Path for output CSV file")
    parser.add_argument("--features", type=str, nargs="+", default=None,
                       help="List of features to use")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                       help="Sentence transformer model name")
    parser.add_argument("--use-apd", action="store_true",
                       help="Use APD partitioning")
    parser.add_argument("--threshold", type=float, default=0.8,
                       help="Similarity threshold")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create sample data and exit")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data(args.data)
        return
    
    # Create pipeline
    pipeline = EntityResolutionPipeline(
        selected_features=args.features,
        model_name=args.model,
        use_apd=args.use_apd,
        similarity_threshold=args.threshold
    )
    
    # Run pipeline
    results = pipeline.run_full_pipeline(args.data)
    
    # Save results
    pipeline.save_results(args.output)
    
    print(f"\n🎯 Final Summary:")
    print(f"   Input file: {args.data}")
    print(f"   Output file: {args.output}")
    print(f"   Records processed: {len(results)}")
    print(f"   Clusters found: {results['cluster_id'].nunique()}")

if __name__ == "__main__":
    main() 