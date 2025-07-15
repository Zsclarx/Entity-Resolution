import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class CandidatePairGenerationComponent:
    """
    Component for generating candidate pairs from HNSW results
    Based on the Scala CandidatePairGenerationComponent
    """
    
    def __init__(self):
        self.SEP = "#"
    
    def extract_candidate_pairs_from_hnsw(
        self,
        knn_df: pd.DataFrame,
        partitioned_vectors: pd.DataFrame,
        blocking_attributes: List[str]
    ) -> pd.DataFrame:
        """
        Extracts candidate pairs from raw HNSW results and enriches with attributes
        
        Args:
            knn_df: Raw HNSW results DataFrame with neighbors column
            partitioned_vectors: Original DataFrame with attributes
            blocking_attributes: List of attributes to include in output
            
        Returns:
            DataFrame with candidate pairs and enriched attributes
        """
        print(f"🔄 Extracting candidate pairs from HNSW results...")
        
        # Convert dots to hashes to match vectorized column names
        aliased_attrs = [attr.replace(".", self.SEP) for attr in blocking_attributes]
        
        # Explode HNSW neighbors to get flat pairs
        flat_pairs = self._explode_neighbors(knn_df)
        
        # Create mini dataset with relevant attributes
        mini = self._prepare_mini_dataset(partitioned_vectors)
        
        # Enrich candidate pairs with attributes
        candidate_pairs = self._enrich_with_attributes(
            flat_pairs, mini, aliased_attrs
        )
        
        pair_count = len(candidate_pairs)
        print(f"✅ Candidate pairs extracted successfully: {pair_count} pairs")
        
        return candidate_pairs
    
    def _explode_neighbors(self, knn_df: pd.DataFrame) -> pd.DataFrame:
        """
        Explode HNSW neighbors to get flat pairs
        
        Args:
            knn_df: DataFrame with neighbors column
            
        Returns:
            DataFrame with src_id, dst_id, distance columns
        """
        flat_pairs = []
        
        for _, row in knn_df.iterrows():
            src_id = row['row_id']
            neighbors = row['neighbors']
            
            if isinstance(neighbors, list):
                for neighbor in neighbors:
                    if isinstance(neighbor, dict):
                        flat_pairs.append({
                            'src_id': src_id,
                            'dst_id': neighbor.get('neighbor'),
                            'distance': neighbor.get('distance', 0.0)
                        })
        
        return pd.DataFrame(flat_pairs)
    
    def _prepare_mini_dataset(self, partitioned_vectors: pd.DataFrame) -> pd.DataFrame:
        """
        Create mini dataset with relevant attributes
        
        Args:
            partitioned_vectors: Original DataFrame with attributes
            
        Returns:
            Mini DataFrame with cleaned columns
        """
        # Drop blocking-specific columns
        columns_to_drop = ['blockingKey', 'normFeatures']
        existing_columns_to_drop = [col for col in columns_to_drop if col in partitioned_vectors.columns]
        
        if existing_columns_to_drop:
            mini = partitioned_vectors.drop(columns=existing_columns_to_drop)
        else:
            mini = partitioned_vectors.copy()
        
        return mini
    
    def _enrich_with_attributes(
        self,
        flat_pairs: pd.DataFrame,
        mini: pd.DataFrame,
        aliased_attrs: List[str]
    ) -> pd.DataFrame:
        """
        Enrich candidate pairs with attributes from both sides
        
        Args:
            flat_pairs: DataFrame with src_id, dst_id pairs
            mini: Mini dataset with attributes
            aliased_attrs: List of aliased attribute names
            
        Returns:
            Enriched DataFrame with left and right attributes
        """
        # Prepare left and right datasets
        mini_l = mini.rename(columns={'row_id': 'l_row_id'})
        mini_r = mini.rename(columns={'row_id': 'r_row_id'})
        
        # Create column selection for final output
        base_cols = ['src_id', 'dst_id']
        
        # Find available aliased attributes in the mini dataset
        available_attrs = [attr for attr in aliased_attrs if attr in mini.columns]
        
        # If no attributes found, show available columns to help debug
        if not available_attrs:
            print(f"⚠️ Warning: No matching attributes found!")
            print(f"   Looking for: {aliased_attrs}")
            print(f"   Available columns: {list(mini.columns)}")
            # Still proceed with empty attributes
        
        left_attrs = [f"{attr}_l" for attr in available_attrs]
        right_attrs = [f"{attr}_r" for attr in available_attrs]
        
        # Join to enrich candidate pairs with attributes
        result = flat_pairs.merge(
            mini_l, left_on='src_id', right_on='l_row_id', how='left'
        ).merge(
            mini_r, left_on='dst_id', right_on='r_row_id', how='left'
        )
        
        # Rename attribute columns with _l and _r suffixes
        for attr in available_attrs:
            if f"{attr}_x" in result.columns:
                result = result.rename(columns={f"{attr}_x": f"{attr}_l"})
            if f"{attr}_y" in result.columns:
                result = result.rename(columns={f"{attr}_y": f"{attr}_r"})
        
        # Select final columns - INCLUDE DISTANCE!
        final_cols = base_cols.copy()
        
        # Add distance column if it exists
        if 'distance' in result.columns:
            final_cols.append('distance')
            
        for attr in available_attrs:
            if f"{attr}_l" in result.columns:
                final_cols.append(f"{attr}_l")
            if f"{attr}_r" in result.columns:
                final_cols.append(f"{attr}_r")
        
        # Filter to only include available columns
        available_final_cols = [col for col in final_cols if col in result.columns]
        
        if available_final_cols:
            result = result[available_final_cols]
        
        return result
    
    def generate_comparison_pairs(
        self,
        knn_df: pd.DataFrame,
        original_df: pd.DataFrame,
        features_to_compare: List[str]
    ) -> pd.DataFrame:
        """
        Generate comparison pairs with specific features for similarity computation
        
        Args:
            knn_df: HNSW results DataFrame
            original_df: Original DataFrame with all features
            features_to_compare: List of feature names to include in comparison
            
        Returns:
            DataFrame ready for similarity computation
        """
        print(f"🔄 Generating comparison pairs for {len(features_to_compare)} features...")
        
        # Explode neighbors
        flat_pairs = self._explode_neighbors(knn_df)
        
        # Prepare comparison data
        comparison_data = original_df[['row_id'] + features_to_compare].copy()
        
        # Create left and right versions
        left_data = comparison_data.add_suffix('_l')
        left_data = left_data.rename(columns={'row_id_l': 'src_id'})
        
        right_data = comparison_data.add_suffix('_r')
        right_data = right_data.rename(columns={'row_id_r': 'dst_id'})
        
        # Join with flat pairs
        result = flat_pairs.merge(left_data, on='src_id', how='left') \
                          .merge(right_data, on='dst_id', how='left')
        
        print(f"✅ Generated {len(result)} comparison pairs")
        
        return result
    
    def filter_by_distance_threshold(self, candidate_pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Filter candidate pairs by distance threshold
        
        Args:
            candidate_pairs: DataFrame with candidate pairs
            threshold: Distance threshold for filtering
            
        Returns:
            Filtered DataFrame
        """
        if 'distance' not in candidate_pairs.columns:
            print("⚠️ Warning: No distance column found, skipping distance filtering")
            return candidate_pairs
        
        initial_count = len(candidate_pairs)
        filtered_pairs = candidate_pairs[candidate_pairs['distance'] <= threshold]
        final_count = len(filtered_pairs)
        
        print(f"🔍 Distance filtering: {initial_count} → {final_count} pairs (threshold: {threshold})")
        
        return filtered_pairs 