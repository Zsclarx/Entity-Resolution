import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import Levenshtein distance
try:
    from Levenshtein import distance as levenshtein_distance
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    # Fallback implementation
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Simple Levenshtein distance implementation"""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    LEVENSHTEIN_AVAILABLE = False

class LevenshteinSimilarityComponent:
    """
    Enhanced similarity component using Levenshtein distance for all attributes
    with configurable per-attribute thresholds
    """
    
    def __init__(self, attribute_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize Levenshtein Similarity component
        
        Args:
            attribute_thresholds: Dictionary mapping attribute names to their similarity thresholds
                                Example: {"title": 0.8, "authors": 0.7, "venue": 0.6}
        """
        self.attribute_thresholds = attribute_thresholds or {}
        self.default_threshold = 0.7  # Default threshold for attributes not specified
        
    def compute_similarity_scores(
        self,
        candidate_pairs: pd.DataFrame,
        attributes: List[str]
    ) -> pd.DataFrame:
        """
        Compute Levenshtein similarity scores for all attributes
        
        Args:
            candidate_pairs: DataFrame with candidate pairs and their attributes
            attributes: List of attribute names to compute similarity for
            
        Returns:
            DataFrame with similarity scores for each attribute
        """
        print(f"🔄 Computing Levenshtein similarity for {len(attributes)} attributes...")
        
        # Create a copy to avoid modifying original
        result = candidate_pairs.copy()
        
        # Compute similarity for each attribute
        for attr in attributes:
            left_col = f"{attr}_l"
            right_col = f"{attr}_r"
            sim_col = f"{attr}_sim"
            
            if left_col in result.columns and right_col in result.columns:
                print(f"   Computing similarity for: {attr}")
                
                # Compute Levenshtein similarity
                similarities = []
                for _, row in result.iterrows():
                    left_val = str(row.get(left_col, "")).strip()
                    right_val = str(row.get(right_col, "")).strip()
                    
                    # Handle empty values
                    if not left_val or not right_val:
                        similarities.append(0.0)
                        continue
                    
                    # Compute normalized Levenshtein similarity
                    max_len = max(len(left_val), len(right_val))
                    if max_len == 0:
                        similarities.append(1.0)
                    else:
                        lev_dist = levenshtein_distance(left_val.lower(), right_val.lower())
                        similarity = 1.0 - (lev_dist / max_len)
                        similarities.append(max(0.0, similarity))  # Ensure non-negative
                
                result[sim_col] = similarities
            else:
                print(f"   ⚠️ Columns not found for attribute: {attr}")
                result[sim_col] = 0.0
        
        print(f"✅ Similarity computation completed")
        return result
    
    def apply_attribute_thresholds(
        self,
        similarity_df: pd.DataFrame,
        attributes: List[str]
    ) -> pd.DataFrame:
        """
        Apply per-attribute thresholds to filter candidate pairs
        
        Args:
            similarity_df: DataFrame with similarity scores
            attributes: List of attribute names
            
        Returns:
            Filtered DataFrame based on attribute thresholds
        """
        print(f"🎯 Applying per-attribute thresholds...")
        
        # Show current thresholds
        for attr in attributes:
            threshold = self.attribute_thresholds.get(attr, self.default_threshold)
            print(f"   {attr}: {threshold:.3f}")
        
        # Apply thresholds
        mask = pd.Series([True] * len(similarity_df))
        
        for attr in attributes:
            sim_col = f"{attr}_sim"
            threshold = self.attribute_thresholds.get(attr, self.default_threshold)
            
            if sim_col in similarity_df.columns:
                attr_mask = similarity_df[sim_col] >= threshold
                mask = mask & attr_mask
                
                # Show filtering statistics
                passed = attr_mask.sum()
                total = len(attr_mask)
                print(f"   {attr}: {passed}/{total} pairs passed threshold ({passed/total*100:.1f}%)")
        
        result = similarity_df[mask].copy()
        
        initial_count = len(similarity_df)
        final_count = len(result)
        print(f"🔍 Threshold filtering: {initial_count} → {final_count} pairs ({final_count/initial_count*100:.1f}% passed)")
        
        return result
    
    def compute_overall_similarity(
        self,
        similarity_df: pd.DataFrame,
        attributes: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Compute overall similarity score as weighted average of attribute similarities
        
        Args:
            similarity_df: DataFrame with individual attribute similarities
            attributes: List of attribute names
            weights: Optional weights for each attribute (defaults to equal weights)
            
        Returns:
            DataFrame with overall similarity score
        """
        if weights is None:
            # Equal weights for all attributes
            weights = {attr: 1.0 / len(attributes) for attr in attributes}
        
        print(f"🧮 Computing overall similarity with weights:")
        for attr, weight in weights.items():
            print(f"   {attr}: {weight:.3f}")
        
        # Compute weighted average
        overall_similarities = []
        
        for _, row in similarity_df.iterrows():
            weighted_sum = 0.0
            total_weight = 0.0
            
            for attr in attributes:
                sim_col = f"{attr}_sim"
                if sim_col in similarity_df.columns:
                    similarity = row.get(sim_col, 0.0)
                    weight = weights.get(attr, 0.0)
                    weighted_sum += similarity * weight
                    total_weight += weight
            
            if total_weight > 0:
                overall_similarities.append(weighted_sum / total_weight)
            else:
                overall_similarities.append(0.0)
        
        result = similarity_df.copy()
        result['overall_similarity'] = overall_similarities
        
        print(f"✅ Overall similarity computed (mean: {np.mean(overall_similarities):.3f})")
        return result
    
    def get_similarity_statistics(
        self,
        similarity_df: pd.DataFrame,
        attributes: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Get detailed statistics about similarity scores
        
        Args:
            similarity_df: DataFrame with similarity scores
            attributes: List of attribute names
            
        Returns:
            Dictionary with statistics for each attribute
        """
        stats = {}
        
        for attr in attributes:
            sim_col = f"{attr}_sim"
            if sim_col in similarity_df.columns:
                similarities = similarity_df[sim_col]
                
                # Handle empty DataFrame or all NaN values
                if len(similarities) == 0 or similarities.isna().all():
                    stats[attr] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'median': 0.0,
                        'count': 0
                    }
                else:
                    # Remove NaN values before computing statistics
                    valid_similarities = similarities.dropna()
                    
                    if len(valid_similarities) == 0:
                        stats[attr] = {
                            'mean': 0.0,
                            'std': 0.0,
                            'min': 0.0,
                            'max': 0.0,
                            'median': 0.0,
                            'count': 0
                        }
                    else:
                        stats[attr] = {
                            'mean': float(valid_similarities.mean()),
                            'std': float(valid_similarities.std()) if len(valid_similarities) > 1 else 0.0,
                            'min': float(valid_similarities.min()),
                            'max': float(valid_similarities.max()),
                            'median': float(valid_similarities.median()),
                            'count': int(len(valid_similarities))
                        }
        
        return stats
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update attribute thresholds
        
        Args:
            new_thresholds: Dictionary mapping attribute names to new thresholds
        """
        self.attribute_thresholds.update(new_thresholds)
        print(f"🔄 Updated thresholds: {self.attribute_thresholds}")
    
    def get_threshold(self, attribute: str) -> float:
        """
        Get threshold for a specific attribute
        
        Args:
            attribute: Attribute name
            
        Returns:
            Threshold value for the attribute
        """
        return self.attribute_thresholds.get(attribute, self.default_threshold) 