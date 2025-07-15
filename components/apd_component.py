import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

@dataclass
class APDHyperplaneNode:
    """
    APD Hyperplane Node data structure
    Based on the Scala APDHyperplaneNode case class
    """
    node_id: int
    hyperplane: np.ndarray
    threshold: float
    left_child: Optional['APDHyperplaneNode'] = None
    right_child: Optional['APDHyperplaneNode'] = None
    is_leaf: bool = False
    partition_id: int = -1
    projection_median: float = 0.0
    projection_spread: float = 0.0

class APDComponent:
    """
    Simplified APD (Adaptive Partitioning for Distributed) Hyperplane Segmenter
    Based on the Scala APDComponent class
    """
    
    def __init__(
        self,
        num_partitions: int,
        sample_size: int = 100000,
        max_tree_depth: int = 6,
        min_leaf_size: int = 1000
    ):
        self.num_partitions = num_partitions
        self.sample_size = sample_size
        self.max_tree_depth = max_tree_depth
        self.min_leaf_size = min_leaf_size
    
    def learn_segmentation_tree(self, df: pd.DataFrame) -> APDHyperplaneNode:
        """
        Learns segmentation tree for partitioning
        
        Args:
            df: Input DataFrame with normalized features
            
        Returns:
            Learned segmentation tree
        """
        print(f"🔄 Learning APD segmentation tree...")
        print(f"   Number of partitions: {self.num_partitions}")
        print(f"   Sample size: {self.sample_size}")
        print(f"   Max tree depth: {self.max_tree_depth}")
        print(f"   Min leaf size: {self.min_leaf_size}")
        
        total_count = len(df)
        print(f"   Total dataset size: {total_count}")
        
        # Step 1: Get representative sample
        print("📊 Getting representative sample...")
        representative_sample = self._get_simple_sample(df, self.sample_size)
        
        print(f"📊 Using {len(representative_sample)} sample points to build tree...")
        
        # Step 2: Build tree structure
        raw_tree = self._learn_tree_driver_only(representative_sample, 0, 0)
        
        # Step 3: Assign clean consecutive partition IDs
        clean_tree = self._assign_clean_partition_ids(raw_tree)
        
        leaf_count = self._count_leaf_nodes(clean_tree)
        print(f"✅ APD segmentation tree learned with {leaf_count} leaf nodes")
        print(f"   Partition IDs: 0 to {leaf_count - 1}")
        
        return clean_tree
    
    def partition_data(self, df: pd.DataFrame, segmentation_tree: APDHyperplaneNode) -> pd.DataFrame:
        """
        Partitions data using the learned segmentation tree
        
        Args:
            df: Input DataFrame
            segmentation_tree: Learned tree structure
            
        Returns:
            DataFrame with partition assignments
        """
        print("🔄 Applying APD segmentation tree...")
        
        result_df = df.copy()
        partitions = []
        
        for _, row in df.iterrows():
            features = np.array(row['normFeatures'])
            partition_id = self._partition_data_point(features, segmentation_tree, 0, 0)
            partitions.append(partition_id)
        
        result_df['partition_id'] = partitions
        
        # Show partition distribution
        print("📊 Partition distribution:")
        partition_dist = result_df['partition_id'].value_counts().sort_index()
        for partition_id, count in partition_dist.items():
            print(f"   Partition {partition_id}: {count} records")
        
        return result_df
    
    def partition_with_natural_spill(
        self,
        df: pd.DataFrame,
        segmentation_tree: APDHyperplaneNode,
        alpha: float = 0.3
    ) -> pd.DataFrame:
        """
        Partitions data with natural spill for multi-query processing
        
        Args:
            df: Input DataFrame
            segmentation_tree: Learned tree structure
            alpha: Spill threshold factor (controls spillage)
            
        Returns:
            DataFrame with query partitions
        """
        print(f"🔄 Applying natural spill multi-query with alpha={alpha}...")
        print("   Natural spill behavior:")
        print("     - Root spill → ALL partitions")
        print("     - Level 1 spill → Half partitions")
        print("     - Level 2 spill → Quarter partitions")
        print("     - etc. (emerges naturally from tree structure)")
        
        result_df = df.copy()
        query_partitions = []
        
        for _, row in df.iterrows():
            features = np.array(row['normFeatures'])
            partitions = self._find_partitions_natural_spill(segmentation_tree, features, alpha)
            query_partitions.append(partitions)
        
        result_df['queryPartitions'] = query_partitions
        
        # Show spill statistics
        print("📊 Natural spill expansion statistics:")
        spill_counts = {}
        for partitions in query_partitions:
            count = len(partitions)
            spill_counts[count] = spill_counts.get(count, 0) + 1
        
        for count, frequency in sorted(spill_counts.items()):
            print(f"   {count} partitions: {frequency} records")
        
        return result_df
    
    def _get_simple_sample(self, df: pd.DataFrame, target_sample_size: int) -> np.ndarray:
        """Get sample using simple limit (faster and more reliable)"""
        print(f"📊 Getting simple sample of {target_sample_size} points...")
        
        sample_size = min(target_sample_size, len(df))
        sample_df = df.head(sample_size)
        
        sample_vectors = np.array([vec for vec in sample_df['normFeatures'].values])
        
        print(f"✅ Simple sampling produced {len(sample_vectors)} points")
        return sample_vectors
    
    def _learn_tree_driver_only(
        self,
        vectors: np.ndarray,
        node_id: int,
        node_level: int
    ) -> APDHyperplaneNode:
        """Driver-only tree learning with SVD-based hyperplane computation"""
        
        # Base case: if we reached max depth or too few samples
        if node_level >= self.max_tree_depth or len(vectors) < self.min_leaf_size:
            return APDHyperplaneNode(
                node_id=node_id,
                hyperplane=np.array([]),
                threshold=0.0,
                is_leaf=True,
                partition_id=node_id % self.num_partitions
            )
        
        print(f"🔧 Building APD node {node_id} at level {node_level} with {len(vectors)} samples")
        
        if len(vectors) == 0:
            return APDHyperplaneNode(
                node_id=node_id,
                hyperplane=np.array([]),
                threshold=0.0,
                is_leaf=True,
                partition_id=node_id % self.num_partitions
            )
        
        # Use TruncatedSVD for dimensionality reduction
        n_components = min(2, vectors.shape[1], vectors.shape[0])
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        try:
            svd.fit(vectors)
            # Get hyperplane h (second singular vector if available, else first)
            if n_components > 1:
                hyperplane_h = svd.components_[1]
            else:
                hyperplane_h = svd.components_[0]
        except:
            # Fallback: use random hyperplane
            hyperplane_h = np.random.randn(vectors.shape[1])
            hyperplane_h = hyperplane_h / np.linalg.norm(hyperplane_h)
        
        # Compute dot products
        dot_prods = np.dot(vectors, hyperplane_h)
        
        # Sort and find median
        sorted_dot_prods = np.sort(dot_prods)
        median = np.median(sorted_dot_prods)
        
        # Store additional statistics for spill decisions
        left_boundary = sorted_dot_prods[0]
        right_boundary = sorted_dot_prods[-1]
        projection_spread = abs(right_boundary - left_boundary)
        
        print(f"   APD Node {node_id}: median={median:.4f}, spread={projection_spread:.4f}")
        
        # Split data
        left_mask = dot_prods <= median
        left_data = vectors[left_mask]
        right_data = vectors[~left_mask]
        
        print(f"   APD Node {node_id}: Left={len(left_data)}, Right={len(right_data)}")
        
        # Recursive calls
        left_child = None
        if len(left_data) > 0:
            left_child = self._learn_tree_driver_only(left_data, 2 * node_id + 1, node_level + 1)
        
        right_child = None
        if len(right_data) > 0:
            right_child = self._learn_tree_driver_only(right_data, 2 * node_id + 2, node_level + 1)
        
        # Return node with statistics
        return APDHyperplaneNode(
            node_id=node_id,
            hyperplane=hyperplane_h,
            threshold=float(median),
            left_child=left_child,
            right_child=right_child,
            is_leaf=False,
            projection_median=float(median),
            projection_spread=float(projection_spread)
        )
    
    def _assign_clean_partition_ids(self, tree: APDHyperplaneNode) -> APDHyperplaneNode:
        """Assign clean consecutive partition IDs to leaf nodes"""
        partition_counter = [0]  # Use list for mutable reference
        
        def reassign_ids(node: APDHyperplaneNode) -> APDHyperplaneNode:
            if node.is_leaf:
                clean_id = partition_counter[0]
                partition_counter[0] += 1
                return APDHyperplaneNode(
                    node_id=node.node_id,
                    hyperplane=node.hyperplane,
                    threshold=node.threshold,
                    left_child=node.left_child,
                    right_child=node.right_child,
                    is_leaf=node.is_leaf,
                    partition_id=clean_id,
                    projection_median=node.projection_median,
                    projection_spread=node.projection_spread
                )
            else:
                new_left = reassign_ids(node.left_child) if node.left_child else None
                new_right = reassign_ids(node.right_child) if node.right_child else None
                return APDHyperplaneNode(
                    node_id=node.node_id,
                    hyperplane=node.hyperplane,
                    threshold=node.threshold,
                    left_child=new_left,
                    right_child=new_right,
                    is_leaf=node.is_leaf,
                    partition_id=node.partition_id,
                    projection_median=node.projection_median,
                    projection_spread=node.projection_spread
                )
        
        clean_tree = reassign_ids(tree)
        print(f"✅ Assigned clean partition IDs: 0 to {partition_counter[0] - 1} ({partition_counter[0]} partitions)")
        return clean_tree
    
    def _partition_data_point(
        self,
        vector: np.ndarray,
        node: APDHyperplaneNode,
        node_id: int,
        node_level: int
    ) -> int:
        """Partition a single data point using tree traversal"""
        if node_level >= self.max_tree_depth or node.is_leaf:
            return node.partition_id
        
        dot_product = np.dot(vector, node.hyperplane)
        
        if dot_product <= node.threshold:
            if node.left_child:
                return self._partition_data_point(vector, node.left_child, 2 * node_id + 1, node_level + 1)
            else:
                return node.partition_id
        else:
            if node.right_child:
                return self._partition_data_point(vector, node.right_child, 2 * node_id + 2, node_level + 1)
            else:
                return node.partition_id
    
    def _find_partitions_natural_spill(
        self,
        tree: APDHyperplaneNode,
        query_vector: np.ndarray,
        alpha: float
    ) -> List[int]:
        """Find partitions using natural spill behavior"""
        found_partitions = set()
        
        def traverse(node: APDHyperplaneNode):
            if node.is_leaf:
                found_partitions.add(node.partition_id)
                return
            
            projection = np.dot(query_vector, node.hyperplane)
            distance_from_median = abs(projection - node.projection_median)
            spill_threshold = alpha * node.projection_spread
            
            should_spill = distance_from_median <= spill_threshold
            
            if should_spill:
                # SPILL: Explore both children
                if node.left_child:
                    traverse(node.left_child)
                if node.right_child:
                    traverse(node.right_child)
            else:
                # NO SPILL: Follow natural path
                if projection <= node.threshold:
                    if node.left_child:
                        traverse(node.left_child)
                else:
                    if node.right_child:
                        traverse(node.right_child)
        
        traverse(tree)
        return list(found_partitions)
    
    def _count_leaf_nodes(self, node: APDHyperplaneNode) -> int:
        """Count the number of leaf nodes in the tree"""
        if node.is_leaf:
            return 1
        else:
            left_count = self._count_leaf_nodes(node.left_child) if node.left_child else 0
            right_count = self._count_leaf_nodes(node.right_child) if node.right_child else 0
            return left_count + right_count

def create_apd_component(num_partitions: Optional[int] = None) -> APDComponent:
    """
    Create APD component with automatic or specified number of partitions
    
    Args:
        num_partitions: Number of partitions (if None, uses a default)
        
    Returns:
        APDComponent instance
    """
    if num_partitions is None:
        # Default to a reasonable number for CPU processing
        import multiprocessing
        num_partitions = multiprocessing.cpu_count()
    
    return APDComponent(num_partitions) 