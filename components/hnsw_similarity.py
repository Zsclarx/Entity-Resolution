import pandas as pd
import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class HNSWComponent:
    """
    HNSW (Hierarchical Navigable Small World) similarity component using FAISS
    Based on the HNSW functionality from the Scala version
    """
    
    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 100,
        ef: int = 30,
        k: int = 30
    ):
        """
        Initialize HNSW component
        
        Args:
            M: Number of bi-directional links created for every new element during construction
            ef_construction: Size of the dynamic candidate list for construction
            ef: Size of the dynamic candidate list for search
            k: Number of nearest neighbors to return
        """
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.k = k
        self.index = None
        self.dimension = None
        
    def build_index(self, df: pd.DataFrame, features_col: str = 'normFeatures') -> None:
        """
        Build HNSW index from normalized features
        
        Args:
            df: DataFrame with normalized feature vectors
            features_col: Column name containing normalized features
        """
        print(f"🔄 Building HNSW index...")
        print(f"   M: {self.M}")
        print(f"   ef_construction: {self.ef_construction}")
        print(f"   ef: {self.ef}")
        print(f"   k: {self.k}")
        
        # Extract features as numpy array
        features_list = df[features_col].tolist()
        features_array = np.array(features_list, dtype=np.float32)
        
        self.dimension = features_array.shape[1]
        print(f"   Dimension: {self.dimension}")
        print(f"   Number of vectors: {len(features_array)}")
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(self.dimension, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        
        # Add vectors to index
        print("🔧 Adding vectors to HNSW index...")
        self.index.add(features_array)
        
        # Set search parameters
        self.index.hnsw.efSearch = self.ef
        
        print(f"✅ HNSW index built successfully with {self.index.ntotal} vectors")
    
    def search_neighbors(self, df: pd.DataFrame, features_col: str = 'normFeatures') -> pd.DataFrame:
        """
        Search for nearest neighbors using the built HNSW index
        
        Args:
            df: DataFrame with normalized feature vectors to search
            features_col: Column name containing normalized features
            
        Returns:
            DataFrame with neighbor information
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        print(f"🔍 Searching for {self.k} nearest neighbors...")
        
        # Extract query features
        query_features = np.array(df[features_col].tolist(), dtype=np.float32)
        
        # Search for neighbors
        distances, indices = self.index.search(query_features, self.k + 1)  # +1 to include self
        
        # Create results DataFrame
        results = []
        for i, (row_idx, row) in enumerate(df.iterrows()):
            row_id = row.get('row_id', i)
            
            # Filter out self-matches and create neighbor list
            neighbors = []
            for j in range(len(indices[i])):
                neighbor_idx = indices[i][j]
                distance = float(distances[i][j])
                
                # Skip self-matches (distance very close to 0)
                if neighbor_idx != i and distance > 1e-6:
                    neighbors.append({
                        'neighbor': int(neighbor_idx),
                        'distance': distance
                    })
            
            # Limit to k neighbors (excluding self)
            neighbors = neighbors[:self.k]
            
            results.append({
                'row_id': int(row_id) if row_id is not None else i,
                'neighbors': neighbors
            })
        
        result_df = pd.DataFrame(results)
        
        total_pairs = sum(len(row['neighbors']) for row in results)
        print(f"✅ Found {total_pairs} neighbor pairs")
        
        return result_df
    
    def transform(self, df: pd.DataFrame, features_col: str = 'normFeatures') -> pd.DataFrame:
        """
        Build index and search for neighbors in one step (similar to Spark ML transform)
        
        Args:
            df: DataFrame with normalized feature vectors
            features_col: Column name containing normalized features
            
        Returns:
            DataFrame with neighbor information
        """
        self.build_index(df, features_col)
        return self.search_neighbors(df, features_col)

class HNSWSimilarity:
    """
    HNSW Similarity class that mimics the Spark ML interface
    Compatible with the pipeline architecture
    """
    
    def __init__(
        self,
        identifier_col: str = "row_id",
        features_col: str = "normFeatures",
        distance_function: str = "inner-product",
        num_partitions: int = 16,
        M: int = 16,
        ef_construction: int = 100,
        ef: int = 30,
        k: int = 30,
        prediction_col: str = "neighbors"
    ):
        """
        Initialize HNSW Similarity component
        
        Args:
            identifier_col: Column name for record identifiers
            features_col: Column name for feature vectors
            distance_function: Distance function (currently supports "inner-product")
            num_partitions: Number of partitions (for compatibility)
            M: HNSW parameter M
            ef_construction: HNSW parameter efConstruction
            ef: HNSW parameter ef
            k: Number of nearest neighbors
            prediction_col: Output column name for predictions
        """
        self.identifier_col = identifier_col
        self.features_col = features_col
        self.distance_function = distance_function
        self.num_partitions = num_partitions
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.k = k
        self.prediction_col = prediction_col
        
        # Internal HNSW component
        self.hnsw_component = HNSWComponent(M, ef_construction, ef, k)
    
    def fit(self, df: pd.DataFrame) -> 'HNSWSimilarityModel':
        """
        Fit the HNSW model on the input data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Fitted HNSWSimilarityModel
        """
        print(f"🔄 Fitting HNSW model...")
        
        # Build the index
        self.hnsw_component.build_index(df, self.features_col)
        
        # Return fitted model
        return HNSWSimilarityModel(
            hnsw_component=self.hnsw_component,
            identifier_col=self.identifier_col,
            features_col=self.features_col,
            prediction_col=self.prediction_col
        )

class HNSWSimilarityModel:
    """
    Fitted HNSW Similarity Model
    """
    
    def __init__(
        self,
        hnsw_component: HNSWComponent,
        identifier_col: str,
        features_col: str,
        prediction_col: str
    ):
        self.hnsw_component = hnsw_component
        self.identifier_col = identifier_col
        self.features_col = features_col
        self.prediction_col = prediction_col
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame to find nearest neighbors
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with neighbor predictions
        """
        print(f"🔄 Transforming data with HNSW model...")
        
        # Search for neighbors
        neighbor_df = self.hnsw_component.search_neighbors(df, self.features_col)
        
        # Merge with original data
        result = df.merge(
            neighbor_df[[self.identifier_col, 'neighbors']], 
            on=self.identifier_col, 
            how='left'
        )
        
        # Rename neighbors column to prediction column
        result = result.rename(columns={'neighbors': self.prediction_col})
        
        return result 