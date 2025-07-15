import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class SimpleSimilarityComponent:
    """
    Simple similarity component using scikit-learn NearestNeighbors
    Fallback implementation when FAISS has issues
    """
    
    def __init__(
        self,
        n_neighbors: int = 30,
        metric: str = 'cosine',
        algorithm: str = 'brute'
    ):
        """
        Initialize Simple Similarity component
        
        Args:
            n_neighbors: Number of nearest neighbors to find
            metric: Distance metric to use
            algorithm: Algorithm to use for neighbor search
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.model = None
        
    def build_index(self, df: pd.DataFrame, features_col: str = 'normFeatures') -> None:
        """
        Build similarity index from normalized features
        
        Args:
            df: DataFrame with normalized feature vectors
            features_col: Column name containing normalized features
        """
        print(f"🔄 Building Simple Similarity index...")
        print(f"   n_neighbors: {self.n_neighbors}")
        print(f"   metric: {self.metric}")
        print(f"   algorithm: {self.algorithm}")
        
        # Extract features as numpy array
        features_list = df[features_col].tolist()
        features_array = np.array(features_list, dtype=np.float32)
        
        print(f"   Dimension: {features_array.shape[1]}")
        print(f"   Number of vectors: {len(features_array)}")
        
        # Create and fit NearestNeighbors model
        self.model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(features_array)),  # +1 for self
            metric=self.metric,
            algorithm=self.algorithm
        )
        
        print("🔧 Fitting similarity model...")
        self.model.fit(features_array)
        
        print(f"✅ Simple Similarity index built successfully with {len(features_array)} vectors")
    
    def search_neighbors(self, df: pd.DataFrame, features_col: str = 'normFeatures') -> pd.DataFrame:
        """
        Search for nearest neighbors using the built index
        
        Args:
            df: DataFrame with normalized feature vectors to search
            features_col: Column name containing normalized features
            
        Returns:
            DataFrame with neighbor information
        """
        if self.model is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        print(f"🔍 Searching for nearest neighbors...")
        
        # Extract query features
        query_features = np.array(df[features_col].tolist(), dtype=np.float32)
        
        # Search for neighbors
        distances, indices = self.model.kneighbors(query_features)
        
        # Create results DataFrame
        results = []
        for i, (row_idx, row) in enumerate(df.iterrows()):
            row_id = row.get('row_id', i)
            
            # Filter out self-matches and create neighbor list
            neighbors = []
            for j in range(len(indices[i])):
                neighbor_idx = indices[i][j]
                distance = float(distances[i][j])
                
                # Skip self-matches (index same as current row)
                if neighbor_idx != i:
                    neighbors.append({
                        'neighbor': int(neighbor_idx),
                        'distance': distance
                    })
            
            # Limit to desired number of neighbors (excluding self)
            neighbors = neighbors[:self.n_neighbors]
            
            results.append({
                'row_id': int(row_id) if row_id is not None else i,
                'neighbors': neighbors
            })
        
        result_df = pd.DataFrame(results)
        
        total_pairs = sum(len(row['neighbors']) for row in results)
        print(f"✅ Found {total_pairs} neighbor pairs")
        
        return result_df

class SimpleSimilarity:
    """
    Simple Similarity class that mimics the HNSW interface
    Compatible with the pipeline architecture
    """
    
    def __init__(
        self,
        identifier_col: str = "row_id",
        features_col: str = "normFeatures",
        distance_function: str = "cosine",
        num_partitions: int = 16,
        k: int = 30,
        prediction_col: str = "neighbors"
    ):
        """
        Initialize Simple Similarity component
        
        Args:
            identifier_col: Column name for record identifiers
            features_col: Column name for feature vectors
            distance_function: Distance function (cosine, euclidean, etc.)
            num_partitions: Number of partitions (for compatibility)
            k: Number of nearest neighbors
            prediction_col: Output column name for predictions
        """
        self.identifier_col = identifier_col
        self.features_col = features_col
        self.distance_function = distance_function
        self.num_partitions = num_partitions
        self.k = k
        self.prediction_col = prediction_col
        
        # Internal similarity component
        self.similarity_component = SimpleSimilarityComponent(
            n_neighbors=k,
            metric=distance_function
        )
    
    def fit(self, df: pd.DataFrame) -> 'SimpleSimilarityModel':
        """
        Fit the similarity model on the input data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Fitted SimpleSimilarityModel
        """
        print(f"🔄 Fitting Simple Similarity model...")
        
        # Build the index
        self.similarity_component.build_index(df, self.features_col)
        
        # Return fitted model
        return SimpleSimilarityModel(
            similarity_component=self.similarity_component,
            identifier_col=self.identifier_col,
            features_col=self.features_col,
            prediction_col=self.prediction_col
        )

class SimpleSimilarityModel:
    """
    Fitted Simple Similarity Model
    """
    
    def __init__(
        self,
        similarity_component: SimpleSimilarityComponent,
        identifier_col: str,
        features_col: str,
        prediction_col: str
    ):
        self.similarity_component = similarity_component
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
        print(f"🔄 Transforming data with Simple Similarity model...")
        
        # Search for neighbors
        neighbor_df = self.similarity_component.search_neighbors(df, self.features_col)
        
        # Merge with original data
        result = df.merge(
            neighbor_df[[self.identifier_col, 'neighbors']], 
            on=self.identifier_col, 
            how='left'
        )
        
        # Rename neighbors column to prediction column
        result = result.rename(columns={'neighbors': self.prediction_col})
        
        return result 