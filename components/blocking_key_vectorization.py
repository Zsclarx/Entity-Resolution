import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BlockingKeyVectorizationComponent:
    """
    Component for creating blocking keys and generating vector embeddings
    Based on the Scala BlockingKeyVectorizationComponent
    """
    
    def __init__(self):
        self.SEP = "#"
        self.model = None
    
    def create_blocking_keys_and_embeddings(
        self,
        df: pd.DataFrame,
        selected_features: List[str],
        num_blocking_cols: int = 12,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 512
    ) -> pd.DataFrame:
        """
        Creates blocking keys and generates embeddings for the input DataFrame
        
        Args:
            df: Input DataFrame with selected features
            selected_features: List of selected feature column names
            num_blocking_cols: Number of top features to use for blocking key
            model_name: Name of the sentence transformer model
            batch_size: Batch size for embedding generation
            
        Returns:
            DataFrame with blocking keys and normalized embeddings
        """
        print(f"🔄 Creating blocking keys and embeddings...")
        print(f"   Selected features: {len(selected_features)}")
        print(f"   Blocking columns: {num_blocking_cols}")
        print(f"   Model: {model_name}")
        print(f"   Batch size: {batch_size}")
        
        # Step 1: Create column alias mapping
        cols_to_select = selected_features.copy()
        # Only add 'id' if it exists in the DataFrame
        if 'id' in df.columns and 'id' not in cols_to_select:
            cols_to_select.append('id')
        
        # Ensure row_id is included if it exists
        if 'row_id' in df.columns and 'row_id' not in cols_to_select:
            cols_to_select.append('row_id')
        
        # Step 2: Create aliased DataFrame
        alias_mapping = {}
        aliased_df = df[cols_to_select].copy()
        
        for col in cols_to_select:
            alias = col.replace(".", self.SEP)
            alias_mapping[col] = alias
            if col != alias:
                aliased_df = aliased_df.rename(columns={col: alias})
        
        # Step 3: Create blocking key from top features
        blocking_cols = selected_features[:num_blocking_cols]
        blocking_col_aliases = [alias_mapping.get(col, col) for col in blocking_cols]
        
        # Handle null values and create blocking key
        blocking_data = []
        for _, row in aliased_df.iterrows():
            blocking_values = []
            for col_alias in blocking_col_aliases:
                value = row.get(col_alias, "")
                if pd.isna(value) or value == "":
                    blocking_values.append("")
                else:
                    blocking_values.append(str(value))
            blocking_data.append("|".join(blocking_values))
        
        aliased_df['blockingKey'] = blocking_data
        aliased_df['row_id'] = range(len(aliased_df))
        
        initial_count = len(aliased_df)
        print(f"📊 Initial DataFrame with blocking keys: {initial_count} records")
        
        # Step 4: Handle empty blocking keys
        num_pipes = len(blocking_cols) - 1
        empty_key_replacement = "|" * num_pipes
        aliased_df['blockingKey'] = aliased_df['blockingKey'].fillna(empty_key_replacement)
        mask = aliased_df['blockingKey'] == ""
        aliased_df.loc[mask, 'blockingKey'] = empty_key_replacement
        
        print(f"📊 Processed DataFrame (empty keys replaced with {num_pipes} pipes): {len(aliased_df)} records")
        
        # Step 5: Generate embeddings
        vector_df = self._generate_embeddings(aliased_df, model_name, batch_size)
        
        # Step 6: Normalize embeddings
        normalized_df = self._normalize_embeddings(vector_df)
        
        print(f"✅ Final DataFrame with embeddings: {len(normalized_df)} records")
        
        return normalized_df
    
    def create_blocking_key_only(
        self,
        df: pd.DataFrame,
        selected_features: List[str],
        num_blocking_cols: int = 12
    ) -> pd.DataFrame:
        """
        Creates a simple blocking key from selected features (without embeddings)
        
        Args:
            df: Input DataFrame
            selected_features: List of selected feature column names
            num_blocking_cols: Number of top features to use for blocking key
            
        Returns:
            DataFrame with blocking key only
        """
        print(f"🔄 Creating blocking keys only...")
        print(f"   Selected features: {len(selected_features)}")
        print(f"   Blocking columns: {num_blocking_cols}")
        
        # Create column alias mapping
        cols_to_select = selected_features.copy()
        if 'id' not in cols_to_select:
            cols_to_select.append('id')
        
        alias_mapping = {}
        result_df = df[cols_to_select].copy()
        
        for col in cols_to_select:
            alias = col.replace(".", self.SEP)
            alias_mapping[col] = alias
            if col != alias:
                result_df = result_df.rename(columns={col: alias})
        
        # Create blocking key
        blocking_cols = selected_features[:num_blocking_cols]
        blocking_col_aliases = [alias_mapping.get(col, col) for col in blocking_cols]
        
        blocking_data = []
        for _, row in result_df.iterrows():
            blocking_values = []
            for col_alias in blocking_col_aliases:
                value = row.get(col_alias, "")
                if pd.isna(value) or value == "":
                    blocking_values.append("")
                else:
                    blocking_values.append(str(value))
            blocking_data.append("|".join(blocking_values))
        
        result_df['blockingKey'] = blocking_data
        result_df['row_id'] = range(len(result_df))
        
        # Handle empty blocking keys
        num_pipes = len(blocking_cols) - 1
        empty_key_replacement = "|" * num_pipes
        result_df['blockingKey'] = result_df['blockingKey'].fillna(empty_key_replacement)
        mask = result_df['blockingKey'] == ""
        result_df.loc[mask, 'blockingKey'] = empty_key_replacement
        
        final_count = len(result_df)
        print(f"✅ Blocking keys created: {final_count} records")
        
        return result_df
    
    def _generate_embeddings(self, df: pd.DataFrame, model_name: str, batch_size: int) -> pd.DataFrame:
        """Generates embeddings using the specified model"""
        print(f"🔧 Generating embeddings...")
        
        if self.model is None:
            print(f"📥 Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
        
        # Extract blocking keys
        blocking_keys = df['blockingKey'].tolist()
        
        # Generate embeddings in batches
        print(f"🔄 Processing {len(blocking_keys)} texts in batches of {batch_size}")
        embeddings = self.model.encode(
            blocking_keys,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add embeddings to DataFrame
        result_df = df.copy()
        result_df['features'] = list(embeddings)
        
        print(f"✅ Embeddings generated successfully")
        return result_df
    
    def _normalize_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes the feature vectors to unit length"""
        print(f"🔧 Normalizing embeddings...")
        
        # Convert features to numpy array for normalization
        features_array = np.vstack(df['features'].values)
        
        # Normalize to unit length (L2 normalization)
        normalized_features = normalize(features_array, norm='l2')
        
        # Add normalized features back to DataFrame
        result_df = df.copy()
        result_df['normFeatures'] = list(normalized_features)
        
        print(f"✅ Embeddings normalized successfully")
        return result_df 