# Entity Resolution Pipeline

A Python implementation of the Adobe Entity Resolution pipeline architecture, converted from Scala/Spark to Python with pandas and scikit-learn.

## 🚀 Features

- **Blocking Key Vectorization**: Creates semantic embeddings using sentence transformers
- **APD Partitioning**: Adaptive Partitioning for Distributed processing (optional)
- **Similarity Search**: Uses scikit-learn's NearestNeighbors (fallback from FAISS HNSW)
- **Candidate Pair Generation**: Extracts and enriches similar record pairs
- **Entity Clustering**: Connected components algorithm for final clustering

## 📁 Project Structure

```
entity_resolution_pipeline/
├── main.py                     # Main pipeline orchestrator
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── components/                 # Pipeline components
│   ├── __init__.py
│   ├── blocking_key_vectorization.py
│   ├── apd_component.py
│   ├── simple_similarity.py
│   ├── hnsw_similarity.py     # Original FAISS implementation
│   └── candidate_pair_generation.py
├── data/                       # Data directory
│   ├── sample_data.csv        # Test data
│   └── results.csv            # Pipeline output
└── pipeline_env/              # Virtual environment
```

## 🔧 Installation

1. **Clone and setup environment**:
   ```bash
   cd entity_resolution_pipeline
   source pipeline_env/bin/activate  # or .\pipeline_env\Scripts\activate on Windows
   ```

2. **Dependencies are already installed in the virtual environment**

## 🏃 Quick Start

### Create Sample Data
```bash
python main.py --create-sample
```

### Run Full Pipeline
```bash
python main.py --data data/sample_data.csv --output data/results.csv
```

### View Results
```bash
head data/results.csv
```

## 📊 Pipeline Stages

### 1. Data Loading
- Loads CSV data
- Auto-detects features (excludes ID columns)
- Adds row_id for tracking

### 2. Blocking Key Vectorization
- Creates blocking keys from top features
- Generates semantic embeddings using sentence transformers
- Normalizes vectors for similarity search

### 3. APD Partitioning (Optional)
- Adaptive hyperplane-based partitioning
- Uses SVD for optimal splits
- Enables distributed processing patterns

### 4. Similarity Search
- Finds k-nearest neighbors for each record
- Uses cosine similarity with brute-force algorithm
- Fallback from FAISS HNSW due to compatibility issues

### 5. Candidate Pair Generation
- Extracts neighbor pairs from similarity results
- Enriches with original attributes
- Applies distance thresholds

### 6. Entity Clustering
- Builds graph from candidate pairs
- Uses connected components for clustering
- Assigns unique cluster IDs

## ⚙️ Configuration Options

```bash
python main.py --help
```

### Key Parameters:
- `--data`: Input CSV file path
- `--output`: Output CSV file path
- `--features`: Specific features to use
- `--model`: Sentence transformer model name
- `--use-apd`: Enable APD partitioning
- `--threshold`: Similarity threshold (0.0-1.0)

### Advanced Usage:
```bash
# Use specific features
python main.py --features title authors venue --threshold 0.9

# Enable APD partitioning
python main.py --use-apd --threshold 0.7

# Use different model
python main.py --model all-mpnet-base-v2
```

## 📈 Performance

**Sample Results** (200 records):
- **Processing Time**: ~6 seconds
- **Throughput**: ~34 records/second
- **Memory Usage**: Minimal (single machine)

**Scalability Notes**:
- Current implementation: Single-machine pandas
- Original Scala version: Distributed Spark
- Suitable for datasets up to ~100K records
- For larger datasets, consider the original Spark implementation

## 🔧 Architecture Comparison

| Component | Original (Scala/Spark) | Python Implementation |
|-----------|------------------------|----------------------|
| Data Processing | Spark DataFrames | Pandas DataFrames |
| Embeddings | DJL + ONNX | Sentence Transformers |
| Similarity Search | FAISS HNSW | Scikit-learn KNN |
| Clustering | GraphX | NetworkX |
| Parallelization | Distributed | Single-machine |

## 🚨 Known Issues

1. **FAISS Compatibility**: FAISS HNSW causes segmentation faults on ARM64
   - **Solution**: Falls back to scikit-learn NearestNeighbors
   - **Impact**: Slower but more reliable

2. **Clustering Sensitivity**: Current parameters may be too permissive
   - **Solution**: Adjust `--threshold` parameter
   - **Recommendation**: Start with higher thresholds (0.9+)

3. **Memory Usage**: All processing in memory
   - **Limitation**: ~100K record limit
   - **Solution**: Use original Spark version for larger datasets

## 🎯 Use Cases

### Academic Papers (Sample Data)
- Identifies duplicate publications
- Handles author name variations
- Venue abbreviation matching
- Year consistency checking

### Business Applications
- Customer record deduplication
- Product catalog matching
- Address standardization
- Contact information consolidation

## 🔄 Extending the Pipeline

### Adding New Similarity Metrics
1. Modify `simple_similarity.py`
2. Add new distance functions
3. Update parameter validation

### Custom Feature Engineering
1. Extend `blocking_key_vectorization.py`
2. Add preprocessing steps
3. Implement domain-specific logic

### Alternative Clustering Methods
1. Replace NetworkX connected components
2. Implement density-based clustering
3. Add hierarchical clustering options

## 📚 Related Work

- **Original Scala Implementation**: Adobe Entity Resolution framework
- **HNSW Algorithm**: Hierarchical Navigable Small World graphs
- **APD**: Adaptive Partitioning for Distributed processing
- **Sentence Transformers**: Semantic text embeddings

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

This implementation is based on the Adobe Entity Resolution pipeline architecture and follows the same design patterns and algorithms while adapting them for Python/pandas usage. 