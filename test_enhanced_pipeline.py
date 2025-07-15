#!/usr/bin/env python3
"""
Test script for enhanced entity resolution pipeline with per-attribute thresholds
"""

import pandas as pd
from main import EntityResolutionPipeline

def test_enhanced_pipeline():
    """Test the pipeline with per-attribute Levenshtein thresholds"""
    
    print("🧪 Testing Enhanced Entity Resolution Pipeline")
    print("=" * 60)
    
    # Configure per-attribute thresholds
    attribute_thresholds = {
        "title": 0.8,      # High threshold for titles (strict matching)
        "authors": 0.7,    # Medium threshold for authors
        "venue": 0.6,      # Lower threshold for venues (more variations)
        "year": 0.9        # Very high threshold for years (should be exact)
    }
    
    print("🎯 Testing with per-attribute thresholds:")
    for attr, threshold in attribute_thresholds.items():
        print(f"   {attr}: {threshold:.2f}")
    
    # Initialize pipeline with Levenshtein similarity
    pipeline = EntityResolutionPipeline(
        selected_features=["title", "authors", "venue", "year"],
        num_blocking_cols=4,
        model_name="all-MiniLM-L6-v2",
        batch_size=256,
        use_apd=False,
        similarity_threshold=0.7,  # Global threshold (fallback)
        attribute_thresholds=attribute_thresholds,
        use_levenshtein=True
    )
    
    # Run the pipeline
    try:
        results = pipeline.run_full_pipeline("data/sample_data.csv")
        
        # Analyze results
        if results is not None:
            cluster_stats = results['cluster_id'].value_counts()
            multi_record_clusters = cluster_stats[cluster_stats > 1]
            
            print(f"\n📊 Enhanced Pipeline Results:")
            print(f"   Total records: {len(results)}")
            print(f"   Total clusters: {len(cluster_stats)}")
            print(f"   Multi-record clusters: {len(multi_record_clusters)}")
            print(f"   Largest cluster: {cluster_stats.max() if len(cluster_stats) > 0 else 0}")
            print(f"   Records in multi-record clusters: {multi_record_clusters.sum() if len(multi_record_clusters) > 0 else 0}")
            
            # Show some example clusters
            if len(multi_record_clusters) > 0:
                print(f"\n🔍 Example Multi-Record Clusters:")
                for cluster_id in multi_record_clusters.head(3).index:
                    cluster_records = results[results['cluster_id'] == cluster_id]
                    print(f"\n   Cluster {cluster_id} ({len(cluster_records)} records):")
                    for _, record in cluster_records.iterrows():
                        title = record.get('title', 'N/A')[:50]
                        authors = record.get('authors', 'N/A')[:30]
                        print(f"     - {title}... by {authors}...")
            
            print(f"\n✅ Enhanced pipeline test completed successfully!")
            return True
        else:
            print("❌ Pipeline returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison():
    """Compare old vs new pipeline performance"""
    
    print("\n🔬 Comparing Old vs Enhanced Pipeline")
    print("=" * 50)
    
    # Test with old method (no Levenshtein)
    print("🔄 Testing without Levenshtein similarity...")
    pipeline_old = EntityResolutionPipeline(
        selected_features=["title", "authors", "venue", "year"],
        use_levenshtein=False,
        similarity_threshold=0.6
    )
    
    try:
        results_old = pipeline_old.run_full_pipeline("data/sample_data.csv")
        clusters_old = results_old['cluster_id'].value_counts() if results_old is not None else pd.Series()
        multi_old = clusters_old[clusters_old > 1] if len(clusters_old) > 0 else pd.Series()
        
        print(f"   Old method: {len(multi_old)} multi-record clusters")
    except Exception as e:
        print(f"   Old method failed: {e}")
        clusters_old = pd.Series()
        multi_old = pd.Series()
    
    # Test with new method (Levenshtein)
    print("🔄 Testing with Levenshtein similarity...")
    attribute_thresholds = {
        "title": 0.8,
        "authors": 0.7,
        "venue": 0.6,
        "year": 0.9
    }
    
    pipeline_new = EntityResolutionPipeline(
        selected_features=["title", "authors", "venue", "year"],
        use_levenshtein=True,
        attribute_thresholds=attribute_thresholds
    )
    
    try:
        results_new = pipeline_new.run_full_pipeline("data/sample_data.csv")
        clusters_new = results_new['cluster_id'].value_counts() if results_new is not None else pd.Series()
        multi_new = clusters_new[clusters_new > 1] if len(clusters_new) > 0 else pd.Series()
        
        print(f"   New method: {len(multi_new)} multi-record clusters")
    except Exception as e:
        print(f"   New method failed: {e}")
        clusters_new = pd.Series()
        multi_new = pd.Series()
    
    # Comparison
    print(f"\n📈 Comparison Results:")
    print(f"   Without Levenshtein: {len(multi_old)} multi-record clusters")
    print(f"   With Levenshtein:    {len(multi_new)} multi-record clusters")
    
    if len(multi_new) > 0 and len(multi_old) > 0:
        improvement = len(multi_new) - len(multi_old)
        print(f"   Difference: {improvement:+d} clusters")
        if improvement > 0:
            print(f"   ✅ Enhanced method found {improvement} more duplicate groups!")
        elif improvement < 0:
            print(f"   🎯 Enhanced method is more precise ({-improvement} fewer clusters)")
        else:
            print(f"   ➡️ Both methods found the same number of clusters")

if __name__ == "__main__":
    # Run tests
    success = test_enhanced_pipeline()
    
    if success:
        test_comparison()
    
    print(f"\n🎉 Testing completed!") 