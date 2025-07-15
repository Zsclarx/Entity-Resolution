#!/usr/bin/env python3
"""
Generate comprehensive sample data for entity resolution pipeline testing.
Creates 300 records with 20+ columns including realistic variations and duplicates.
"""

import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
import re

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Base data for realistic variations
RESEARCH_AREAS = [
    "Machine Learning", "Deep Learning", "Computer Vision", "Natural Language Processing",
    "Artificial Intelligence", "Data Mining", "Neural Networks", "Reinforcement Learning",
    "Information Retrieval", "Robotics", "Bioinformatics", "Computational Biology",
    "Database Systems", "Software Engineering", "Human-Computer Interaction", "Security",
    "Distributed Systems", "Algorithms", "Theory", "Graphics", "Networks"
]

VENUES = {
    "NIPS": ["NIPS", "NeurIPS", "Neural Information Processing Systems"],
    "ICML": ["ICML", "International Conference on Machine Learning"],
    "ICLR": ["ICLR", "International Conference on Learning Representations"],
    "AAAI": ["AAAI", "Association for the Advancement of Artificial Intelligence"],
    "IJCAI": ["IJCAI", "International Joint Conference on Artificial Intelligence"],
    "ACL": ["ACL", "Association for Computational Linguistics"],
    "EMNLP": ["EMNLP", "Empirical Methods in Natural Language Processing"],
    "ICCV": ["ICCV", "International Conference on Computer Vision"],
    "CVPR": ["CVPR", "Computer Vision and Pattern Recognition"],
    "ECCV": ["ECCV", "European Conference on Computer Vision"],
    "SIGIR": ["SIGIR", "Information Retrieval"],
    "WWW": ["WWW", "World Wide Web", "Web Conference"],
    "KDD": ["KDD", "Knowledge Discovery and Data Mining"],
    "ICDE": ["ICDE", "International Conference on Data Engineering"],
    "VLDB": ["VLDB", "Very Large Data Bases"]
}

INSTITUTIONS = [
    "MIT", "Stanford University", "Harvard University", "UC Berkeley", "CMU",
    "Google Research", "Microsoft Research", "OpenAI", "DeepMind", "Facebook AI",
    "IBM Research", "Amazon Research", "Apple", "NVIDIA", "Tesla",
    "University of Washington", "NYU", "Princeton", "Yale", "Columbia",
    "University of Toronto", "McGill University", "ETH Zurich", "University of Cambridge",
    "University of Oxford", "Max Planck Institute", "INRIA", "University of Tokyo"
]

COUNTRIES = [
    "USA", "Canada", "UK", "Germany", "France", "Japan", "China", "South Korea",
    "Australia", "Netherlands", "Switzerland", "Sweden", "Italy", "Spain", "Israel"
]

PUBLISHERS = [
    "IEEE", "ACM", "Springer", "Elsevier", "Nature", "Science", "AAAI Press",
    "MIT Press", "Cambridge University Press", "Oxford University Press", "Wiley",
    "Taylor & Francis", "PMLR", "arXiv", "bioRxiv"
]

FUNDING_AGENCIES = [
    "NSF", "NIH", "DARPA", "DOE", "NASA", "NSERC", "EPSRC", "DFG", "ANR",
    "JST", "NSFC", "ERC", "Horizon 2020", "Google", "Microsoft", "Amazon"
]

# Author name variations
AUTHOR_NAMES = [
    ("John", "Smith"), ("Jane", "Doe"), ("Michael", "Johnson"), ("Sarah", "Williams"),
    ("David", "Brown"), ("Emily", "Davis"), ("Robert", "Miller"), ("Lisa", "Wilson"),
    ("James", "Moore"), ("Maria", "Taylor"), ("Christopher", "Anderson"), ("Jennifer", "Thomas"),
    ("Matthew", "Jackson"), ("Ashley", "White"), ("Daniel", "Harris"), ("Jessica", "Martin"),
    ("Andrew", "Thompson"), ("Amanda", "Garcia"), ("Joshua", "Martinez"), ("Stephanie", "Robinson"),
    ("Kevin", "Clark"), ("Michelle", "Rodriguez"), ("Brian", "Lewis"), ("Nicole", "Lee"),
    ("William", "Walker"), ("Kimberly", "Hall"), ("Steven", "Allen"), ("Amy", "Young"),
    ("Joseph", "Hernandez"), ("Angela", "King"), ("Thomas", "Wright"), ("Brenda", "Lopez"),
    ("Charles", "Hill"), ("Emma", "Scott"), ("Anthony", "Green"), ("Olivia", "Adams"),
    ("Mark", "Baker"), ("Sophia", "Gonzalez"), ("Donald", "Nelson"), ("Isabella", "Carter"),
    ("Paul", "Mitchell"), ("Mia", "Perez"), ("Kenneth", "Roberts"), ("Charlotte", "Turner"),
    ("Jason", "Phillips"), ("Amelia", "Campbell"), ("Ryan", "Parker"), ("Harper", "Evans"),
    ("Gary", "Edwards"), ("Evelyn", "Collins"), ("Nicholas", "Stewart"), ("Abigail", "Sanchez"),
    ("Eric", "Morris"), ("Madison", "Rogers"), ("Stephen", "Reed"), ("Ella", "Cook"),
    ("Jonathan", "Morgan"), ("Scarlett", "Bailey"), ("Larry", "Cooper"), ("Grace", "Richardson")
]

def generate_author_variations(first_name, last_name):
    """Generate realistic author name variations."""
    variations = []
    
    # Full name
    variations.append(f"{first_name} {last_name}")
    
    # First initial + last name
    variations.append(f"{first_name[0]}. {last_name}")
    
    # First name + last initial
    variations.append(f"{first_name} {last_name[0]}.")
    
    # Both initials
    variations.append(f"{first_name[0]}. {last_name[0]}.")
    
    # Last name, first name
    variations.append(f"{last_name}, {first_name}")
    
    # Last name, first initial
    variations.append(f"{last_name}, {first_name[0]}.")
    
    # With middle initial
    middle_initial = chr(ord('A') + random.randint(0, 25))
    variations.append(f"{first_name} {middle_initial}. {last_name}")
    variations.append(f"{first_name[0]}. {middle_initial}. {last_name}")
    
    return variations

def generate_title_variations(base_title):
    """Generate realistic title variations."""
    variations = [base_title]
    
    # Common abbreviations
    abbreviations = {
        "Natural Language Processing": "NLP",
        "Computer Vision": "CV",
        "Machine Learning": "ML",
        "Deep Learning": "DL",
        "Artificial Intelligence": "AI",
        "Neural Networks": "NN",
        "Reinforcement Learning": "RL",
        "Information Retrieval": "IR"
    }
    
    # Apply abbreviations
    for full, abbr in abbreviations.items():
        if full in base_title:
            variations.append(base_title.replace(full, abbr))
    
    # Add/remove articles
    if base_title.startswith("A "):
        variations.append(base_title[2:])
    elif base_title.startswith("An "):
        variations.append(base_title[3:])
    elif base_title.startswith("The "):
        variations.append(base_title[4:])
    else:
        variations.append(f"A {base_title}")
    
    # Add common prefixes/suffixes
    variations.append(f"{base_title}: A Survey")
    variations.append(f"{base_title} Revisited")
    variations.append(f"On {base_title}")
    variations.append(f"Towards {base_title}")
    
    return variations

def generate_sample_data():
    """Generate comprehensive sample data with 300 records and 20+ columns."""
    
    records = []
    
    # Generate base papers with intended duplicates
    base_papers = []
    
    # Create 100 unique base papers
    for i in range(100):
        area = random.choice(RESEARCH_AREAS)
        venue_key = random.choice(list(VENUES.keys()))
        venue_variations = VENUES[venue_key]
        
        # Generate base paper
        base_paper = {
            "title": f"{area} for {random.choice(['Classification', 'Prediction', 'Analysis', 'Optimization', 'Enhancement'])}",
            "authors": random.sample(AUTHOR_NAMES, random.randint(1, 4)),
            "venue": venue_key,
            "year": random.randint(2015, 2024),
            "pages": f"{random.randint(1, 500)}-{random.randint(501, 1000)}",
            "volume": random.randint(1, 50),
            "issue": random.randint(1, 12),
            "doi": f"10.{random.randint(1000, 9999)}/{random.randint(1000000, 9999999)}",
            "isbn": f"978-{random.randint(0, 9)}-{random.randint(100, 999)}-{random.randint(10000, 99999)}-{random.randint(0, 9)}",
            "publisher": random.choice(PUBLISHERS),
            "institution": random.choice(INSTITUTIONS),
            "country": random.choice(COUNTRIES),
            "language": random.choice(["English", "English", "English", "English", "Chinese", "German", "French"]),
            "keywords": ", ".join(random.sample(["machine learning", "deep learning", "neural networks", "classification", "clustering", "optimization", "prediction", "analysis", "computer vision", "nlp", "ai", "data mining", "algorithms", "statistics"], random.randint(3, 8))),
            "abstract": f"This paper presents a novel approach to {area.lower()} using advanced techniques. Our method shows significant improvements over existing approaches.",
            "funding": random.choice(FUNDING_AGENCIES),
            "citation_count": random.randint(0, 1000),
            "h_index": random.randint(5, 100),
            "impact_factor": round(random.uniform(0.5, 10.0), 2),
            "open_access": random.choice([True, False]),
            "conference_type": random.choice(["International", "National", "Regional", "Workshop"]),
            "peer_reviewed": random.choice([True, True, True, False])  # Mostly peer-reviewed
        }
        
        base_papers.append(base_paper)
    
    # Now generate 300 records with variations
    record_id = 1
    
    for base_paper in base_papers:
        # Generate 2-4 variations of each base paper (some will be exact duplicates)
        num_variations = random.randint(2, 4)
        
        for var_idx in range(num_variations):
            if record_id > 300:
                break
                
            record = {
                "id": record_id,
                "row_id": record_id  # Add row_id for pipeline compatibility
            }
            
            # Title variations
            title_variations = generate_title_variations(base_paper["title"])
            record["title"] = random.choice(title_variations)
            
            # Author variations
            author_list = []
            for first, last in base_paper["authors"]:
                variations = generate_author_variations(first, last)
                author_list.append(random.choice(variations))
            record["authors"] = "; ".join(author_list)
            
            # Venue variations
            venue_variations = VENUES[base_paper["venue"]]
            record["venue"] = random.choice(venue_variations)
            
            # Year variations (sometimes off by 1)
            if random.random() < 0.1:  # 10% chance of year variation
                record["year"] = base_paper["year"] + random.choice([-1, 1])
            else:
                record["year"] = base_paper["year"]
            
            # Pages variations
            if random.random() < 0.2:  # 20% chance of page variation
                start_page = random.randint(1, 500)
                end_page = start_page + random.randint(10, 50)
                record["pages"] = f"{start_page}-{end_page}"
            else:
                record["pages"] = base_paper["pages"]
            
            # Volume and issue variations
            record["volume"] = base_paper["volume"]
            record["issue"] = base_paper["issue"]
            
            # DOI variations (sometimes missing or different)
            if random.random() < 0.1:  # 10% chance of different DOI
                record["doi"] = f"10.{random.randint(1000, 9999)}/{random.randint(1000000, 9999999)}"
            elif random.random() < 0.05:  # 5% chance of missing DOI
                record["doi"] = ""
            else:
                record["doi"] = base_paper["doi"]
            
            # ISBN variations
            record["isbn"] = base_paper["isbn"]
            
            # Publisher variations
            record["publisher"] = base_paper["publisher"]
            
            # Institution variations (sometimes abbreviated)
            institution = base_paper["institution"]
            if "University" in institution and random.random() < 0.2:
                institution = institution.replace("University", "Univ.")
            record["institution"] = institution
            
            # Country variations
            record["country"] = base_paper["country"]
            
            # Language variations
            record["language"] = base_paper["language"]
            
            # Keywords variations (sometimes reordered or with synonyms)
            keywords = base_paper["keywords"].split(", ")
            if random.random() < 0.3:  # 30% chance of reordering
                keywords = random.sample(keywords, len(keywords))
            record["keywords"] = ", ".join(keywords)
            
            # Abstract variations (sometimes truncated or with minor changes)
            abstract = base_paper["abstract"]
            if random.random() < 0.2:  # 20% chance of variation
                abstract = abstract.replace("novel approach", "new method")
                abstract = abstract.replace("significant improvements", "better performance")
            record["abstract"] = abstract
            
            # Funding variations
            record["funding"] = base_paper["funding"]
            
            # Citation count variations (realistic spread)
            base_citations = base_paper["citation_count"]
            record["citation_count"] = max(0, base_citations + random.randint(-50, 50))
            
            # H-index variations
            record["h_index"] = base_paper["h_index"]
            
            # Impact factor variations
            record["impact_factor"] = base_paper["impact_factor"]
            
            # Open access variations
            record["open_access"] = base_paper["open_access"]
            
            # Conference type variations
            record["conference_type"] = base_paper["conference_type"]
            
            # Peer reviewed variations
            record["peer_reviewed"] = base_paper["peer_reviewed"]
            
            records.append(record)
            record_id += 1
            
            if record_id > 300:
                break
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Ensure we have exactly 300 records
    if len(df) > 300:
        df = df.head(300)
    
    # Add some completely unique records to fill up to 300 if needed
    while len(df) < 300:
        unique_record = {
            "id": len(df) + 1,
            "row_id": len(df) + 1,
            "title": f"Unique Research on {random.choice(RESEARCH_AREAS)} {random.randint(1000, 9999)}",
            "authors": f"{random.choice(AUTHOR_NAMES)[0]} {random.choice(AUTHOR_NAMES)[1]}",
            "venue": random.choice(list(VENUES.keys())),
            "year": random.randint(2015, 2024),
            "pages": f"{random.randint(1, 500)}-{random.randint(501, 1000)}",
            "volume": random.randint(1, 50),
            "issue": random.randint(1, 12),
            "doi": f"10.{random.randint(1000, 9999)}/{random.randint(1000000, 9999999)}",
            "isbn": f"978-{random.randint(0, 9)}-{random.randint(100, 999)}-{random.randint(10000, 99999)}-{random.randint(0, 9)}",
            "publisher": random.choice(PUBLISHERS),
            "institution": random.choice(INSTITUTIONS),
            "country": random.choice(COUNTRIES),
            "language": "English",
            "keywords": ", ".join(random.sample(["machine learning", "deep learning", "neural networks", "classification", "clustering", "optimization", "prediction", "analysis", "computer vision", "nlp", "ai", "data mining", "algorithms", "statistics"], random.randint(3, 8))),
            "abstract": f"This unique paper explores novel aspects of research methodology and applications.",
            "funding": random.choice(FUNDING_AGENCIES),
            "citation_count": random.randint(0, 100),
            "h_index": random.randint(5, 50),
            "impact_factor": round(random.uniform(0.5, 5.0), 2),
            "open_access": random.choice([True, False]),
            "conference_type": random.choice(["International", "National", "Regional", "Workshop"]),
            "peer_reviewed": True
        }
        df = pd.concat([df, pd.DataFrame([unique_record])], ignore_index=True)
    
    return df

if __name__ == "__main__":
    # Generate the sample data
    print("Generating comprehensive sample data...")
    df = generate_sample_data()
    
    # Save to CSV
    output_path = "data/sample_data.csv"
    df.to_csv(output_path, index=False)
    
    # Print statistics
    print(f"\nGenerated {len(df)} records with {len(df.columns)} columns")
    print(f"Saved to: {output_path}")
    print(f"\nColumns: {', '.join(df.columns)}")
    
    # Show some sample records
    print(f"\nFirst 5 records:")
    print(df.head().to_string())
    
    # Show data types
    print(f"\nData types:")
    print(df.dtypes)
    
    # Show some statistics
    print(f"\nStatistics:")
    print(f"- Unique titles: {df['title'].nunique()}")
    print(f"- Unique authors: {df['authors'].nunique()}")
    print(f"- Unique venues: {df['venue'].nunique()}")
    print(f"- Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"- Citation count range: {df['citation_count'].min()} - {df['citation_count'].max()}") 