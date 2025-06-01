import json
import os
from typing import Dict, Any
from tqdm import tqdm
import chardet


def detect_encoding(file_path: str, sample_size: int = 10000) -> str:
    """Detect the encoding of a file by reading a sample."""
    with open(file_path, 'rb') as f:
        sample = f.read(sample_size)
        result = chardet.detect(sample)
        return result['encoding']


def create_whole_corpus():
    """
    Creates a JSONL corpus file and a PMID lookup mapping from the allMeSH_2022.json file.
    
    This function:
    1. Checks if data/allMeSH_2022.json exists
    2. Iterates over the JSON file to create pubmed.jsonl with sequential IDs and formatted contents
    3. Creates data/pubmed-lookupPMID.json mapping PMIDs to sequential IDs
    """
    # Check if the source file exists
    source_file = "data/allMeSH_2022.json"
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file {source_file} does not exist")

    print(f"Processing {source_file}...")

    # Detect file encoding
    print("Detecting file encoding...")
    detected_encoding = detect_encoding(source_file)
    print(f"Detected encoding: {detected_encoding}")

    # Output files
    corpus_file = "data/pubmed.jsonl"
    lookup_file = "data/pubmed-lookupPMID.json"

    # Create lookup mapping
    pmid_to_id = {}

    # Process the JSON file with detected encoding
    encodings_to_try = [
        detected_encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1'
    ]

    data = None
    for encoding in encodings_to_try:
        if encoding is None:
            continue
        try:
            print(f"Trying to read file with encoding: {encoding}")
            with open(source_file, 'r', encoding=encoding,
                      errors='replace') as f:
                data = json.load(f)
            print(f"Successfully loaded file with encoding: {encoding}")
            break
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"Failed with encoding {encoding}: {e}")
            continue

    if data is None:
        raise ValueError(
            "Could not read the file with any of the attempted encodings")

    print(f"Loaded {len(data)} documents from {source_file}")

    # Write JSONL corpus file
    with open(corpus_file, 'w', encoding='utf-8') as corpus_f:
        for idx, document in enumerate(
                tqdm(data['articles'], desc="Processing documents")):
            # Extract title and abstract
            title = document.get("title", "")
            abstract = document.get("abstractText", "")
            pmid = document.get("pmid", "")

            # Create formatted contents
            contents = f"TITLE: {title}. ABSTRACT: {abstract}"

            # Create corpus entry
            corpus_entry = {"id": idx, "contents": contents}

            # Write to JSONL file
            corpus_f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')

            # Store PMID mapping
            if pmid:
                pmid_to_id[str(pmid)] = idx

    # Save PMID lookup mapping
    with open(lookup_file, 'w', encoding='utf-8') as lookup_f:
        json.dump(pmid_to_id, lookup_f, indent=2, ensure_ascii=False)

    print(f"Created corpus file: {corpus_file} with {len(data)} documents")
    print(
        f"Created lookup file: {lookup_file} with {len(pmid_to_id)} PMID mappings"
    )


if __name__ == "__main__":
    create_whole_corpus()
