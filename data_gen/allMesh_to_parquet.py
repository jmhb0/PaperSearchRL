
"""
python -m ipdb data_gen/allMesh_to_parquet.py
Needs `data/allMeSH_2022.json` to exist, downloaded from bioASQ from 2022 challenge https://ceur-ws.org/Vol-3180/paper-10.pdf 

After running, then other files can load it with:

from data_gen.allMesh_to_parquet import return_indexer
indexer = return_indexer('data/allMeSH_2022.parquet')

and then:

doc0 = indexer.loc('12345678')
doc1 = indexer.iloc(0)

"""
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import ijson
import json
import pickle
import os
from typing import Union, List, Dict, Any
import numpy as np

class ArrowPMIDIndexer:
    """Fast Arrow-based indexer with both row number and PMID access"""
    
    def __init__(self, arrow_file: str):
        """Load Arrow table and build/load PMID index"""
        self.arrow_file = arrow_file
        self.index_file = arrow_file.replace('.parquet', '_pmid_index.pkl')
        
        print(f"Loading Arrow table from {arrow_file}")
        self.table = pq.read_table(arrow_file)
        
        # Try to load existing index, otherwise build it
        if self._load_pmid_index():
            print("Loaded existing PMID index")
        else:
            print("Building new PMID index...")
            self._build_pmid_index()
            self._save_pmid_index()
    
    def _load_pmid_index(self) -> bool:
        """Try to load existing PMID index from disk"""
        if not os.path.exists(self.index_file):
            return False
        
        try:
            # Check if index file is newer than parquet file
            index_mtime = os.path.getmtime(self.index_file)
            parquet_mtime = os.path.getmtime(self.arrow_file)
            
            if index_mtime < parquet_mtime:
                print("Index file is older than parquet file, rebuilding...")
                return False
            
            with open(self.index_file, 'rb') as f:
                self.pmid_to_row = pickle.load(f)
            return True
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False
    
    def _save_pmid_index(self):
        """Save PMID index to disk"""
        try:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.pmid_to_row, f)
            print(f"Saved PMID index to {self.index_file}")
        except Exception as e:
            print(f"Failed to save index: {e}")
    
    def _build_pmid_index(self):
        """Build hash map from pmid to row number for O(1) lookups"""
        import time
        print("Building PMID index...")
        start_time = time.time()
        
        print("Extracting PMIDs from Arrow table...")
        pmids = self.table['pmid'].to_pylist()
        extract_time = time.time() - start_time
        print(f"PMID extraction took {extract_time:.2f} seconds")
        
        print("Building hash map...")
        map_start = time.time()
        self.pmid_to_row = {str(pmid): i for i, pmid in enumerate(pmids)}
        map_time = time.time() - map_start
        
        total_time = time.time() - start_time
        print(f"Hash map creation took {map_time:.2f} seconds")
        print(f"Total index build time: {total_time:.2f} seconds")
        print(f"Built index for {len(self.pmid_to_row)} PMIDs")
    
    def iloc(self, row_num: Union[int, slice, List[int]]) -> Union[Dict, List[Dict]]:
        """Access by row number (like pandas iloc)"""
        if isinstance(row_num, int):
            # Single row
            row = self.table.slice(row_num, 1)
            return row.to_pylist()[0]
        elif isinstance(row_num, slice):
            # Slice of rows
            start, stop, step = row_num.indices(len(self.table))
            if step == 1:
                # Efficient slice
                rows = self.table.slice(start, stop - start)
                return rows.to_pylist()
            else:
                # Handle step != 1 - process individually to avoid Arrow issues
                indices = list(range(start, stop, step))
                result = []
                for idx in indices:
                    row = self.table.slice(idx, 1).to_pylist()[0]
                    result.append(row)
                return result
        elif isinstance(row_num, list):
            # List of row numbers - process individually to avoid Arrow concatenation issues
            result = []
            for idx in row_num:
                row = self.table.slice(idx, 1).to_pylist()[0]
                result.append(row)
            return result
    
    def loc(self, pmid: Union[str, int, List[Union[str, int]]]) -> Union[Dict, List[Dict]]:
        """Access by PMID (like pandas loc)"""
        if isinstance(pmid, (str, int)):
            # Single PMID
            pmid_str = str(pmid)
            if pmid_str not in self.pmid_to_row:
                raise KeyError(f"PMID {pmid} not found")
            row_num = self.pmid_to_row[pmid_str]
            return self.iloc(row_num)
        elif isinstance(pmid, list):
            # List of PMIDs
            row_nums = []
            for p in pmid:
                p_str = str(p)
                if p_str not in self.pmid_to_row:
                    raise KeyError(f"PMID {p} not found")
                row_nums.append(self.pmid_to_row[p_str])
            return self.iloc(row_nums)
    
    def filter_pmids(self, pmid_list: List[Union[str, int]]) -> List[Dict]:
        """Efficiently filter by multiple PMIDs"""
        pmid_strs = [str(p) for p in pmid_list]
        # Use Arrow's compute functions for efficient filtering
        mask = pc.is_in(self.table['pmid'], value_set=pa.array(pmid_strs))
        filtered_table = self.table.filter(mask)
        return filtered_table.to_pylist()
    
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, key):
        """Support indexer[key] syntax"""
        if isinstance(key, (int, slice, list)) and not isinstance(key, str):
            return self.iloc(key)
        else:
            return self.loc(key)


def convert_json_to_arrow(input_file: str, output_file: str):
    """Convert JSON with 'articles' key to Arrow format"""
    print(f"Converting {input_file} to Arrow format...")
    
    # Try different encodings to handle UTF-8 issues
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    
    articles = None
    for encoding in encodings_to_try:
        try:
            print(f"Trying encoding: {encoding}")
            with open(input_file, 'r', encoding=encoding, errors='replace') as f:
                articles = ijson.items(f, 'articles.item')
                # Test if we can read at least one item
                first_item = next(articles)
                print(f"Successfully opened with {encoding}")
                
                # Reopen and process all items
                with open(input_file, 'r', encoding=encoding, errors='replace') as f:
                    articles = ijson.items(f, 'articles.item')
                    article_list = list(articles)
                break
        except (UnicodeDecodeError, ijson.common.IncompleteJSONError) as e:
            print(f"Failed with {encoding}: {e}")
            continue
    
    if articles is None:
        raise ValueError("Could not decode file with any encoding")
    
    print(f"Loaded {len(article_list)} articles")
    
    table = pa.Table.from_pylist(article_list)
    
    # Write to Arrow/Parquet format
    pq.write_table(table, output_file)
    print(f"Saved to {output_file}")


def convert_json_to_arrow_with_index(input_file: str, output_file: str):
    """Convert JSON to Arrow format and prebuild the PMID index"""
    # First convert to parquet
    convert_json_to_arrow(input_file, output_file)
    
    # Then build and save the index
    print("Building PMID index...")
    indexer = ArrowPMIDIndexer(output_file)
    print("Index building complete!")
    return indexer
    """Create and return an ArrowPMIDIndexer for the given parquet file"""
    return ArrowPMIDIndexer(parquet_file)


def return_indexer(parquet_file: str) -> ArrowPMIDIndexer:
    """Create and return an ArrowPMIDIndexer for the given parquet file"""
    return ArrowPMIDIndexer(parquet_file)


# Usage example
if __name__ == "__main__":
    # Step 1: Convert JSON to Arrow AND build index (one-time setup)
    indexer = convert_json_to_arrow_with_index('data/allMeSH_2022.json', 'data/allMeSH_2022.parquet')
    
    # Step 2: For subsequent uses, just load the indexer (fast!)
    indexer = return_indexer('data/allMeSH_2022.parquet')

    ipdb.set_trace()
    
    # Usage examples:
    
    # Access by row number (like iloc)
    article_0 = indexer.iloc(0)                    # First article
    articles_slice = indexer.iloc(slice(100, 200))  # Articles 100-199
    specific_rows = indexer.iloc([0, 50, 100])     # Specific row numbers
    
    # Access by PMID (like loc)
    article_by_pmid = indexer.loc('34823483')      # Single PMID
    articles_by_pmids = indexer.loc(['34823483', '34821622'])  # Multiple PMIDs
    
    # Alternative syntax
    article = indexer[0]           # Same as iloc(0)
    article = indexer['34823483']  # Same as loc('34823483')
    
    # Efficient filtering
    filtered = indexer.filter_pmids(['34823483', '34821622'])
    
    print(f"Total articles: {len(indexer)}")
    print(f"Sample pmid: {article_by_pmid.get('pmid', 'Not found')}")