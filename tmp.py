import lmdb
import os
import json

def salvage_lmdb_cache(corrupted_path, salvaged_path):
    """
    Attempts to salvage data from a corrupted LMDB database.

    It reads entries from the corrupted database in read-only mode and
    writes the successfully read entries to a new database.
    """
    print(f"Attempting to salvage cache from: {corrupted_path}")
    print(f"Salvaged data will be written to: {salvaged_path}")

    # Ensure the parent directory for the salvaged path exists
    os.makedirs(os.path.dirname(salvaged_path), exist_ok=True)
    
    # Check if corrupted path exists
    if not os.path.exists(corrupted_path):
        print(f"Error: Corrupted cache path does not exist: {corrupted_path}")
        return

    salvaged_count = 0
    error_count = 0

    try:
        # Open the corrupted environment in read-only mode.
        # lock=False is advisable when dealing with potentially corrupted dbs.
        corrupted_env = lmdb.open(corrupted_path, readonly=True, lock=False)

        # Open a new environment for the salvaged data
        salvaged_env = lmdb.open(salvaged_path, map_size=1024 * 1024 * 1024) # 1GB

        with corrupted_env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    # Your cache values are JSON, so let's try to parse them
                    # to ensure they are not corrupted.
                    json.loads(value.decode())

                    # If valid, write to the new database
                    with salvaged_env.begin(write=True) as salvaged_txn:
                        salvaged_txn.put(key, value)
                    salvaged_count += 1
                except Exception:
                    # This entry is likely corrupted or invalid
                    error_count += 1
        
        print("\nSalvage operation complete.")
        print(f"  Successfully salvaged entries: {salvaged_count}")
        print(f"  Skipped corrupted entries: {error_count}")

    except lmdb.Error as e:
        print(f"\nAn LMDB error occurred during salvage: {e}")
        print("This may indicate severe corruption, and salvage might not be possible.")
    finally:
        if 'corrupted_env' in locals():
            corrupted_env.close()
        if 'salvaged_env' in locals():
            salvaged_env.close()

    if salvaged_count > 0:
        print("\nTo use the salvaged cache:")
        print(f"1. Back up and remove the old cache: mv {corrupted_path} {corrupted_path}.bak")
        print(f"2. Rename the new cache: mv {salvaged_path} {corrupted_path}")


if __name__ == "__main__":
    # From your api.py, the cache path is likely './cache/llm_cache.lmdb'
    CORRUPTED_CACHE_PATH = "./cache/llm_cache.lmdb"
    SALVAGED_CACHE_PATH = "./cache/llm_cache.lmdb.salvaged"
    salvage_lmdb_cache(CORRUPTED_CACHE_PATH, SALVAGED_CACHE_PATH)
