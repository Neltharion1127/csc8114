import os
import shutil
from pathlib import Path

def merge_distributed_data():
    """
    Merge split results from vps_results/pi_results and vps_bestweights/pi_bestweights
    into central 'results/' and 'bestweights/' directories for plotting and evaluation.
    """
    base_dir = Path(".")
    
    # Define source and destination pairs
    # (Source Directory, Central Destination)
    mapping = [
        (base_dir / "results" / "vps_results", base_dir / "results"),
        (base_dir / "results" / "pi_results", base_dir / "results"),
        (base_dir / "bestweights" / "vps_bestweights", base_dir / "bestweights"),
        (base_dir / "bestweights" / "pi_bestweights", base_dir / "bestweights"),
    ]

    print("Starting Distributed Results Merge...")

    for src_parent, dest_parent in mapping:
        if not src_parent.exists():
            print(f"Skipping: Source directory {src_parent} does not exist.")
            continue

        print(f"📁 Scanning {src_parent} ...")
        
        # Each child in src_parent is a Session ID (e.g., 2026-04-06_16-40-02)
        for session_dir in src_parent.iterdir():
            if not session_dir.is_dir():
                continue
            
            session_id = session_dir.name
            dest_session_path = dest_parent / session_id
            
            print(f"  Merging Session: {session_id}")
            
            # Recursive sync
            sync_directories(session_dir, dest_session_path)

    print("\n All sessions merged! You can now run 'make evaluate' or 'make plot-session'.")

def sync_directories(src: Path, dest: Path):
    """Deeply merge src into dest without deleting existing files in dest."""
    dest.mkdir(parents=True, exist_ok=True)
    
    for item in src.iterdir():
        if item.is_dir():
            sync_directories(item, dest / item.name)
        else:
            # Use copy2 to preserve metadata (timestamps) if possible
            shutil.copy2(item, dest / item.name)

if __name__ == "__main__":
    merge_distributed_data()
