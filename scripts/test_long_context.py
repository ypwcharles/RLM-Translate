#!/usr/bin/env python3
"""
Long Context Test Script for DeepTrans-RLM.

Merges multiple chapters into a single long text to test:
1. Analyzer's ability to handle large context.
2. Chunker's ability to split back correctly.
3. RLM's state management across long sessions.
"""

import sys
import logging
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

from src.graphs.main_graph import create_main_graph

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_chapters(data_dir: Path, chapters: List[str]) -> str:
    """Load and merge chapters."""
    merged_text = ""
    for chapter in chapters:
        file_path = data_dir / chapter
        if not file_path.exists():
            logger.error(f"Chapter not found: {file_path}")
            continue
            
        logger.info(f"Loading {chapter}...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            # Ensure separation between chapters
            merged_text += f"\n\n{text}\n\n"
            
    return merged_text

def main():
    logger.info("Starting Long Context Test...")
    
    data_dir = project_root / "data"
    output_dir = project_root / "output" / "long_context_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select chapters to merge (01-05)
    chapter_files = [
        "01_Prologue_Holden.md",
        "02_Chapter_One_Elvi.md",
        "03_Chapter_Two_Naomi.md",
        "04_Chapter_Three_Alex.md",
        "05_Chapter_Four_Teresa.md"
    ]
    
    # 1. Merge Text
    logger.info("Merging chapters...")
    full_text = load_chapters(data_dir, chapter_files)
    logger.info(f"Total merged length: {len(full_text)} characters")
    
    # 2. Initialize Graph
    # Using defaults which now load config/models.yaml
    graph = create_main_graph()
    
    # 3. Run Translation
    logger.info("Invoking MainGraph with merged text...")
    try:
        result = graph.translate_text(
            text=full_text,
            target_audience="Devoted sci-fi fans",
            style_guide="Maintain the hard sci-fi tone and Belter distinctiveness."
        )
        
        # 4. Save Results
        translations = result.get("translations", [])
        
        # Save individual chunks
        for i, trans in enumerate(translations):
            out_file = output_dir / f"chunk_{i+1}_translated.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(trans)
            logger.info(f"Saved chunk {i+1} to {out_file}")
            
        # Save merged result
        full_out_file = output_dir / "merged_translation.txt"
        with open(full_out_file, "w", encoding="utf-8") as f:
            f.write(result.get("full_translation", ""))
        logger.info(f"Saved full translation to {full_out_file}")
        
        # Save glossary
        import json
        glossary_file = output_dir / "glossary.json"
        with open(glossary_file, "w", encoding="utf-8") as f:
            json.dump(result.get("glossary", {}), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved glossary to {glossary_file}")
        
        logger.info("Long Context Test Completed Successfully.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
