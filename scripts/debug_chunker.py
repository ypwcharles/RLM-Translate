
from pathlib import Path
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_chapters(data_dir: Path, chapters: list) -> str:
    merged_text = ""
    for chapter in chapters:
        file_path = data_dir / chapter
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            merged_text += f"\n\n{text}\n\n"
    return merged_text

def test_chunking():
    project_root = Path.cwd()
    data_dir = project_root / "data"
    
    chapter_files = [
        "01_Prologue_Holden.md",
        "02_Chapter_One_Elvi.md",
        "03_Chapter_Two_Naomi.md",
        "04_Chapter_Three_Alex.md",
        "05_Chapter_Four_Teresa.md"
    ]
    
    # Check if files exist
    for f in chapter_files:
        if not (data_dir / f).exists():
             logger.error(f"File not found: {f}")
             return

    full_text = load_chapters(data_dir, chapter_files)
    logger.info(f"Full text length: {len(full_text)}")
    
    # patterns from ChunkerConfig
    chapter_patterns = [
        r'^#*\s*第[一二三四五六七八九十百千万\d]+章.*$',
        r'^#*\s*Chapter\s+.*$',
        r'^#*\s*CHAPTER\s+.*$',
        r'^#*\s*Prologue.*$',
        r'^#*\s*Epilogue.*$',
    ]
    
    combined_pattern = '|'.join(f'({p})' for p in chapter_patterns)
    logger.info(f"Pattern: {combined_pattern}")
    
    matches = list(re.finditer(combined_pattern, full_text, re.MULTILINE))
    logger.info(f"Found {len(matches)} matches")
    
    for i, match in enumerate(matches):
        logger.info(f"Match {i}: '{match.group().strip()}' at {match.start()}")
        
    # Simulate logic
    chapters = []
    if not matches:
        logger.info("No matches found.")
    else:
        if matches[0].start() > 0:
             logger.info(f"Pre-chapter content: 0 to {matches[0].start()}")
             
        for i, match in enumerate(matches):
            start = match.start()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(full_text)
            
            logger.info(f"Chunk {i}: {start} to {end} (Length: {end - start})")
            logger.info(f"Head: {full_text[start:start+50]}...")

if __name__ == "__main__":
    test_chunking()
