#!/usr/bin/env python3
"""
DeepTrans-RLM 翻译命令行入口

Usage:
    python run_translation.py input.txt --output output/
    python run_translation.py input.txt --debug --verbose
    python run_translation.py input.txt --resume-from checkpoint_name
    python run_translation.py input.txt --chapters 1-10
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 自动加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass  # python-dotenv 未安装则跳过

from src.core.state import create_initial_state
from src.core.chunker import ChunkerConfig
from src.graphs.main_graph import create_main_graph
from src.utils.file_handler import FileHandler
from src.utils.checkpoint import CheckpointManager
from src.utils.debugger import TranslationDebugger


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """设置日志。"""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def parse_chapter_range(chapter_str: str) -> tuple:
    """解析章节范围字符串。
    
    Args:
        chapter_str: 如 "1-10" 或 "5"
        
    Returns:
        (start, end) 元组
    """
    if "-" in chapter_str:
        parts = chapter_str.split("-")
        return int(parts[0]), int(parts[1])
    else:
        n = int(chapter_str)
        return n, n


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="DeepTrans-RLM: Recursive Multi-Agent Translation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.txt                    # Basic translation
  %(prog)s input.txt -o output/         # Specify output directory
  %(prog)s input.txt --debug            # Enable debug mode
  %(prog)s input.txt --chapters 1-10    # Translate specific chapters
  %(prog)s input.txt --resume-from cp1  # Resume from checkpoint
        """
    )
    
    # 必选参数
    parser.add_argument(
        "input",
        help="Input file path (text file to translate)",
    )
    
    # 输出选项
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)",
    )
    
    # 翻译选项
    parser.add_argument(
        "--source-lang",
        default="English",
        help="Source language (default: English)",
    )
    parser.add_argument(
        "--target-lang",
        default="Chinese",
        help="Target language (default: Chinese)",
    )
    parser.add_argument(
        "--style-guide",
        default="",
        help="Translation style guide",
    )
    parser.add_argument(
        "--target-audience",
        default="一般读者",
        help="Target audience (default: 一般读者)",
    )
    
    # 章节选项
    parser.add_argument(
        "--chapters",
        help="Chapter range to translate (e.g., 1-10 or 5)",
    )
    
    # 调试选项
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (save prompts and responses)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (don't actually call LLM)",
    )
    
    # 断点恢复
    parser.add_argument(
        "--resume-from",
        help="Resume from checkpoint name",
    )
    
    # API 选项
    parser.add_argument(
        "--api-key",
        help="Google API key (or set GOOGLE_API_KEY env var)",
    )
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = Path(args.output) / "translation.log" if args.debug else None
    setup_logging(args.verbose, str(log_file) if log_file else None)
    
    logger = logging.getLogger(__name__)
    logger.info("DeepTrans-RLM Translation System Starting...")
    
    # 验证输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
        
    # 读取输入文件
    logger.info(f"Reading input file: {input_path}")
    file_handler = FileHandler()
    raw_text = file_handler.read_text(str(input_path))
    logger.info(f"Input text length: {len(raw_text)} characters")
    
    # 设置输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化调试器
    debugger = None
    if args.debug:
        debug_dir = output_dir / "debug"
        debugger = TranslationDebugger(
            config={"debug_dir": str(debug_dir)},
            enabled=True,
        )
        logger.info(f"Debug mode enabled, outputs to: {debug_dir}")
        
    # 初始化检查点管理器
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_manager = CheckpointManager(str(checkpoint_dir))
    
    # 检查断点恢复
    initial_state = None
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        initial_state = checkpoint_manager.load_checkpoint(args.resume_from)
        if initial_state is None:
            logger.error(f"Checkpoint not found: {args.resume_from}")
            sys.exit(1)
        # 需要重新加载原文
        initial_state["raw_text"] = raw_text
    else:
        # 创建初始状态
        initial_state = create_initial_state(
            raw_text=raw_text,
            style_guide=args.style_guide,
            target_audience=args.target_audience,
            raw_text_path=str(input_path),
        )
        
    # Dry run 模式
    if args.dry_run:
        logger.info("Dry run mode - skipping actual translation")
        from src.core.chunker import TextChunker
        chunker = TextChunker()
        chunks = chunker.plan_chunks(raw_text)
        logger.info(f"Would create {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            logger.info(f"  Chunk {i + 1}: {chunk.get('title', 'Unknown')} ({chunk.get('token_count', 0)} tokens)")
        sys.exit(0)
        
    # 创建主图
    logger.info("Initializing translation graph...")
    try:
        main_graph = create_main_graph(
            api_key=args.api_key,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            debugger=debugger,
        )
    except Exception as e:
        logger.error(f"Failed to initialize translation graph: {e}")
        sys.exit(1)
        
    # 执行翻译
    logger.info("Starting translation...")
    try:
        final_state = main_graph.invoke(initial_state)
    except KeyboardInterrupt:
        logger.warning("Translation interrupted by user")
        # 保存检查点
        checkpoint_manager.save_checkpoint(initial_state, "interrupted")
        logger.info("Checkpoint saved as 'interrupted'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        # 保存检查点
        checkpoint_manager.save_checkpoint(initial_state, "error")
        logger.info("Checkpoint saved as 'error'")
        raise
        
    # 保存结果
    logger.info("Saving translation results...")
    
    translations = final_state.get("completed_translations", [])
    full_translation = "\n\n".join(translations)
    
    output_file = output_dir / f"{input_path.stem}_translated.txt"
    file_handler = FileHandler(str(output_dir))
    file_handler.write_text(output_file.name, full_translation)
    
    logger.info(f"Translation saved to: {output_file}")
    logger.info(f"Total chunks translated: {len(translations)}")
    
    # 保存术语表
    glossary = final_state.get("glossary", {})
    if glossary:
        glossary_file = output_dir / "glossary.json"
        import json
        glossary_file.write_text(
            json.dumps(glossary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Glossary saved to: {glossary_file}")
        
    # 保存最终检查点
    checkpoint_manager.save_checkpoint(final_state, "final")
    
    logger.info("Translation completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
