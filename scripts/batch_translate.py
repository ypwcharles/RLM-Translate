#!/usr/bin/env python3
"""
批量翻译脚本

翻译 data/ 目录下的指定章节。
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Batch translate chapters")
    parser.add_argument("--start", type=int, default=1, help="Start chapter index")
    parser.add_argument("--end", type=int, default=10, help="End chapter index")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="output/batch_test", help="Output directory")
    args = parser.parse_args()

    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有 Markdown 文件并排序
    files = sorted(list(data_dir.glob("*.md")))
    
    # 过滤掉目录文件 (00_目录.md)
    chapter_files = [f for f in files if f.name.split('_')[0].isdigit() and int(f.name.split('_')[0]) > 0]

    count = 0
    for file_path in chapter_files:
        # 提取序号
        try:
            idx = int(file_path.name.split('_')[0])
        except ValueError:
            continue
            
        if args.start <= idx <= args.end:
            print(f"\n[{count+1}] Processing Chapter {idx}: {file_path.name}")
            print("=" * 60)
            
            cmd = [
                sys.executable,
                str(project_root / "scripts/run_translation.py"),
                str(file_path),
                "--output", str(output_dir),
                "--target-audience", "科幻小说爱好者, The Expanse 粉丝",
                "--style-guide", "保持硬科幻风格，专有名词参考现有中文版翻译",
                "--verbose"
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"✓ Completed: {file_path.name}")
                count += 1
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed: {file_path.name} (Exit code: {e.returncode})")
                # 可以选择 continue 或 break
                # break 

    print("\n" + "=" * 60)
    print(f"Batch processing completed. Total translated: {count}")


if __name__ == "__main__":
    main()
