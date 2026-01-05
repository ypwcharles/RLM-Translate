#!/usr/bin/env python3
"""
EPUB 章节提取脚本 v2

使用 toc.ncx 获取准确的章节标题。
"""

import os
import re
from pathlib import Path
from html.parser import HTMLParser
import xml.etree.ElementTree as ET


class HTMLToMarkdown(HTMLParser):
    """HTML 到 Markdown 转换器"""
    
    def __init__(self):
        super().__init__()
        self.result = []
        self.current_tag = None
        self.skip_tags = {'style', 'script', 'head', 'meta', 'link'}
        self.skip_content = False
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag in self.skip_tags:
            self.skip_content = True
        elif tag == 'p':
            self.result.append('\n\n')
        elif tag == 'br':
            self.result.append('\n')
        elif tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            level = int(tag[1])
            self.result.append('\n\n' + '#' * level + ' ')
        elif tag == 'em' or tag == 'i':
            self.result.append('*')
        elif tag == 'strong' or tag == 'b':
            self.result.append('**')
        elif tag == 'blockquote':
            self.result.append('\n\n> ')
            
    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.skip_content = False
        elif tag == 'em' or tag == 'i':
            self.result.append('*')
        elif tag == 'strong' or tag == 'b':
            self.result.append('**')
        elif tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.result.append('\n\n')
        self.current_tag = None
            
    def handle_data(self, data):
        if self.skip_content:
            return
        text = data.strip()
        if text:
            self.result.append(text)
            
    def get_markdown(self):
        text = ''.join(self.result)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


def html_to_markdown(html_content: str) -> str:
    """将 HTML 转换为 Markdown"""
    parser = HTMLToMarkdown()
    parser.feed(html_content)
    return parser.get_markdown()


def parse_toc(toc_path: Path) -> dict:
    """解析 toc.ncx 获取章节标题映射"""
    tree = ET.parse(toc_path)
    root = tree.getroot()
    
    # 处理命名空间
    ns = {'ncx': 'http://www.daisy.org/z3986/2005/ncx/'}
    
    toc_map = {}
    nav_points = root.findall('.//ncx:navPoint', ns)
    
    for nav in nav_points:
        text_elem = nav.find('ncx:navLabel/ncx:text', ns)
        content_elem = nav.find('ncx:content', ns)
        
        if text_elem is not None and content_elem is not None:
            title = text_elem.text
            src = content_elem.get('src', '')
            
            # 提取文件名 (去除锚点)
            filename = src.split('#')[0].replace('Text/', '')
            toc_map[filename] = title
            
    return toc_map


def count_words(text: str) -> int:
    """统计英文单词数"""
    return len(re.findall(r'[a-zA-Z]+', text))


def main():
    """主函数"""
    base_dir = Path('/Users/peiwenyang/Development/RLM-Translate')
    epub_dir = base_dir / 'epub_temp/OEBPS/Text'
    toc_path = base_dir / 'epub_temp/OEBPS/toc.ncx'
    output_dir = base_dir / 'chapters'
    
    # 清空输出目录
    for f in output_dir.glob('*.md'):
        f.unlink()
    output_dir.mkdir(exist_ok=True)
    
    # 解析 TOC
    toc_map = parse_toc(toc_path)
    
    # 获取所有 HTML 文件
    html_files = sorted(epub_dir.glob('*.html'))
    
    print(f"找到 {len(html_files)} 个 HTML 文件")
    print(f"TOC 包含 {len(toc_map)} 个章节\n")
    print("=" * 90)
    print(f"{'序号':<6}{'章节标题':<45}{'字数':>8}")
    print("=" * 90)
    
    chapters = []
    total_words = 0
    chapter_num = 0
    
    for html_file in html_files:
        filename = html_file.name
        
        # 从 TOC 获取标题
        title = toc_map.get(filename)
        
        # 跳过没有在 TOC 中的文件（如 cover）
        if not title:
            continue
            
        # 跳过一些非章节内容
        if title in ('Title', 'Copyright', 'Contents', 'Dedication'):
            continue
            
        chapter_num += 1
        
        # 读取并转换
        content = html_file.read_text(encoding='utf-8')
        markdown = html_to_markdown(content)
        
        # 统计字数
        word_count = count_words(markdown)
        total_words += word_count
        
        # 生成输出文件名
        safe_title = re.sub(r'[^\w\s\-]', '', title)[:45].strip()
        safe_title = safe_title.replace(' ', '_')
        output_name = f"{chapter_num:02d}_{safe_title}.md"
        output_path = output_dir / output_name
        
        # 添加 Markdown 标题
        if not markdown.startswith('#'):
            markdown = f"# {title}\n\n{markdown}"
        
        # 保存文件
        output_path.write_text(markdown, encoding='utf-8')
        
        chapters.append({
            'num': chapter_num,
            'title': title,
            'output': output_name,
            'words': word_count,
        })
        
        # 显示
        display_title = title[:42] + '...' if len(title) > 45 else title
        print(f"{chapter_num:<6}{display_title:<45}{word_count:>8}")
    
    print("=" * 90)
    print(f"{'总计':<6}{f'{len(chapters)} 个章节':<45}{total_words:>8}")
    print("=" * 90)
    
    # 生成目录文件
    toc_content = "# Tiamat's Wrath - 章节目录\n\n"
    toc_content += f"**作者**: James S.A. Corey\n"
    toc_content += f"**系列**: The Expanse #8\n"
    toc_content += f"**总字数**: {total_words:,} 词\n\n"
    toc_content += "| 序号 | 章节标题 | 字数 |\n"
    toc_content += "|------|----------|------|\n"
    
    for ch in chapters:
        toc_content += f"| {ch['num']:02d} | [{ch['title']}]({ch['output']}) | {ch['words']:,} |\n"
    
    (output_dir / '00_目录.md').write_text(toc_content, encoding='utf-8')
    
    print(f"\n✓ 已生成 {len(chapters)} 个 Markdown 章节到 chapters/ 目录")
    print(f"✓ 目录文件: chapters/00_目录.md")
    print(f"✓ 总字数: {total_words:,} 词")


if __name__ == '__main__':
    main()
