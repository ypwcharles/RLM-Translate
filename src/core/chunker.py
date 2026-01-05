"""
TextChunker - 文本切分逻辑

实现 PRD 中的两级切分策略：
- Level 1: 识别章节头
- Level 2: 若单章超过阈值，则按段落进行二次切分
"""

import re
from typing import List, Optional
from dataclasses import dataclass

from .state import ChunkInfo


@dataclass
class ChunkerConfig:
    """切分器配置"""
    max_source_tokens: int = 200000  # 源文本块 Token 上限
    chapter_patterns: Optional[List[str]] = None
    paragraph_separator: str = "\n\n"
    
    def __post_init__(self):
        if self.chapter_patterns is None:
            self.chapter_patterns = [
                r'^#*\s*第[一二三四五六七八九十百千万\d]+章.*$',
                r'^#*\s*Chapter\s+.*$',  # 匹配 "Chapter 1" 或 "Chapter One"
                r'^#*\s*CHAPTER\s+.*$',
                r'^#*\s*Prologue.*$',
                r'^#*\s*Epilogue.*$',
            ]


class TextChunker:
    """文本切分器。
    
    实现 RLM 风格的两级切分策略：
    1. Level 1: 基于章节标题识别
    2. Level 2: 若单章超过 Token 阈值，按段落二次切分
    
    Attributes:
        config: 切分器配置
        tokenizer: Token 计数函数
        
    Example:
        >>> chunker = TextChunker(config, tokenizer=count_tokens)
        >>> chunks = chunker.plan_chunks(full_text)
    """
    
    def __init__(
        self, 
        config: Optional[ChunkerConfig] = None,
        tokenizer: Optional[callable] = None,
    ):
        """初始化切分器。
        
        Args:
            config: 切分器配置
            tokenizer: Token 计数函数，签名 (str) -> int
        """
        self.config = config or ChunkerConfig()
        self.tokenizer = tokenizer or self._simple_tokenizer
        
    def _simple_tokenizer(self, text: str) -> int:
        """简单的 Token 估算。
        
        实际使用时应替换为 tiktoken 等精确计数器。
        """
        # 英文按空格分词 + 中文/日文按字符
        word_count = len(text.split())
        cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff]', text))
        return word_count + cjk_count
        
    def plan_chunks(self, full_text: str) -> List[ChunkInfo]:
        """规划文本切分。
        
        两级切分策略：
        1. Level 1: 基于章节模式识别边界
        2. Level 2: 对超限章节按段落二次切分
        
        Args:
            full_text: 完整原文
            
        Returns:
            ChunkInfo 列表
        """
        # Level 1: 章节切分
        chapters = self._split_by_chapters(full_text)
        
        # Level 2: 对超限章节进行二次切分
        chunks = []
        for title, start, end in chapters:
            chapter_text = full_text[start:end]
            token_count = self.tokenizer(chapter_text)
            
            if token_count <= self.config.max_source_tokens:
                # 章节未超限，直接作为一个块
                chunks.append(ChunkInfo(
                    title=title,
                    start=start,
                    end=end,
                    token_count=token_count,
                ))
            else:
                # 章节超限，按段落二次切分
                sub_chunks = self._split_by_paragraphs(
                    chapter_text, 
                    title, 
                    start,
                )
                chunks.extend(sub_chunks)
                
        return chunks
    
    def _split_by_chapters(
        self, 
        full_text: str,
    ) -> List[tuple]:
        """基于章节模式切分。
        
        Args:
            full_text: 完整原文
            
        Returns:
            章节列表 [(title, start, end), ...]
        """
        # 合并所有章节模式
        combined_pattern = '|'.join(
            f'({p})' for p in self.config.chapter_patterns
        )
        
        chapters = []
        matches = list(re.finditer(combined_pattern, full_text, re.MULTILINE))
        
        if not matches:
            # 无章节标记，整个文本作为一个块
            return [("Full Text", 0, len(full_text))]
            
        # 处理第一章之前的内容
        if matches[0].start() > 0:
            chapters.append((
                "Prologue",
                0,
                matches[0].start(),
            ))
            
        # 处理各章节
        for i, match in enumerate(matches):
            title = match.group().strip()
            start = match.start()
            
            # 结束位置：下一章开始或文本结尾
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(full_text)
                
            chapters.append((title, start, end))
            
        return chapters
    
    def _split_by_paragraphs(
        self, 
        text: str, 
        base_title: str,
        base_offset: int,
    ) -> List[ChunkInfo]:
        """按段落切分超限章节。
        
        Args:
            text: 章节文本
            base_title: 章节标题
            base_offset: 章节在原文中的起始偏移
            
        Returns:
            ChunkInfo 列表
        """
        paragraphs = text.split(self.config.paragraph_separator)
        chunks = []
        
        current_chunk_text = ""
        current_chunk_start = base_offset
        chunk_index = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_with_sep = paragraph
            if i < len(paragraphs) - 1:
                paragraph_with_sep += self.config.paragraph_separator
                
            test_text = current_chunk_text + paragraph_with_sep
            test_tokens = self.tokenizer(test_text)
            
            if test_tokens <= self.config.max_source_tokens:
                # 未超限，继续累积
                current_chunk_text = test_text
            else:
                # 超限，保存当前块并开始新块
                if current_chunk_text:
                    chunk_index += 1
                    chunks.append(ChunkInfo(
                        title=f"{base_title} (Part {chunk_index})",
                        start=current_chunk_start,
                        end=current_chunk_start + len(current_chunk_text),
                        token_count=self.tokenizer(current_chunk_text),
                    ))
                    current_chunk_start += len(current_chunk_text)
                    
                current_chunk_text = paragraph_with_sep
                
        # 保存最后一个块
        if current_chunk_text:
            chunk_index += 1
            title = base_title if chunk_index == 1 else f"{base_title} (Part {chunk_index})"
            chunks.append(ChunkInfo(
                title=title,
                start=current_chunk_start,
                end=current_chunk_start + len(current_chunk_text),
                token_count=self.tokenizer(current_chunk_text),
            ))
            
        return chunks
    
    def estimate_total_tokens(self, full_text: str) -> int:
        """估算全文 Token 数。
        
        Args:
            full_text: 完整原文
            
        Returns:
            估算的 Token 数
        """
        return self.tokenizer(full_text)
    
    def get_chunk_text(self, full_text: str, chunk: ChunkInfo) -> str:
        """获取块文本内容。
        
        Args:
            full_text: 完整原文
            chunk: 块信息
            
        Returns:
            块文本内容
        """
        return full_text[chunk["start"]:chunk["end"]]
