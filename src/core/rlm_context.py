"""
RLMContext - RLM 风格的上下文管理器

基于 MIT RLM 论文核心设计原则：
- Prompt-as-Variable: Prompt 作为 REPL 环境中的变量
- Selective Viewing: 按需查看上下文片段
- Recursive Sub-calling: 递归分解复杂任务
- Code-driven Filtering: 利用模型先验知识过滤信息
"""

import re
from typing import List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """搜索结果"""
    start: int
    end: int
    content: str
    context_before: str = ""
    context_after: str = ""


@dataclass
class TextMetadata:
    """文本元数据"""
    total_length: int
    line_count: int
    chapter_count: int
    estimated_tokens: int


class RLMContext:
    """RLM 风格的上下文管理器。
    
    将原文作为环境变量管理，而非直接注入 LLM 上下文。
    核心理念：长 Prompt 不应直接喂入神经网络，而应作为环境的一部分
    让 LLM 以符号方式交互。
    
    Attributes:
        full_text: 完整文本内容
        metadata: 文本元数据
        
    Example:
        >>> ctx = RLMContext(full_text)
        >>> chunk = ctx.peek(0, 5000)  # 查看前 5000 字符
        >>> results = ctx.search(r"Harry Potter")  # 搜索关键词
    """
    
    def __init__(
        self, 
        full_text: str,
        context_window: int = 500,
    ):
        """初始化 RLM 上下文。
        
        Args:
            full_text: 完整文本内容
            context_window: 搜索结果的上下文窗口大小
        """
        self.full_text = full_text
        self.context_window = context_window
        self._metadata: Optional[TextMetadata] = None
        self._line_offsets: Optional[List[int]] = None
        
    @property
    def metadata(self) -> TextMetadata:
        """获取文本元数据（懒加载）。"""
        if self._metadata is None:
            self._metadata = self._extract_metadata()
        return self._metadata
    
    @property
    def line_offsets(self) -> List[int]:
        """获取行偏移表（懒加载）。"""
        if self._line_offsets is None:
            self._line_offsets = self._build_line_offsets()
        return self._line_offsets
        
    def _extract_metadata(self) -> TextMetadata:
        """提取文本元数据。"""
        lines = self.full_text.split('\n')
        
        # 简单的章节计数（基于常见模式）
        chapter_patterns = [
            r'^第[一二三四五六七八九十百千万\d]+章',
            r'^Chapter\s+\d+',
            r'^CHAPTER\s+\d+',
        ]
        chapter_count = 0
        for line in lines:
            for pattern in chapter_patterns:
                if re.match(pattern, line.strip()):
                    chapter_count += 1
                    break
        
        # 估算 Token 数（粗略：英文按空格分词，中文按字符）
        # 实际使用时应调用 tokenizer
        estimated_tokens = len(self.full_text.split()) + len(
            re.findall(r'[\u4e00-\u9fff]', self.full_text)
        )
        
        return TextMetadata(
            total_length=len(self.full_text),
            line_count=len(lines),
            chapter_count=chapter_count,
            estimated_tokens=estimated_tokens,
        )
    
    def _build_line_offsets(self) -> List[int]:
        """构建行偏移表，用于快速定位行。"""
        offsets = [0]
        for i, char in enumerate(self.full_text):
            if char == '\n':
                offsets.append(i + 1)
        return offsets
        
    def peek(
        self, 
        start: int, 
        end: int,
        expand_to_line: bool = False,
    ) -> str:
        """按需查看文本片段（Selective Viewing）。
        
        Args:
            start: 起始字符位置
            end: 结束字符位置
            expand_to_line: 是否扩展到完整行边界
            
        Returns:
            指定范围的文本内容
        """
        # 边界检查
        start = max(0, start)
        end = min(len(self.full_text), end)
        
        if expand_to_line:
            # 扩展到行边界
            while start > 0 and self.full_text[start - 1] != '\n':
                start -= 1
            while end < len(self.full_text) and self.full_text[end] != '\n':
                end += 1
                
        return self.full_text[start:end]
    
    def peek_lines(self, start_line: int, end_line: int) -> str:
        """按行号查看文本。
        
        Args:
            start_line: 起始行号（1-indexed）
            end_line: 结束行号（1-indexed，inclusive）
            
        Returns:
            指定行范围的文本内容
        """
        offsets = self.line_offsets
        
        # 转换为 0-indexed
        start_idx = max(0, start_line - 1)
        end_idx = min(len(offsets) - 1, end_line)
        
        if start_idx >= len(offsets):
            return ""
            
        start_pos = offsets[start_idx]
        end_pos = offsets[end_idx] if end_idx < len(offsets) else len(self.full_text)
        
        return self.full_text[start_pos:end_pos]
        
    def search(
        self, 
        pattern: str,
        is_regex: bool = True,
        case_sensitive: bool = True,
        max_results: int = 100,
    ) -> List[SearchResult]:
        """基于模型先验的关键词搜索（Code-driven Filtering）。
        
        Args:
            pattern: 搜索模式（正则表达式或纯文本）
            is_regex: 是否作为正则表达式处理
            case_sensitive: 是否区分大小写
            max_results: 最大结果数
            
        Returns:
            SearchResult 列表
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        
        if not is_regex:
            pattern = re.escape(pattern)
            
        results = []
        for match in re.finditer(pattern, self.full_text, flags):
            if len(results) >= max_results:
                break
                
            start, end = match.start(), match.end()
            
            # 提取上下文
            ctx_start = max(0, start - self.context_window)
            ctx_end = min(len(self.full_text), end + self.context_window)
            
            results.append(SearchResult(
                start=start,
                end=end,
                content=match.group(),
                context_before=self.full_text[ctx_start:start],
                context_after=self.full_text[end:ctx_end],
            ))
            
        return results
    
    def find_chapters(
        self, 
        patterns: Optional[List[str]] = None,
    ) -> List[Tuple[int, int, str]]:
        """查找章节边界。
        
        Args:
            patterns: 章节标题正则模式列表
            
        Returns:
            章节列表 [(start, end, title), ...]
        """
        if patterns is None:
            patterns = [
                r'^第[一二三四五六七八九十百千万\d]+章.*$',
                r'^Chapter\s+\d+.*$',
                r'^CHAPTER\s+\d+.*$',
                r'^Prologue.*$',
                r'^Epilogue.*$',
            ]
            
        # 合并所有模式
        combined_pattern = '|'.join(f'({p})' for p in patterns)
        
        chapters = []
        prev_end = 0
        prev_title = "Prologue"
        
        for match in re.finditer(combined_pattern, self.full_text, re.MULTILINE):
            if prev_end > 0 or match.start() > 0:
                # 保存前一章节
                chapters.append((prev_end, match.start(), prev_title))
            
            prev_end = match.start()
            prev_title = match.group().strip()
            
        # 添加最后一章
        if prev_end < len(self.full_text):
            chapters.append((prev_end, len(self.full_text), prev_title))
            
        return chapters
    
    def chunk_and_delegate(
        self, 
        chunks: List[Tuple[int, int]], 
        sub_task: Callable[[str], Any],
    ) -> List[Any]:
        """分块并递归子调用（Recursive Sub-calling）。
        
        Args:
            chunks: 块边界列表 [(start, end), ...]
            sub_task: 处理单个块的任务函数
            
        Returns:
            每个块的处理结果列表
        """
        results = []
        for start, end in chunks:
            chunk_text = self.peek(start, end)
            result = sub_task(chunk_text)
            results.append(result)
        return results
    
    def get_surrounding_context(
        self, 
        position: int, 
        before_chars: int = 1000, 
        after_chars: int = 1000,
    ) -> Tuple[str, str]:
        """获取指定位置的上下文。
        
        Args:
            position: 字符位置
            before_chars: 前文字符数
            after_chars: 后文字符数
            
        Returns:
            (前文, 后文) 元组
        """
        before_start = max(0, position - before_chars)
        after_end = min(len(self.full_text), position + after_chars)
        
        return (
            self.full_text[before_start:position],
            self.full_text[position:after_end],
        )
