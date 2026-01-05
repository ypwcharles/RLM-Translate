"""
Structure Scanner - 结构扫描工具

负责扫描全文，提取结构骨架（Structure Skeleton），供 Analyzer Agent 进行语义切分分析。
骨架包含：
1. 潜在的章节标题（以 # 开头，或短文本，或特定关键词开头）
2. 空行（Visual Gaps）
3. 段落的起始信息（用于定位）
4. 不包含正文全文，以节省 Token
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class StructureNode:
    """结构节点，代表文本中的一行或一个片段。"""
    line_number: int       # 行号 (0-indexed)
    content_preview: str   # 内容预览 (如果是标题则全字，如果是正文则截断)
    type: str              # "header_candidate" | "empty" | "text"
    length: int            # 该行字符数
    
    def to_dict(self):
        return asdict(self)


class StructureScanner:
    """结构扫描器。"""
    
    def __init__(self):
        # 预编译一些正则，用于粗略识别标题候选
        self.header_patterns = [
            r'^#+\s.*',                    # Markdown Header
            r'^\s*Chapter\s.*',            # English Standard
            r'^\s*CHAPTER\s.*',            # English Caps
            r'^\s*Prologue.*',             # Prologue
            r'^\s*Epilogue.*',             # Epilogue
            r'^\s*第[0-9一二三四五六七八九十百千万]+章.*', # Chinese Standard
            r'^\s*[0-9]+\.\s.*',           # Numbered list/header
            r'^[A-Z\s\d:,-]{2,50}$'        # All Caps short line (potential title)
        ]
        
    def scan(self, text: str) -> List[Dict[str, Any]]:
        """扫描文本，生成骨架。
        
        Args:
            text: 完整文本
            
        Returns:
            结构节点列表 (JSON artifacts)
        """
        lines = text.split('\n')
        skeleton = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 1. 空行
            if not stripped:
                skeleton.append(StructureNode(
                    line_number=i,
                    content_preview="[EMPTY LINE]",
                    type="empty",
                    length=0
                ).to_dict())
                continue
                
            # 2. 潜在标题 (Header Candidate)
            is_header_candidate = False
            
            # 规则 A: 匹配正则
            for pattern in self.header_patterns:
                if re.match(pattern, stripped):
                    is_header_candidate = True
                    break
            
            # 规则 B: 非常短的行 (可能是标题)，且不以标点结尾 (除了冒号)
            if not is_header_candidate and len(stripped) < 50:
                 # 排除显然的对话或句子片段
                 if not stripped[-1] in ".。,，!！?？\"'”’":
                     is_header_candidate = True

            if is_header_candidate:
                skeleton.append(StructureNode(
                    line_number=i,
                    content_preview=stripped,
                    type="header_candidate",
                    length=len(line)
                ).to_dict())
            else:
                # 3. 普通正文 (Text)
                # 为了节省 Token，只保留前 50 个字符作为预览，并标记长度
                preview = stripped[:50] + "..." if len(stripped) > 50 else stripped
                skeleton.append(StructureNode(
                    line_number=i,
                    content_preview=preview,
                    type="text",
                    length=len(line)
                ).to_dict())
                
        # 压缩骨架：
        # 连续的 "text" 节点可以合并，只保留第一行和最后一行，或者每隔 N 行采样
        # 这里为了简单起见，暂不合并，交给 LLM 自己看。
        # 但为了防止 Token 爆炸，我们可以过滤掉纯 "text" 节点，只保留 header 和它们周围的 context？
        # 或者：保留所有 header, empty, 以及 text 的第一行（代表段落开始）。
        
        return self._compress_skeleton(skeleton)

    def _compress_skeleton(self, raw_skeleton: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """压缩骨架，移除冗余的正文行，保留结构信息。"""
        compressed = []
        
        for j, node in enumerate(raw_skeleton):
            # 保留 header
            if node['type'] == 'header_candidate':
                compressed.append(node)
                continue
            
            # 保留 empty (作为分隔符重要)
            if node['type'] == 'empty':
                # 如果连续空行，只保留一个? 不，保留所有吧，也没多少 token
                compressed.append(node)
                continue
                
            # 对于 type="text"
            # 如果主要目的是切分章节，我们其实不需要知道正文的具体内容，
            # 只需要知道 "这里有一大块文本"。
            # 但是如果 LLM 要判断 "Scene Break" (场景转换)，可能需要看正文开头。
            
            # 策略：保留每个 Text Block 的第一行。
            # 检查上一行是否是 empty 或 header。如果是，说明这是新段落的开始。
            prev_node = raw_skeleton[j-1] if j > 0 else None
            if prev_node is None or prev_node['type'] in ['empty', 'header_candidate']:
                 compressed.append(node)
            else:
                # 这是一个大段落中间的行，或者连续段落。
                # 我们可以忽略它，或者每隔 20 行采样一个？
                # 暂且忽略，用 "..." 节点代替?
                # 为了简化，我们只保留 "Block Start"。
                pass
                
        return compressed
