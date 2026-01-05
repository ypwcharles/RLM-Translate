"""
短期记忆管理

管理当前对话/翻译块的上下文信息。
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    """对话轮次"""
    role: str           # "drafter", "critic", "editor"
    content: str        # 内容
    metadata: Dict = field(default_factory=dict)


class ShortTermMemory:
    """短期记忆管理。
    
    管理当前翻译块的对话历史和临时上下文。
    在 Addition-by-Subtraction 协作中用于传递迭代信息。
    
    Attributes:
        max_turns: 保留的最大对话轮次
        history: 对话历史
        
    Example:
        >>> memory = ShortTermMemory()
        >>> memory.add_turn("drafter", "翻译初稿...")
        >>> memory.add_turn("critic", "审查意见...")
        >>> history = memory.get_history()
    """
    
    def __init__(self, max_turns: int = 10):
        """初始化短期记忆。
        
        Args:
            max_turns: 保留的最大对话轮次
        """
        self.max_turns = max_turns
        self.history: List[ConversationTurn] = []
        self._context: Dict[str, Any] = {}
        
    def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ):
        """添加一轮对话。
        
        Args:
            role: 角色名称
            content: 内容
            metadata: 元数据
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.history.append(turn)
        
        # 保持历史长度限制
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
            
    def get_history(
        self,
        roles: Optional[List[str]] = None,
        last_n: Optional[int] = None,
    ) -> List[ConversationTurn]:
        """获取对话历史。
        
        Args:
            roles: 筛选指定角色的对话
            last_n: 只返回最近 N 轮
            
        Returns:
            对话历史列表
        """
        history = self.history
        
        if roles:
            history = [t for t in history if t.role in roles]
            
        if last_n:
            history = history[-last_n:]
            
        return history
        
    def get_history_as_text(
        self,
        separator: str = "\n\n",
        include_role: bool = True,
    ) -> str:
        """将历史格式化为文本。
        
        Args:
            separator: 轮次分隔符
            include_role: 是否包含角色标签
            
        Returns:
            格式化的历史文本
        """
        lines = []
        for turn in self.history:
            if include_role:
                lines.append(f"[{turn.role}]: {turn.content}")
            else:
                lines.append(turn.content)
        return separator.join(lines)
        
    def get_last_turn(self, role: Optional[str] = None) -> Optional[ConversationTurn]:
        """获取最后一轮对话。
        
        Args:
            role: 筛选指定角色
            
        Returns:
            最后一轮对话，不存在时返回 None
        """
        if role:
            for turn in reversed(self.history):
                if turn.role == role:
                    return turn
            return None
        return self.history[-1] if self.history else None
        
    def set_context(self, key: str, value: Any):
        """设置上下文变量。
        
        Args:
            key: 变量名
            value: 变量值
        """
        self._context[key] = value
        
    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文变量。
        
        Args:
            key: 变量名
            default: 默认值
            
        Returns:
            变量值
        """
        return self._context.get(key, default)
        
    def clear_history(self):
        """清空对话历史。"""
        self.history.clear()
        
    def clear_context(self):
        """清空上下文变量。"""
        self._context.clear()
        
    def clear(self):
        """清空所有记忆。"""
        self.clear_history()
        self.clear_context()
        
    def to_dict(self) -> Dict:
        """导出为字典。
        
        Returns:
            包含历史和上下文的字典
        """
        return {
            "history": [
                {
                    "role": t.role,
                    "content": t.content,
                    "metadata": t.metadata,
                }
                for t in self.history
            ],
            "context": self._context,
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> "ShortTermMemory":
        """从字典恢复。
        
        Args:
            data: 数据字典
            
        Returns:
            ShortTermMemory 实例
        """
        memory = cls()
        
        for item in data.get("history", []):
            memory.add_turn(
                role=item["role"],
                content=item["content"],
                metadata=item.get("metadata", {}),
            )
            
        memory._context = data.get("context", {})
        
        return memory
