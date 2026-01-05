"""
长期记忆管理

管理全书范围的持久化信息：Glossary, Summary, Character Profiles。
基于 TransAgents 论文的长期记忆设计。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class CharacterProfile:
    """角色档案"""
    name: str                           # 角色名
    description: str = ""               # 角色描述
    current_state: str = ""             # 当前状态
    relationships: Dict[str, str] = field(default_factory=dict)  # 关系网络
    aliases: List[str] = field(default_factory=list)  # 别名列表


class LongTermMemory:
    """长期记忆管理。
    
    基于 TransAgents 论文，管理翻译指南（Translation Guidelines）：
    1. Glossary: 专有名词/术语映射
    2. Book Summary: 全书剧情概要
    3. Plot Summaries: 各章节剧情摘要
    4. Character Profiles: 角色档案
    5. Style Guide: 风格定调
    6. Target Audience: 目标读者
    
    Attributes:
        glossary: 术语表
        plot_summaries: 剧情摘要列表
        characters: 角色档案
        style_guide: 风格指南
        
    Example:
        >>> memory = LongTermMemory()
        >>> memory.add_glossary_entry("Harry Potter", "哈利·波特")
        >>> memory.add_plot_summary("第一章：男孩生存下来")
    """
    
    def __init__(
        self,
        style_guide: str = "",
        target_audience: str = "一般读者",
        book_summary: str = "",
    ):
        """初始化长期记忆。
        
        Args:
            style_guide: 风格指南
            target_audience: 目标读者
            book_summary: 全书概要
        """
        self.glossary: Dict[str, str] = {}
        self.plot_summaries: List[str] = []
        self.characters: Dict[str, CharacterProfile] = {}
        self.style_guide = style_guide
        self.target_audience = target_audience
        self.book_summary = book_summary
        
    # === Glossary 管理 ===
    
    def add_glossary_entry(self, source: str, target: str):
        """添加术语表条目。
        
        Args:
            source: 原文术语
            target: 译文术语
        """
        self.glossary[source] = target
        
    def add_glossary_batch(self, entries: Dict[str, str]):
        """批量添加术语表条目。
        
        Args:
            entries: 术语映射字典
        """
        self.glossary.update(entries)
        
    def get_glossary_entry(self, source: str) -> Optional[str]:
        """获取术语翻译。
        
        Args:
            source: 原文术语
            
        Returns:
            译文，不存在时返回 None
        """
        return self.glossary.get(source)
        
    def format_glossary(self, max_entries: Optional[int] = None) -> str:
        """格式化术语表为 Prompt 文本。
        
        Args:
            max_entries: 最大条目数
            
        Returns:
            格式化的术语表文本
        """
        entries = list(self.glossary.items())
        if max_entries:
            entries = entries[:max_entries]
            
        return "\n".join(f"- {src} → {tgt}" for src, tgt in entries)
        
    # === Plot Summary 管理 ===
    
    def add_plot_summary(self, summary: str):
        """添加章节剧情摘要。
        
        Args:
            summary: 剧情摘要
        """
        self.plot_summaries.append(summary)
        
    def get_recent_summaries(self, count: int = 10) -> List[str]:
        """获取最近的剧情摘要。
        
        Args:
            count: 返回数量
            
        Returns:
            摘要列表
        """
        return self.plot_summaries[-count:]
        
    def format_plot_summaries(self, count: int = 10) -> str:
        """格式化剧情摘要为 Prompt 文本。
        
        Args:
            count: 包含的摘要数量
            
        Returns:
            格式化的摘要文本
        """
        summaries = self.get_recent_summaries(count)
        return "\n\n".join(summaries)
        
    # === Character Profile 管理 ===
    
    def add_character(
        self,
        name: str,
        description: str = "",
        current_state: str = "",
    ) -> CharacterProfile:
        """添加角色。
        
        Args:
            name: 角色名
            description: 描述
            current_state: 当前状态
            
        Returns:
            创建的角色档案
        """
        profile = CharacterProfile(
            name=name,
            description=description,
            current_state=current_state,
        )
        self.characters[name] = profile
        return profile
        
    def update_character_state(self, name: str, state: str):
        """更新角色状态。
        
        Args:
            name: 角色名
            state: 新状态
        """
        if name in self.characters:
            self.characters[name].current_state = state
            
    def get_character(self, name: str) -> Optional[CharacterProfile]:
        """获取角色档案。
        
        Args:
            name: 角色名
            
        Returns:
            角色档案，不存在时返回 None
        """
        return self.characters.get(name)
        
    def format_character_profiles(self) -> str:
        """格式化角色档案为 Prompt 文本。
        
        Returns:
            格式化的角色档案文本
        """
        lines = []
        for name, profile in self.characters.items():
            line = f"- {name}"
            if profile.description:
                line += f": {profile.description}"
            if profile.current_state:
                line += f" (当前状态: {profile.current_state})"
            lines.append(line)
        return "\n".join(lines)
        
    # === 综合注入 ===
    
    def inject_into_prompt(
        self,
        prompt_template: str,
        glossary_max: Optional[int] = None,
        summary_count: int = 10,
    ) -> str:
        """将长期记忆注入 Prompt 模板。
        
        Args:
            prompt_template: Prompt 模板（包含占位符）
            glossary_max: 术语表最大条目数
            summary_count: 剧情摘要数量
            
        Returns:
            填充后的 Prompt
        """
        return prompt_template.format(
            glossary=self.format_glossary(glossary_max),
            plot_summary=self.format_plot_summaries(summary_count),
            style_guide=self.style_guide,
            target_audience=self.target_audience,
            book_summary=self.book_summary,
            character_profiles=self.format_character_profiles(),
        )
        
    # === 持久化 ===
    
    def save(self, filepath: str):
        """保存到文件。
        
        Args:
            filepath: 文件路径
        """
        data = self.to_dict()
        Path(filepath).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
    @classmethod
    def load(cls, filepath: str) -> "LongTermMemory":
        """从文件加载。
        
        Args:
            filepath: 文件路径
            
        Returns:
            LongTermMemory 实例
        """
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        return cls.from_dict(data)
        
    def to_dict(self) -> Dict:
        """导出为字典。
        
        Returns:
            数据字典
        """
        return {
            "glossary": self.glossary,
            "plot_summaries": self.plot_summaries,
            "characters": {
                name: {
                    "name": p.name,
                    "description": p.description,
                    "current_state": p.current_state,
                    "relationships": p.relationships,
                    "aliases": p.aliases,
                }
                for name, p in self.characters.items()
            },
            "style_guide": self.style_guide,
            "target_audience": self.target_audience,
            "book_summary": self.book_summary,
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> "LongTermMemory":
        """从字典恢复。
        
        Args:
            data: 数据字典
            
        Returns:
            LongTermMemory 实例
        """
        memory = cls(
            style_guide=data.get("style_guide", ""),
            target_audience=data.get("target_audience", "一般读者"),
            book_summary=data.get("book_summary", ""),
        )
        
        memory.glossary = data.get("glossary", {})
        memory.plot_summaries = data.get("plot_summaries", [])
        
        for name, char_data in data.get("characters", {}).items():
            profile = CharacterProfile(
                name=char_data["name"],
                description=char_data.get("description", ""),
                current_state=char_data.get("current_state", ""),
                relationships=char_data.get("relationships", {}),
                aliases=char_data.get("aliases", []),
            )
            memory.characters[name] = profile
            
        return memory
