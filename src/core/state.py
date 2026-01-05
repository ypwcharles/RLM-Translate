"""
TranslationState - 全局翻译状态定义

基于 RLM (Recursive Language Models) 和 TransAgents 论文设计。
- RLM：将长文本作为环境变量，按需访问而非直接注入上下文
- TransAgents：维护长期记忆（Glossary, Summary）确保一致性
"""

from typing import TypedDict, List, Dict, Optional, Any


class ChunkInfo(TypedDict):
    """章节/块信息"""
    title: str          # 章节标题
    start: int          # 起始字符位置
    end: int            # 结束字符位置
    token_count: int    # Token 数量


class TranslationState(TypedDict, total=False):
    """全局翻译状态，贯穿整个工作流生命周期。
    
    基于 RLM 范式：将长文本作为环境变量，而非直接注入上下文。
    基于 TransAgents：维护长期记忆（Glossary, Summary）确保一致性。
    
    Attributes:
        === RLM 环境变量 (Prompt-as-Variable) ===
        raw_text: 完整原文内容（作为环境变量，按需访问）
        raw_text_path: 原文文件路径（可选，用于大文件）
        raw_text_length: 原文长度（Token 计数）
        
        === 长期记忆 (Long-Term Memory - TransAgents) ===
        glossary: 术语表 {原文: 译文}
        style_guide: 风格指南
        plot_summary: 剧情摘要链（递增）
        character_profiles: 角色状态
        book_summary: 全书概要
        target_audience: 目标读者
        
        === 运行时变量 (循环控制) ===
        chapter_map: 章节映射表
        total_chunks: 总任务块数
        current_chunk_index: 当前处理索引
        current_source_text: 当前待处理文本块（RLM peek 结果）
        current_chunk_title: 当前块标题
        
        === Agent 协作区 (Addition-by-Subtraction) ===
        draft_translation: Addition Agent 初稿
        critique_comments: Subtraction Agent 反馈
        iteration_count: 当前协作迭代次数
        final_chunk_translation: 当前块定稿
        
        === 输出结果 ===
        completed_translations: 已完成译文列表
        
        === 错误处理 ===
        errors: 错误记录列表
    """
    
    # === RLM 环境变量 (Prompt-as-Variable) ===
    raw_text: str
    raw_text_path: Optional[str]
    raw_text_length: int
    
    # === 长期记忆 (Long-Term Memory - TransAgents) ===
    glossary: Dict[str, str]
    style_guide: str
    plot_summary: List[str]
    character_profiles: Dict[str, str]
    book_summary: str
    target_audience: str
    
    # === 运行时变量 (循环控制) ===
    chapter_map: List[ChunkInfo]
    total_chunks: int
    current_chunk_index: int
    current_source_text: str
    current_chunk_title: str
    
    # === Agent 协作区 (Addition-by-Subtraction) ===
    draft_translation: str
    critique_comments: str
    iteration_count: int
    final_chunk_translation: str
    
    # === 输出结果 ===
    completed_translations: List[str]
    
    # === 错误处理 ===
    errors: List[Dict[str, Any]]


def create_initial_state(
    raw_text: str,
    style_guide: str = "",
    target_audience: str = "一般读者",
    raw_text_path: Optional[str] = None,
) -> TranslationState:
    """创建初始翻译状态。
    
    Args:
        raw_text: 完整原文内容
        style_guide: 风格指南
        target_audience: 目标读者
        raw_text_path: 原文文件路径（可选）
        
    Returns:
        初始化的 TranslationState
    """
    return TranslationState(
        # RLM 环境变量
        raw_text=raw_text,
        raw_text_path=raw_text_path,
        raw_text_length=0,  # 将由 tokenizer 填充
        
        # 长期记忆
        glossary={},
        style_guide=style_guide,
        plot_summary=[],
        character_profiles={},
        book_summary="",
        target_audience=target_audience,
        
        # 运行时变量
        chapter_map=[],
        total_chunks=0,
        current_chunk_index=0,
        current_source_text="",
        current_chunk_title="",
        
        # Agent 协作区
        draft_translation="",
        critique_comments="",
        iteration_count=0,
        final_chunk_translation="",
        
        # 输出结果
        completed_translations=[],
        
        # 错误处理
        errors=[],
    )


def update_state(state: TranslationState, **updates) -> TranslationState:
    """不可变方式更新状态。
    
    Args:
        state: 当前状态
        **updates: 要更新的字段
        
    Returns:
        更新后的新状态
    """
    return TranslationState(**{**state, **updates})


def append_to_list_field(
    state: TranslationState, 
    field: str, 
    value: Any
) -> TranslationState:
    """向列表字段追加元素。
    
    Args:
        state: 当前状态
        field: 列表字段名
        value: 要追加的值
        
    Returns:
        更新后的新状态
    """
    current_list = state.get(field, [])
    return update_state(state, **{field: [*current_list, value]})
