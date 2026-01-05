"""
Editor Agent - 润色 Agent

负责结合评审反馈产出最终定稿翻译。
"""

from typing import Dict, Any, Optional, List

import json
import re
import logging

from .base import BaseAgent
from ..core.client import ChatGoogleGenerativeAI

from pathlib import Path

# 获取 prompts 目录的绝对路径
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

logger = logging.getLogger(__name__)

class EditorAgent(BaseAgent):
    """润色 Agent (Precision Surgeon)。
    
    负责结合评审反馈，进行精准的手术式修改 (Surgical Editing)。
    它不仅生成最终文本，而是生成修改一系列 Patch，并将其应用到 Draft 上。
    
    Attributes:
        source_lang: 源语言
        target_lang: 目标语言
        
    Example:
        >>> editor = EditorAgent(client)
        >>> final = await editor.process(
        ...     source_text, context, 
        ...     draft=draft, feedback=critique
        ... )
    """
    
    def __init__(
        self,
        client: ChatGoogleGenerativeAI,
        source_lang: str = "English",
        target_lang: str = "Chinese",
        prompt_template: Optional[str] = None,
        prompt_file: Optional[str] = None,
    ):
        """初始化 Editor Agent。
        
        Args:
            client: LLM 客户端
            source_lang: 源语言
            target_lang: 目标语言
            prompt_template: 自定义 Prompt 模板
            prompt_file: Prompt 模板文件 (默认为 prompts/editor.txt)
        """
        # 如果未指定 prompt_file 且没有 template，则使用默认文件
        if not prompt_template and not prompt_file:
            prompt_file = str(PROMPTS_DIR / "editor.txt")
            
        super().__init__(
            name="editor",
            client=client,
            prompt_template=prompt_template,
            prompt_file=prompt_file,
        )
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def _get_default_template(self) -> str:
        """获取默认 Prompt 模板。"""
        return "Error: Prompt template not found."
        
    def apply_patches(self, source_text: str, patches: List[Dict[str, Any]]) -> str:
        """应用补丁 (逻辑同 PatcherAgent)。
        
        Args:
            source_text: 原始文本 (Draft)
            patches: 补丁列表
            
        Returns:
            修改后的文本
        """
        if not patches:
            return source_text
            
        current_text = source_text
        applied_count = 0
        skipped_count = 0
        
        # 按 original_span 长度降序排序，优先处理长文本，减少嵌套替换问题
        sorted_patches = sorted(patches, key=lambda x: len(x.get("original_span", "")), reverse=True)
        
        for patch in sorted_patches:
            original = patch.get("original_span")
            replacement = patch.get("replacement")
            
            if not original or replacement is None:
                logger.warning(f"[Editor] Invalid patch: {patch}")
                skipped_count += 1
                continue
                
            # 检查出现次数
            count = current_text.count(original)
            
            if count == 1:
                # 唯一匹配，应用替换
                current_text = current_text.replace(original, replacement)
                applied_count += 1
                logger.info(f"[Editor] Applied patch: '{original[:20]}...' -> '{replacement[:20]}...'")
            elif count == 0:
                logger.warning(f"[Editor] Span not found: '{original[:50]}...'")
                skipped_count += 1
                # TODO: Fuzzy matching fallback if needed
            else:
                logger.warning(f"[Editor] Span matches multiple times ({count}): '{original[:50]}...'")
                skipped_count += 1
                
        logger.info(f"[Editor] Finished. Applied: {applied_count}, Skipped: {skipped_count}")
        return current_text

    def parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """提取 JSON (Patches)。忽略 Task List 文本。"""
        # 尝试提取 Markdown 代码块 ```json ... ```
        match = re.search(r"```json\s*(.*?)```", response, re.DOTALL)
        if match:
            try:
                content = match.group(1).strip()
                return json.loads(content)
            except json.JSONDecodeError:
                pass
                
        # 尝试提取 [...]
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
             try:
                content = match.group(0).strip()
                return json.loads(content)
             except json.JSONDecodeError:
                pass
                
        return []

    async def process(
        self,
        source_text: str,
        context: Dict[str, Any],
        draft_translation: str = "",
        critique_comments: str = "",
        **kwargs,
    ) -> str:
        """润色翻译 (Surgical Edit)。
        
        Args:
            source_text: 源文本
            context: 上下文
            draft_translation: 初稿翻译
            critique_comments: 评审意见
            
        Returns:
            润色后的最终翻译
        """
        # 1. 构造 Prompt 并调用 LLM
        prompt = self.build_prompt(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            source_text=source_text,
            draft_translation=draft_translation,
            critique_comments=critique_comments,
            glossary=context.get("glossary", ""),
            style_guide=context.get("style_guide", ""),
        )
        
        response_text = await self.invoke(prompt)
        
        # 2. 解析 Patch
        patches = self.parse_json_response(response_text)
        
        if not patches:
            logger.warning("[Editor] No valid patches found in response.")
            # 如果没有 patch，可能意味着没有修改，或者格式错误
            # 这里保守起见，返回原 draft (或考虑直接返回 response_text 如果它包含了全文? 
            # 但我们的 Prompt 要求输出 Task List + JSON Patch，不包含全文)
            return draft_translation
            
        # 3. 应用 Patch 到 Draft
        final_translation = self.apply_patches(draft_translation, patches)
        
        return final_translation
        
    def process_sync(
        self,
        source_text: str,
        context: Dict[str, Any],
        draft_translation: str = "",
        critique_comments: str = "",
        **kwargs,
    ) -> str:
        """同步润色翻译 (Surgical Edit)。"""
        prompt = self.build_prompt(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            source_text=source_text,
            draft_translation=draft_translation,
            critique_comments=critique_comments,
            glossary=context.get("glossary", ""),
            style_guide=context.get("style_guide", ""),
        )
        
        response_text = self.invoke_sync(prompt)
        
        patches = self.parse_json_response(response_text)
        
        if not patches:
            logger.warning("[Editor] No valid patches found in response.")
            return draft_translation
            
        final_translation = self.apply_patches(draft_translation, patches)
        
        return final_translation
