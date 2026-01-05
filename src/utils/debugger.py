"""
翻译流程调试器

提供 Prompt、Response 保存和状态快照功能，用于调试和断点恢复。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any


class TranslationDebugger:
    """翻译流程调试器。
    
    用于保存调试信息，包括 Prompt、LLM Response 和状态快照。
    
    Attributes:
        enabled: 是否启用调试
        debug_dir: 调试输出目录
        session_id: 当前会话 ID
        
    Example:
        >>> debugger = TranslationDebugger({"debug_dir": "./debug_output"})
        >>> debugger.save_prompt("drafter", prompt, chunk_index=0)
        >>> debugger.save_response("drafter", response, chunk_index=0)
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        enabled: bool = True,
    ):
        """初始化调试器。
        
        Args:
            config: 配置字典，包含 debug_dir 等设置
            enabled: 是否启用调试
        """
        self.enabled = enabled
        self.config = config or {}
        self.debug_dir = Path(self.config.get("debug_dir", "./debug_output"))
        
        if self.enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    @property
    def session_id(self) -> str:
        """获取当前会话 ID。"""
        return self._session_id
        
    def _get_subdir(self, name: str) -> Path:
        """获取子目录路径。"""
        subdir = self.debug_dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir
        
    def save_prompt(
        self,
        agent_name: str,
        prompt: str,
        chunk_index: int,
        iteration: int = 0,
        metadata: Optional[Dict] = None,
    ) -> Optional[Path]:
        """保存 Prompt 到文件用于调试。
        
        Args:
            agent_name: Agent 名称（drafter, critic, editor 等）
            prompt: Prompt 内容
            chunk_index: 当前块索引
            iteration: 协作迭代次数
            metadata: 额外元数据
            
        Returns:
            保存的文件路径，禁用时返回 None
        """
        if not self.enabled:
            return None
            
        filename = f"{self._session_id}_{agent_name}_chunk{chunk_index:03d}_iter{iteration}.txt"
        filepath = self._get_subdir("prompts") / filename
        
        content = prompt
        if metadata:
            header = f"# Metadata: {json.dumps(metadata, ensure_ascii=False)}\n\n"
            content = header + content
            
        filepath.write_text(content, encoding="utf-8")
        return filepath
        
    def save_response(
        self,
        agent_name: str,
        response: str,
        chunk_index: int,
        iteration: int = 0,
        metadata: Optional[Dict] = None,
    ) -> Optional[Path]:
        """保存 LLM 响应到文件。
        
        Args:
            agent_name: Agent 名称
            response: LLM 响应内容
            chunk_index: 当前块索引
            iteration: 协作迭代次数
            metadata: 额外元数据（如 token 使用量、耗时等）
            
        Returns:
            保存的文件路径，禁用时返回 None
        """
        if not self.enabled:
            return None
            
        data = {
            "agent": agent_name,
            "chunk_index": chunk_index,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "metadata": metadata or {},
        }
        
        filename = f"{self._session_id}_{agent_name}_chunk{chunk_index:03d}_iter{iteration}.json"
        filepath = self._get_subdir("responses") / filename
        
        filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return filepath
        
    def save_state_snapshot(
        self,
        state: Dict,
        label: str = "",
    ) -> Optional[Path]:
        """保存状态快照用于断点恢复。
        
        Args:
            state: TranslationState 状态字典
            label: 快照标签（如 "chunk_005"）
            
        Returns:
            保存的文件路径，禁用时返回 None
        """
        if not self.enabled:
            return None
            
        # 排除大型字段避免快照过大
        snapshot = {}
        for k, v in state.items():
            if k == "raw_text":
                snapshot[k] = f"<text: {len(v)} chars>"
            elif k == "completed_translations":
                snapshot[k] = f"<translations: {len(v)} items>"
            elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                snapshot[k] = v
            else:
                snapshot[k] = str(v)
                
        snapshot["_snapshot_time"] = datetime.now().isoformat()
        snapshot["_session_id"] = self._session_id
        snapshot["_label"] = label
        
        filename = f"{self._session_id}_snapshot_{label}.json"
        filepath = self._get_subdir("snapshots") / filename
        
        filepath.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return filepath
        
    def log_event(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict] = None,
    ) -> Optional[Path]:
        """记录事件日志。
        
        Args:
            event_type: 事件类型（如 "start", "error", "complete"）
            message: 事件消息
            data: 事件数据
            
        Returns:
            日志文件路径
        """
        if not self.enabled:
            return None
            
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self._session_id,
            "event_type": event_type,
            "message": message,
            "data": data or {},
        }
        
        log_file = self._get_subdir("logs") / f"{self._session_id}_events.jsonl"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        return log_file
        
    def list_snapshots(self) -> list:
        """列出所有状态快照。
        
        Returns:
            快照文件信息列表
        """
        snapshot_dir = self.debug_dir / "snapshots"
        if not snapshot_dir.exists():
            return []
            
        snapshots = []
        for filepath in sorted(snapshot_dir.glob("*.json")):
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                snapshots.append({
                    "filename": filepath.name,
                    "path": str(filepath),
                    "label": data.get("_label", ""),
                    "time": data.get("_snapshot_time", ""),
                    "chunk_index": data.get("current_chunk_index", 0),
                })
            except Exception:
                continue
                
        return snapshots
