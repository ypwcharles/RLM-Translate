"""
检查点管理器

支持断点续传的检查点保存和加载功能。
"""

import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class CheckpointManager:
    """支持断点续传的检查点管理器。
    
    用于保存翻译状态，支持从中断点恢复处理。
    
    Attributes:
        checkpoint_dir: 检查点目录
        max_checkpoints: 保留的最大检查点数
        
    Example:
        >>> manager = CheckpointManager("./checkpoints")
        >>> manager.save_checkpoint(state, "chunk_005")
        >>> state = manager.load_checkpoint("chunk_005")
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 10,
    ):
        """初始化检查点管理器。
        
        Args:
            checkpoint_dir: 检查点目录路径
            max_checkpoints: 保留的最大检查点数
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
    def save_checkpoint(
        self,
        state: Dict,
        checkpoint_name: str = "latest",
        include_translations: bool = False,
    ) -> Path:
        """保存检查点。
        
        Args:
            state: TranslationState 状态字典
            checkpoint_name: 检查点名称
            include_translations: 是否包含已完成的翻译
            
        Returns:
            检查点文件路径
        """
        # 序列化状态
        serializable_state = self._make_serializable(state, include_translations)
        
        # 添加元数据
        serializable_state["_checkpoint_meta"] = {
            "name": checkpoint_name,
            "timestamp": datetime.now().isoformat(),
            "chunk_index": state.get("current_chunk_index", 0),
            "total_chunks": state.get("total_chunks", 0),
        }
        
        filepath = self.checkpoint_dir / f"{checkpoint_name}.json"
        filepath.write_text(
            json.dumps(serializable_state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
        # 保留最近的检查点
        self._cleanup_old_checkpoints()
        
        return filepath
        
    def load_checkpoint(
        self,
        checkpoint_name: str = "latest",
    ) -> Optional[Dict]:
        """加载检查点以恢复状态。
        
        Args:
            checkpoint_name: 检查点名称
            
        Returns:
            恢复的状态字典，不存在时返回 None
        """
        filepath = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        if not filepath.exists():
            return None
            
        data = json.loads(filepath.read_text(encoding="utf-8"))
        
        # 移除元数据字段
        data.pop("_checkpoint_meta", None)
        
        return data
        
    def resume_from_chunk(self, chunk_index: int) -> Optional[Dict]:
        """从指定块恢复处理。
        
        查找最接近指定块索引的检查点。
        
        Args:
            chunk_index: 目标块索引
            
        Returns:
            恢复的状态字典，不存在时返回 None
        """
        checkpoints = self.list_checkpoints()
        
        # 查找最接近且不超过目标索引的检查点
        best_checkpoint = None
        best_index = -1
        
        for cp in checkpoints:
            cp_index = cp.get("chunk_index", 0)
            if cp_index <= chunk_index and cp_index > best_index:
                best_checkpoint = cp
                best_index = cp_index
                
        if best_checkpoint:
            return self.load_checkpoint(best_checkpoint["name"])
            
        return None
        
    def list_checkpoints(self) -> List[Dict]:
        """列出所有检查点。
        
        Returns:
            检查点信息列表，按时间倒序排列
        """
        checkpoints = []
        
        for filepath in self.checkpoint_dir.glob("*.json"):
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                meta = data.get("_checkpoint_meta", {})
                checkpoints.append({
                    "name": meta.get("name", filepath.stem),
                    "path": str(filepath),
                    "timestamp": meta.get("timestamp", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "total_chunks": meta.get("total_chunks", 0),
                })
            except Exception:
                continue
                
        # 按时间倒序排列
        checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return checkpoints
        
    def get_latest_checkpoint(self) -> Optional[Dict]:
        """获取最新的检查点。
        
        Returns:
            最新检查点的状态字典
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return self.load_checkpoint(checkpoints[0]["name"])
        
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """删除指定检查点。
        
        Args:
            checkpoint_name: 检查点名称
            
        Returns:
            是否成功删除
        """
        filepath = self.checkpoint_dir / f"{checkpoint_name}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False
        
    def clear_all_checkpoints(self) -> int:
        """清空所有检查点。
        
        Returns:
            删除的检查点数量
        """
        count = 0
        for filepath in self.checkpoint_dir.glob("*.json"):
            filepath.unlink()
            count += 1
        return count
        
    def _make_serializable(
        self,
        state: Dict,
        include_translations: bool = False,
    ) -> Dict:
        """转换为可序列化格式。
        
        Args:
            state: 原始状态字典
            include_translations: 是否包含翻译结果
            
        Returns:
            可序列化的字典
        """
        result = {}
        
        for k, v in state.items():
            if k == "raw_text":
                # 不保存完整原文，节省空间
                result[k] = f"<text: {len(v)} chars>"
                result["_raw_text_length"] = len(v)
            elif k == "completed_translations" and not include_translations:
                # 只保存翻译数量
                result[k] = f"<translations: {len(v)} items>"
                result["_completed_count"] = len(v)
            elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                result[k] = v
            else:
                result[k] = str(v)
                
        return result
        
    def _cleanup_old_checkpoints(self):
        """清理旧检查点，保留最新的 N 个。"""
        checkpoints = self.list_checkpoints()
        
        # 跳过 "latest" 检查点
        checkpoints = [cp for cp in checkpoints if cp["name"] != "latest"]
        
        if len(checkpoints) > self.max_checkpoints:
            # 删除最旧的检查点
            for cp in checkpoints[self.max_checkpoints:]:
                self.delete_checkpoint(cp["name"])
