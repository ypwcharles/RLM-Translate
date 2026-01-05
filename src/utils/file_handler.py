"""
文件处理工具

提供文本文件的读取、写入和目录管理功能。
"""

import os
from pathlib import Path
from typing import Optional, List
import json


class FileHandler:
    """文件处理器。
    
    提供常用的文件操作功能，包括读取、写入、目录管理等。
    
    Attributes:
        base_dir: 基础目录路径
        encoding: 文件编码
        
    Example:
        >>> handler = FileHandler("./output")
        >>> handler.write_text("result.txt", "Hello, world!")
        >>> content = handler.read_text("result.txt")
    """
    
    def __init__(
        self,
        base_dir: str = ".",
        encoding: str = "utf-8",
    ):
        self.base_dir = Path(base_dir)
        self.encoding = encoding
        
    def ensure_dir(self, path: Optional[str] = None) -> Path:
        """确保目录存在。
        
        Args:
            path: 相对路径或绝对路径，为 None 时使用 base_dir
            
        Returns:
            目录的 Path 对象
        """
        if path is None:
            dir_path = self.base_dir
        elif Path(path).is_absolute():
            dir_path = Path(path)
        else:
            dir_path = self.base_dir / path
            
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
        
    def _resolve_path(self, filename: str) -> Path:
        """解析文件路径。
        
        Args:
            filename: 文件名或相对路径
            
        Returns:
            完整的 Path 对象
        """
        file_path = Path(filename)
        if file_path.is_absolute():
            return file_path
        return self.base_dir / filename
        
    def read_text(self, filename: str) -> str:
        """读取文本文件。
        
        Args:
            filename: 文件名或路径
            
        Returns:
            文件内容
            
        Raises:
            FileNotFoundError: 文件不存在
        """
        file_path = self._resolve_path(filename)
        return file_path.read_text(encoding=self.encoding)
        
    def write_text(
        self,
        filename: str,
        content: str,
        ensure_parent: bool = True,
    ) -> Path:
        """写入文本文件。
        
        Args:
            filename: 文件名或路径
            content: 文件内容
            ensure_parent: 是否自动创建父目录
            
        Returns:
            写入的文件路径
        """
        file_path = self._resolve_path(filename)
        
        if ensure_parent:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
        file_path.write_text(content, encoding=self.encoding)
        return file_path
        
    def append_text(
        self,
        filename: str,
        content: str,
        ensure_parent: bool = True,
    ) -> Path:
        """追加文本到文件。
        
        Args:
            filename: 文件名或路径
            content: 要追加的内容
            ensure_parent: 是否自动创建父目录
            
        Returns:
            文件路径
        """
        file_path = self._resolve_path(filename)
        
        if ensure_parent:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
        with open(file_path, "a", encoding=self.encoding) as f:
            f.write(content)
            
        return file_path
        
    def read_json(self, filename: str) -> dict:
        """读取 JSON 文件。
        
        Args:
            filename: 文件名或路径
            
        Returns:
            解析后的字典
        """
        content = self.read_text(filename)
        return json.loads(content)
        
    def write_json(
        self,
        filename: str,
        data: dict,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> Path:
        """写入 JSON 文件。
        
        Args:
            filename: 文件名或路径
            data: 要写入的数据
            indent: 缩进空格数
            ensure_ascii: 是否转义非 ASCII 字符
            
        Returns:
            写入的文件路径
        """
        content = json.dumps(
            data,
            indent=indent,
            ensure_ascii=ensure_ascii,
        )
        return self.write_text(filename, content)
        
    def list_files(
        self,
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[Path]:
        """列出匹配的文件。
        
        Args:
            pattern: 文件匹配模式（glob 语法）
            recursive: 是否递归搜索
            
        Returns:
            匹配的文件路径列表
        """
        if recursive:
            return list(self.base_dir.rglob(pattern))
        return list(self.base_dir.glob(pattern))
        
    def exists(self, filename: str) -> bool:
        """检查文件是否存在。
        
        Args:
            filename: 文件名或路径
            
        Returns:
            是否存在
        """
        return self._resolve_path(filename).exists()
        
    def delete(self, filename: str) -> bool:
        """删除文件。
        
        Args:
            filename: 文件名或路径
            
        Returns:
            是否成功删除
        """
        file_path = self._resolve_path(filename)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
        
    def copy(self, src: str, dst: str) -> Path:
        """复制文件。
        
        Args:
            src: 源文件
            dst: 目标文件
            
        Returns:
            目标文件路径
        """
        import shutil
        src_path = self._resolve_path(src)
        dst_path = self._resolve_path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return dst_path
        
    def get_size(self, filename: str) -> int:
        """获取文件大小。
        
        Args:
            filename: 文件名或路径
            
        Returns:
            文件大小（字节）
        """
        return self._resolve_path(filename).stat().st_size
