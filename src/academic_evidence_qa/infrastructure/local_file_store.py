"""模块：原件本地存储

流式 SHA-256、服务生成对象键、同文件系统原子发布与内容复用。

核心设计：
  - 内容哈希确定 file_id；
  - 临时文件完整 fsync 后用硬链接原子发布，绝不覆盖既有对象。

工程要点：
  - 仅接收二进制流/UUID，不接收外部路径；
  - 失败清理暂存，数据库回滚残留原件供重试复用。
"""

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid5

FILE_NAMESPACE = UUID("1c588220-f8cf-4fc9-9087-52409dc5b675")
CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class StoredObject:
    """存储对象描述

    传递原件身份与完整性信息。

    Args:
        file_id, object_key: 服务生成的 UUID 与相对键，不是外部指定路径。
        file_sha256, size_bytes: 流式计算的完整哈希与实际字节数。
    Returns:
        不可变的对象描述，供数据库登记与复用校验使用。
    """

    file_id: UUID
    file_sha256: str
    object_key: str
    size_bytes: int


class FileIntegrityError(Exception):
    """已有对象内容与登记不符；阻止复用，不自动修复或覆盖证据。"""


class LocalFileStore:
    """本地原件存储

    在服务自有目录写入和读取不可变对象。

    Args:
        data_root: 可信部署配置，暂存与原件必须位于同一文件系统。
    Returns:
        存储实例；不执行 PDF 预检，不宣称 M1-B 安全防护已实现。
    """

    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()
        self.staging = self.data_root / "staging"
        self.staging.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def object_key(file_id: UUID) -> str:
        """生成原件键

        仅从文件 UUID 构造相对存储位置。

        Args:
            file_id: 服务生成的 UUID；字符串路径与路径穿越输入被拒绝。
        Returns:
            固定 originals 分区内的相对对象键。
        Raises:
            TypeError: 输入不是 UUID。
        """
        if not isinstance(file_id, UUID):
            raise TypeError("file_id 必须为 UUID，不能传入本地路径")
        return f"originals/{file_id.hex[:2]}/{file_id.hex}"

    def write(self, source: BinaryIO) -> StoredObject:
        """流式保存原件

        计算哈希并原子发布或验证复用。

        Args:
            source: 当前游标起的二进制内容流，调用方保有流的生命周期。
        Returns:
            服务生成的对象身份、相对键、哈希和实际字节数。
        Raises:
            FileIntegrityError: 相同身份的已有对象损坏。
            OSError: 磁盘/硬链接不可用；不退化为覆盖式复制。
        并发：硬链接创建具有排他性；竞争者只能验证既有对象，不能覆盖。
        """
        digest = hashlib.sha256()
        size_bytes = 0
        descriptor, temporary = tempfile.mkstemp(prefix="original-", dir=self.staging)
        try:
            with os.fdopen(descriptor, "wb") as output:
                while chunk := source.read(CHUNK_BYTES):
                    output.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            file_sha256 = digest.hexdigest()
            file_id = uuid5(FILE_NAMESPACE, file_sha256)
            stored = StoredObject(file_id, file_sha256, self.object_key(file_id), size_bytes)
            destination = self.data_root / stored.object_key
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                self.verify(stored)
            return stored
        finally:
            Path(temporary).unlink(missing_ok=True)

    def open(self, file_id: UUID) -> BinaryIO:
        """按身份读取

        仅解析服务生成的 UUID 键。

        Args:
            file_id: 数据库已经登记的文件身份，授权检查由上层完成。
        Returns:
            只读二进制文件流，调用方负责关闭；不支持任意路径。
        """
        return (self.data_root / self.object_key(file_id)).open("rb")

    def verify(self, stored: StoredObject) -> None:
        """验证复用对象

        逐块核实大小与哈希，损坏则拒绝。

        Args:
            stored: 已登记或刚计算的对象描述。
        Returns:
            None；成功意味着当前内容一致，不代表断电恢复已验收。
        """
        digest = hashlib.sha256()
        size_bytes = 0
        with self.open(stored.file_id) as source:
            while chunk := source.read(CHUNK_BYTES):
                digest.update(chunk)
                size_bytes += len(chunk)
        if size_bytes != stored.size_bytes or digest.hexdigest() != stored.file_sha256:
            raise FileIntegrityError("已有原件的大小或 SHA-256 不符")
