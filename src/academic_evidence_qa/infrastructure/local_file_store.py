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


@dataclass(frozen=True)
class StoredArtifact:
    """解析产物描述，键与 SHA-256 绑定且永不覆盖。"""

    object_key: str
    sha256: str
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

    def write_verified_staged(
        self,
        staging_path: Path,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> StoredObject:
        """保存已受限预检的服务暂存原件。

        Args:
            staging_path: 必须位于服务生成的 staging/<upload_id>/ 中的文件。
            expected_sha256, expected_size_bytes: 接收层逐块计算的实际值。
        Returns:
            已原子保存或经完整性验证复用的原件描述。
        Raises:
            ValueError: 路径不属于服务暂存区或完整性不一致。
        """
        path = staging_path.resolve()
        try:
            path.relative_to(self.staging)
        except ValueError as error:
            raise ValueError("暂存路径不属于服务 staging 根目录") from error
        if len(path.relative_to(self.staging).parts) != 2 or not path.is_file():
            raise ValueError("暂存路径不是服务生成的单个上传文件")
        with path.open("rb") as source:
            stored = self.write(source)
        if stored.file_sha256 != expected_sha256 or stored.size_bytes != expected_size_bytes:
            raise FileIntegrityError("预检前后的原件完整性不一致")
        return stored

    def write_artifact(self, category: str, source: BinaryIO, *, max_bytes: int) -> StoredArtifact:
        """保存不可变 parser 原始或标准化产物。

        Args:
            category: 固定服务类别，如 raw_parse 或 normalized_parse。
            source: 已在受限子进程中生成的内容流。
            max_bytes: 写入期间再次执行的实际字节上限。
        Returns:
            按内容哈希命名的不可变产物描述。
        """
        if not category.isidentifier() or max_bytes <= 0:
            raise ValueError("无效产物类别或大小上限")
        digest = hashlib.sha256()
        size_bytes = 0
        descriptor, temporary = tempfile.mkstemp(prefix=f"{category}-", dir=self.staging)
        try:
            with os.fdopen(descriptor, "wb") as output:
                while chunk := source.read(CHUNK_BYTES):
                    size_bytes += len(chunk)
                    if size_bytes > max_bytes:
                        raise ValueError("artifact_size_exceeded")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            sha256 = digest.hexdigest()
            key = f"artifacts/{category}/{sha256[:2]}/{sha256}"
            destination = self.data_root / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                artifact = StoredArtifact(key, sha256, size_bytes)
                self.verify_artifact(artifact)
                return artifact
            return StoredArtifact(key, sha256, size_bytes)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def verify_artifact(self, artifact: StoredArtifact) -> None:
        """验证既有不可变产物，不允许损坏内容被缓存复用。"""
        self.verify_artifact_reference(
            artifact.object_key,
            expected_sha256=artifact.sha256,
            expected_size_bytes=artifact.size_bytes,
        )

    def verify_artifact_reference(
        self,
        object_key: str,
        *,
        expected_sha256: str,
        expected_size_bytes: int | None = None,
    ) -> int:
        """验证数据库中已记录的产物引用。

        Args:
            object_key: 数据库保存的相对产物键。
            expected_sha256: 数据库保存的 SHA-256。
            expected_size_bytes: 可选的登记大小；未提供时返回实际大小供调用方记录。
        Returns:
            实际字节数。
        Raises:
            FileIntegrityError: 路径越界、文件不存在或哈希/大小不一致。
        """
        path = (self.data_root / object_key).resolve()
        if self.data_root not in path.parents:
            raise FileIntegrityError("产物键越界")
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with path.open("rb") as source:
                while chunk := source.read(CHUNK_BYTES):
                    digest.update(chunk)
                    size_bytes += len(chunk)
        except OSError as error:
            raise FileIntegrityError("产物文件不可读") from error
        if digest.hexdigest() != expected_sha256 or (
            expected_size_bytes is not None and size_bytes != expected_size_bytes
        ):
            raise FileIntegrityError("已有产物的大小或 SHA-256 不符")
        return size_bytes

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
        try:
            with self.open(stored.file_id) as source:
                while chunk := source.read(CHUNK_BYTES):
                    digest.update(chunk)
                    size_bytes += len(chunk)
        except OSError as error:
            raise FileIntegrityError("已有原件不可读") from error
        if size_bytes != stored.size_bytes or digest.hexdigest() != stored.file_sha256:
            raise FileIntegrityError("已有原件的大小或 SHA-256 不符")
