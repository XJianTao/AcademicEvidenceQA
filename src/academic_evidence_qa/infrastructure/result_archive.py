"""模块：受限解析结果归档处理。

用途概要：在 Windows Job Object 子进程中检查并展开 MinerU ZIP 结果，避免 Zip Slip、链接、重复路径和压缩炸弹。

核心设计：不使用 extractall；每个成员先校验规范化路径，再以排他创建和实际读取计数写入 attempt 目录。

工程要点：失败只影响本次 target_dir，调用者不得让目标目录越出服务控制的 attempt 根目录。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from academic_evidence_qa.infrastructure.process_guard import (
    GuardedProcessRequest,
    ResourceGuardError,
    WindowsProcessGuard,
)


class ArchiveError(RuntimeError):
    """归档错误，仅暴露稳定 error_code。"""

    def __init__(self, error_code: str):
        self.error_code = error_code
        super().__init__(error_code)


@dataclass(frozen=True)
class ExtractedArchive:
    """安全解包结果。"""

    file_count: int
    extracted_bytes: int


def extract_result_archive(
    archive_path: Path,
    target_dir: Path,
    *,
    max_files: int,
    max_extracted_bytes: int,
    max_json_bytes: int,
    timeout_seconds: int,
    memory_mib: int,
    guard: WindowsProcessGuard | None = None,
) -> ExtractedArchive:
    """通过受限子进程展开 ZIP。

    Args:
        archive_path: 已下载且由服务控制的 ZIP。
        target_dir: 当前 attempt 的空目录。
        max_files, max_extracted_bytes, max_json_bytes: 声明和实际写入限额。
        timeout_seconds, memory_mib: 解包子进程限制。
    Returns:
        实际文件数和展开字节数。
    Raises:
        ArchiveError: 路径、大小、格式或资源防护异常。
    """
    result_path = target_dir.parent / f"{target_dir.name}-extract-result.json"
    request = GuardedProcessRequest(
        command=[
            sys.executable,
            "-m",
            "academic_evidence_qa.infrastructure.result_archive",
            "--extract",
            str(archive_path),
            str(target_dir),
            str(result_path),
            str(max_files),
            str(max_extracted_bytes),
            str(max_json_bytes),
        ],
        working_directory=target_dir.parent,
        timeout_seconds=timeout_seconds,
        memory_mib=memory_mib,
    )
    try:
        completed = (guard or WindowsProcessGuard()).run(request)
    except ResourceGuardError as error:
        raise ArchiveError(error.error_code) from error
    if completed.exit_code != 0 or not result_path.is_file() or result_path.stat().st_size > 4096:
        raise ArchiveError("archive_extract_failed")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ArchiveError("archive_extract_failed") from error
    if error_code := payload.get("error_code"):
        raise ArchiveError(str(error_code))
    if not isinstance(payload.get("file_count"), int) or not isinstance(
        payload.get("extracted_bytes"), int
    ):
        raise ArchiveError("archive_extract_failed")
    return ExtractedArchive(payload["file_count"], payload["extracted_bytes"])


def _safe_member_path(name: str) -> PurePosixPath:
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        not name
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or any(":" in part for part in posix.parts)
    ):
        raise ArchiveError("archive_member_path_invalid")
    return posix


def _extract_child(
    archive_path: Path,
    target_dir: Path,
    result_path: Path,
    max_files: int,
    max_extracted_bytes: int,
    max_json_bytes: int,
) -> int:
    target_dir.mkdir(parents=True, exist_ok=False)
    extracted_bytes = 0
    file_count = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            seen: set[str] = set()
            entries = archive.infolist()
            if len(entries) > max_files:
                raise ArchiveError("archive_file_limit_exceeded")
            for entry in entries:
                if entry.is_dir():
                    continue
                relative = _safe_member_path(entry.filename)
                normalized = str(relative).casefold()
                if normalized in seen:
                    raise ArchiveError("archive_duplicate_path")
                seen.add(normalized)
                mode = entry.external_attr >> 16
                if mode and (mode & 0o170000) in {0o120000, 0o060000}:
                    raise ArchiveError("archive_link_member")
                if entry.file_size > max_json_bytes and relative.suffix.lower() == ".json":
                    raise ArchiveError("archive_json_limit_exceeded")
                if entry.file_size > max_extracted_bytes - extracted_bytes:
                    raise ArchiveError("archive_extracted_limit_exceeded")
                destination = (target_dir / Path(*relative.parts)).resolve()
                if target_dir.resolve() not in destination.parents:
                    raise ArchiveError("archive_member_path_invalid")
                destination.parent.mkdir(parents=True, exist_ok=True)
                member_bytes = 0
                with archive.open(entry, "r") as source, destination.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        extracted_bytes += len(chunk)
                        member_bytes += len(chunk)
                        if extracted_bytes > max_extracted_bytes:
                            raise ArchiveError("archive_extracted_limit_exceeded")
                        if relative.suffix.lower() == ".json" and member_bytes > max_json_bytes:
                            raise ArchiveError("archive_json_limit_exceeded")
                        output.write(chunk)
                file_count += 1
        result = {"file_count": file_count, "extracted_bytes": extracted_bytes}
    except (ArchiveError, OSError, zipfile.BadZipFile) as error:
        shutil.rmtree(target_dir, ignore_errors=True)
        result = {"error_code": getattr(error, "error_code", "archive_invalid")}
    result_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    return 0


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", nargs=6)
    args = parser.parse_args()
    if args.extract is None:
        return 2
    archive, target, result, files, total, json_limit = args.extract
    return _extract_child(
        Path(archive), Path(target), Path(result), int(files), int(total), int(json_limit)
    )


if __name__ == "__main__":
    raise SystemExit(_main())
