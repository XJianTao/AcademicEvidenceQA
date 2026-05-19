"""模块：最小解析任务契约。

用途概要：保存 M1-B 显式执行所需的状态、检查点和稳定错误码。

核心设计：任务只支持单个显式执行器；不定义队列、租约或重领协议。

工程要点：远端任务标识和取消请求均须持久化，敏感 token 不进入此契约。
"""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from academic_evidence_qa.contracts.documents import DocumentError

ParseTaskState = Literal["queued", "executing", "failed", "cancelled", "completed"]


@dataclass(frozen=True)
class ParseTaskRequest:
    """创建解析任务。

    Args:
        document_version_id: 固定的不可变输入版本。
        parser_backend, parser_config_id: 稳定且不含凭据的解析配置身份。
        allow_remote_processing: 每次导入的明确外发授权。
    Returns:
        可持久化的请求，不启动后台执行。
    """

    document_version_id: UUID
    parser_backend: str
    parser_config_id: str
    allow_remote_processing: bool

    def __post_init__(self) -> None:
        if not self.parser_backend.strip() or not self.parser_config_id.strip():
            raise DocumentError("invalid_parser_selection", 422)


@dataclass(frozen=True)
class ParseTaskCheckpoint:
    """任务检查点；attempt_id 是数据库任务的权威执行身份。"""

    parse_task_id: UUID
    attempt_id: UUID
    state: ParseTaskState
    stage: str
    remote_task_id: str | None
    cancel_requested: bool
    error_code: str | None
