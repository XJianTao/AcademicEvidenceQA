"""模块名称：解析任务与索引任务契约。

用途概要：保存 M1-B 显式执行所需的状态、检查点和稳定错误码。

核心设计：解析任务沿用显式执行身份；索引任务使用独立不可变租约身份。

工程要点：远端任务标识和取消请求均须持久化，敏感 token 不进入此契约。
"""

from dataclasses import dataclass
import re
from typing import Literal
from uuid import UUID

from academic_evidence_qa.contracts.documents import DocumentError

ParseTaskState = Literal["queued", "executing", "failed", "cancelled", "completed"]

IndexJobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class JobLeaseError(Exception):
    """租约或取消边界错误；仅携带稳定错误码，不含第三方响应。"""


def index_error_code(error):
    """索引失败仅保留稳定码，第三方消息、URL 和输入不进入数据库。"""
    code = str(error)
    if (
        isinstance(error, ValueError) or type(error).__module__.startswith("academic_evidence_qa")
    ) and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code):
        return code
    return "index_job_failed"


@dataclass(frozen=True)
class IndexJobLease:
    """名称：索引任务租约。用途：不可变地绑定一次领取身份。

    Args:
        job_id, library_id, membership_revision: 持久化意图的固定身份。
        owner, version, attempt_id: 执行器、单调 fencing token 与本次专属 attempt。
    Returns:
        可传入短事务核验的上下文；到期时间以数据库为准。
    """

    job_id: UUID
    library_id: UUID
    membership_revision: int
    owner: str
    version: int
    attempt_id: UUID


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
