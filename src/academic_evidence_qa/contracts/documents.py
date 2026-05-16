"""模块：文档与成员契约

定义与 ORM 无关的登记结果、成员命令和可映射 HTTP 的领域错误。

核心设计：
  - 冻结输入/输出；
  - 文件身份、论文身份和每库选版明确分离。

工程要点：
  - 修订冲突使用 409；
  - 此处不定义 HTTP 路由或完整 Job 契约。
"""

from dataclasses import dataclass
from uuid import UUID


class DocumentError(Exception):
    """文档领域异常

    提供稳定错误码而不泄漏数据库详情。

    Args:
        error_code: 可由后续 API 映射的英文错误码。
        status_code: 预期 HTTP 状态，仅为契约，当前不创建 API。
    Returns:
        可由调用方捕获的异常实例。
    """

    def __init__(self, error_code: str, status_code: int = 409):
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(error_code)


@dataclass(frozen=True)
class ParserSelection:
    """解析配置身份

    记录选版对应的明确配置，不执行解析。

    Args:
        parser_backend: 主解析器或离线解析器标识。
        parser_config_id: 已固定参数的配置标识；后续快照按此匹配。
    Returns:
        不可变的配置身份，不隐含远端外发授权。
    """

    parser_backend: str
    parser_config_id: str

    def __post_init__(self) -> None:
        for value in (self.parser_backend, self.parser_config_id):
            if not value.strip() or len(value) > 128:
                raise DocumentError("invalid_parser_selection", 422)


@dataclass(frozen=True)
class MemberCommand:
    """成员管理命令

    把库、论文与期望修订绑定。

    Args:
        library_id: 目标库身份。
        paper_id: 论文身份，不能通过标题自动合并。
        expected_membership_revision: 非负的客户端期望修订。
    Returns:
        可用于加入、显式恢复/选版或软移除的输入。
    """

    library_id: UUID
    paper_id: UUID
    expected_membership_revision: int

    def __post_init__(self) -> None:
        if self.expected_membership_revision < 0:
            raise DocumentError("invalid_membership_revision", 422)


@dataclass(frozen=True)
class RegisteredDocument:
    """登记结果

    返回固定文件和版本身份，不返回 ORM 对象。

    Args:
        paper_id, document_version_id, file_id: 三种独立身份。
        file_sha256, size_bytes: 内容校验信息。
        reused: 本次是否复用了已有数据库版本。
    Returns:
        可跨会话使用的不可变登记结果。
    """

    paper_id: UUID
    document_version_id: UUID
    file_id: UUID
    file_sha256: str
    size_bytes: int
    reused: bool


@dataclass(frozen=True)
class MemberResult:
    """成员操作结果

    区分幂等与实际修订变化。

    Args:
        library_id, paper_id: 成员身份。
        selected_document_version_id: 当前待发布选版，不表示已发布。
        status: active 或 removed。
        membership_revision: 操作后的库修订。
        changed: 是否产生新修订及对应待构建意图。
    Returns:
        与数据库会话生命周期无关的成员结果。
    """

    library_id: UUID
    paper_id: UUID
    selected_document_version_id: UUID
    status: str
    membership_revision: int
    changed: bool
