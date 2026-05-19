"""模块：文档、解析与成员契约

定义与 ORM 无关的登记结果、成员命令和可映射 HTTP 的领域错误。

核心设计：
  - 冻结输入/输出；
  - 文件身份、论文身份和每库选版明确分离。

工程要点：
  - 修订冲突使用 409；
  - 此处不定义 HTTP 路由或完整 Job 契约。
"""

from dataclasses import dataclass, field
from typing import Any, Literal
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


QualityState = Literal["usable", "usable_with_warnings", "rejected"]
PositionReliability = Literal["located", "page_only", "unlocated", "range_incomplete"]


@dataclass(frozen=True)
class ReceivedPdf:
    """名称：受限接收结果。用途概要：描述尚未登记的可信暂存 PDF。

    Args:
        upload_id: 服务生成的暂存目录身份。
        staging_path: 仅服务内部使用的可信绝对路径。
        size_bytes, sha256: 接收层逐块计算的实际值。
    Returns:
        仅能由上传模块创建并交给预检/原件存储的对象。
    """

    upload_id: UUID
    staging_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PdfPreflightResult:
    """名称：PDF 预检结果。用途概要：记录受限子进程给出的结构结论。"""

    page_count: int
    is_encrypted: bool


@dataclass(frozen=True)
class SourceLocation:
    """名称：来源位置。用途概要：保存 parser 原始位置和规范化位置。

    Args:
        page_index: 0-based 物理页；未知时为 None。
        bbox: 旋转后 CropBox 显示坐标中的 0–1 左上原点矩形。
        reliability: 明确区分页级、无位置和不完整范围。
        raw_provenance: 保留可得 parser 字段，不能替代规范化坐标。
    Returns:
        可序列化的不可变来源位置。
    """

    page_index: int | None
    bbox: tuple[float, float, float, float] | None
    reliability: PositionReliability
    raw_provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedBlock:
    """名称：解析块。用途概要：统一文本、表格、公式和算法的保存格式。"""

    block_type: str
    reading_order: int
    text: str
    locations: tuple[SourceLocation, ...]
    raw_provenance: dict[str, Any] = field(default_factory=dict)
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedSection:
    """名称：解析章节。用途概要：保存明确的标题层级，不推断缺失关系。"""

    title: str | None
    level: int | None
    reading_order: int
    locations: tuple[SourceLocation, ...]


@dataclass(frozen=True)
class ParsedDocument:
    """名称：标准解析文档。用途概要：解析器到不可变快照的中间契约。

    Args:
        sections, blocks: 统一阅读顺序的结构，原文始终保留在块内。
        quality_state, quality_metrics: 可复核质量结论及计数。
        metadata: 仅直接从文档确认的元数据和来源。
    Returns:
        可安全序列化为标准化产物的不可变解析结果。
    """

    sections: tuple[ParsedSection, ...]
    blocks: tuple[ParsedBlock, ...]
    quality_state: QualityState
    quality_metrics: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParseSnapshot:
    """名称：解析快照。用途概要：绑定固定版本与固定解析产物身份。"""

    parse_snapshot_id: UUID
    document_version_id: UUID
    input_sha256: str
    parser_backend: str
    parser_config_id: str
    adapter_version: str
    quality_state: QualityState
    raw_artifact_sha256: str
    normalized_artifact_sha256: str
