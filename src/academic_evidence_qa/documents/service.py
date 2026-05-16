"""模块：文档与论文库事务服务

登记不可变版本，编排成员加入、软移除、显式恢复与选版。

核心设计：
  - 修订检查发生在库行锁内；
  - 成员、递增修订、待构建意图同事务 flush。

工程要点：
  - 不自行提交；
  - 异常须回滚整个调用事务；
  - 文件落盘不具备数据库事务性，残留保留供复用。
"""

import asyncio
from datetime import datetime, timezone
from typing import BinaryIO
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import SessionTransactionOrigin

from academic_evidence_qa.contracts.documents import (
    DocumentError,
    MemberCommand,
    MemberResult,
    ParserSelection,
    RegisteredDocument,
)
from academic_evidence_qa.infrastructure.database import repositories
from academic_evidence_qa.infrastructure.database.models import (
    BuildIntent,
    DocumentVersion,
    Library,
    LibraryPaper,
    Paper,
    StoredFile,
)
from academic_evidence_qa.infrastructure.local_file_store import LocalFileStore, StoredObject


class DocumentService:
    """名称：文档服务。用途概要：集中实施身份复用和成员修订规则。

    Args:
        file_store: 可信根目录下的原件存储。
    Returns:
        不持有数据库会话、不加载模型的服务实例。
    事务：每个公共操作传入显式 begin 事务，不允许先登记后捕获冲突并提交。
    """

    def __init__(self, file_store: LocalFileStore):
        self.file_store = file_store

    @staticmethod
    def _require_transaction(session: AsyncSession) -> None:
        transaction = session.sync_session.get_transaction()
        if transaction is None or transaction.origin != SessionTransactionOrigin.BEGIN:
            raise RuntimeError("文档服务需要调用方显式开启事务")

    async def create_library(self, session: AsyncSession) -> UUID:
        """创建库

        创建修订为零、尚无发布指针的库。

        Args:
            session: 调用方显式事务。
        Returns:
            新 library_id；调用方提交后才持久生效。
        """
        self._require_transaction(session)
        library = Library()
        session.add(library)
        await session.flush()
        return library.library_id

    async def register_document(
        self,
        session: AsyncSession,
        source: BinaryIO,
        *,
        paper_id: UUID | None = None,
        title: str | None = None,
    ) -> RegisteredDocument:
        """登记原件/版本

        按哈希复用；新内容可关联已有论文。

        Args:
            session: 显式事务；不要在同一事务登记后反向获取 Library 锁。
            source: 二进制流，本阶段假定上层已确认内容；不替代 PDF 预检。
            paper_id: 明确身份；与已有哈希所属论文冲突则拒绝。
            title: 仅用于新论文的可选元数据，不用于自动合并。
        Returns:
            固定文件/版本身份；没有成员变更或构建意图。
        Raises:
            DocumentError: paper_identity_conflict 或 paper_not_found。
        """
        self._require_transaction(session)
        stored = await asyncio.to_thread(self.file_store.write, source)
        return await self._register_object(session, stored, paper_id, title)

    async def register_in_library(
        self,
        session: AsyncSession,
        source: BinaryIO,
        *,
        library_id: UUID,
        expected_membership_revision: int,
        parser: ParserSelection,
        paper_id: UUID | None = None,
        title: str | None = None,
    ) -> tuple[RegisteredDocument, MemberResult]:
        """登记到库

        保存候选，并且仅为不存在的成员默认选择本次版本。

        Args:
            session: 显式事务；失败必须整体回滚。
            source: 二进制流，不接受外部路径。
            library_id, expected_membership_revision: 库身份及期望修订，先锁库再锁哈希。
            parser: 新成员配置；已有成员配置不被普通登记改变。
            paper_id, title: 可选明确论文身份和新论文元数据。
        Returns:
            登记与成员结果；已有 removed 成员保持 removed，已有选版保持不变。
        """
        library = await self._checked_library(session, library_id, expected_membership_revision)
        stored = await asyncio.to_thread(self.file_store.write, source)
        document = await self._register_object(session, stored, paper_id, title)
        member = await repositories.find_member(session, library_id, document.paper_id)
        changed = member is None
        if member is None:
            member = self._new_member(
                library_id, document.paper_id, document.document_version_id, parser
            )
            session.add(member)
            await self._record_change(session, library)
        return document, self._member_result(library, member, changed)

    async def set_member(
        self,
        session: AsyncSession,
        command: MemberCommand,
        selected_document_version_id: UUID,
        parser: ParserSelection,
    ) -> MemberResult:
        """显式加入/恢复/选版

        唯一允许改变已有选版或恢复的操作。

        Args:
            session: 显式事务，锁保留至提交或回滚。
            command: 库、论文和期望修订。
            selected_document_version_id: 必须属于 command.paper_id 的版本。
            parser: 选定配置；配置变化同样递增修订。
        Returns:
            active 成员结果，幂等操作 changed=false 且不新增意图。
        Raises:
            DocumentError: 修订陈旧、版本不存在或版本不属于论文。
        """
        library = await self._checked_library(
            session,
            command.library_id,
            command.expected_membership_revision,
        )
        version = await session.get(DocumentVersion, selected_document_version_id)
        if version is None or version.paper_id != command.paper_id:
            raise DocumentError("document_version_paper_mismatch", 422)
        member = await repositories.find_member(session, command.library_id, command.paper_id)
        desired = (
            selected_document_version_id,
            parser.parser_backend,
            parser.parser_config_id,
            "active",
        )
        changed = member is None or self._member_state(member) != desired
        if member is None:
            member = self._new_member(
                command.library_id, command.paper_id, selected_document_version_id, parser
            )
            session.add(member)
        elif changed:
            (
                member.selected_document_version_id,
                member.parser_backend,
                member.parser_config_id,
                member.status,
            ) = desired
            member.updated_at = datetime.now(timezone.utc)
        if changed:
            await self._record_change(session, library)
        return self._member_result(library, member, changed)

    async def remove_member(self, session: AsyncSession, command: MemberCommand) -> MemberResult:
        """软移除

        只变成员状态，不删除文件/版本或活动指针。

        Args:
            session: 显式事务。
            command: 库、论文和期望修订。
        Returns:
            removed 结果；重复移除不递增，陈旧修订仍被拒绝。
        """
        library = await self._checked_library(
            session,
            command.library_id,
            command.expected_membership_revision,
        )
        member = await repositories.find_member(session, command.library_id, command.paper_id)
        if member is None:
            raise DocumentError("member_not_found", 404)
        changed = member.status != "removed"
        if changed:
            member.status = "removed"
            member.updated_at = datetime.now(timezone.utc)
            await self._record_change(session, library)
        return self._member_result(library, member, changed)

    async def _checked_library(
        self,
        session: AsyncSession,
        library_id: UUID,
        expected_revision: int,
    ) -> Library:
        self._require_transaction(session)
        if expected_revision < 0:
            raise DocumentError("invalid_membership_revision", 422)
        library = await repositories.lock_library(session, library_id)
        if library is None:
            raise DocumentError("library_not_found", 404)
        if library.membership_revision != expected_revision:
            raise DocumentError("stale_membership_revision")
        return library

    async def _register_object(
        self,
        session: AsyncSession,
        stored: StoredObject,
        paper_id: UUID | None,
        title: str | None,
    ) -> RegisteredDocument:
        await repositories.lock_file_hash(session, stored.file_sha256)
        version = await repositories.find_version_by_hash(session, stored.file_sha256)
        reused = version is not None
        if version is not None:
            if paper_id is not None and paper_id != version.paper_id:
                raise DocumentError("paper_identity_conflict")
            file = await session.get(StoredFile, version.file_id)
            if file is None or (
                file.file_id,
                file.file_sha256,
                file.size_bytes,
                file.object_key,
            ) != (
                stored.file_id,
                stored.file_sha256,
                stored.size_bytes,
                stored.object_key,
            ):
                raise DocumentError("file_registration_mismatch")
        else:
            if paper_id is None:
                paper = Paper(title=title)
                session.add(paper)
                await session.flush()
                paper_id = paper.paper_id
            elif await session.get(Paper, paper_id) is None:
                raise DocumentError("paper_not_found", 404)
            session.add(
                StoredFile(
                    file_id=stored.file_id,
                    file_sha256=stored.file_sha256,
                    object_key=stored.object_key,
                    size_bytes=stored.size_bytes,
                )
            )
            await session.flush()
            version = DocumentVersion(
                paper_id=paper_id, file_id=stored.file_id, file_sha256=stored.file_sha256
            )
            session.add(version)
            await session.flush()
        return RegisteredDocument(
            version.paper_id,
            version.document_version_id,
            version.file_id,
            version.file_sha256,
            stored.size_bytes,
            reused,
        )

    @staticmethod
    def _new_member(
        library_id: UUID,
        paper_id: UUID,
        version_id: UUID,
        parser: ParserSelection,
    ) -> LibraryPaper:
        return LibraryPaper(
            library_id=library_id,
            paper_id=paper_id,
            selected_document_version_id=version_id,
            parser_backend=parser.parser_backend,
            parser_config_id=parser.parser_config_id,
            status="active",
        )

    @staticmethod
    def _member_state(member: LibraryPaper) -> tuple[UUID, str, str, str]:
        return (
            member.selected_document_version_id,
            member.parser_backend,
            member.parser_config_id,
            member.status,
        )

    @staticmethod
    async def _record_change(session: AsyncSession, library: Library) -> None:
        """登记实际修订

        将成员变化与待构建意图原子持久化。

        Args:
            session: 同一调用方事务，成员已加入该会话。
            library: 已通过期望修订校验且持有行锁的库。
        Returns:
            None；flush 不等于 commit，后续失败仍回滚成员、修订与意图。
        """
        # 一个实际修订一个持久意图；flush 失败或调用方回滚时，三者一起撤销。
        library.membership_revision += 1
        session.add(
            BuildIntent(
                library_id=library.library_id, membership_revision=library.membership_revision
            )
        )
        await session.flush()

    @staticmethod
    def _member_result(library: Library, member: LibraryPaper, changed: bool) -> MemberResult:
        return MemberResult(
            library.library_id,
            member.paper_id,
            member.selected_document_version_id,
            member.status,
            library.membership_revision,
            changed,
        )
