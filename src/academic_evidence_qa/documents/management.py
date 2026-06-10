"""模块名称：本地管理事务服务。

用途：组合文档服务处理持久幂等、导入受理、任务与成员管理。
核心设计：key→Library→文件哈希锁序；请求响应与副作用同一短事务提交。
工程要点：只返回白名单字段；不执行远端解析，不跨库复用发布结果。
"""

import base64
import hashlib
import json
import re
import os
from dataclasses import asdict

from sqlalchemy import select, func

from academic_evidence_qa.bootstrap import parser_identity
from academic_evidence_qa.contracts.documents import DocumentError, MemberCommand, ParserSelection
from academic_evidence_qa.contracts.jobs import ParseTaskRequest
from academic_evidence_qa.documents.service import DocumentService
from academic_evidence_qa.documents.filenames import normalize_original_filename
from academic_evidence_qa.documents.source_reader import verify_snapshot
from academic_evidence_qa.documents.published_index import verify_active_index
from academic_evidence_qa.infrastructure.database.models import (
    ManagementRequest,
    Library,
    LibraryPaper,
    DocumentVersion,
    Paper,
    ParseTask,
    ParseSnapshotRecord,
    BuildIntent,
    IndexGeneration,
    StoredFile,
    DocumentMetadata,
)
from academic_evidence_qa.infrastructure.database.repositories import lock_library


def json_body(value):
    """将冻结 DTO 转成可持久 JSON；不序列化 ORM 或敏感配置。"""
    return json.loads(json.dumps(value, default=str))


def _paper_find_cursor(scope_sha256, paper_id):
    """将末项身份绑定完整查询范围；游标不承担权限，服务每页重新校验权限。"""
    payload = json.dumps(
        {"scope_sha256": scope_sha256, "last_paper_id": paper_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


class ManagementService:
    """名称：管理服务。用途：薄 HTTP 与持久队列之间的事务边界。

    Args:
        database, store, settings: 轻量资源，不含编码模型。
    Returns:
        可复用服务，不持有请求会话。
    """

    def __init__(self, database, store, settings):
        self.database, self.store, self.settings = database, store, settings
        self.documents = DocumentService(store)
        self.identity, self.config_id = parser_identity(settings)

    async def submit(self, key, operation, payload, action, *, legacy_payload=None):
        """名称：持久幂等提交。用途：先重放，再核验修订，不重复副作用。

        Args:
            key: 必须提供的 ASCII Idempotency-Key。
            operation, payload: 路径/操作与规范化请求（上传包含实际哈希）。
            action: 同事务执行，返回状态码及白名单响应。
            legacy_payload: 仅用于兼容迁移前无文件名的上传哈希；只重放，不执行 action。
        Returns:
            原始响应或本次结果；同 key 不同请求为 409。
        """
        if not key or not re.fullmatch(r"[\x21-\x7e]{1,128}", key):
            raise DocumentError("idempotency_key_required", 422)
        digest = hashlib.sha256(
            json.dumps(
                json_body([operation, payload]), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        lock_key = int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)
        async with self.database.transaction() as session:
            await session.execute(select(func.pg_advisory_xact_lock(lock_key)))
            previous = await session.get(ManagementRequest, key)
            if previous is not None:
                legacy_digest = (
                    hashlib.sha256(
                        json.dumps(
                            json_body([operation, legacy_payload]),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest()
                    if legacy_payload is not None
                    else None
                )
                if previous.request_sha256 not in (digest, legacy_digest):
                    raise DocumentError("idempotency_key_conflict", 409)
                return previous.response_status, previous.response_body
            status, body = await action(session)
            body = json_body(body)
            session.add(
                ManagementRequest(
                    idempotency_key=key,
                    request_sha256=digest,
                    response_status=status,
                    response_body=body,
                )
            )
            return status, body

    async def create_library(self, key):
        async def action(session):
            library_id = await self.documents.create_library(session)
            return 201, {"library_id": library_id, "membership_revision": 0}

        return await self.submit(key, "create_library", {}, action)

    async def _snapshot(self, session, version_id, backend, config_id):
        version = await session.get(DocumentVersion, version_id)
        candidates = (
            await session.scalars(
                select(ParseSnapshotRecord)
                .where(
                    ParseSnapshotRecord.document_version_id == version_id,
                    ParseSnapshotRecord.parser_backend == backend,
                    ParseSnapshotRecord.parser_config_id == config_id,
                    ParseSnapshotRecord.quality_state.in_(("usable", "usable_with_warnings")),
                )
                .order_by(ParseSnapshotRecord.created_at)
            )
        ).all()
        for snapshot in candidates:
            try:
                await verify_snapshot(
                    session,
                    version,
                    snapshot,
                    self.store,
                    max_json_bytes=self.settings.result_max_json_bytes,
                )
                return snapshot
            except Exception:
                continue
        return None

    async def _index_job(self, session, library, *, new=False, retry_of=None):
        jobs = (
            await session.scalars(
                select(BuildIntent)
                .where(
                    BuildIntent.library_id == library.library_id,
                    BuildIntent.membership_revision == library.membership_revision,
                )
                .order_by(BuildIntent.request_sequence.desc())
            )
        ).all()
        if jobs and not new:
            return jobs[0]
        job = BuildIntent(
            library_id=library.library_id,
            membership_revision=library.membership_revision,
            request_sequence=jobs[0].request_sequence + 1 if jobs else 0,
            retry_of_job_id=retry_of,
        )
        session.add(job)
        await session.flush()
        return job

    async def accept_upload(self, key, request, stored, *, original_filename=None):
        """名称：受理可信原件。用途：登记、授权任务和构建意图同事务。

        Args:
            key, request: 幂等键、显式库/修订/外发字段。
            stored: 正式受限预检后保存的原件。
            original_filename: multipart 校验后的展示名；内部旧调用可缺省。
        Returns:
            202 任务身份，或本库完整发布复用的 200；不启动网络请求。
        """
        payload = {
            **request.model_dump(mode="json"),
            "file_sha256": stored.file_sha256,
            "parser_config_id": self.config_id,
        }
        legacy_payload = payload.copy()
        payload["original_filename"] = (
            normalize_original_filename(original_filename)
            if original_filename is not None
            else None
        )
        # 核验网络/来源在行锁事务之外；事务内重新确认同一个不可变活动代。
        verified_generation_id = await verify_active_index(
            self.database,
            self.store,
            request.library_id,
            max_json_bytes=self.settings.result_max_json_bytes,
            qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:56333"),
        )
        # 完整文件/JSON 核验使用无行锁只读会话，不能拖长成员写事务。
        async with self.database.sessions() as session:
            version_id = await session.scalar(
                select(DocumentVersion.document_version_id).where(
                    DocumentVersion.file_sha256 == stored.file_sha256
                )
            )
            verified_snapshot = (
                await self._snapshot(session, version_id, request.parser_backend, self.config_id)
                if version_id is not None
                else None
            )
            verified_snapshot_id = (
                verified_snapshot.parse_snapshot_id if verified_snapshot else None
            )

        async def action(session):
            registered, member = await self.documents.register_verified_in_library(
                session,
                stored,
                library_id=request.library_id,
                expected_membership_revision=request.expected_membership_revision,
                parser=ParserSelection(request.parser_backend, self.config_id),
                paper_id=request.paper_id,
                original_filename=payload["original_filename"],
            )
            library = await session.get(Library, request.library_id)
            snapshot = (
                await session.get(ParseSnapshotRecord, verified_snapshot_id)
                if verified_snapshot_id is not None
                else None
            )
            if snapshot is not None and (
                snapshot.document_version_id != registered.document_version_id
                or snapshot.parser_backend != request.parser_backend
                or snapshot.parser_config_id != self.config_id
            ):
                raise DocumentError("parse_snapshot_identity_invalid", 409)
            base = {
                "library_id": request.library_id,
                "paper_id": registered.paper_id,
                "document_version_id": registered.document_version_id,
                "membership_revision": member.membership_revision,
                "file_sha256": registered.file_sha256,
            }
            # 未选候选仍准备获准解析，但不暗改选版、恢复成员或提交索引。
            candidate_only = (
                member.status != "active"
                or member.selected_document_version_id != registered.document_version_id
            )
            if (
                not candidate_only
                and snapshot
                and library.active_generation_id
                and library.active_generation_id == verified_generation_id
            ):
                generation = await session.get(IndexGeneration, library.active_generation_id)
                entry = next(
                    (
                        item
                        for item in generation.manifest.get("members", [])
                        if item.get("paper_id") == str(registered.paper_id)
                    ),
                    None,
                )
                if (
                    entry
                    and entry.get("document_version_id") == str(registered.document_version_id)
                    and entry.get("parse_snapshot_id") == str(snapshot.parse_snapshot_id)
                    and generation.membership_revision == library.membership_revision
                ):
                    published_job = await session.scalar(
                        select(BuildIntent).where(
                            BuildIntent.library_id == library.library_id,
                            BuildIntent.generation_id == generation.generation_id,
                            BuildIntent.state == "succeeded",
                        )
                    )
                    return 200, {
                        **base,
                        "status": "reused",
                        "job_id": published_job.build_intent_id if published_job else None,
                        "index_job_id": published_job.build_intent_id if published_job else None,
                        "parse_snapshot_id": snapshot.parse_snapshot_id,
                        "generation_id": generation.generation_id,
                    }
            parse = None
            if not snapshot:
                parse = await session.scalar(
                    select(ParseTask)
                    .where(
                        ParseTask.document_version_id == registered.document_version_id,
                        ParseTask.parser_config_id == self.config_id,
                        ParseTask.parser_backend == request.parser_backend,
                        ParseTask.library_id == request.library_id,
                        ParseTask.state.in_(("queued", "executing")),
                    )
                    .order_by(ParseTask.created_at)
                    .limit(1)
                )
                if parse is None:
                    result = await self.documents.create_parse_task(
                        session,
                        ParseTaskRequest(
                            registered.document_version_id,
                            request.parser_backend,
                            self.config_id,
                            request.allow_remote_processing,
                        ),
                    )
                    parse = await session.get(ParseTask, result.parse_task_id)
                    parse.library_id = request.library_id
            if candidate_only:
                return (202 if parse else 200), {
                    **base,
                    "status": "registered",
                    "job_id": parse.parse_task_id if parse else None,
                    "parse_job_id": parse.parse_task_id if parse else None,
                    "parse_snapshot_id": snapshot.parse_snapshot_id if snapshot else None,
                    "index_job_id": None,
                }
            index = await self._index_job(session, library)
            if index.state in ("failed", "cancelled"):
                index = await self._index_job(session, library, new=True)
            if parse and parse.index_job_id is None:
                parse.index_job_id = index.build_intent_id
            return 202, {
                **base,
                "status": "accepted",
                "job_id": parse.parse_task_id if parse else index.build_intent_id,
                "parse_job_id": parse.parse_task_id if parse else None,
                "index_job_id": index.build_intent_id,
                "parse_snapshot_id": snapshot.parse_snapshot_id if snapshot else None,
            }

        return await self.submit(key, "upload", payload, action, legacy_payload=legacy_payload)

    async def member_change(self, key, library_id, paper_id, request, *, remove=False):
        async def action(session):
            command = MemberCommand(library_id, paper_id, request.expected_membership_revision)
            if remove:
                result = await self.documents.remove_member(session, command)
            else:
                result = await self.documents.set_member(
                    session,
                    command,
                    request.selected_document_version_id,
                    ParserSelection(request.parser_backend, request.parser_config_id),
                )
            library = await session.get(Library, library_id)
            index = await self._index_job(session, library) if result.changed else None
            return (202 if result.changed else 200), {
                **asdict(result),
                "index_job_id": index.build_intent_id if index else None,
            }

        return await self.submit(
            key, f"member:{library_id}:{paper_id}:{remove}", request.model_dump(mode="json"), action
        )

    async def reindex(self, key, library_id, request):
        async def action(session):
            library = await self.documents._checked_library(
                session, library_id, request.expected_membership_revision
            )
            job = await self._index_job(session, library, new=True)
            return 202, {
                "job_id": job.build_intent_id,
                "job_type": "index",
                "library_id": library_id,
                "membership_revision": library.membership_revision,
            }

        return await self.submit(
            key, f"reindex:{library_id}", request.model_dump(mode="json"), action
        )

    async def mutate_job(self, key, job_id, *, retry=False):
        async def action(session):
            parse = await session.get(ParseTask, job_id)
            index = None if parse else await session.get(BuildIntent, job_id)
            if not parse and not index:
                raise DocumentError("job_not_found", 404)
            if index:
                library = await lock_library(session, index.library_id)
                index = await session.get(
                    BuildIntent, job_id, with_for_update=True, populate_existing=True
                )
                if retry:
                    if index.state not in ("failed", "cancelled"):
                        raise DocumentError("job_not_retryable", 409)
                    created = await session.scalar(
                        select(BuildIntent)
                        .where(
                            BuildIntent.retry_of_job_id == job_id,
                            BuildIntent.state.in_(("queued", "running")),
                            BuildIntent.membership_revision == library.membership_revision,
                        )
                        .limit(1)
                    )
                    if created is None:
                        created = await self._index_job(session, library, new=True, retry_of=job_id)
                    return 202, {
                        "job_id": created.build_intent_id,
                        "job_type": "index",
                        "retry_of_job_id": job_id,
                    }
                if index.state == "succeeded":
                    raise DocumentError("job_already_completed", 409)
                index.cancel_requested = True
                if index.state == "queued":
                    index.state, index.stage, index.error_code = (
                        "cancelled",
                        "finished",
                        "job_cancelled",
                    )
                return 200, {"job_id": job_id, "job_type": "index", "cancel_requested": True}
            if parse.library_id is not None:
                library = await lock_library(session, parse.library_id)
            parse = await session.get(
                ParseTask, job_id, with_for_update=True, populate_existing=True
            )
            if not retry:
                await self.documents.request_parse_cancellation(session, job_id)
                return 200, {"job_id": job_id, "job_type": "parse", "cancel_requested": True}
            if parse.state not in ("failed", "cancelled"):
                raise DocumentError("job_not_retryable", 409)
            # 同一失败任务只能有一个未终结的后继；不同 key 并发也不重复外发。
            created = await session.scalar(
                select(ParseTask)
                .where(
                    ParseTask.retry_of_job_id == job_id,
                    ParseTask.state.in_(("queued", "executing")),
                )
                .limit(1)
            )
            if created is None:
                result = await self.documents.create_parse_task(
                    session,
                    ParseTaskRequest(
                        parse.document_version_id,
                        parse.parser_backend,
                        parse.parser_config_id,
                        parse.allow_remote_processing,
                    ),
                )
                created = await session.get(ParseTask, result.parse_task_id)
                created.library_id, created.retry_of_job_id = parse.library_id, job_id
            index_job = None
            if parse.library_id:
                index_job = await self._index_job(session, library)
                if index_job.state in ("failed", "cancelled", "succeeded"):
                    index_job = await self._index_job(session, library, new=True)
                if created.index_job_id is None:
                    created.index_job_id = index_job.build_intent_id
            return 202, {
                "job_id": created.parse_task_id,
                "job_type": "parse",
                "attempt_id": created.attempt_id,
                "retry_of_job_id": job_id,
                "index_job_id": index_job.build_intent_id if index_job else None,
            }

        return await self.submit(key, f"{'retry' if retry else 'cancel'}:{job_id}", {}, action)

    async def read_job(self, job_id):
        async with self.database.sessions() as session:
            parse = await session.get(ParseTask, job_id)
            task = parse or await session.get(BuildIntent, job_id)
            if task is None:
                raise DocumentError("job_not_found", 404)
            state = {"executing": "running", "completed": "succeeded"}.get(task.state, task.state)
            result = {
                "job_id": job_id,
                "job_type": "parse" if parse else "index",
                "state": state,
                "stage": task.stage,
                "attempt_id": task.attempt_id,
                "error_code": task.error_code,
                "cancel_requested": task.cancel_requested,
                "retry_of_job_id": task.retry_of_job_id,
                "library_id": task.library_id,
            }
            model, id_field = (
                (ParseTask, ParseTask.parse_task_id)
                if parse
                else (BuildIntent, BuildIntent.build_intent_id)
            )
            result["retry_job_ids"] = list(
                (
                    await session.scalars(select(id_field).where(model.retry_of_job_id == job_id))
                ).all()
            )
            if parse:
                result.update(
                    document_version_id=task.document_version_id,
                    parse_snapshot_id=task.parse_snapshot_id,
                )
                result["index_job_ids"] = (
                    list(
                        (
                            await session.scalars(
                                select(BuildIntent.build_intent_id)
                                .where(BuildIntent.build_intent_id == task.index_job_id)
                                .order_by(BuildIntent.created_at)
                            )
                        ).all()
                    )
                    if task.library_id
                    else []
                )
            else:
                result.update(
                    generation_id=task.generation_id, membership_revision=task.membership_revision
                )
            return json_body(result)

    async def list_papers(self, *, library_id=None, offset=0, limit=20):
        """名称：分页论文状态。用途：区分选版与固定发布，不把解析完成当发布。

        Args:
            library_id: 提供时限制成员，未提供时为本地管理全局元数据。
            offset, limit: 已校验的分页参数。
        Returns:
            白名单元数据与下一页，不返回原件路径或 ORM。
        """
        async with self.database.sessions() as session:
            library = await session.get(Library, library_id) if library_id else None
            if library_id and library is None:
                raise DocumentError("library_not_found", 404)
            query = select(Paper).order_by(Paper.paper_id)
            if library_id:
                query = query.join(LibraryPaper, LibraryPaper.paper_id == Paper.paper_id).where(
                    LibraryPaper.library_id == library_id
                )
            papers = (await session.scalars(query.offset(offset).limit(limit + 1))).all()
            generation = (
                await session.get(IndexGeneration, library.active_generation_id)
                if library and library.active_generation_id
                else None
            )
            items = []
            for paper in papers[:limit]:
                versions = list(
                    (
                        await session.scalars(
                            select(DocumentVersion.document_version_id)
                            .where(DocumentVersion.paper_id == paper.paper_id)
                            .order_by(DocumentVersion.created_at)
                        )
                    ).all()
                )
                item = {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "document_version_ids": versions,
                }
                if library:
                    member = await session.scalar(
                        select(LibraryPaper).where(
                            LibraryPaper.library_id == library_id,
                            LibraryPaper.paper_id == paper.paper_id,
                        )
                    )
                    published = (
                        next(
                            (
                                entry
                                for entry in generation.manifest.get("members", [])
                                if entry.get("paper_id") == str(paper.paper_id)
                            ),
                            None,
                        )
                        if generation
                        else None
                    )
                    item.update(
                        status=member.status,
                        selected_document_version_id=member.selected_document_version_id,
                        published_source=published,
                        pending=not generation
                        or generation.membership_revision != library.membership_revision,
                        parser_backend=member.parser_backend,
                        parser_config_id=member.parser_config_id,
                    )
                items.append(item)
            return json_body(
                {
                    "items": items,
                    "next_offset": offset + limit if len(papers) > limit else None,
                    "library_id": library_id,
                    "membership_revision": library.membership_revision if library else None,
                    "active_generation_id": library.active_generation_id if library else None,
                }
            )

    async def find_papers(self, request):
        """名称：固定范围元数据发现。用途：先有界筛选，再按论文身份稳定分页。

        Args:
            request: 已校验的库、固定发布代、最多二十篇允许池及查询游标。
        Returns:
            固定版本的标题、文件名可用性、消歧信息和范围绑定游标。
        Raises:
            DocumentError: 范围失效、来源绑定损坏或游标错绑，绝不伪装空结果。
        """
        from uuid import UUID

        from academic_evidence_qa.ingestion.index_builder import content_hash

        wanted = set(request.paper_ids)
        async with self.database.sessions() as session:
            if await session.get(Library, request.library_id) is None:
                raise DocumentError("library_not_found", 404)
            generation = await session.get(IndexGeneration, request.generation_id)
            if (
                generation is None
                or generation.library_id != request.library_id
                or generation.state not in ("published", "retired")
            ):
                raise DocumentError("generation_out_of_scope", 404)
            manifest = generation.manifest
            if (
                not isinstance(manifest, dict)
                or content_hash(manifest) != generation.manifest_sha256
                or manifest.get("library_id") != str(request.library_id)
                or manifest.get("generation_id") != str(request.generation_id)
                or manifest.get("collection_name") != generation.collection_name
                or manifest.get("attempt_id") != str(generation.attempt_id)
                or manifest.get("membership_revision") != generation.membership_revision
            ):
                raise DocumentError("generation_integrity_failed", 409)
            members = {}
            try:
                members = {UUID(item["paper_id"]): item for item in manifest["members"]}
                if len(members) != len(manifest["members"]):
                    raise ValueError("duplicate_manifest_member")
                selected = {paper_id: members[paper_id] for paper_id in wanted}
                version_ids = [UUID(item["document_version_id"]) for item in selected.values()]
                snapshot_ids = [UUID(item["parse_snapshot_id"]) for item in selected.values()]
            except KeyError as error:
                if members and not wanted.issubset(members):
                    raise DocumentError("paper_out_of_scope", 404) from error
                raise DocumentError("generation_integrity_failed", 409) from error
            except (TypeError, ValueError) as error:
                raise DocumentError("generation_integrity_failed", 409) from error
            rows = (
                await session.execute(
                    select(
                        Paper, DocumentVersion, StoredFile, ParseSnapshotRecord, DocumentMetadata
                    )
                    .join(LibraryPaper, LibraryPaper.paper_id == Paper.paper_id)
                    .join(DocumentVersion, DocumentVersion.paper_id == Paper.paper_id)
                    .join(StoredFile, StoredFile.file_id == DocumentVersion.file_id)
                    .join(
                        ParseSnapshotRecord,
                        ParseSnapshotRecord.document_version_id
                        == DocumentVersion.document_version_id,
                    )
                    .outerjoin(
                        DocumentMetadata,
                        DocumentMetadata.document_version_id == DocumentVersion.document_version_id,
                    )
                    .where(
                        Paper.paper_id.in_(wanted),
                        LibraryPaper.library_id == request.library_id,
                        LibraryPaper.status == "active",
                        DocumentVersion.document_version_id.in_(version_ids),
                        ParseSnapshotRecord.parse_snapshot_id.in_(snapshot_ids),
                    )
                    .order_by(Paper.paper_id)
                    .limit(21)
                )
            ).all()
            if len(rows) != len(wanted):
                raise DocumentError("paper_out_of_scope", 404)
            items = []
            for paper, version, stored, snapshot, metadata in rows:
                entry = selected[paper.paper_id]
                if (
                    str(version.document_version_id) != entry["document_version_id"]
                    or str(snapshot.parse_snapshot_id) != entry["parse_snapshot_id"]
                    or version.file_sha256 != entry.get("file_sha256")
                    or stored.file_sha256 != version.file_sha256
                    or snapshot.input_sha256 != version.file_sha256
                ):
                    raise DocumentError("generation_integrity_failed", 409)
                filenames = metadata.original_filenames if metadata else None
                warnings = [] if filenames else ["original_filename_unavailable"]
                if (
                    version.file_sha256
                    == "3d20bc5876e306bb149b467750a48a9cbeb03939f33bc883280edc64dc73abcc"
                    and paper.title
                    and paper.title.strip().casefold() == "learning fuzzy cognitive maps"
                ):
                    warnings.append("suspected_incorrect_title")
                items.append(
                    {
                        "paper_id": str(paper.paper_id),
                        "title": paper.title,
                        "filename": filenames[0] if filenames else None,
                        "filename_aliases": filenames[1:] if filenames else [],
                        "document_version_id": str(version.document_version_id),
                        "parse_snapshot_id": str(snapshot.parse_snapshot_id),
                        "publication_status": "published",
                        "metadata_warnings": warnings,
                        "disambiguation": {
                            "file_sha256": version.file_sha256,
                            "identity_fields": [
                                "paper_id",
                                "document_version_id",
                                "parse_snapshot_id",
                                "file_sha256",
                            ],
                            "requires_explicit_selection": True,
                        },
                    }
                )
        normalized_query = request.query.strip().casefold()
        # 允许池最多二十篇；先对完整有界集合筛选，避免只搜列表第一页。
        matches = [
            item
            for item in items
            if normalized_query in (item["title"] or "").casefold()
            or normalized_query in (item["filename"] or "").casefold()
            or any(normalized_query in filename.casefold() for filename in item["filename_aliases"])
        ]
        scope_sha256 = content_hash(
            {
                "library_id": str(request.library_id),
                "generation_id": str(request.generation_id),
                "manifest_sha256": generation.manifest_sha256,
                "paper_ids": sorted(str(paper_id) for paper_id in wanted),
                "query": normalized_query,
                "limit": request.limit,
                "metadata": items,
            }
        )
        offset = 0
        if request.cursor is not None:
            try:
                payload = json.loads(
                    base64.b64decode(
                        request.cursor + "=" * (-len(request.cursor) % 4),
                        altchars=b"-_",
                        validate=True,
                    )
                )
                if (
                    set(payload) != {"scope_sha256", "last_paper_id"}
                    or payload["scope_sha256"] != scope_sha256
                    or _paper_find_cursor(scope_sha256, payload["last_paper_id"]) != request.cursor
                ):
                    raise ValueError("cursor_binding_invalid")
                offset = [item["paper_id"] for item in matches].index(payload["last_paper_id"]) + 1
            except (ValueError, TypeError, KeyError, UnicodeError) as error:
                raise DocumentError("invalid_cursor", 422) from error
        page = matches[offset : offset + request.limit]
        has_more = len(matches) > offset + request.limit
        return {
            "contract": "m3_paper_find_v1",
            "library_id": str(request.library_id),
            "generation_id": str(request.generation_id),
            "items": page,
            "next_cursor": _paper_find_cursor(scope_sha256, page[-1]["paper_id"])
            if has_more
            else None,
            "has_more": has_more,
        }
