"""模块名称：M1-C1 索引代迁移。

用途：保存独立构建身份、冻结 chunk 与活动/构建指针。
核心设计：PostgreSQL 事务是唯一发布权威；已发布清单及 chunk 禁止改写。
工程要点：不创建 Worker、租约或管理 API。
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0003_m1c1"
down_revision: str | None = "0002_m1b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """名称：升级 C1 数据结构。用途：在现有 M1-B 库增量建立索引记录。"""
    op.add_column("libraries", sa.Column("building_generation_id", sa.Uuid(), nullable=True))
    op.create_table(
        "index_generations",
        sa.Column("generation_id", sa.Uuid(), primary_key=True),
        sa.Column("library_id", sa.Uuid(), sa.ForeignKey("libraries.library_id"), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("membership_revision", sa.BigInteger(), nullable=False),
        sa.Column("expected_active_generation_id", sa.Uuid(), nullable=True),
        sa.Column("collection_name", sa.String(160), nullable=False),
        sa.Column("state", sa.String(16), server_default="building", nullable=False),
        sa.Column("manifest", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=False),
        sa.Column("manifest_sha256", sa.String(64), nullable=True),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "state IN ('building', 'published', 'retired', 'failed')", name="ck_generation_state"
        ),
        sa.CheckConstraint("membership_revision >= 0", name="ck_generation_revision"),
        sa.UniqueConstraint("attempt_id", name="uq_generation_attempt"),
        sa.UniqueConstraint("collection_name", name="uq_generation_collection"),
    )
    op.create_table(
        "index_chunks",
        sa.Column("chunk_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "generation_id",
            sa.Uuid(),
            sa.ForeignKey("index_generations.generation_id"),
            nullable=False,
        ),
        sa.Column(
            "parse_snapshot_id",
            sa.Uuid(),
            sa.ForeignKey("parse_snapshots.parse_snapshot_id"),
            nullable=False,
        ),
        sa.Column(
            "document_version_id",
            sa.Uuid(),
            sa.ForeignKey("document_versions.document_version_id"),
            nullable=False,
        ),
        sa.Column("paper_id", sa.Uuid(), sa.ForeignKey("papers.paper_id"), nullable=False),
        sa.Column("reading_order", sa.Integer(), nullable=False),
        sa.Column("block_type", sa.String(128), nullable=False),
        sa.Column("retrieval_text", sa.Text(), nullable=False),
        sa.Column("source_ranges", sa.JSON(), nullable=False),
        sa.Column("source_locations", sa.JSON(), nullable=False),
        sa.Column("quality_flags", sa.JSON(), nullable=False),
        sa.Column("embedding_tokens", sa.Integer(), nullable=False),
        sa.Column("reranker_tokens", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "embedding_tokens > 0 AND embedding_tokens <= 600", name="ck_chunk_tokens"
        ),
        sa.CheckConstraint("reranker_tokens > 0", name="ck_chunk_reranker_tokens"),
        sa.UniqueConstraint("generation_id", "reading_order", name="uq_index_chunk_order"),
    )
    op.create_foreign_key(
        "fk_library_active_generation",
        "libraries",
        "index_generations",
        ["active_generation_id"],
        ["generation_id"],
    )
    op.create_foreign_key(
        "fk_library_building_generation",
        "libraries",
        "index_generations",
        ["building_generation_id"],
        ["generation_id"],
    )
    op.execute("""
        CREATE FUNCTION guard_index_generation_change() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'immutable_generation_delete' USING ERRCODE = '23514';
            END IF;
            IF OLD.state IN ('published', 'retired') AND
               (NEW.manifest::jsonb IS DISTINCT FROM OLD.manifest::jsonb OR
                NEW.manifest_sha256 IS DISTINCT FROM OLD.manifest_sha256 OR
                NEW.collection_name IS DISTINCT FROM OLD.collection_name OR
                NEW.attempt_id IS DISTINCT FROM OLD.attempt_id OR
                NEW.library_id IS DISTINCT FROM OLD.library_id OR
                NEW.membership_revision IS DISTINCT FROM OLD.membership_revision OR
                NEW.expected_active_generation_id IS DISTINCT FROM OLD.expected_active_generation_id OR
                NEW.state NOT IN ('published', 'retired')) THEN
                RAISE EXCEPTION 'immutable_published_generation' USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END; $$
    """)
    op.execute(
        "CREATE TRIGGER guard_generation_change BEFORE UPDATE OR DELETE ON index_generations FOR EACH ROW EXECUTE FUNCTION guard_index_generation_change()"
    )
    op.execute(
        "CREATE TRIGGER guard_index_chunk_change BEFORE UPDATE OR DELETE ON index_chunks FOR EACH ROW EXECUTE FUNCTION reject_immutable_parse_snapshot_change()"
    )
    op.execute("""
        CREATE FUNCTION guard_index_chunk_insert() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE generation_state text;
        BEGIN
            SELECT state INTO generation_state FROM index_generations
                WHERE generation_id = NEW.generation_id FOR UPDATE;
            IF generation_state IS DISTINCT FROM 'building' THEN
                RAISE EXCEPTION 'immutable_generation_chunks' USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END; $$
    """)
    op.execute(
        "CREATE TRIGGER guard_chunk_insert BEFORE INSERT ON index_chunks FOR EACH ROW EXECUTE FUNCTION guard_index_chunk_insert()"
    )


def downgrade() -> None:
    """名称：回退 C1 结构。用途：仅供管理员显式迁移操作。"""
    op.execute("DROP TRIGGER IF EXISTS guard_index_chunk_change ON index_chunks")
    op.execute("DROP TRIGGER IF EXISTS guard_chunk_insert ON index_chunks")
    op.execute("DROP FUNCTION IF EXISTS guard_index_chunk_insert()")
    op.execute("DROP TRIGGER IF EXISTS guard_generation_change ON index_generations")
    op.execute("DROP FUNCTION IF EXISTS guard_index_generation_change()")
    op.drop_constraint("fk_library_building_generation", "libraries", type_="foreignkey")
    op.drop_constraint("fk_library_active_generation", "libraries", type_="foreignkey")
    op.drop_table("index_chunks")
    op.drop_table("index_generations")
    op.drop_column("libraries", "building_generation_id")
