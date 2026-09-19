"""访问控制（多租户 / 角色 / 组）与审计。

背景
----
知识库里同时存在公开标准（GB/T、SJ/T 等）与厂商私有资料（IPC-2581C、供应商规范等），
Dify / HTTP 接口对外暴露时必须保证「不同租户只能看到自己的资料」。

本模块提供三层防护：

1. **入库侧**（``acl_metadata_for_ingest``）：为每个文档写入 ``tenant_id`` /
   ``visibility`` / ``acl_groups`` 等字段，作为后续过滤的依据
2. **检索侧（向量库）**（``build_acl_filters``）：把权限条件编译成 Milvus 过滤表达式，
   在召回阶段就把无权文档挡掉，避免「先召回后过滤」导致的 top_k 稀释
3. **检索侧（内存兜底）**（``filter_nodes`` / ``node_allowed``）：对召回结果逐条复核，
   即使向量库表达式被服务端忽略（或降级为全量召回）也不会越权

设计取舍
--------
- 组字段用 ``|group|other|`` 这种**首尾带分隔符**的字符串存放，而不是数组：
  Milvus 动态字段对数组支持不一致，而 ``like "%|g|%"`` 在各版本都可用，
  且首尾分隔符能避免 ``dev`` 误命中 ``devops``
- ``security.py`` 不导入 llama-index（延迟导入），因此可被单元测试轻量引用

用法::

    from pcb_rag.security import resolve_principal, build_acl_filters, filter_nodes

    principal = resolve_principal(request.headers)
    nodes = retriever.retrieve(query, filters=build_acl_filters(principal))
    nodes = filter_nodes(nodes, principal)          # 内存兜底
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

__all__ = [
    "ACL_ENABLED",
    "Principal",
    "acl_metadata_for_ingest",
    "audit_log",
    "build_acl_filters",
    "current_principal",
    "describe_acl",
    "filter_nodes",
    "merge_filters",
    "node_allowed",
    "reset_request_principal",
    "resolve_principal",
    "set_request_principal",
]


# ---------------------------------------------------------------------------
# 1. 配置
# ---------------------------------------------------------------------------
def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


ACL_ENABLED = _env_bool("ACL_ENABLED", False)

# metadata 字段名（入库与检索必须一致）
ACL_TENANT_FIELD = os.getenv("ACL_TENANT_FIELD", "tenant_id").strip() or "tenant_id"
ACL_GROUPS_FIELD = os.getenv("ACL_GROUPS_FIELD", "acl_groups").strip() or "acl_groups"
ACL_VISIBILITY_FIELD = os.getenv("ACL_VISIBILITY_FIELD", "visibility").strip() or "visibility"
ACL_OWNER_FIELD = os.getenv("ACL_OWNER_FIELD", "owner_id").strip() or "owner_id"

# 公开标记值：visibility 等于该值的文档对所有租户可见
ACL_PUBLIC_VALUE = os.getenv("ACL_PUBLIC_VALUE", "public").strip() or "public"
# 默认可视性（入库时未指定则用此值）：private 更安全
ACL_DEFAULT_VISIBILITY = os.getenv("ACL_DEFAULT_VISIBILITY", "private").strip() or "private"
# 拥有这些角色之一的用户可跨租户访问
ACL_ADMIN_ROLES = {
    r.strip().lower()
    for r in os.getenv("ACL_ADMIN_ROLES", "admin,root").split(",")
    if r.strip()
}

# 请求头名称（Dify / 网关注入）
ACL_HEADER_USER = os.getenv("ACL_HEADER_USER", "X-User-Id").strip()
ACL_HEADER_TENANT = os.getenv("ACL_HEADER_TENANT", "X-Tenant-Id").strip()
ACL_HEADER_ROLES = os.getenv("ACL_HEADER_ROLES", "X-User-Roles").strip()
ACL_HEADER_GROUPS = os.getenv("ACL_HEADER_GROUPS", "X-User-Groups").strip()
# 允许客户端直接指定租户（单机 / 内网场景默认开启；生产建议关闭并由网关注入）
ACL_TRUST_HEADERS = _env_bool("ACL_TRUST_HEADERS", True)
# 未携带任何身份信息时的兜底租户（ACL_ENABLED=1 时用于「匿名用户」）
ACL_ANONYMOUS_TENANT = os.getenv("ACL_ANONYMOUS_TENANT", "public").strip() or "public"
# 段分隔符
ACL_GROUP_SEP = "|"

# 审计日志
ACL_AUDIT_ENABLED = _env_bool("ACL_AUDIT_ENABLED", True)
ACL_AUDIT_PATH = os.getenv("ACL_AUDIT_PATH", "./data/acl_audit.jsonl").strip()


def _env_bool_extra() -> bool:
    return ACL_ENABLED


# ---------------------------------------------------------------------------
# 2. 主体（Principal）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Principal:
    """一次请求的访问主体。

    - ``tenant_id``：租户标识，决定默认可访问的数据边界
    - ``roles`` / ``groups``：授权维度，命中文档的 ``acl_groups`` 即放行
    - ``user_id``：用于审计与 owner 判定
    """

    user_id: str = "anonymous"
    tenant_id: str = ACL_ANONYMOUS_TENANT
    roles: frozenset[str] = field(default_factory=frozenset)
    groups: frozenset[str] = field(default_factory=frozenset)
    authenticated: bool = False

    # ---- 判定 ----
    @property
    def is_admin(self) -> bool:
        return bool(self.roles & ACL_ADMIN_ROLES)

    def group_tokens(self) -> Set[str]:
        """返回可用于文档级匹配的授权 token（组 + 角色 + 租户）。"""

        tokens = {g.lower() for g in self.groups}
        tokens |= {r.lower() for r in self.roles}
        if self.tenant_id:
            tokens.add(self.tenant_id.lower())
        if self.user_id:
            tokens.add(self.user_id.lower())
        return {t for t in tokens if t}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "roles": sorted(self.roles),
            "groups": sorted(self.groups),
            "authenticated": self.authenticated,
            "is_admin": self.is_admin,
        }

    def fingerprint(self) -> str:
        """稳定的短摘要，用于日志与缓存键（避免明文用户名落盘）。"""

        raw = f"{self.tenant_id}|{','.join(sorted(self.roles))}|{','.join(sorted(self.groups))}"
        return hashlib.md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def _split_tokens(raw: Optional[str]) -> frozenset[str]:
    if not raw:
        return frozenset()
    parts: List[str] = []
    for chunk in str(raw).replace(";", ",").replace("|", ",").split(","):
        token = chunk.strip().lower()
        if token:
            parts.append(token)
    return frozenset(parts)


def resolve_principal(
    headers: Optional[Any] = None,
    *,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    roles: Optional[Iterable[str]] = None,
    groups: Optional[Iterable[str]] = None,
) -> Principal:
    """从 HTTP 头或显式参数构造访问主体。

    ``headers`` 支持 dict 与 Starlette/FastAPI 的 ``Headers``（大小写不敏感查找由本函数统一处理）。
    """

    def _get(name: str) -> Optional[str]:
        if headers is None:
            return None
        try:
            value = headers.get(name)
        except Exception:
            value = None
        if value is None:
            # dict 可能存在大小写不一致
            try:
                lowered = {str(k).lower(): v for k, v in dict(headers).items()}
                value = lowered.get(name.lower())
            except Exception:
                value = None
        return value

    if not ACL_TRUST_HEADERS and headers is not None:
        resolved_user = user_id
        resolved_tenant = tenant_id
        resolved_roles = roles
        resolved_groups = groups
    else:
        resolved_user = user_id or _get(ACL_HEADER_USER)
        resolved_tenant = tenant_id or _get(ACL_HEADER_TENANT)
        resolved_roles = roles if roles is not None else _split_tokens(_get(ACL_HEADER_ROLES))
        resolved_groups = groups if groups is not None else _split_tokens(_get(ACL_HEADER_GROUPS))

    role_set = frozenset({r.strip().lower() for r in (resolved_roles or []) if str(r).strip()})
    group_set = frozenset({g.strip().lower() for g in (resolved_groups or []) if str(g).strip()})
    authenticated = bool(resolved_user or resolved_tenant)

    return Principal(
        user_id=(resolved_user or "anonymous").strip(),
        tenant_id=(resolved_tenant or ACL_ANONYMOUS_TENANT).strip(),
        roles=role_set,
        groups=group_set,
        authenticated=authenticated,
    )


# ---------------------------------------------------------------------------
# 2.1 请求级主体（ContextVar）
# ---------------------------------------------------------------------------
_current_principal: ContextVar[Optional[Principal]] = ContextVar("pcb_rag_principal", default=None)


def set_request_principal(principal: Optional[Principal]) -> Any:
    """把主体绑定到当前请求上下文（HTTP 中间件里调用），返回可用于重置的 token。

    FastAPI 用线程池执行同步端点时会复制上下文（anyio 的 ``copy_context``），
    因此中间件里设置的 principal 在同步端点函数内同样可见。
    """

    return _current_principal.set(principal)


def reset_request_principal(token: Any) -> None:
    """恢复上下文（中间件收尾调用）；token 非法时静默忽略。"""

    if token is None:
        return
    try:
        _current_principal.reset(token)
    except Exception:
        pass


def current_principal() -> Principal:
    """返回当前请求主体；未绑定（CLI / 无中间件）时回退为匿名主体。"""

    principal = _current_principal.get()
    if principal is not None:
        return principal
    return resolve_principal(None)


# ---------------------------------------------------------------------------
# 3. 入库侧：ACL 字段
# ---------------------------------------------------------------------------
def encode_groups(groups: Iterable[str]) -> str:
    """把组列表编码成 ``|g1|g2|`` 形式（首尾分隔符防止子串误匹配）。"""

    tokens = sorted({str(g).strip().lower() for g in groups if str(g).strip()})
    if not tokens:
        return ""
    return ACL_GROUP_SEP + ACL_GROUP_SEP.join(tokens) + ACL_GROUP_SEP


def acl_metadata_for_ingest(
    *,
    tenant_id: Optional[str] = None,
    visibility: Optional[str] = None,
    owner_id: Optional[str] = None,
    groups: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    """构造写入 chunk metadata 的权限字段。

    未显式指定时从环境变量读取（``ACL_DEFAULT_TENANT`` / ``ACL_DEFAULT_VISIBILITY``），
    便于「整个知识库属于某个租户」的部署方式。
    """

    resolved_tenant = (tenant_id or os.getenv("ACL_DEFAULT_TENANT", "") or ACL_ANONYMOUS_TENANT).strip()
    resolved_visibility = (visibility or ACL_DEFAULT_VISIBILITY).strip() or ACL_DEFAULT_VISIBILITY
    resolved_owner = (owner_id or os.getenv("ACL_DEFAULT_OWNER", "") or "").strip()

    metadata: Dict[str, str] = {
        ACL_TENANT_FIELD: resolved_tenant,
        ACL_VISIBILITY_FIELD: resolved_visibility,
    }
    if resolved_owner:
        metadata[ACL_OWNER_FIELD] = resolved_owner

    encoded = encode_groups(groups or [])
    if encoded:
        metadata[ACL_GROUPS_FIELD] = encoded
    return metadata


# ---------------------------------------------------------------------------
# 4. 检索侧：Milvus 过滤表达式
# ---------------------------------------------------------------------------
def build_acl_filters(principal: Optional[Principal]) -> Optional[Any]:
    """把权限条件编译为 llama-index ``MetadataFilters``（Milvus 侧粗过滤）。

    - 管理员：返回 ``None``（不限制）
    - 普通用户：``tenant_id == <租户> OR visibility == public``
    - 关闭 ACL：返回 ``None``

    返回类型为 ``MetadataFilters | None``；由于延迟导入 llama-index，
    本函数在无重型依赖的环境（如纯逻辑单测）中也能被调用（仅当 ACL 关闭时返回 None）。
    """

    if not ACL_ENABLED or principal is None or principal.is_admin:
        return None

    from llama_index.core.vector_stores.types import (
        FilterCondition,
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )

    clauses = [
        MetadataFilter(key=ACL_TENANT_FIELD, operator=FilterOperator.EQ, value=principal.tenant_id),
        MetadataFilter(key=ACL_VISIBILITY_FIELD, operator=FilterOperator.EQ, value=ACL_PUBLIC_VALUE),
    ]
    return MetadataFilters(filters=clauses, condition=FilterCondition.OR)


def merge_filters(base: Optional[Any], extra: Optional[Any]) -> Optional[Any]:
    """以 AND 语义合并两组过滤条件（任一侧为空时返回另一侧）。

    注意：``MetadataFilters`` 自身带有 ``condition``（AND / OR）。合并时**不能**把子组的
    ``filters`` 摊平进外层 AND —— 否则 ``build_acl_filters`` 产出的
    ``tenant == X OR visibility == public`` 会退化成 ``tenant == X AND visibility == public``，
    导致用户对本租户的私有文档全部不可见。这里的策略：

    - 子组本身就是 AND 组 → 可以安全摊平（AND 满足结合律）
    - 子组是 OR 组（或单个 ``MetadataFilter``）→ 作为一个整体挂到外层 AND 之下（嵌套）

    llama-index 的 Milvus 适配器支持嵌套 ``MetadataFilters``，会编译成带括号的表达式。
    """

    if base is None:
        return extra
    if extra is None:
        return base

    from llama_index.core.vector_stores.types import FilterCondition, MetadataFilters

    def _as_and_operands(node: Any) -> List[Any]:
        """把一侧拆成可以直接挂到外层 AND 下的操作数列表。"""
        if node is None:
            return []
        if isinstance(node, MetadataFilters):
            if not node.filters:
                return []
            cond = getattr(node, "condition", None)
            if cond is None or cond == FilterCondition.AND:
                # AND 组可以摊平（结合律），子元素可能仍是嵌套组，保持原样
                return list(node.filters)
            if cond == FilterCondition.OR and len(node.filters) == 1:
                # 单元素 OR 组等价于该元素本身（NOT 组不能这样拆）
                return [node.filters[0]]
            # OR / NOT 组：整体作为一个操作数，保留其内部语义
            return [node]
        return [node]

    merged = _as_and_operands(base) + _as_and_operands(extra)
    if not merged:
        return None
    if len(merged) == 1:
        only = merged[0]
        # 顶层必须是 MetadataFilters（下游 retriever 期望该类型）
        return only if isinstance(only, MetadataFilters) else MetadataFilters(filters=[only], condition=FilterCondition.AND)
    return MetadataFilters(filters=merged, condition=FilterCondition.AND)


# ---------------------------------------------------------------------------
# 5. 检索侧：内存兜底复核
# ---------------------------------------------------------------------------
def node_allowed(metadata: Dict[str, Any], principal: Optional[Principal]) -> bool:
    """判断单条 chunk 是否允许该主体访问（内存侧最终裁决）。"""

    if not ACL_ENABLED or principal is None or principal.is_admin:
        return True

    meta = metadata or {}
    visibility = str(meta.get(ACL_VISIBILITY_FIELD, "") or "").strip().lower()
    tenant = str(meta.get(ACL_TENANT_FIELD, "") or "").strip().lower()
    owner = str(meta.get(ACL_OWNER_FIELD, "") or "").strip().lower()
    groups_raw = str(meta.get(ACL_GROUPS_FIELD, "") or "")

    # 1) 公开文档直接放行
    if visibility == ACL_PUBLIC_VALUE:
        return True
    # 2) 同租户放行
    if tenant and tenant == principal.tenant_id.lower():
        return True
    # 3) owner 放行
    if owner and owner == principal.user_id.lower():
        return True
    # 4) 组 / 角色 / 用户 token 命中文档授权列表
    if groups_raw:
        tokens = principal.group_tokens()
        for group in groups_raw.split(ACL_GROUP_SEP):
            g = group.strip().lower()
            if g and g in tokens:
                return True
    # 5) 没有标注任何权限字段的旧数据：默认放行以保持向后兼容，
    #    但可通过 ACL_STRICT_LEGACY=1 改为拒绝
    if not visibility and not tenant and not groups_raw:
        return not _env_bool("ACL_STRICT_LEGACY", False)
    return False


def filter_nodes(nodes: Optional[Sequence[Any]], principal: Optional[Principal]) -> List[Any]:
    """对召回结果做权限复核，丢弃无权访问的节点。"""

    if not nodes:
        return []
    if not ACL_ENABLED or principal is None or principal.is_admin:
        return list(nodes)

    kept: List[Any] = []
    dropped = 0
    for item in nodes:
        node = getattr(item, "node", item)
        metadata = getattr(node, "metadata", None) or {}
        if node_allowed(metadata, principal):
            kept.append(item)
        else:
            dropped += 1

    if dropped:
        from pcb_rag.observability import counter, record_event

        counter("acl.filtered_nodes", dropped)
        record_event("acl.denied_nodes", count=dropped, principal=principal.fingerprint())
    return kept


# ---------------------------------------------------------------------------
# 6. 审计
# ---------------------------------------------------------------------------
_audit_lock = threading.Lock()


def audit_log(action: str, principal: Optional[Principal], **fields: Any) -> None:
    """把一次访问行为写入审计日志（JSONL）。

    审计失败绝不影响主流程；只记录指纹而不落明文用户名，降低日志泄露风险。
    """

    if not ACL_ENABLED or not ACL_AUDIT_ENABLED:
        return
    record = {
        "ts": time.time(),
        "action": action,
        "principal": principal.fingerprint() if principal else None,
        "tenant": principal.tenant_id if principal else None,
        "authenticated": principal.authenticated if principal else False,
        **{k: v for k, v in fields.items() if v is not None},
    }
    try:
        path = os.path.abspath(ACL_AUDIT_PATH)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _audit_lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        return


# ---------------------------------------------------------------------------
# 7. 诊断
# ---------------------------------------------------------------------------
def describe_acl() -> Dict[str, Any]:
    """返回当前权限配置摘要，供启动日志与 ``/health`` 展示。"""

    return {
        "enabled": ACL_ENABLED,
        "tenant_field": ACL_TENANT_FIELD,
        "visibility_field": ACL_VISIBILITY_FIELD,
        "groups_field": ACL_GROUPS_FIELD,
        "public_value": ACL_PUBLIC_VALUE,
        "default_visibility": ACL_DEFAULT_VISIBILITY,
        "admin_roles": sorted(ACL_ADMIN_ROLES),
        "trust_headers": ACL_TRUST_HEADERS,
        "audit": ACL_AUDIT_ENABLED,
        "anonymous_tenant": ACL_ANONYMOUS_TENANT,
    }
