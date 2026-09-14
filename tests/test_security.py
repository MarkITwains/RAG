"""访问控制单元测试。

覆盖：主体解析、组编码、入库权限字段、单节点放行判定、结果集复核。
不依赖 llama-index（`build_acl_filters` 在 ACL 关闭时返回 None，可在无该依赖的环境下调用）。
"""

from types import SimpleNamespace

import pytest

import pcb_rag.security as security
from pcb_rag.security import (
    Principal,
    acl_metadata_for_ingest,
    build_acl_filters,
    describe_acl,
    encode_groups,
    filter_nodes,
    node_allowed,
    resolve_principal,
)


@pytest.fixture
def acl_on(monkeypatch):
    monkeypatch.setattr(security, "ACL_ENABLED", True)
    return security


def _node(text: str, metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(node=SimpleNamespace(text=text, metadata=metadata), score=0.5)


# ---------------------------------------------------------------------------
# 主体解析
# ---------------------------------------------------------------------------
class TestResolvePrincipal:
    def test_reads_headers(self):
        headers = {
            "X-User-Id": "u1",
            "X-Tenant-Id": "acme",
            "X-User-Roles": "engineer,qa",
            "X-User-Groups": "pcb_team",
        }
        principal = resolve_principal(headers)
        assert principal.user_id == "u1"
        assert principal.tenant_id == "acme"
        assert principal.roles == frozenset({"engineer", "qa"})
        assert principal.groups == frozenset({"pcb_team"})
        assert principal.authenticated is True

    def test_headers_are_case_insensitive(self):
        principal = resolve_principal({"x-tenant-id": "acme"})
        assert principal.tenant_id == "acme"

    def test_explicit_args_override_headers(self):
        principal = resolve_principal({"X-Tenant-Id": "acme"}, tenant_id="other")
        assert principal.tenant_id == "other"

    def test_anonymous_defaults(self, monkeypatch):
        monkeypatch.setattr(security, "ACL_ANONYMOUS_TENANT", "public")
        principal = resolve_principal(None)
        assert principal.user_id == "anonymous"
        assert principal.tenant_id == "public"
        assert principal.authenticated is False

    def test_admin_detection(self):
        assert resolve_principal(None, roles=["admin"]).is_admin is True
        assert resolve_principal(None, roles=["engineer"]).is_admin is False

    def test_group_tokens_include_tenant_and_roles(self):
        principal = Principal(user_id="u1", tenant_id="acme", roles=frozenset({"qa"}), groups=frozenset({"pcb"}))
        tokens = principal.group_tokens()
        assert {"acme", "qa", "pcb", "u1"} <= tokens

    def test_fingerprint_is_stable_and_short(self):
        principal = Principal(user_id="u1", tenant_id="acme")
        assert principal.fingerprint() == principal.fingerprint()
        assert len(principal.fingerprint()) == 12

    def test_to_dict(self):
        data = Principal(user_id="u1", tenant_id="acme", roles=frozenset({"qa"})).to_dict()
        assert data["roles"] == ["qa"]
        assert data["tenant_id"] == "acme"


# ---------------------------------------------------------------------------
# 入库权限字段
# ---------------------------------------------------------------------------
class TestAclMetadata:
    def test_encode_groups_wraps_with_separator(self):
        assert encode_groups(["pcb", "qa"]) == "|pcb|qa|"

    def test_encode_groups_dedupes_and_lowercases(self):
        assert encode_groups(["PCB", "pcb", " QA "]) == "|pcb|qa|"

    def test_encode_empty(self):
        assert encode_groups([]) == ""

    def test_metadata_contains_required_fields(self, monkeypatch):
        monkeypatch.setattr(security, "ACL_DEFAULT_VISIBILITY", "private")
        metadata = acl_metadata_for_ingest(tenant_id="acme", groups=["pcb"])
        assert metadata[security.ACL_TENANT_FIELD] == "acme"
        assert metadata[security.ACL_VISIBILITY_FIELD] == "private"
        assert metadata[security.ACL_GROUPS_FIELD] == "|pcb|"

    def test_visibility_override(self):
        metadata = acl_metadata_for_ingest(visibility="public")
        assert metadata[security.ACL_VISIBILITY_FIELD] == "public"

    def test_owner_written_when_provided(self):
        metadata = acl_metadata_for_ingest(owner_id="u9")
        assert metadata[security.ACL_OWNER_FIELD] == "u9"


# ---------------------------------------------------------------------------
# 单节点放行判定
# ---------------------------------------------------------------------------
class TestNodeAllowed:
    def test_disabled_acl_allows_everything(self):
        assert node_allowed({}, Principal()) is True

    def test_admin_allows_everything(self, acl_on):
        admin = Principal(user_id="root", tenant_id="root", roles=frozenset({"admin"}))
        assert node_allowed({"tenant_id": "secret", "visibility": "private"}, admin) is True

    def test_public_document_visible_to_all(self, acl_on):
        assert node_allowed({"tenant_id": "other", "visibility": "public"}, Principal(tenant_id="acme")) is True

    def test_same_tenant_allowed(self, acl_on):
        assert node_allowed({"tenant_id": "acme", "visibility": "private"}, Principal(tenant_id="acme")) is True

    def test_cross_tenant_denied(self, acl_on):
        assert node_allowed({"tenant_id": "other", "visibility": "private"}, Principal(tenant_id="acme")) is False

    def test_group_token_grants_access(self, acl_on):
        metadata = {"tenant_id": "other", "visibility": "private", "acl_groups": "|pcb|qa|"}
        principal = Principal(tenant_id="acme", groups=frozenset({"qa"}))
        assert node_allowed(metadata, principal) is True

    def test_group_token_not_matching_denied(self, acl_on):
        metadata = {"tenant_id": "other", "visibility": "private", "acl_groups": "|pcb|qa|"}
        principal = Principal(tenant_id="acme", groups=frozenset({"sales"}))
        assert node_allowed(metadata, principal) is False

    def test_group_substring_does_not_grant(self, acl_on):
        # |dev| 不应命中 |devops|
        metadata = {"tenant_id": "other", "visibility": "private", "acl_groups": "|devops|"}
        assert node_allowed(metadata, Principal(tenant_id="acme", groups=frozenset({"dev"}))) is False

    def test_owner_grants_access(self, acl_on):
        metadata = {"tenant_id": "other", "visibility": "private", "owner_id": "u1"}
        assert node_allowed(metadata, Principal(user_id="u1", tenant_id="acme")) is True

    def test_legacy_document_without_acl_fields_allowed(self, acl_on):
        assert node_allowed({"source_path": "a.pdf"}, Principal(tenant_id="acme")) is True

    def test_legacy_document_strict_mode_denied(self, acl_on, monkeypatch):
        monkeypatch.setenv("ACL_STRICT_LEGACY", "1")
        assert node_allowed({"source_path": "a.pdf"}, Principal(tenant_id="acme")) is False


# ---------------------------------------------------------------------------
# 结果集复核
# ---------------------------------------------------------------------------
class TestFilterNodes:
    def test_drops_denied_nodes(self, acl_on):
        nodes = [
            _node("公开资料", {"tenant_id": "other", "visibility": "public"}),
            _node("本租户资料", {"tenant_id": "acme", "visibility": "private"}),
            _node("他租户资料", {"tenant_id": "other", "visibility": "private"}),
        ]
        kept = filter_nodes(nodes, Principal(tenant_id="acme"))
        assert len(kept) == 2
        assert all("他租户" not in item.node.text for item in kept)

    def test_admin_sees_everything(self, acl_on):
        nodes = [_node("他租户资料", {"tenant_id": "other", "visibility": "private"})]
        admin = Principal(roles=frozenset({"admin"}))
        assert len(filter_nodes(nodes, admin)) == 1

    def test_empty_input(self, acl_on):
        assert filter_nodes([], Principal()) == []

    def test_disabled_acl_keeps_all(self):
        nodes = [_node("x", {"tenant_id": "other", "visibility": "private"})]
        assert len(filter_nodes(nodes, Principal(tenant_id="acme"))) == 1

    def test_accepts_bare_nodes(self, acl_on):
        nodes = [SimpleNamespace(text="公开", metadata={"visibility": "public"})]
        assert len(filter_nodes(nodes, Principal(tenant_id="acme"))) == 1


# ---------------------------------------------------------------------------
# 过滤构造与诊断
# ---------------------------------------------------------------------------
class TestBuildAclFilters:
    def test_returns_none_when_disabled(self):
        assert build_acl_filters(Principal(tenant_id="acme")) is None

    def test_returns_none_for_admin(self, acl_on):
        assert build_acl_filters(Principal(roles=frozenset({"admin"}))) is None

    def test_returns_none_for_missing_principal(self, acl_on):
        assert build_acl_filters(None) is None


class TestDescribeAcl:
    def test_reports_configuration(self):
        info = describe_acl()
        assert "enabled" in info
        assert info["tenant_field"] == security.ACL_TENANT_FIELD
        assert isinstance(info["admin_roles"], list)
