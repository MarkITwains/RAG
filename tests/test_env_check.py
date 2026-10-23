from scripts import check_env


def test_missing_selected_adapter_fails_dependency_check(monkeypatch):
    seen = []
    monkeypatch.setattr(check_env, "load_env", lambda path: None)
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.setenv("EMBED_BACKEND", "api")
    monkeypatch.setenv("RERANK_ENABLED", "0")

    def check(module):
        seen.append(module)
        return module != "llama_index.embeddings.openai"

    monkeypatch.setattr(check_env, "check_import", check)
    assert check_env.main(["--dependencies-only"]) == 1
    assert "llama_index.llms.ollama" in seen
    assert "llama_index.llms.openai_like" not in seen
    assert "llama_index.embeddings.ollama" not in seen


def test_configuration_failure_affects_self_check_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(check_env, "__file__", str(tmp_path / "scripts" / "check_env.py"))
    (tmp_path / ".env").touch()
    (tmp_path / "corpus").mkdir()
    monkeypatch.setenv("DATA_DIR", "corpus")
    monkeypatch.setattr(check_env, "load_env", lambda path: None)
    monkeypatch.setattr(check_env, "check_import", lambda name: True)
    for name in ("check_embed", "check_rerank", "check_milvus"):
        monkeypatch.setattr(check_env, name, lambda: True)
    monkeypatch.setattr(check_env, "check_llm", lambda: False)
    assert check_env.main([]) == 1
