"""Reproducible experiment records must preserve failures and provenance."""

import pytest

from scripts.experiment_common import digest, percentile, public_config, summarize_retrieval
from scripts.experiment_aux import parallel_judge


def test_public_configuration_keeps_token_limits_but_not_secrets():
    result = public_config(
        {
            "LLM_MODEL": "test",
            "LLM_MAX_TOKENS": "8192",
            "LLM_API_KEY": "secret",
            "DIFY_API_TOKEN": "secret",
            "RERANK_API_KEY": "secret",
            "EMBED_API_KEY": "secret",
            "COLLECTION": "frozen",
        }
    )
    assert result == {"COLLECTION": "frozen", "LLM_MAX_TOKENS": "8192", "LLM_MODEL": "test"}


def test_bm25_parameters_are_recorded_in_experiment_provenance():
    result = public_config({"BM25_K1": "2.4", "BM25_B": "0.35"})

    assert result == {"BM25_K1": "2.4", "BM25_B": "0.35"}


def test_experiment_provenance_covers_effective_config_families_without_secrets():
    result = public_config(
        {
            "OLLAMA_TIMEOUT": "300",
            "HF_RERANK_MODEL": "model",
            "COMPRESSION_BUDGET_CHARS": "12000",
            "CONTEXT_DEDUPE_THRESHOLD": "0.85",
            "MULTIMODAL_VLM_MODEL": "vision",
            "TABLE_MIN_ROWS": "2",
            "STEPBACK_ROUTE_WEIGHT": "0.7",
            "DECOMPOSE_MAX_SUB_QUERIES": "3",
            "CITATION_ENABLED": "1",
            "MULTIMODAL_VLM_API_KEY": "secret",
            "DIFY_API_TOKEN": "secret",
            "UNRELATED_HOST_SETTING": "do-not-record",
        }
    )

    assert "MULTIMODAL_VLM_API_KEY" not in result and "DIFY_API_TOKEN" not in result
    assert "UNRELATED_HOST_SETTING" not in result
    assert result["OLLAMA_TIMEOUT"] == "300"
    assert result["HF_RERANK_MODEL"] == "model"
    assert result["COMPRESSION_BUDGET_CHARS"] == "12000"
    assert result["CONTEXT_DEDUPE_THRESHOLD"] == "0.85"
    assert result["MULTIMODAL_VLM_MODEL"] == "vision"
    assert result["TABLE_MIN_ROWS"] == "2"
    assert result["STEPBACK_ROUTE_WEIGHT"] == "0.7"
    assert result["DECOMPOSE_MAX_SUB_QUERIES"] == "3"
    assert result["CITATION_ENABLED"] == "1"


def record(rank, wall, error=None, fallback=0):
    return {
        "mrr": 1 / rank if rank else 0,
        "first_hit_rank": rank,
        "ap": 1 / rank if rank else 0,
        "ndcg@10": 0.5,
        "hits": {f"hit@{k}": float(0 < rank <= k) for k in [1, 3, 5, 10]},
        "wall_s": wall,
        "got_ids": ["id"] if rank else [],
        "timing": {"stage": wall},
        "error": error,
        "rerank_failures": fallback,
    }


def test_retrieval_metrics_record_empty_and_degraded_queries():
    result = summarize_retrieval([record(1, 1), record(20, 2), record(0, 3, fallback=1)])
    assert result["mrr"] == 0.35
    assert result["mrr@10"] == round(1 / 3, 6)
    assert result["empty_results"] == 1 and result["rerank_fallbacks"] == 1
    assert result["invalid"] is True
    assert result["wall_s_p95"] == 3


def test_percentile_handles_boundaries():
    assert percentile([], 95) == 0
    assert percentile([3], 99) == 3
    assert percentile([3, 1, 2], 0) == 1
    assert percentile([3, 1, 2], 100) == 3


class Judge:
    def complete(self, prompt):
        from types import SimpleNamespace

        if '"relevant"' in prompt:
            text = '{"relevant":' + ("false" if "irrelevant" in prompt else "true") + "}"
        elif '"supported"' in prompt:
            text = '{"statements":[{"text":"fact","supported":true}]}'
        elif '"found"' in prompt:
            text = '{"statements":[{"text":"fact","found":true}]}'
        else:
            text = '{"score":0.8}'
        return SimpleNamespace(text=text)


def test_parallel_judging_preserves_the_original_metric_formulas():
    from eval.metrics import evaluate_case

    kwargs = {
        "question": "question",
        "answer": "answer",
        "contexts": ["relevant", "irrelevant", "relevant"],
        "ground_truth": "reference",
        "llm": Judge(),
    }
    sequential = evaluate_case(**kwargs)
    concurrent = parallel_judge(
        kwargs["question"], kwargs["answer"], kwargs["contexts"], kwargs["ground_truth"], Judge()
    )
    assert concurrent == sequential


def test_parallel_judging_keeps_failures_as_unavailable():
    class Failed:
        def complete(self, prompt):
            raise RuntimeError("provider error")

    result = parallel_judge("question", "answer", ["context"], "reference", Failed())
    assert set(result["judge_failed"]) == {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    assert "overall" not in result


def test_freeze_guard_preserves_bytes_and_rejects_changed_input(tmp_path):
    import json
    import pytest
    from scripts.experiment_common import assert_frozen_inputs, digest, write_json

    (tmp_path / "datasets").mkdir()
    dataset = tmp_path / "datasets/regression_v1.json"
    dataset.write_text("[]", encoding="utf-8")
    cache = tmp_path / "corpus.jsonl"
    cache.write_text('{"text":"original"}\n', encoding="utf-8")
    manifest = {"dataset_sha256": digest(dataset), "lexical_sha256": digest(cache), "code_fingerprints": {}}
    write_json(tmp_path / "manifest.json", manifest)
    original = cache.read_bytes()
    assert_frozen_inputs(tmp_path, {"LEXICAL_CACHE_PATH": str(cache)})
    assert cache.read_bytes() == original
    cache.write_text(json.dumps({"text": "changed"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen lexical corpus changed"):
        assert_frozen_inputs(tmp_path, {"LEXICAL_CACHE_PATH": str(cache)})


def test_scenarios_have_three_per_type_and_explicit_ai_provenance():
    import json
    from collections import Counter
    from scripts.experiment_common import ROOT

    data = json.loads((ROOT / "scripts/experiment_scenarios.json").read_text(encoding="utf-8"))
    assert data["authored_by"] == "AI" and data["human_authored"] is False
    assert len(data["questions"]) == 30
    assert len({row["id"] for row in data["questions"]}) == 30
    assert set(Counter(row["type"] for row in data["questions"]).values()) == {3}


def test_frozen_workers_do_not_fall_back_to_production_dotenv(monkeypatch):
    from pcb_rag import env_loader
    from scripts.experiment_common import isolate_project_env

    calls = []

    def original(path=None):
        calls.append(path)
        return 7

    monkeypatch.setattr(env_loader, "load_project_env", original)
    monkeypatch.setenv("PCB_RAG_EXPERIMENT_FROZEN", "1")
    isolate_project_env()
    isolate_project_env()
    assert env_loader.load_project_env() == 0
    assert env_loader.load_project_env("explicit.env") == 7
    assert calls == ["explicit.env"]


def test_frozen_base_env_does_not_inherit_unfrozen_project_settings(monkeypatch):
    from scripts.experiment_queue import frozen_base_env

    monkeypatch.setenv("RECALL_TOP_K", "999")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")

    env = frozen_base_env({"LLM_MODEL": "frozen"})

    assert env["LLM_MODEL"] == "frozen"
    assert "RECALL_TOP_K" not in env
    assert env["PATH"] == "/usr/bin"
    assert env["HTTPS_PROXY"] == "http://proxy.invalid"


def test_frozen_env_digest_is_required_on_resume(tmp_path):
    from scripts.experiment_common import digest
    from scripts.experiment_queue import load_frozen_env

    path = tmp_path / "private.json"
    path.write_text('{"LLM_MODEL":"frozen"}', encoding="utf-8")
    expected = digest(path)
    assert load_frozen_env(path, expected)["LLM_MODEL"] == "frozen"
    with pytest.raises(RuntimeError, match="environment"):
        load_frozen_env(path, "different")
    with pytest.raises(RuntimeError, match="digest"):
        load_frozen_env(path, None)


def test_parallel_scheduler_limits_capacity_and_keeps_timing_stages_exclusive():
    from scripts.experiment_queue import can_start

    ablation = {"parallel_safe": True}
    baseline = {"parallel_safe": False}
    assert can_start(ablation, [ablation, ablation], 3)
    assert not can_start(ablation, [ablation, ablation, ablation], 3)
    assert not can_start(baseline, [ablation], 3)
    assert not can_start(ablation, [baseline], 3)
    assert not can_start({}, [ablation], 3)
    assert can_start(baseline, [], 3)


def test_parallel_scheduler_preserves_dependency_and_failure_gates():
    from scripts.experiment_queue import ready_jobs

    manifest = {
        "steps": [
            {"name": "baseline", "status": "running"},
            {"name": "gate", "status": "pending", "depends_on": ["baseline"]},
            {"name": "independent", "status": "interrupted"},
            {"name": "summary", "status": "pending", "depends_on": ["baseline"], "run_after_failed_dependencies": True},
        ]
    }
    assert [job["name"] for job in ready_jobs(manifest)] == ["independent"]
    manifest["steps"][0]["status"] = "done"
    assert [job["name"] for job in ready_jobs(manifest)] == ["gate", "independent", "summary"]
    manifest["steps"][0]["status"] = "failed"
    assert [job["name"] for job in ready_jobs(manifest)] == ["independent", "summary"]


def test_queue_runs_three_workers_together_then_an_exclusive_worker(tmp_path):
    import json
    import subprocess
    import sys
    from scripts.experiment_common import ROOT, write_json

    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, pathlib, sys, time\n"
        "started = time.monotonic()\n"
        "time.sleep(1)\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'start': started, 'end': time.monotonic()}))\n",
        encoding="utf-8",
    )
    (tmp_path / "logs").mkdir()
    private = tmp_path / "private.json"
    write_json(private, {})
    names = ["a", "b", "c", "exclusive"]
    steps = [
        {
            "name": name,
            "stage": "test",
            "status": "pending",
            "parallel_safe": name != "exclusive",
            "argv": [str(worker), str(tmp_path / (name + ".json"))],
            "depends_on": [] if name != "exclusive" else names[:3],
        }
        for name in names
    ]
    write_json(
        tmp_path / "manifest.json",
        {
            "run_id": "scheduler-integration",
            "status": "ready",
            "private_env_path": str(private),
            "private_env_sha256": digest(private),
            "max_parallel_steps": 3,
            "plan_complete": True,
            "steps": steps,
        },
    )
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/experiment_queue.py"), "--run-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    intervals = {name: json.loads((tmp_path / (name + ".json")).read_text()) for name in names}
    assert max(intervals[name]["start"] for name in names[:3]) < min(intervals[name]["end"] for name in names[:3])
    assert intervals["exclusive"]["start"] > max(intervals[name]["end"] for name in names[:3])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["status"] == "complete" and heartbeat["active_steps"] == []
    assert {job["max_observed_parallel_steps"] for job in manifest["steps"][:3]} == {3}
    assert manifest["steps"][-1]["max_observed_parallel_steps"] == 1


def test_queue_halts_all_workers_when_the_shared_rerank_circuit_opens(tmp_path):
    import json
    import subprocess
    import sys
    from scripts.experiment_common import ROOT, write_json

    worker = tmp_path / "worker.py"
    worker.write_text(
        "import pathlib, sys, time\n"
        "from pcb_rag.rerank_control import RerankControl\n"
        "if sys.argv[1] == 'trip':\n"
        "    time.sleep(.3)\n"
        "    RerankControl.from_env().trip('http_402')\n"
        "else:\n"
        "    time.sleep(20)\n"
        "    pathlib.Path(sys.argv[2]).write_text('should not finish')\n",
        encoding="utf-8",
    )
    (tmp_path / "logs").mkdir()
    private = tmp_path / "private.json"
    write_json(
        private,
        {
            "RERANK_CONTROL_DB": str(tmp_path / "control.sqlite3"),
            "RERANK_BUDGET_SCOPE": "test-shared-halt",
            "RERANK_TOKEN_BUDGET": "1000000",
            "RERANK_REQUEST_BUDGET": "100",
        },
    )
    marker = tmp_path / "must-not-exist"
    write_json(
        tmp_path / "manifest.json",
        {
            "run_id": "halt-integration",
            "status": "ready",
            "private_env_path": str(private),
            "private_env_sha256": digest(private),
            "max_parallel_steps": 3,
            "plan_complete": True,
            "steps": [
                {
                    "name": name,
                    "stage": "test",
                    "status": "pending",
                    "parallel_safe": True,
                    "argv": [str(worker), name, str(marker)],
                }
                for name in ["trip", "sleeping"]
            ],
        },
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/experiment_queue.py"), "--run-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "halted" and manifest["halt_reason"] == "http_402"
    assert not marker.exists()
    assert json.loads((tmp_path / "heartbeat.json").read_text())["active_steps"] == []
