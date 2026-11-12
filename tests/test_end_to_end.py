"""Synthetic end-to-end run through the CLI (prepare -> baselines -> train -> compare)."""
import json
from pathlib import Path

import pytest

from steam_nffmtl.cli import cmd_smoke

CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")


@pytest.mark.slow
def test_smoke_pipeline(tmp_path: Path) -> None:
    pytest.importorskip("implicit")
    work = cmd_smoke(str(tmp_path), CONFIG)
    results = list((work / "r").glob("*/results/*__test-positive.json"))
    metas = [json.loads(p.read_text()) for p in results]
    models = {m["model"] for m in metas}
    assert {"Random", "Popularity", "TagKNN", "ItemKNN-CF", "BPR-MF", "ALS", "EASE", "NFF-MTL"} <= models
    assert len({m["split_fingerprint"] for m in metas}) == 1      # identical protocol for every model
    assert all(m["valid"] and m["coverage"]["failed_users"] == 0 for m in metas)
    ndcg = {m["model"]: m["summary"]["ndcg@20"] for m in metas}
    assert ndcg["Popularity"] > ndcg["Random"]
    stats = list((work / "r").glob("*compare*/comparison_stats.json"))
    assert stats, "compare step did not write statistics"
