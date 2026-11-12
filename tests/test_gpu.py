"""Small CUDA checks; skipped automatically when no GPU is visible (run them on Colab)."""
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")]


def test_cuda_sampler_exact_and_seeded() -> None:
    from steam_nffmtl.train import ExactNegativeSampler

    dense = np.random.default_rng(0).random((60, 40)) < 0.6
    m = sp.csr_matrix(dense.astype(np.float32))
    m.sort_indices()
    dev = torch.device("cuda")
    users = torch.arange(60, device=dev).repeat_interleave(50)
    a = ExactNegativeSampler(m, dev, torch.Generator(device=dev).manual_seed(3)).sample(users)
    b = ExactNegativeSampler(m, dev, torch.Generator(device=dev).manual_seed(3)).sample(users)
    assert not dense[users.cpu().numpy(), a.cpu().numpy()].any()
    assert torch.equal(a, b)


def test_cuda_scores_match_cpu() -> None:
    from steam_nffmtl.model import NFFMTL, NFFMTLScorer

    torch.manual_seed(0)
    tags = (np.random.default_rng(1).random((33, 6)) < 0.4).astype(np.float32)
    model = NFFMTL(20, 33, tags, 8, (16, 8, 4)).eval()
    cpu = NFFMTLScorer(model, "cpu", 9).score(np.arange(20))
    gpu = NFFMTLScorer(model.to("cuda"), "cuda", 9).score(np.arange(20))
    np.testing.assert_allclose(cpu, gpu, atol=1e-4)


def test_cuda_training_epoch(tmp_path: Path) -> None:
    pytest.importorskip("implicit", reason="BPR-MF/ALS need implicit (preflight fails without it)")
    from steam_nffmtl import cli

    root = Path(__file__).resolve().parents[1]
    work = cli.cmd_smoke(str(tmp_path), str(root / "configs" / "smoke.yaml"))
    train_runs = [p for p in (work / "r").iterdir() if "train-full" in p.name]
    env = (train_runs[0] / "env.json").read_text()
    assert '"cuda_available": true' in env
