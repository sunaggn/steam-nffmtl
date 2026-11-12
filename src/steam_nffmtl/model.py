"""NFF-MTL: user/game embeddings + tag projection fused in a shared MLP with two heads.

The first fusion layer acts on ``[e_u ; e_i ; h_c]``, so it decomposes as
``W_u e_u + W_i e_i + W_c h_c + b``. Full-catalog scoring precomputes the game
part once and broadcasts it against each user, which is exactly equivalent to
calling :meth:`NFFMTL.forward` on every (user, game) pair in eval mode.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NFFMTL(nn.Module):
    """Neural feature fusion with an optional auxiliary playtime-regression head."""

    def __init__(self, n_users: int, n_items: int, tag_matrix: np.ndarray, embedding_dim: int = 64,
                 hidden: tuple[int, int, int] | list[int] = (128, 64, 32),
                 dropout: tuple[float, float] | list[float] = (0.3, 0.2),
                 use_content: bool = True, use_mtl: bool = True) -> None:
        super().__init__()
        h1, h2, h3 = hidden
        d = embedding_dim
        self.d, self.use_content, self.use_mtl = d, use_content, use_mtl
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.register_buffer("tags", torch.as_tensor(tag_matrix, dtype=torch.float32))
        self.content = nn.Sequential(nn.Linear(tag_matrix.shape[1], d), nn.ReLU()) if use_content else None
        self.fc1 = nn.Linear(d * (3 if use_content else 2), h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.drop1 = nn.Dropout(dropout[0])
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.drop2 = nn.Dropout(dropout[1])
        self.fc3 = nn.Linear(h2, h3)
        self.rec_head = nn.Linear(h3, 1)
        self.hour_head = nn.Linear(h3, 1) if use_mtl else None

    def _trunk(self, z1: torch.Tensor) -> torch.Tensor:
        """Layers after the first linear map (BN -> ReLU -> Dropout -> ...)."""
        x = self.drop1(F.relu(self.bn1(z1)))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return F.relu(self.fc3(x))

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return recommendation logits and (if enabled) predicted log-playtime, each shape (B,)."""
        parts = [self.user_emb(users), self.item_emb(items)]
        if self.content is not None:
            parts.append(self.content(self.tags[items]))
        h = self._trunk(self.fc1(torch.cat(parts, dim=1)))
        logits = self.rec_head(h).squeeze(-1)
        hours = self.hour_head(h).squeeze(-1) if self.hour_head is not None else None
        return logits, hours

    @torch.no_grad()
    def item_part(self) -> torch.Tensor:
        """Game-dependent part of the first layer for the whole catalog, shape (n_items, h1)."""
        w = self.fc1.weight
        d = self.d
        z = self.item_emb.weight @ w[:, d:2 * d].T + self.fc1.bias
        if self.content is not None:
            z = z + self.content(self.tags) @ w[:, 2 * d:].T
        return z

    @torch.no_grad()
    def score_users(self, users: torch.Tensor, item_chunk: int = 4096,
                    item_part: torch.Tensor | None = None) -> torch.Tensor:
        """Logits for every catalog game, shape (len(users), n_items). Requires eval mode."""
        if self.training:
            raise RuntimeError("score_users must be called in eval mode")
        zi = self.item_part() if item_part is None else item_part
        zu = self.user_emb(users) @ self.fc1.weight[:, : self.d].T
        n_items, h1 = zi.shape
        out = torch.empty(len(users), n_items, device=zu.device)
        for s in range(0, n_items, item_chunk):
            z = zu[:, None, :] + zi[None, s:s + item_chunk, :]
            b, c, _ = z.shape
            h = self._trunk(z.reshape(b * c, h1))
            out[:, s:s + c] = self.rec_head(h).reshape(b, c)
        return out


class NFFMTLScorer:
    """Adapter that exposes a trained :class:`NFFMTL` through the common Scorer interface."""

    def __init__(self, model: NFFMTL, device: torch.device | str, item_chunk: int = 4096,
                 name: str = "NFF-MTL") -> None:
        self.model, self.device, self.item_chunk, self.name = model, torch.device(device), item_chunk, name
        self._item_part: torch.Tensor | None = None

    def refresh(self) -> None:
        """Recompute the cached game part (call after the weights change)."""
        self.model.eval()
        self._item_part = self.model.item_part()

    def score(self, users: np.ndarray) -> np.ndarray:
        """Full-catalog logits for a batch of user indices."""
        if self._item_part is None:
            self.refresh()
        u = torch.as_tensor(users, dtype=torch.long, device=self.device)
        return self.model.score_users(u, self.item_chunk, self._item_part).double().cpu().numpy()

    @torch.no_grad()
    def score_pairs(self, users: np.ndarray, items: np.ndarray, batch: int = 65536) -> np.ndarray:
        """Logits for aligned (user, game) pairs."""
        self.model.eval()
        out = []
        for s in range(0, len(users), batch):
            u = torch.as_tensor(users[s:s + batch], dtype=torch.long, device=self.device)
            i = torch.as_tensor(items[s:s + batch], dtype=torch.long, device=self.device)
            out.append(self.model(u, i)[0].double().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    @torch.no_grad()
    def predict_hours(self, users: np.ndarray, items: np.ndarray, batch: int = 65536) -> np.ndarray:
        """Predicted log1p(hours) for aligned pairs (requires the regression head)."""
        if self.model.hour_head is None:
            raise RuntimeError("Model was trained without the playtime head")
        self.model.eval()
        out = []
        for s in range(0, len(users), batch):
            u = torch.as_tensor(users[s:s + batch], dtype=torch.long, device=self.device)
            i = torch.as_tensor(items[s:s + batch], dtype=torch.long, device=self.device)
            out.append(self.model(u, i)[1].double().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
