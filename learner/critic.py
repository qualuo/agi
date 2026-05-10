"""Trace-quality critic.

A small specialist model that learns: given (prompt, response), predict the
probability that the response is good (passed eval / user thumbs-up). Plugs
into the architecture as the verifier component — instead of relying on
LLM-as-judge (drift) or only-eval-pass (sparse), the critic generalizes from
collected traces.

This is a deliberate first specialist: same training infrastructure that any
other small model would use, useful artifact at the end (filters bad
generations from Opus before showing them to the user; produces dense training
signal for the LoRA loop).

Architecture:
- Featurizer: hashed character n-grams, 4096-dim sparse → dense float vector.
  Cheap, language-agnostic, captures local lexical structure.
- Model: 2-layer MLP, ~500K params. Tiny; trains on CPU in seconds.
- Loss: BCE with logits.

Why not a transformer encoder? Could be — but for v1 the hashed-ngram + MLP
is simpler, faster to train, and a strong baseline. Upgrade when this saturates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CriticConfig:
    n_buckets: int = 4096
    ngram_min: int = 1
    ngram_max: int = 3
    hidden: int = 128
    dropout: float = 0.1


class CharHashFeaturizer:
    """Hash character n-grams into a fixed-size dense bag-of-features vector.

    Uses Python's hash() with stable seeding via PYTHONHASHSEED would be ideal,
    but for a single-process demo we just rely on builtin hash which is stable
    within a process run. Persisted models include the config so loaders
    rebuild a featurizer with matching parameters.
    """

    def __init__(self, cfg: CriticConfig) -> None:
        self.cfg = cfg

    def featurize(self, text: str) -> torch.Tensor:
        v = torch.zeros(self.cfg.n_buckets, dtype=torch.float32)
        for n in range(self.cfg.ngram_min, self.cfg.ngram_max + 1):
            for i in range(len(text) - n + 1):
                ngram = text[i : i + n]
                bucket = hash(ngram) % self.cfg.n_buckets
                v[bucket] += 1.0
        # L2 normalize so long inputs don't dominate
        norm = v.norm().clamp_min(1e-6)
        return v / norm

    def featurize_batch(self, texts: Iterable[str]) -> torch.Tensor:
        return torch.stack([self.featurize(t) for t in texts])


class CriticMLP(nn.Module):
    def __init__(self, cfg: CriticConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_buckets, cfg.hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(cfg.hidden, cfg.hidden)
        self.fc3 = nn.Linear(cfg.hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        h = F.gelu(self.fc2(h))
        return self.fc3(h).squeeze(-1)


class Critic:
    """End-to-end critic: featurize → score. Trains and predicts on
    (prompt, response) pairs."""

    def __init__(self, cfg: CriticConfig | None = None) -> None:
        self.cfg = cfg or CriticConfig()
        self.featurizer = CharHashFeaturizer(self.cfg)
        self.model = CriticMLP(self.cfg)

    @staticmethod
    def _join(prompt: str, response: str) -> str:
        # Separator token unlikely to appear naturally
        return f"{prompt}\x1e{response}"

    def predict_proba(self, prompt: str, response: str) -> float:
        self.model.eval()
        with torch.no_grad():
            x = self.featurizer.featurize(self._join(prompt, response)).unsqueeze(0)
            return torch.sigmoid(self.model(x)).item()

    def fit(
        self,
        examples: list[tuple[str, str, int]],
        *,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        verbose: bool = True,
    ) -> dict:
        """examples: list of (prompt, response, label) where label is 0 or 1."""
        joined = [self._join(p, r) for p, r, _ in examples]
        labels = torch.tensor([float(y) for _, _, y in examples])
        X = self.featurizer.featurize_batch(joined)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        n = len(examples)
        history = {"loss": [], "acc": []}
        self.model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            losses = []
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                xb, yb = X[idx], labels[idx]
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                preds = (torch.sigmoid(self.model(X)) > 0.5).float()
                acc = (preds == labels).float().mean().item()
            self.model.train()
            avg_loss = sum(losses) / len(losses)
            history["loss"].append(avg_loss)
            history["acc"].append(acc)
            if verbose:
                print(f"epoch {epoch + 1:3d}: loss={avg_loss:.4f} acc={acc:.3f}")
        return history

    def evaluate(self, examples: list[tuple[str, str, int]]) -> dict:
        joined = [self._join(p, r) for p, r, _ in examples]
        labels = torch.tensor([float(y) for _, _, y in examples])
        X = self.featurizer.featurize_batch(joined)
        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X))
            preds = (probs > 0.5).float()
            acc = (preds == labels).float().mean().item()
            tp = ((preds == 1) & (labels == 1)).sum().item()
            fp = ((preds == 1) & (labels == 0)).sum().item()
            fn = ((preds == 0) & (labels == 1)).sum().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
        return {"accuracy": acc, "precision": precision, "recall": recall, "n": len(examples)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "cfg": self.cfg.__dict__}, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "Critic":
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        critic = cls(CriticConfig(**ckpt["cfg"]))
        critic.model.load_state_dict(ckpt["state_dict"])
        return critic
