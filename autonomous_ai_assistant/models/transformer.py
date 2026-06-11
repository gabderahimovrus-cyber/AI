from __future__ import annotations

from dataclasses import dataclass, asdict

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None
    F = None


@dataclass(slots=True)
class TransformerConfig:
    vocab_size: int = 32_000
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    context_length: int = 512
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


if nn is not None:
    class DecoderBlock(nn.Module):
        def __init__(self, config: TransformerConfig):
            super().__init__()
            self.ln_1 = nn.LayerNorm(config.hidden_size)
            self.attn = nn.MultiheadAttention(
                config.hidden_size,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.ln_2 = nn.LayerNorm(config.hidden_size)
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 4),
                nn.GELU(),
                nn.Linear(config.hidden_size * 4, config.hidden_size),
                nn.Dropout(config.dropout),
            )

        def forward(self, x, causal_mask):
            h = self.ln_1(x)
            attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
            x = x + attn_out
            return x + self.mlp(self.ln_2(x))


    class DecoderOnlyTransformer(nn.Module):
        """Small GPT-style decoder-only Transformer trainable on 4GB GPUs."""

        def __init__(self, config: TransformerConfig | None = None):
            super().__init__()
            self.config = config or TransformerConfig()
            self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
            self.position_embedding = nn.Embedding(self.config.context_length, self.config.hidden_size)
            self.drop = nn.Dropout(self.config.dropout)
            self.blocks = nn.ModuleList([DecoderBlock(self.config) for _ in range(self.config.num_layers)])
            self.ln_f = nn.LayerNorm(self.config.hidden_size)
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight

        def forward(self, input_ids, labels=None):
            _, seq_len = input_ids.shape
            if seq_len > self.config.context_length:
                raise ValueError(f"Sequence length {seq_len} exceeds context length {self.config.context_length}")
            pos = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
            x = self.drop(self.token_embedding(input_ids) + self.position_embedding(pos))
            mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device).triu(1)
            for block in self.blocks:
                x = block(x, mask)
            logits = self.lm_head(self.ln_f(x))
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            return {"loss": loss, "logits": logits}

        @property
        def parameter_count(self) -> int:
            return sum(p.numel() for p in self.parameters())

        @torch.no_grad()
        def generate(self, input_ids, max_new_tokens: int = 64, temperature: float = 0.8):
            self.eval()
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.config.context_length :]
                logits = self(context)["logits"][:, -1, :] / max(temperature, 1e-5)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_id], dim=1)
            return input_ids
else:
    class DecoderOnlyTransformer:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise ImportError("PyTorch is required for DecoderOnlyTransformer. Install torch to train the local model.")
