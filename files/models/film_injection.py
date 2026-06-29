"""
FiLM Injection Module for YOLOv8n (Nano)
──────────────────────────────────────────
Feature-wise Linear Modulation (FiLM) layers injected into
YOLOv8n neck via PyTorch forward hooks — no ultralytics internals modified.

YOLOv8n Nano channel sizes (width_multiple = 0.25):
    P3 neck output  → model.model[15] → 64  channels
    P4 neck output  → model.model[18] → 128 channels
    P5 neck output  → model.model[21] → 256 channels

Reference:
    FiLM: Visual Reasoning with a General Conditioning Layer
    Perez et al., 2018  https://arxiv.org/abs/1709.07871
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from configs.task_config import FILM_CONFIG


# ── FiLM Layer ─────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Applies feature-wise affine transform conditioned on task embedding.

        output = γ(e) * x + β(e)

    where γ and β are MLP projections of the task embedding e,
    and x is the spatial feature map from the YOLO neck.

    Parameters
    ----------
    embed_dim    : int  - dimension of input task embedding (384)
    feature_dim  : int  - number of channels in the feature map (64/128/256)
    hidden_dim   : int  - MLP hidden size
    dropout      : float
    """
    def __init__(
        self,
        embed_dim:   int   = 384,
        feature_dim: int   = 64,
        hidden_dim:  int   = 128,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # γ network (scale)
        self.gamma_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        # β network (shift)
        self.beta_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Init gamma→1, beta→0 so FiLM starts as identity."""
        for layer in [self.gamma_net, self.beta_net]:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        # Final gamma bias → 1 (multiplicative identity)
        nn.init.ones_(self.gamma_net[-1].bias)

    def forward(self, x: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : [B, C, H, W]  feature map from YOLO neck
        task_emb : [B, embed_dim] task embedding (one per image in batch)

        Returns
        -------
        modulated : [B, C, H, W]
        """
        gamma = self.gamma_net(task_emb)           # [B, C]
        beta  = self.beta_net(task_emb)            # [B, C]

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta  = beta.unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]

        return gamma * x + beta                    # broadcast over H, W


# ── Hook Manager ───────────────────────────────────────────────────────────────

class FiLMHookManager:
    """
    Registers forward hooks on YOLOv8n neck layers and applies
    FiLM modulation in-place during the forward pass.

    Usage
    -----
        manager = FiLMHookManager(yolo_model)
        manager.register_hooks()
        manager.set_task_embedding(task_emb_batch)  # before each forward
        output = yolo_model(images)
        manager.remove_hooks()                       # when done training
    """

    # YOLOv8n (nano) neck module indices → channel sizes
    LAYER_CHANNELS = {
        15: 64,     # P3
        18: 128,    # P4
        21: 256,    # P5
    }

    def __init__(self, yolo_model: nn.Module):
        self.yolo_model    = yolo_model
        self._task_emb     = None    # set before each forward pass
        self._hooks        = []
        self.film_layers   = nn.ModuleDict()
        self._build_film_layers()

    def _build_film_layers(self):
        hidden_dims = FILM_CONFIG.get("film_hidden_dims", {})
        for layer_idx, ch in self.LAYER_CHANNELS.items():
            self.film_layers[f"layer_{layer_idx}"] = FiLMLayer(
                embed_dim   = FILM_CONFIG["embed_dim"],
                feature_dim = ch,
                hidden_dim  = hidden_dims.get(layer_idx, 128),
            )

    def film_parameters(self):
        """Return only FiLM layer parameters (for optimizer)."""
        return self.film_layers.parameters()

    def set_task_embedding(self, task_emb: torch.Tensor):
        """
        Call this before each forward pass.
        task_emb : [B, 384] — one embedding per image in the batch.
                   With batch-by-task, all images share the same embedding.
        """
        self._task_emb = task_emb

    def _make_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer index."""
        film = self.film_layers[f"layer_{layer_idx}"]

        def hook_fn(module, input, output):
            if self._task_emb is None:
                return output
            emb = self._task_emb.to(output.device)
            return film(output, emb)

        return hook_fn

    def register_hooks(self):
        """Attach hooks to YOLOv8n neck layers."""
        model_seq = self.yolo_model.model.model   # nn.Sequential inside YOLO
        for layer_idx in self.LAYER_CHANNELS:
            layer = model_seq[layer_idx]
            h     = layer.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(h)
        print(f"✅ FiLM hooks registered on layers {list(self.LAYER_CHANNELS.keys())}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        print("🔴 FiLM hooks removed.")

    def to(self, device):
        self.film_layers = self.film_layers.to(device)
        return self

    def train(self):
        self.film_layers.train()

    def eval(self):
        self.film_layers.eval()

    def state_dict(self):
        return self.film_layers.state_dict()

    def load_state_dict(self, sd):
        self.film_layers.load_state_dict(sd)
