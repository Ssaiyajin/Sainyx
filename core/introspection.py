"""
Model introspection for the "neuron activity" visualization.

Runs one real generation pass and returns:
  - per-token, per-layer activation strength for all 6 transformer blocks
    (how strongly each block actually fired for each token)
  - the real attention weight matrices from the final block's heads
    (which tokens each head actually attended to)

This is read-only instrumentation - it doesn't change generation behavior,
just observes it via forward hooks + the last_attn stash on AttentionHead.
"""

from typing import Dict, List

import torch


def run_with_activations(model, idx: torch.Tensor, encode=None, decode=None) -> Dict:
    """
    model: a loaded Sainyx instance (eval mode)
    idx: input token indices, shape (1, T)

    Returns:
      {
        "tokens": [str, ...]                # decoded input tokens, if decode given
        "layer_activations": [[float, ...]] # one list per block, one value per token
        "attention": [[[float, ...]]]       # one matrix per head in the last block, (T, T)
      }
    """
    model.eval()

    layer_activations: List[List[float]] = [None] * len(model.blocks)  # type: ignore
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            # Mean absolute activation per token position -> one scalar per token
            per_token = output.detach().abs().mean(dim=-1)[0]  # (T,)
            layer_activations[layer_idx] = per_token.cpu().tolist()
        return hook

    for i, block in enumerate(model.blocks):
        h = block.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(idx)

    for h in hooks:
        h.remove()

    # Real attention weights from the last block's heads (post-forward stash)
    last_block = model.blocks[-1]
    attention = [
        head.last_attn[0].cpu().tolist()  # (T, T) for this head
        for head in last_block.self_attention.heads
    ]

    tokens = None
    if decode is not None:
        tokens = [decode([t]) for t in idx[0].cpu().tolist()]

    return {
        "tokens": tokens,
        "layer_activations": layer_activations,
        "attention": attention,
    }