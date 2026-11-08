"""Experimental helpers for grouping question heads around a shared state.

This module deliberately does not alter :class:`Agent`'s default prediction path.
Its grouped execution helpers and ModernBERT K/V path are experimental and require
checkpoint-level parity checks before their results are treated as production output.
"""
from dataclasses import dataclass
from itertools import product
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch


GROUPING_METHODS = ("order", "random", "cosine", "oracle")
POSITION_MODES = ("packed", "head_local")
WITHIN_GROUP_MODES = ("full", "block_diagonal")


@dataclass(frozen=True)
class PackedGroup:
    """One packed sequence and the question-to-token routing needed to read it out."""

    question_indices: Tuple[int, ...]
    input_ids: Tuple[int, ...]
    position_ids: Tuple[int, ...]
    question_spans: Tuple[Tuple[int, int], ...]
    marker_positions: Tuple[Tuple[int, ...], ...]
    state_span: Tuple[int, int]
    attention_mask: torch.Tensor
    state_tokens_used: int
    state_tokens_dropped: int


@dataclass
class StateKVCache:
    """K/V tensors and identity metadata for a query-unaware state encoding.

    Keys are post-RoPE. The cache is tied to a state token sequence and model ID.
    """

    key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    state_token_ids: Tuple[int, ...]
    position_ids: Tuple[int, ...]
    model_id: Optional[str] = None
    final_hidden: Optional[torch.Tensor] = None

    @property
    def nbytes(self) -> int:
        return sum(key.numel() * key.element_size() + value.numel() * value.element_size()
                   for key, value in self.key_values)


def _ngrams(text: str, n: int = 3) -> Dict[str, float]:
    value = " " + " ".join(str(text).lower().split()) + " "
    grams = [value[i:i + n] for i in range(max(1, len(value) - n + 1))]
    counts: Dict[str, float] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {key: val / norm for key, val in counts.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(key, 0.0) for key, value in a.items())


def _balanced_assignments(n: int, k: int):
    """Yield restricted-growth labels with group sizes differing by at most one."""
    target_sizes = sorted((n // k + (i < n % k) for i in range(k)), reverse=True)
    for labels in product(range(k), repeat=n):
        if labels[0] != 0 or set(labels) != set(range(k)):
            continue
        sizes = [labels.count(group) for group in range(k)]
        if sorted(sizes, reverse=True) == target_sizes:
            yield labels


def group_questions(texts: Sequence[str], k: int, method: str = "order", seed: int = 0,
                    max_oracle_questions: int = 8) -> List[List[int]]:
    """Partition question texts into ``k`` non-empty groups.

    ``cosine`` uses normalized character trigrams and deterministic greedy assignment.
    ``oracle`` exhaustively maximizes within-group pairwise cosine and is intentionally
    restricted to small ``n``. Returned indices always cover ``range(len(texts))`` once.
    """
    n = len(texts)
    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k <= n:
        raise ValueError("k must be an integer between 0 and the number of questions")
    if method not in GROUPING_METHODS:
        raise ValueError("method must be one of %s" % (", ".join(GROUPING_METHODS)))
    if n == 0:
        if k != 0:
            raise ValueError("k must be 0 when there are no questions")
        return []
    if k == 0:
        return [list(range(n))]
    if k == n or method == "order":
        if k == n:
            return [[i] for i in range(n)]
        groups, offset = [], 0
        for group in range(k):
            size = n // k + (group < n % k)
            groups.append(list(range(offset, offset + size)))
            offset += size
        return groups

    vectors = [_ngrams(text) for text in texts]
    if method == "random":
        indices = list(range(n))
        random.Random(seed).shuffle(indices)
        groups = [[] for _ in range(k)]
        for offset, index in enumerate(indices):
            groups[offset % k].append(index)
        return [sorted(group) for group in groups]

    if method == "cosine":
        # Seed groups from the most dissimilar questions, then greedily attach each
        # remaining question to its most similar group while keeping sizes balanced.
        seeds = [0]
        while len(seeds) < k:
            seeds.append(max((i for i in range(n) if i not in seeds),
                             key=lambda i: min(_cosine(vectors[i], vectors[s]) for s in seeds)))
        groups = [[i] for i in seeds]
        for index in sorted((i for i in range(n) if i not in seeds),
                            key=lambda i: -max(_cosine(vectors[i], vectors[s]) for s in seeds)):
            eligible = [g for g in range(k) if len(groups[g]) < math.ceil(n / k)]
            chosen = max(eligible, key=lambda g: sum(_cosine(vectors[index], vectors[j])
                                                      for j in groups[g]) / len(groups[g]))
            groups[chosen].append(index)
        return [sorted(group) for group in groups]

    if n > max_oracle_questions:
        raise ValueError("oracle grouping supports at most %d questions" % max_oracle_questions)
    best_labels, best_score = None, float("-inf")
    for labels in _balanced_assignments(n, k):
        score = sum(_cosine(vectors[i], vectors[j]) for i in range(n) for j in range(i + 1, n)
                    if labels[i] == labels[j])
        if score > best_score:
            best_labels, best_score = labels, score
    groups = [[] for _ in range(k)]
    for i, group in enumerate(best_labels):
        groups[group].append(i)
    return groups


def build_group_mask(head_lengths: Sequence[int], state_length: int,
                     mode: str = "full", device=None) -> torch.Tensor:
    """Boolean ``[query, key]`` mask for heads followed by one shared state.

    All heads can read state tokens. In ``block_diagonal`` mode, a head can read
    shared prefix tokens (its own [CLS]/instruction/[SEP]) and its own option spans;
    state tokens can read every head and the complete state. ``full`` permits all
    tokens to interact.
    """
    if mode not in WITHIN_GROUP_MODES:
        raise ValueError("mode must be one of %s" % (", ".join(WITHIN_GROUP_MODES)))
    if not head_lengths or any(length <= 0 for length in head_lengths) or state_length < 0:
        raise ValueError("head_lengths must be positive and state_length non-negative")
    total = sum(head_lengths) + state_length
    mask = torch.ones((total, total), dtype=torch.bool, device=device)
    if mode == "full":
        return mask
    # The caller supplies each question as a standalone head. Within each head all
    # positions communicate; different heads are isolated. State queries may attend
    # to every head and state key, while head queries may attend to all state keys.
    cursor = 0
    for length in head_lengths:
        lo, hi = cursor, cursor + length
        mask[lo:hi, :sum(head_lengths)] = False
        mask[lo:hi, lo:hi] = True
        mask[lo:hi, sum(head_lengths):] = True
        cursor = hi
    return mask


def build_query_unaware_mask(head_lengths: Sequence[int], state_length: int, device=None) -> torch.Tensor:
    """Reference asymmetric mask: heads read own tokens plus state; state reads only itself."""
    if not head_lengths or any(length <= 0 for length in head_lengths) or state_length <= 0:
        raise ValueError("head_lengths and state_length must be positive")
    head_total = sum(head_lengths)
    mask = torch.ones((head_total + state_length, head_total + state_length),
                      dtype=torch.bool, device=device)
    cursor = 0
    for length in head_lengths:
        lo, hi = cursor, cursor + length
        mask[lo:hi, :head_total] = False
        mask[lo:hi, lo:hi] = True
        mask[lo:hi, head_total:] = True
        cursor = hi
    mask[head_total:, :head_total] = False
    return mask


def modernbert_group_attention_masks(attention_mask: torch.Tensor, position_ids: torch.Tensor,
                                     head_ids: torch.Tensor, sliding_window: Optional[int],
                                     query_unaware: bool = False) -> Dict[str, torch.Tensor]:
    """Build Transformers 5 ModernBERT full/local masks for packed question groups.

    ``head_ids`` is zero on the shared state and positive per question head. In
    query-unaware mode, state queries are also prevented from reading any head.
    Local-window comparisons use the caller's RoPE position IDs, matching Laya's
    existing parallel-option mask behavior.
    """
    if attention_mask.ndim != 2 or position_ids.shape != attention_mask.shape or head_ids.shape != attention_mask.shape:
        raise ValueError("attention_mask, position_ids and head_ids must have the same [batch, length] shape")
    valid_keys = attention_mask.bool()[:, None, :]
    q_head, k_head = head_ids[:, :, None], head_ids[:, None, :]
    other_head = (q_head > 0) & (k_head > 0) & (q_head != k_head)
    allowed = valid_keys[:, None] & ~other_head[:, None]
    if query_unaware:
        state_query_to_head = (q_head == 0) & (k_head > 0)
        allowed = allowed & ~state_query_to_head[:, None]
    masks = {"full_attention": allowed}
    if sliding_window is not None:
        near = (position_ids[:, :, None] - position_ids[:, None, :]).abs() <= sliding_window
        masks["sliding_attention"] = allowed & near[:, None]
    return masks


def route_marker_outputs(hidden: torch.Tensor, marker_positions: Sequence[Sequence[int]]) -> List[torch.Tensor]:
    """Gather each packed question's marker vectors in original option order."""
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [sequence, hidden_size]")
    routed = []
    for positions in marker_positions:
        if any(position < 0 or position >= hidden.shape[0] for position in positions):
            raise ValueError("marker position falls outside the packed hidden sequence")
        index = torch.tensor(positions, dtype=torch.long, device=hidden.device)
        routed.append(hidden.index_select(0, index))
    return routed


def _require_modernbert5(encoder):
    import transformers

    if transformers.__version__.split(".", 1)[0] != "5":
        raise RuntimeError("k-state ModernBERT attention masks and K/V reuse currently require transformers>=5")
    if type(encoder).__module__ != "transformers.models.modernbert.modeling_modernbert":
        raise TypeError("k-state execution currently supports Hugging Face ModernBertModel only")
    if encoder.training:
        raise RuntimeError("k-state inference requires encoder.eval() so cached state K/V is reusable")
    if getattr(encoder.config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError("k-state execution currently requires the ModernBERT SDPA implementation")
    layer_types = getattr(encoder.config, "layer_types", None)
    if (not layer_types or len(layer_types) != len(encoder.layers)
            or set(layer_types) - {"full_attention", "sliding_attention"}):
        raise RuntimeError("k-state execution needs ModernBERT full/sliding layer_types from Transformers 5")
    if ("sliding_attention" in layer_types
            and not isinstance(getattr(encoder.config, "sliding_window", None), int)):
        raise RuntimeError("k-state local-attention execution needs an integer sliding_window")


def _rotary_for(encoder, hidden: torch.Tensor, position_ids: torch.Tensor):
    return {kind: encoder.rotary_emb(hidden, position_ids, kind)
            for kind in set(encoder.config.layer_types)}


def _layer_qkv(layer, hidden, rotary, position_ids):
    attn = layer.attn
    normalized = layer.attn_norm(hidden)
    batch, length = normalized.shape[:2]
    values = attn.Wqkv(normalized).view(batch, length, 3, attn.num_heads, attn.head_dim)
    q, k, v = (part.transpose(1, 2) for part in values.unbind(dim=2))
    from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
    q, k = apply_rotary_pos_emb(q, k, *rotary[layer.attention_type], unsqueeze_dim=1)
    return q, k, v


def _layer_finish(layer, hidden, attended):
    attn = layer.attn
    projected = attn.out_drop(attn.Wo(attended.transpose(1, 2).reshape_as(hidden).contiguous()))
    hidden = hidden + projected
    return hidden + layer.mlp(layer.mlp_norm(hidden))


def _attention_mask(position_ids: torch.Tensor, key_positions: torch.Tensor,
                    key_valid: torch.Tensor, attention_type: str, sliding_window: int):
    allowed = key_valid[:, None, None, :].expand(-1, 1, position_ids.shape[1], -1)
    if attention_type == "sliding_attention":
        near = (position_ids[:, :, None] - key_positions[:, None, :]).abs() <= sliding_window
        allowed = allowed & near[:, None]
    # SDPA bool masks use True for allowed positions.
    return allowed


@torch.no_grad()
def encode_state_kv(encoder, state_token_ids: Sequence[int], model_id: Optional[str] = None) -> StateKVCache:
    """Encode one state and cache per-layer K/V after rotary position embedding.

    State position IDs start at zero. Query heads are offset by the encoded state
    length (state-first convention), which makes this cache reusable across requests.
    """
    _require_modernbert5(encoder)
    if not state_token_ids:
        raise ValueError("state_token_ids must not be empty")
    device = next(encoder.parameters()).device
    input_ids = torch.tensor([list(state_token_ids)], dtype=torch.long, device=device)
    hidden = encoder.embeddings(input_ids=input_ids)
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    rotary = _rotary_for(encoder, hidden, positions)
    cache = []
    valid = torch.ones_like(input_ids, dtype=torch.bool)
    for layer in encoder.layers:
        q, key, value = _layer_qkv(layer, hidden, rotary, positions)
        cache.append((key, value))
        mask = _attention_mask(positions, positions, valid, layer.attention_type,
                               encoder.config.sliding_window)
        attended = torch.nn.functional.scaled_dot_product_attention(
            q, key, value, attn_mask=mask, dropout_p=0.0)
        hidden = _layer_finish(layer, hidden, attended)
    hidden = encoder.final_norm(hidden)
    return StateKVCache(tuple(cache), tuple(int(i) for i in state_token_ids),
                        tuple(range(len(state_token_ids))), model_id, hidden)


@torch.no_grad()
def encode_head_with_state_cache(encoder, cache: StateKVCache, head_token_ids: Sequence[int]) -> torch.Tensor:
    """Encode one question head against a cached state; return head hidden states only."""
    _require_modernbert5(encoder)
    if not head_token_ids:
        raise ValueError("head_token_ids must not be empty")
    if len(cache.key_values) != len(encoder.layers):
        raise ValueError("state cache layer count does not match the encoder")
    device = next(encoder.parameters()).device
    head_ids = torch.tensor([list(head_token_ids)], dtype=torch.long, device=device)
    hidden = encoder.embeddings(input_ids=head_ids)
    state_len = len(cache.state_token_ids)
    positions = torch.arange(state_len, state_len + len(head_token_ids), device=device).unsqueeze(0)
    rotary = _rotary_for(encoder, hidden, positions)
    state_positions = torch.arange(state_len, device=device).unsqueeze(0)
    key_valid = torch.ones((1, state_len + len(head_token_ids)), dtype=torch.bool, device=device)
    key_positions = torch.cat([state_positions, positions], dim=1)
    for index, layer in enumerate(encoder.layers):
        q, head_key, head_value = _layer_qkv(layer, hidden, rotary, positions)
        state_key, state_value = cache.key_values[index]
        key = torch.cat([state_key.to(device=device, dtype=head_key.dtype), head_key], dim=2)
        value = torch.cat([state_value.to(device=device, dtype=head_value.dtype), head_value], dim=2)
        mask = _attention_mask(positions, key_positions, key_valid, layer.attention_type,
                               encoder.config.sliding_window)
        attended = torch.nn.functional.scaled_dot_product_attention(q, key, value,
                                                                     attn_mask=mask, dropout_p=0.0)
        hidden = _layer_finish(layer, hidden, attended)
    return encoder.final_norm(hidden)


@torch.no_grad()
def encode_query_unaware_reference(encoder, state_token_ids: Sequence[int],
                                  head_token_ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-pass asymmetric-mask reference for validating the state K/V cache path."""
    _require_modernbert5(encoder)
    if not state_token_ids or not head_token_ids:
        raise ValueError("state and head token ids must not be empty")
    device = next(encoder.parameters()).device
    state_len = len(state_token_ids)
    ids = torch.tensor([list(state_token_ids) + list(head_token_ids)], dtype=torch.long, device=device)
    hidden = encoder.embeddings(input_ids=ids)
    positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    rotary = _rotary_for(encoder, hidden, positions)
    state_query_to_head = torch.zeros((1, ids.shape[1], ids.shape[1]), dtype=torch.bool, device=device)
    state_query_to_head[:, :state_len, state_len:] = True
    allowed = ~state_query_to_head[:, None]
    for layer in encoder.layers:
        q, key, value = _layer_qkv(layer, hidden, rotary, positions)
        layer_mask = allowed
        if layer.attention_type == "sliding_attention":
            near = (positions[:, :, None] - positions[:, None, :]).abs() <= encoder.config.sliding_window
            layer_mask = layer_mask & near[:, None]
        attended = torch.nn.functional.scaled_dot_product_attention(
            q, key, value, attn_mask=layer_mask, dropout_p=0.0)
        hidden = _layer_finish(layer, hidden, attended)
    hidden = encoder.final_norm(hidden)
    return hidden[:, :state_len], hidden[:, state_len:]


@torch.no_grad()
def encode_packed_groups(encoder, groups: Sequence[PackedGroup], pad_token_id: int,
                         within_group: str = "full", query_unaware: bool = False):
    """Encode packed group rows with ModernBERT's per-layer full/local masks."""
    _require_modernbert5(encoder)
    if not groups:
        raise ValueError("groups must not be empty")
    device = next(encoder.parameters()).device
    max_length = max(len(group.input_ids) for group in groups)
    ids = torch.full((len(groups), max_length), pad_token_id, dtype=torch.long, device=device)
    valid = torch.zeros_like(ids)
    positions = torch.zeros_like(ids)
    head_ids = torch.zeros_like(ids)
    for row, group in enumerate(groups):
        length = len(group.input_ids)
        ids[row, :length] = torch.tensor(group.input_ids, dtype=torch.long, device=device)
        valid[row, :length] = 1
        positions[row, :length] = torch.tensor(group.position_ids, dtype=torch.long, device=device)
        if within_group == "full":
            for start, end in group.question_spans:
                head_ids[row, start:end] = 1
        elif within_group == "block_diagonal":
            for index, (start, end) in enumerate(group.question_spans, start=1):
                head_ids[row, start:end] = index
        else:
            raise ValueError("within_group must be one of %s" % (", ".join(WITHIN_GROUP_MODES)))
    masks = modernbert_group_attention_masks(valid, positions, head_ids,
                                             getattr(encoder.config, "sliding_window", None),
                                             query_unaware=query_unaware)
    hidden = encoder(input_ids=ids, attention_mask=masks, position_ids=positions).last_hidden_state
    return hidden, valid, positions, head_ids


def pack_group(question_indices: Sequence[int], heads: Sequence[Sequence[int]],
               markers: Sequence[Sequence[int]], state_ids: Sequence[int], sep_token_id: int,
               max_len: int, position_mode: str = "packed", within_group: str = "full",
               truncate_left: bool = False, device=None) -> PackedGroup:
    """Concatenate question heads and one state, returning exact marker routing."""
    if position_mode not in POSITION_MODES:
        raise ValueError("position_mode must be one of %s" % (", ".join(POSITION_MODES)))
    if max_len <= 0 or not question_indices:
        raise ValueError("max_len and question_indices must be non-empty")
    selected_heads = [list(heads[i]) for i in question_indices]
    selected_markers = [list(markers[i]) for i in question_indices]
    head_total = sum(map(len, selected_heads))
    if head_total + 1 > max_len:
        raise ValueError("packed question heads leave no room for the state closing token")
    room = max(0, max_len - head_total - 1)
    state = list(state_ids[-room:] if truncate_left and room else state_ids[:room])
    if room == 0:
        state = []
    ids, positions, spans, routed_markers = [], [], [], []
    cursor = 0
    max_head = max(map(len, selected_heads))
    for head, head_markers in zip(selected_heads, selected_markers):
        if any(marker < 0 or marker >= len(head) for marker in head_markers):
            raise ValueError("marker position falls outside its question head")
        ids.extend(head)
        local_positions = list(range(len(head)))
        positions.extend(local_positions if position_mode == "head_local"
                         else range(cursor, cursor + len(head)))
        spans.append((cursor, cursor + len(head)))
        routed_markers.append(tuple(cursor + marker for marker in head_markers))
        cursor += len(head)
    state_start = len(ids)
    ids.extend(state)
    ids.append(sep_token_id)
    if position_mode == "head_local":
        positions.extend(range(max_head, max_head + len(state) + 1))
    else:
        positions.extend(range(state_start, state_start + len(state) + 1))
    mask = build_group_mask([len(head) for head in selected_heads], len(state) + 1,
                            mode=within_group, device=device)
    return PackedGroup(
        question_indices=tuple(question_indices), input_ids=tuple(ids), position_ids=tuple(positions),
        question_spans=tuple(spans), marker_positions=tuple(routed_markers),
        state_span=(state_start, len(ids)), attention_mask=mask,
        state_tokens_used=len(state), state_tokens_dropped=max(0, len(state_ids) - len(state)),
    )

