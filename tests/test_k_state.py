"""Small deterministic checks for experimental k-state packing helpers."""
import torch
import pytest

from laya.k_state import (
    build_group_mask,
    build_query_unaware_mask,
    group_questions,
    modernbert_group_attention_masks,
    pack_group,
    route_marker_outputs,
)


def test_grouping_covers_each_question_and_is_seeded():
    texts = ["billing refund", "refund invoice", "technical error", "bug report", "account access"]
    groups = group_questions(texts, 2, method="random", seed=17)
    assert groups == group_questions(texts, 2, method="random", seed=17)
    assert sorted(index for group in groups for index in group) == list(range(len(texts)))
    assert sorted(map(len, groups)) == [2, 3]
    assert group_questions(texts, 2, method="order") == [[0, 1, 2], [3, 4]]
    assert group_questions(texts, 2, method="cosine") == group_questions(texts, 2, method="cosine")


def test_oracle_grouping_is_bounded_and_covers_questions():
    texts = ["refund", "invoice", "bug", "crash"]
    groups = group_questions(texts, 2, method="oracle")
    assert sorted(index for group in groups for index in group) == list(range(4))
    try:
        group_questions(texts * 3, 2, method="oracle")
    except ValueError as exc:
        assert "at most" in str(exc)
    else:
        raise AssertionError("oracle grouping must refuse large exhaustive searches")


def test_attention_masks_keep_questions_separate_or_allow_full_interaction():
    diagonal = build_group_mask([2, 2], 2, mode="block_diagonal")
    assert diagonal[0, 1] and diagonal[1, 0]
    assert not diagonal[0, 2] and not diagonal[2, 0]
    assert diagonal[0, 4] and diagonal[4, 0]
    full = build_group_mask([2, 2], 2, mode="full")
    assert full.all()


def test_query_unaware_mask_is_asymmetric():
    mask = build_query_unaware_mask([2, 2], 2)
    assert mask[0, 0] and mask[0, 1] and mask[0, 4]
    assert not mask[0, 2]
    assert mask[4, 4] and mask[4, 5]
    assert not mask[4, 0] and not mask[5, 3]


def test_modernbert_masks_support_batch_padding_and_window_positions():
    valid = torch.tensor([[1, 1, 1, 1, 1, 0]])
    pos = torch.tensor([[0, 1, 0, 1, 2, 3]])
    slots = torch.tensor([[1, 1, 2, 2, 0, 0]])
    masks = modernbert_group_attention_masks(valid, pos, slots, sliding_window=1,
                                             query_unaware=True)
    assert masks["full_attention"].shape == (1, 1, 6, 6)
    assert masks["sliding_attention"].shape == (1, 1, 6, 6)
    full = masks["full_attention"][0, 0]
    assert full[0, 1] and full[0, 4]
    assert not full[0, 2] and not full[4, 0]
    assert not full[:, 5].any()
    local = masks["sliding_attention"][0, 0]
    assert not local[0, 4] and local[1, 4]


def test_pack_routes_markers_and_tracks_state_truncation():
    packed = pack_group(
        question_indices=[1, 0],
        heads=[[10, 11, 12], [20, 21]],
        markers=[[1], [0]],
        state_ids=[30, 31, 32, 33],
        sep_token_id=2,
        max_len=7,
        position_mode="head_local",
        within_group="block_diagonal",
    )
    assert packed.input_ids == (20, 21, 10, 11, 12, 30, 2)
    assert packed.marker_positions == ((0,), (2,))
    assert packed.state_span == (5, 7)
    assert packed.position_ids == (0, 1, 0, 1, 2, 3, 4)
    assert (packed.state_tokens_used, packed.state_tokens_dropped) == (1, 3)
    assert packed.attention_mask[0, 2]


def test_marker_readout_routes_each_questions_slots():
    hidden = torch.arange(30).reshape(10, 3)
    routed = route_marker_outputs(hidden, [(7, 2), (5,)])
    assert routed[0].tolist() == [hidden[7].tolist(), hidden[2].tolist()]
    assert routed[1].tolist() == [hidden[5].tolist()]


def test_k_equals_question_count_delegates_to_the_original_prediction_path():
    from laya.agent import Agent

    agent = object.__new__(Agent)
    expected = {"answers": {"q": {"choice": "a"}}}
    calls = []

    def original(state, questions, **kwargs):
        calls.append((state, questions, kwargs))
        return expected

    agent.system_one = original
    questions = {"q": {"type": "noul", "instructions": "x"}}
    result = agent.predict_k_state("state", questions, k=1)
    assert result is expected
    assert calls == [("state", questions, {"max_len": None, "head_max_len": None,
                                           "lang": None, "min_confidence": None})]


def test_two_pass_k0_matches_asymmetric_reference():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__.split(".", 1)[0] != "5":
        pytest.skip("manual ModernBERT K/V path currently targets Transformers 5")
    from transformers import ModernBertConfig, ModernBertModel
    from laya.k_state import encode_head_with_state_cache, encode_query_unaware_reference, encode_state_kv

    torch.manual_seed(29)
    config = ModernBertConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, local_attention=8, global_attn_every_n_layers=2,
        max_position_embeddings=128, attn_implementation="sdpa",
    )
    state = [1, 7, 9, 10, 11]
    head = [2, 12, 13, 14]
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        encoder = ModernBertModel(config).to(device=device, dtype=dtype).eval()
        cache = encode_state_kv(encoder, state)
        two_pass = encode_head_with_state_cache(encoder, cache, head)
        reference_state, reference_head = encode_query_unaware_reference(encoder, state, head)
        tolerance = 2e-2 if dtype == torch.float16 else 1e-5
        assert torch.allclose(cache.final_hidden, reference_state, atol=tolerance, rtol=tolerance)
        assert torch.allclose(two_pass, reference_head[0], atol=tolerance, rtol=tolerance)
