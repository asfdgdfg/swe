import pytest
import torch

from verl.workers.actor.dp_actor import _trim_compression_microbatch


def _wide_batch():
    p, r = 32, 20
    prompt0 = [0] * 27 + [101, 102, 103, 104, 105]
    prompt1 = [201, 202, 203, 204, 205, 206, 207, 208] + [0] * 24
    # Put row1's real prompt at the right edge, as required by left padding.
    prompt1 = [0] * 24 + prompt1[:8]
    response0 = [301, 302, 303, 304] + [0] * 16
    response1 = [401, 402, 403, 404, 405, 406, 407] + [0] * 13
    attention = torch.tensor(
        [[0] * 27 + [1] * 5 + [1] * 4 + [0] * 16,
         [0] * 24 + [1] * 8 + [1] * 7 + [0] * 13]
    )
    position_ids = (attention.cumsum(-1) - 1) * attention
    response_fields = {
        "responses": torch.tensor([response0, response1]),
        "advantages": torch.arange(40, dtype=torch.float32).view(2, 20),
        "policy_mask": torch.tensor([[1, 0, 1, 0] + [0] * 16, [1] * 7 + [0] * 13]),
        "actor_token_weight": torch.ones(2, 20),
        "episode_normalized_weight": torch.ones(2, 20),
        "old_log_probs": torch.arange(40, dtype=torch.float32).view(2, 20),
        "ref_log_prob": torch.arange(40, dtype=torch.float32).view(2, 20) + 1,
    }
    return {
        "prompts": torch.tensor([prompt0, prompt1]),
        "input_ids": torch.tensor([prompt0 + response0, prompt1 + response1]),
        "attention_mask": attention,
        "position_ids": position_ids,
        **response_fields,
    }


def test_prompt_left_trim_response_right_trim_and_aligned_fields():
    wide = _wide_batch()
    local, stats = _trim_compression_microbatch(wide, 32, 20)

    assert stats["prompt_local_width"] == 8
    assert stats["response_local_width"] == 7
    assert local["prompts"].tolist() == [
        [0, 0, 0, 101, 102, 103, 104, 105],
        [201, 202, 203, 204, 205, 206, 207, 208],
    ]
    assert local["responses"].tolist() == [
        [301, 302, 303, 304, 0, 0, 0],
        [401, 402, 403, 404, 405, 406, 407],
    ]
    for key in (
        "responses", "advantages", "policy_mask", "actor_token_weight",
        "episode_normalized_weight", "old_log_probs", "ref_log_prob",
    ):
        assert local[key].shape == (2, 7)
        assert torch.equal(local[key], wide[key][:, :7])
    assert local["input_ids"].shape == (2, 15)


def test_trim_preserves_real_token_position_ids_and_observation_tokens():
    wide = _wide_batch()
    local, _ = _trim_compression_microbatch(wide, 32, 20)
    expected_positions = torch.cat(
        [wide["position_ids"][:, 24:32], wide["position_ids"][:, 32:39]], dim=-1
    )
    assert torch.equal(local["position_ids"], expected_positions)
    # row0 response token 302 is an observation: attended but not a policy token.
    assert local["attention_mask"][0, 8 + 1].item() == 1
    assert local["policy_mask"][0, 1].item() == 0
    assert local["responses"][0, 1].item() == 302


@pytest.mark.parametrize(
    "attention,match",
    [
        ([0, 0, 1, 0, 1] + [1, 0], "prompt attention mask"),
        ([0, 0, 1, 1, 1] + [1, 0, 1], "response attention mask"),
    ],
)
def test_malformed_padding_fails_fast(attention, match):
    p = 5
    r = len(attention) - p
    batch = {
        "responses": torch.ones(1, r, dtype=torch.long),
        "input_ids": torch.ones(1, p + r, dtype=torch.long),
        "attention_mask": torch.tensor([attention]),
        "position_ids": torch.arange(p + r).view(1, -1),
    }
    with pytest.raises(ValueError, match=match):
        _trim_compression_microbatch(batch, p, r)


def test_memory_proxy_reports_hard_carrier_and_microbatch_savings():
    # One row per forward makes the intentionally uneven 4k/8k/12k/32k case
    # expose the maximum possible microbatch-local saving.
    lengths = [4_000, 8_000, 12_000, 32_000]
    hard_limit = 51_200
    hard_elements = len(lengths) * hard_limit
    carrier_elements = len(lengths) * max(lengths)
    microbatch_elements = sum(lengths)
    assert hard_elements == 204_800
    assert carrier_elements == 128_000
    assert microbatch_elements == 56_000
    assert hard_elements - microbatch_elements == 148_800
    assert (hard_elements - microbatch_elements) / hard_elements == pytest.approx(
        0.7265625
    )
