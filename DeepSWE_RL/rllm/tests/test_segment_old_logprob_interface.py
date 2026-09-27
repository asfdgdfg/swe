import inspect
from types import SimpleNamespace

import torch

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


class _EvalOnlyModule:
    def eval(self):
        return self


def test_segment_rows_recompute_old_logprob_under_their_own_real_prompts():
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = _EvalOnlyModule()
    actor.ulysses_sequence_parallel_size = 1
    calls = []

    def forward(micro_batch, temperature, calculate_entropy):
        calls.append(
            {
                key: micro_batch[key].clone()
                for key in ("input_ids", "attention_mask", "position_ids", "responses")
            }
        )
        response_length = micro_batch["responses"].shape[-1]
        shape = (micro_batch["responses"].shape[0], response_length)
        return torch.full(shape, 0.5), torch.full(shape, -0.25)

    actor._forward_micro_batch = forward
    segment_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[10, 11, 12, 20, 21], [30, 31, 32, 40, 41]]),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
            # The second response includes an observation token. It remains in
            # the autoregressive sequence and receives a logprob.
            "responses": torch.tensor([[20, 21], [40, 41]]),
        },
        meta_info={
            "micro_batch_size": 1,
            "temperature": 1.0,
            "use_dynamic_bsz": False,
            "compression_segment_training": True,
        },
    )

    compute_log_prob = inspect.unwrap(DataParallelPPOActor.compute_log_prob)
    old_log_probs, entropies = compute_log_prob(actor, segment_batch, calculate_entropy=True)

    assert len(calls) == 2
    assert calls[0]["input_ids"].tolist() == [[10, 11, 12, 20, 21]]
    assert calls[1]["input_ids"].tolist() == [[30, 31, 32, 40, 41]]
    assert calls[0]["responses"].tolist() == [[20, 21]]
    assert calls[1]["responses"].tolist() == [[40, 41]]
    assert calls[0]["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert calls[1]["position_ids"].tolist() == [[0, 1, 2, 3, 4]]
    assert old_log_probs.shape == (2, 2)
    assert entropies.shape == (2, 2)
    assert torch.all(old_log_probs == -0.25)
