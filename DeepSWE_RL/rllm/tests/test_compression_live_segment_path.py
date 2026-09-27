import torch

from verl import DataProto
from rllm.trainer.verl.agent_ppo_trainer import _select_unique_sp_result


def test_sp8_results_restore_one_logical_row_set_not_s_times_sp():
    logical = DataProto.from_dict(tensors={"old_log_probs": torch.zeros(3, 7)})
    result = _select_unique_sp_result([logical for _ in range(8)], operation="oldlog")
    assert result.batch["old_log_probs"].shape == (3, 7)


def test_sp_rank_shape_disagreement_fails_fast():
    rank0 = DataProto.from_dict(tensors={"old_log_probs": torch.zeros(3, 7)})
    rank1 = DataProto.from_dict(tensors={"old_log_probs": torch.zeros(24, 7)})
    try:
        _select_unique_sp_result([rank0, rank1], operation="oldlog")
    except ValueError as exc:
        assert "logical row count" in str(exc)
    else:
        raise AssertionError("SP-expanded logical rows were silently accepted")
