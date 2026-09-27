import pytest
import torch

from verl.utils.ulysses import ulysses_pad


@pytest.mark.parametrize("sp_size", [1, 2, 8])
@pytest.mark.parametrize("total_nnz", [8, 9, 15])
def test_ulysses_padding_is_temporary_flattened_axis_only(sp_size, total_nnz):
    flat = torch.arange(total_nnz).view(1, -1)
    padded, _, pad_size = ulysses_pad(flat, sp_size=sp_size)
    assert pad_size == (sp_size - total_nnz % sp_size) % sp_size
    assert padded.shape[-1] == total_nnz + pad_size
    assert torch.equal(padded[:, :total_nnz], flat)


def test_dummy_only_shape_and_sp_padding_do_not_create_policy_tokens():
    prompt = torch.tensor([[7]])
    response = torch.tensor([[7]])
    policy_mask = torch.zeros(1, 1)
    actor_weight = torch.zeros(1, 1)
    flat = torch.cat([prompt, response], dim=-1)
    padded, _, pad_size = ulysses_pad(flat, sp_size=4)
    assert prompt.shape[-1] == response.shape[-1] == 1
    assert pad_size == 2
    assert padded.shape[-1] == 4
    assert torch.sum(policy_mask * actor_weight).item() == 0.0
