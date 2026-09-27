import torch

from verl.workers.actor.dp_actor import _restore_compression_response_width


def test_different_local_response_widths_restore_and_concat_exactly():
    micro0 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    micro1 = torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]])
    restored = torch.cat(
        [
            _restore_compression_response_width(micro0, 9),
            _restore_compression_response_width(micro1, 9),
        ],
        dim=0,
    )
    assert restored.shape == (2, 9)
    assert torch.equal(restored[0, :4], micro0[0])
    assert torch.equal(restored[1, :7], micro1[0])
    assert torch.count_nonzero(restored[0, 4:]) == 0
    assert torch.count_nonzero(restored[1, 7:]) == 0


def test_restored_padding_is_excluded_by_policy_mask():
    local = torch.tensor([[0.5, -0.5]])
    restored = _restore_compression_response_width(local, 5)
    policy_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
    assert torch.sum(restored * policy_mask).item() == torch.sum(local).item()
