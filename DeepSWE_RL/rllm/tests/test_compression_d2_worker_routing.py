import numpy as np
import pytest
import torch

from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class _DPDimension:
    def __init__(self, coordinate):
        self.coordinate = coordinate

    def get_local_rank(self):
        return self.coordinate


class _Mesh:
    def __init__(self, coordinate):
        self.coordinate = coordinate

    def __getitem__(self, name):
        assert name == "dp"
        return _DPDimension(self.coordinate)


def _batch(targets):
    return DataProto.from_dict(
        tensors={"responses": torch.ones(len(targets), 1, dtype=torch.long)},
        non_tensors={"target_dp_replica": np.array(targets, dtype=np.int64)},
        meta_info={"compression_segment_training_stage": "7C-D2"},
    )


def test_mesh_4x2_rank_mapping_is_row_major():
    ranks = torch.arange(8).view(4, 2)
    assert ranks.tolist() == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_worker_accepts_only_its_mesh_dp_coordinate_before_and_after_gather():
    worker = object.__new__(ActorRolloutRefWorker)
    worker.ulysses_device_mesh = _Mesh(2)
    worker._validate_compression_d2_rows(_batch([2, 2]), after_sp_gather=False)
    worker._validate_compression_d2_rows(_batch([2, 2, 2, 2]), after_sp_gather=True)
    with pytest.raises(ValueError, match="worker_dp=2"):
        worker._validate_compression_d2_rows(_batch([1, 2]), after_sp_gather=False)
