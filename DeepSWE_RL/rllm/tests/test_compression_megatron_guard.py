import inspect

import pytest

# This test exercises the real Megatron worker only when its runtime is
# installed. A missing optional backend is a skip, never a collection error.
pytest.importorskip("megatron", reason="Megatron runtime is not installed")

from verl import DataProto
from verl.workers.megatron_workers import ActorRolloutRefWorker


def test_megatron_compression_guard_runs_before_actor_or_cuda_work():
    worker = object.__new__(ActorRolloutRefWorker)
    guarded_update = inspect.unwrap(ActorRolloutRefWorker.update_actor)
    compression_batch = DataProto(meta_info={"compression_segment_training": True})
    with pytest.raises(
        NotImplementedError,
        match="compression segment training currently supports FSDP only",
    ):
        guarded_update(worker, compression_batch)

    # Feature-disabled execution passes the new guard and reaches the original
    # role assertion on this deliberately incomplete worker, before CUDA use.
    worker._is_actor = False
    with pytest.raises(AssertionError):
        guarded_update(worker, DataProto(meta_info={"compression_segment_training": False}))
