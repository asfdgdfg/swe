import pytest

from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer


def test_d4_fails_with_stage_specific_message_before_rollout():
    trainer = object.__new__(AgentPPOTrainer)
    trainer.config = type("Config", (), {})()
    trainer.config.actor_rollout_ref = type("ARR", (), {})()
    trainer.config.actor_rollout_ref.actor = type("Actor", (), {"use_dynamic_bsz": False})()
    trainer.config.agent = {"compression_episode_mini_batch_size": 2, "use_stepwise_advantage": False}
    with pytest.raises(RuntimeError, match="effective DP degree == 1"):
        trainer._validate_compression_d1_config(4)


@pytest.mark.parametrize("value", [None, 0, -1, 1.5, True])
def test_compression_episode_batch_size_is_strict(value):
    trainer = object.__new__(AgentPPOTrainer)
    trainer.config = type("Config", (), {})()
    trainer.config.actor_rollout_ref = type("ARR", (), {})()
    trainer.config.actor_rollout_ref.actor = type("Actor", (), {"use_dynamic_bsz": False})()
    trainer.config.agent = {"compression_episode_mini_batch_size": value, "use_stepwise_advantage": False}
    with pytest.raises(ValueError, match="compression_episode_mini_batch_size"):
        trainer._validate_compression_d1_config(1)
