"""Standalone two-GPU real-Qwen Ulysses smoke."""

import json
import os

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import compute_policy_loss_weighted
from verl.utils.ulysses import set_ulysses_sequence_parallel_group
from verl.workers.actor.dp_actor import DataParallelPPOActor, _trim_compression_microbatch


MODEL_PATH = os.environ.get(
    "QWEN3_MODEL_PATH", "/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B"
)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True,
    ).to(device).eval()
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.use_remove_padding = True
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = True
    actor.ulysses_sequence_parallel_size = 2
    actor.compute_entropy_from_logits = verl_F.entropy_from_logits
    actor.device_name = "cuda"

    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prompt = enc("System: coding agent.\nUser: inspect and fix parser.")
    assistant = enc("<function=execute_bash><parameter=command>pwd</parameter></function>")
    observation = enc("Tool output: /workspace/project\n")
    response = assistant + observation
    # Make total_nnz odd without changing token identity semantics.
    if (len(prompt) + len(response)) % 2 == 0:
        prompt = enc("Extra context. ") + prompt
        if (len(prompt) + len(response)) % 2 == 0:
            prompt = prompt[1:]
    assert (len(prompt) + len(response)) % 2 == 1
    p_carrier, r_carrier = len(prompt) + 9, len(response) + 7
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    prompt_wide = [pad] * (p_carrier - len(prompt)) + prompt
    response_wide = response + [pad] * (r_carrier - len(response))
    attention = torch.tensor(
        [[0] * (p_carrier - len(prompt)) + [1] * len(prompt)
         + [1] * len(response) + [0] * (r_carrier - len(response))],
        device=device,
    )
    policy = [1] * len(assistant) + [0] * len(observation)
    wide = {
        "responses": torch.tensor([response_wide], device=device),
        "input_ids": torch.tensor([prompt_wide + response_wide], device=device),
        "attention_mask": attention,
        "position_ids": (attention.cumsum(-1) - 1) * attention,
        "policy_mask": torch.tensor([policy + [0] * (r_carrier - len(policy))], device=device),
    }
    minimal, stats = _trim_compression_microbatch(wide, p_carrier, r_carrier)
    with torch.no_grad():
        _, wide_lp = actor._forward_micro_batch(wide, 1.0, False)
        _, trim_lp = actor._forward_micro_batch(minimal, 1.0, False)
    diff = (wide_lp[:, : len(response)] - trim_lp[:, : len(response)]).abs()
    policy_selector = minimal["policy_mask"].bool()
    obs_selector = minimal["attention_mask"][:, -stats["response_local_width"] :].bool() & ~policy_selector

    eos = tokenizer.eos_token_id
    dummy_attention = torch.ones(1, 2, dtype=torch.long, device=device)
    dummy = {
        "responses": torch.tensor([[eos]], device=device),
        "input_ids": torch.tensor([[eos, eos]], device=device),
        "attention_mask": dummy_attention,
        "position_ids": torch.tensor([[0, 1]], device=device),
        "policy_mask": torch.zeros(1, 1, dtype=torch.long, device=device),
    }
    with torch.no_grad():
        _, dummy_lp = actor._forward_micro_batch(dummy, 1.0, False)
    dummy_loss = compute_policy_loss_weighted(
        dummy_lp, dummy_lp, torch.zeros_like(dummy_lp), dummy["policy_mask"],
        torch.zeros_like(dummy_lp), 0.2, 0.2, 0.2, 3.0,
    )[0]
    result = {
        "world": dist.get_world_size(), "sp": 2, "d": 1,
        "p_carrier": p_carrier, "r_carrier": r_carrier,
        "p_local": stats["prompt_local_width"],
        "r_local": stats["response_local_width"],
        "total_nnz": stats["total_nnz"],
        "sp_pad": (2 - stats["total_nnz"] % 2) % 2,
        "wide_shape": list(wide["input_ids"].shape),
        "trim_shape": list(minimal["input_ids"].shape),
        "output_shape": list(trim_lp.shape),
        "policy_shape": list(minimal["policy_mask"].shape),
        "max_logprob_diff": float(diff.max().item()),
        "mean_logprob_diff": float(diff.mean().item()),
        "policy_max_diff": float(diff[policy_selector].max().item()),
        "observation_max_diff": float(diff[obs_selector].max().item()),
        "dummy_output_shape": list(dummy_lp.shape),
        "dummy_loss": float(dummy_loss.item()),
    }
    gathered = [None] * 2
    dist.all_gather_object(gathered, result)
    if rank == 0:
        print(json.dumps({"ranks": gathered}, indent=2))
    assert result["sp_pad"] == 1
    assert result["output_shape"] == [1, len(response)]
    assert result["max_logprob_diff"] < 5e-3
    assert result["dummy_output_shape"] == [1, 1]
    assert result["dummy_loss"] == 0.0
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
