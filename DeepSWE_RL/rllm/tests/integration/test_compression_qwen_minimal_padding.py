"""Standalone real-Qwen GPU smoke; intentionally excluded from normal pytest."""

import json
import math
import os

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import compute_policy_loss_weighted
from verl.workers.actor.dp_actor import DataParallelPPOActor, _trim_compression_microbatch


MODEL_PATH = os.environ.get(
    "QWEN3_MODEL_PATH", "/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B"
)


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.use_remove_padding = True
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False
    actor.ulysses_sequence_parallel_size = 1
    actor.compute_entropy_from_logits = verl_F.entropy_from_logits
    actor.device_name = "cuda"

    prompts = [
        encode(tokenizer, "System: You are a coding agent.\nUser: Fix the parser bug."),
        encode(
            tokenizer,
            "System: You are a coding agent with tools.\nUser: Inspect the repository, preserve tests, and fix the parser bug carefully.",
        ),
    ]
    action = encode(
        tokenizer,
        "<function=execute_bash><parameter=command>pwd</parameter></function>",
    )
    observation = encode(tokenizer, "Tool output: /workspace/project\n")
    compression = encode(
        tokenizer,
        "<function=compress_context><parameter=summary>Inspected repository and found the target parser file.</parameter></function>",
    )
    ordinary = encode(
        tokenizer,
        "<function=execute_bash><parameter=command>pytest -q</parameter></function>",
    )
    responses = [action + observation, compression + observation + ordinary]
    policy_masks = [
        [1] * len(action) + [0] * len(observation),
        [1] * len(compression) + [0] * len(observation) + [1] * len(ordinary),
    ]
    category_masks = {
        "policy": [
            [1] * len(action) + [0] * len(observation),
            [1] * len(compression) + [0] * len(observation) + [1] * len(ordinary),
        ],
        "observation": [
            [0] * len(action) + [1] * len(observation),
            [0] * len(compression) + [1] * len(observation) + [0] * len(ordinary),
        ],
        "compression": [
            [0] * len(responses[0]),
            [1] * len(compression) + [0] * (len(observation) + len(ordinary)),
        ],
    }

    p_carrier = max(map(len, prompts)) + 16
    r_carrier = max(map(len, responses)) + 12
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_tensor, response_tensor, attention, masks = [], [], [], []
    for prompt, response, mask in zip(prompts, responses, policy_masks, strict=True):
        prompt_tensor.append([pad] * (p_carrier - len(prompt)) + prompt)
        response_tensor.append(response + [pad] * (r_carrier - len(response)))
        attention.append(
            [0] * (p_carrier - len(prompt)) + [1] * len(prompt)
            + [1] * len(response) + [0] * (r_carrier - len(response))
        )
        masks.append(mask + [0] * (r_carrier - len(mask)))
    attention = torch.tensor(attention, device=device)
    wide = {
        "prompts": torch.tensor(prompt_tensor, device=device),
        "responses": torch.tensor(response_tensor, device=device),
        "input_ids": torch.tensor(
            [p + r for p, r in zip(prompt_tensor, response_tensor, strict=True)],
            device=device,
        ),
        "attention_mask": attention,
        "position_ids": (attention.cumsum(-1) - 1) * attention,
        "policy_mask": torch.tensor(masks, device=device),
    }
    minimal, stats = _trim_compression_microbatch(wide, p_carrier, r_carrier)

    def run_forward(batch, grad=False):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            entropy, logprob = actor._forward_micro_batch(
                batch, temperature=1.0, calculate_entropy=True
            )
        torch.cuda.synchronize()
        return entropy, logprob, {
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        }

    wide_entropy, wide_logprob, wide_memory = run_forward(wide)
    trim_entropy, trim_logprob, trim_memory = run_forward(minimal)
    valid_lengths = [len(response) for response in responses]
    diffs, entropy_diffs = [], []
    category_diffs = {key: [] for key in category_masks}
    for row, length in enumerate(valid_lengths):
        row_diff = (wide_logprob[row, :length] - trim_logprob[row, :length]).abs()
        diffs.append(row_diff)
        entropy_diffs.append(
            (wide_entropy[row, :length] - trim_entropy[row, :length]).abs()
        )
        for category, row_masks in category_masks.items():
            selector = torch.tensor(row_masks[row], device=device, dtype=torch.bool)
            if selector.any():
                category_diffs[category].append(row_diff[selector])

    policy_mask_wide = wide["policy_mask"].to(torch.long)
    policy_count = int(policy_mask_wide.sum().item())
    weights_wide = policy_mask_wide.float() / policy_count
    advantages_wide = policy_mask_wide.float() * 0.75
    old_wide = wide_logprob.detach() - 0.01
    wide_pg = compute_policy_loss_weighted(
        old_wide, wide_logprob, advantages_wide, policy_mask_wide,
        weights_wide, 0.2, 0.2, 0.2, 3.0,
    )[0]
    r_local = stats["response_local_width"]
    trim_pg = compute_policy_loss_weighted(
        old_wide[:, :r_local], trim_logprob, advantages_wide[:, :r_local],
        policy_mask_wide[:, :r_local], weights_wide[:, :r_local],
        0.2, 0.2, 0.2, 3.0,
    )[0]
    wide_weighted_entropy = torch.sum(wide_entropy * weights_wide)
    trim_weighted_entropy = torch.sum(
        trim_entropy * weights_wide[:, :r_local]
    )

    # Real backward while allocating gradients for only one representative
    # parameter. Model weights and input tokens remain identical.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    representative = model.model.norm.weight
    representative.requires_grad_(True)

    def backward_grad(batch, old, advantages, mask, weight):
        representative.grad = None
        _, current, _ = run_forward(batch, grad=True)
        loss = compute_policy_loss_weighted(
            old, current, advantages, mask, weight, 0.2, 0.2, 0.2, 3.0
        )[0]
        loss.backward()
        grad = representative.grad.detach().float().clone()
        return loss.detach(), grad

    wide_backward_loss, wide_grad = backward_grad(
        wide, old_wide, advantages_wide, policy_mask_wide, weights_wide
    )
    trim_backward_loss, trim_grad = backward_grad(
        minimal, old_wide[:, :r_local], advantages_wide[:, :r_local],
        policy_mask_wide[:, :r_local], weights_wide[:, :r_local]
    )

    all_diffs = torch.cat(diffs)
    all_entropy_diffs = torch.cat(entropy_diffs)
    result = {
        "model": MODEL_PATH,
        "dtype": str(next(model.parameters()).dtype),
        "attention_backend": "flash_attention_2",
        "prompt_lengths": list(map(len, prompts)),
        "response_lengths": valid_lengths,
        "p_carrier": p_carrier,
        "r_carrier": r_carrier,
        "p_local": stats["prompt_local_width"],
        "r_local": r_local,
        "wide_shape": list(wide["input_ids"].shape),
        "trim_shape": list(minimal["input_ids"].shape),
        "valid_tokens": int(attention.sum().item()),
        "max_logprob_diff": float(all_diffs.max().item()),
        "mean_logprob_diff": float(all_diffs.mean().item()),
        "category_max_diff": {
            key: float(torch.cat(value).max().item()) for key, value in category_diffs.items()
        },
        "weighted_pg_wide": float(wide_pg.item()),
        "weighted_pg_trim": float(trim_pg.item()),
        "weighted_pg_diff": float((wide_pg - trim_pg).abs().item()),
        "weighted_entropy_diff": float(
            (wide_weighted_entropy - trim_weighted_entropy).abs().item()
        ),
        "max_entropy_diff": float(all_entropy_diffs.max().item()),
        "wide_backward_loss": float(wide_backward_loss.item()),
        "trim_backward_loss": float(trim_backward_loss.item()),
        "grad_norm_wide": float(wide_grad.norm().item()),
        "grad_norm_trim": float(trim_grad.norm().item()),
        "grad_max_abs_diff": float((wide_grad - trim_grad).abs().max().item()),
        "grad_relative_norm_diff": float(
            (wide_grad.norm() - trim_grad.norm()).abs().item()
            / max(wide_grad.norm().item(), 1e-12)
        ),
        "wide_peak_memory": wide_memory,
        "trim_peak_memory": trim_memory,
        "wide_elements": int(wide["input_ids"].numel()),
        "trim_elements": int(minimal["input_ids"].numel()),
    }
    print(json.dumps(result, indent=2))
    assert result["max_logprob_diff"] < 5e-3
    assert result["weighted_pg_diff"] < 5e-4
    assert result["weighted_entropy_diff"] < 5e-3
    assert result["grad_relative_norm_diff"] < 5e-3


if __name__ == "__main__":
    main()
