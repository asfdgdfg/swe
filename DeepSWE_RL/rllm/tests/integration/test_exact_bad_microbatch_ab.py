"""Standalone exact-token SP1/SP2 NaN diagnostic; no backward or optimizer."""

import json
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


MODEL_PATH = "/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B"
FIXTURE_PATH = "/tmp/stage8c_d2_bad_microbatch.pt"


def counts(tensor):
    return {
        "shape": list(tensor.shape),
        "nan": int(torch.isnan(tensor).sum()),
        "posinf": int(torch.isposinf(tensor).sum()),
        "neginf": int(torch.isneginf(tensor).sum()),
    }


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    sp = int(os.environ["DIAGNOSTIC_SP"])
    assert world == sp
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    fixture = torch.load(FIXTURE_PATH, map_location="cpu")
    micro_batch = {
        key: fixture[key].to(device)
        for key in ("input_ids", "attention_mask", "position_ids", "responses")
    }
    assert tuple(micro_batch["input_ids"].shape) == (1, 23)
    assert fixture["total_nnz"] == 23

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model = FSDP(model, device_id=device, use_orig_params=False)
    config = OmegaConf.create(
        {
            "use_remove_padding": True,
            "use_fused_kernels": False,
            "ulysses_sequence_parallel_size": sp,
            "use_torch_compile": False,
        }
    )
    actor = DataParallelPPOActor(config, model)
    mesh = None
    if sp > 1:
        mesh = init_device_mesh("cuda", (1, sp), mesh_dim_names=("dp", "sp"))
    manager = FSDPUlyssesShardingManager(mesh)

    hook_records = []

    def lm_head_hook(module, inputs, output):
        hidden = inputs[0].detach()
        logits = output.detach()
        # Prediction for response col0 (token at global flattened index 18)
        # is produced at flattened prediction index 17.
        global_prediction_index = 17
        padded_total = 24 if sp == 2 else 23
        shard_width = padded_total // sp
        owner = global_prediction_index // shard_width
        local_index = global_prediction_index % shard_width
        record = {
            "hidden": counts(hidden),
            "hidden_max_abs": float(hidden.float().abs().max()),
            "hidden_norm": float(hidden.float().norm()),
            "logits": counts(logits),
            "global_prediction_index": global_prediction_index,
            "owner_sp_rank": owner,
            "local_prediction_index": local_index,
        }
        if rank == owner:
            vector = logits.reshape(-1, logits.size(-1))[local_index]
            selected = vector[1810]
            log_z_current = torch.logsumexp(vector, dim=-1)
            log_z_fp32 = torch.logsumexp(vector.float(), dim=-1)
            record.update(
                {
                    "vocab_vector": counts(vector),
                    "selected_raw_logit": float(selected.float()),
                    "selected_isfinite": bool(torch.isfinite(selected)),
                    "log_z_current": float(log_z_current.float()),
                    "log_z_current_isfinite": bool(torch.isfinite(log_z_current)),
                    "log_z_fp32": float(log_z_fp32),
                    "log_z_fp32_isfinite": bool(torch.isfinite(log_z_fp32)),
                    "manual_fp32_logprob": float(selected.float() - log_z_fp32),
                }
            )
        hook_records.append(record)

    handle = model.module.lm_head.register_forward_hook(lm_head_hook)
    production_runs = []
    with manager:
        for run in range(3):
            with torch.no_grad():
                _, log_probs = actor._forward_micro_batch(
                    micro_batch, temperature=1.0, calculate_entropy=False
                )
            production_runs.append(
                {
                    "run": run + 1,
                    "shape": list(log_probs.shape),
                    "bad_position_logprob": str(float(log_probs[0, 0])),
                    "bad_position_isfinite": bool(torch.isfinite(log_probs[0, 0])),
                    "all": counts(log_probs),
                }
            )
    handle.remove()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    local_report = {
        "rank": rank,
        "world": world,
        "sp": sp,
        "fixture": {
            "p_carrier": fixture["p_carrier"],
            "r_carrier": fixture["r_carrier"],
            "p_local": fixture["p_local"],
            "r_local": fixture["r_local"],
            "total_nnz": fixture["total_nnz"],
            "sp_pad": fixture["sp_pad"] if sp == 2 else 0,
            "prompt_token_ids": fixture["input_ids"][0, :18].tolist(),
            "response_token_ids": fixture["responses"][0].tolist(),
            "response_decoded": [tokenizer.decode([int(x)]) for x in fixture["responses"][0]],
            "position_ids": fixture["position_ids"].tolist(),
            "attention_mask": fixture["attention_mask"].tolist(),
        },
        "production_runs": production_runs,
        "hook_records": hook_records,
    }
    reports = [None] * world
    dist.all_gather_object(reports, local_report)
    if rank == 0:
        print(json.dumps({"exact_ab_report": reports}, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
