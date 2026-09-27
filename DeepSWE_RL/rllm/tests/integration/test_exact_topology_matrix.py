"""Exact-token topology/content forward-only matrix."""
import json, os
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

MODEL_PATH="/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B"
FIXTURE="/tmp/stage8c_d2_bad_microbatch.pt"

def stat(t):
    return {"nan":int(torch.isnan(t).sum()),"posinf":int(torch.isposinf(t).sum()),"neginf":int(torch.isneginf(t).sum())}

def main():
    dist.init_process_group("nccl")
    rank,world=dist.get_rank(),dist.get_world_size(); torch.cuda.set_device(rank)
    sp=int(os.environ["MATRIX_SP"]); d=world//sp; mode=os.environ["MATRIX_MODE"]
    dp,sp_coord=rank//sp,rank%sp
    x=torch.load(FIXTURE,map_location="cpu")
    exact={k:x[k].cuda(rank) for k in ("input_ids","attention_mask","position_ids","responses")}
    eos=151645
    dummy={
      "input_ids":torch.tensor([[eos,eos]],device=rank),
      "attention_mask":torch.ones((1,2),dtype=torch.long,device=rank),
      "position_ids":torch.tensor([[0,1]],device=rank),
      "responses":torch.tensor([[eos]],device=rank),
    }
    model=AutoModelForCausalLM.from_pretrained(MODEL_PATH,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",trust_remote_code=True)
    model=FSDP(model,device_id=torch.device(f"cuda:{rank}"),use_orig_params=False)
    cfg=OmegaConf.create({"use_remove_padding":True,"use_fused_kernels":False,"ulysses_sequence_parallel_size":sp,"use_torch_compile":False})
    actor=DataParallelPPOActor(cfg,model)
    hook_state={"active":False}
    handles=[]
    save_tag=os.environ.get("QKV_SAVE_TAG")
    if save_tag:
      for name in ("q_proj","k_proj","v_proj"):
        module=getattr(model.module.model.layers[0].self_attn,name)
        def make_hook(n):
          def hook(module,inputs,output):
            if hook_state["active"] and rank==1:
              torch.save(output.detach().cpu(),f"/tmp/{save_tag}_{n}.pt")
          return hook
        handles.append(module.register_forward_hook(make_hook(name)))
    mesh=init_device_mesh("cuda",(d,sp),mesh_dim_names=("dp","sp")) if sp>1 else None
    manager=FSDPUlyssesShardingManager(mesh)
    runs=[]
    with manager:
      for call in range(8):
        use_exact=(mode=="all_exact") or (mode=="dp0_bad_others_dummy" and dp==0 and call==3)
        hook_state["active"]=use_exact and call==3
        mb=exact if use_exact else dummy
        with torch.no_grad(): _,lp=actor._forward_micro_batch(mb,temperature=1.0,calculate_entropy=False)
        runs.append({"call":call,"exact":use_exact,"shape":list(lp.shape),**stat(lp),"col0":str(float(lp[0,0]))})
    report={"rank":rank,"world":world,"d":d,"sp":sp,"dp":dp,"sp_coord":sp_coord,"mode":mode,"runs":runs}
    reports=[None]*world; dist.all_gather_object(reports,report)
    if rank==0: print(json.dumps({"matrix":reports},indent=2),flush=True)
    dist.barrier(); dist.destroy_process_group()
if __name__=="__main__": main()
