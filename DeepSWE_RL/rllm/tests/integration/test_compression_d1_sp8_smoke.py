"""World8/D1/SP8 FA2 old-logprob-only smoke with the original real segments."""
import json, time
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from rllm.trainer.verl.agent_ppo_trainer import _transform_policy_segments
from tests.integration.test_compression_d2_logprob_smoke import episode, nonfinite_counts
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker

MODEL="/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B"

def main():
 dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size(); assert world==8
 torch.cuda.set_device(rank)
 cfg=OmegaConf.create({
  "model":{"lora_rank":0},
  "actor":{"fsdp_config":{"fsdp_size":-1},"ulysses_sequence_parallel_size":8,"ppo_mini_batch_size":8,"ppo_micro_batch_size":None,"ppo_micro_batch_size_per_gpu":1},
  "rollout":{"n":1,"log_prob_micro_batch_size":None,"log_prob_micro_batch_size_per_gpu":1,"log_prob_max_token_len_per_gpu":256,"log_prob_use_dynamic_bsz":False,"temperature":1.0},
  "ref":{"log_prob_micro_batch_size":None},
 })
 worker=ActorRolloutRefWorker(cfg,role="actor")
 coord=worker.ulysses_device_mesh.get_coordinate(); assert tuple(coord)==(0,rank)
 tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
 specs=[
  (0,[[(15,7),(15,7),(15,7)],[(25,8),(25,8)],[(45,9)],[(10,5)],[(20,6)]]),
  (1,[[(12,5)],[(30,7),(18,5),(16,5)],[(38,8),(22,6)],[(9,4)],[(24,7)]])]
 episodes=[]; opt={}; source=0
 for step,groups in specs:
  for ss in groups:
   e=episode(source,step,ss); episodes.append(e); opt[source]=step; source+=1
 batch=_transform_policy_segments(episodes,max_prompt_length=128,max_response_length=64,pad_token_id=tok.pad_token_id or tok.eos_token_id,optimizer_step_by_source=opt,effective_dp_degree=1)
 batch.meta_info.update({"temperature":1.0,"multi_turn":False,"global_token_num":batch.batch["attention_mask"].sum(-1).tolist()})
 model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",trust_remote_code=True)
 model=FSDP(model,device_id=torch.device(f"cuda:{rank}"),use_orig_params=False)
 actor_cfg=OmegaConf.create({"use_remove_padding":True,"use_fused_kernels":False,"ulysses_sequence_parallel_size":8,"use_torch_compile":False})
 actor=DataParallelPPOActor(actor_cfg,model); worker.actor_module_fsdp=model; worker.actor=actor
 torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
 out=worker.compute_compression_log_prob_d1(batch)
 torch.cuda.synchronize(); elapsed=time.perf_counter()-start
 old=out.batch["old_log_probs"]; active=(batch.batch["actor_token_weight"]>0).cpu()
 report={"rank":rank,"dp":0,"sp":rank,"rows":len(batch),"shape":list(old.shape),"active":nonfinite_counts(old,active),"elapsed":elapsed,"allocated":torch.cuda.max_memory_allocated(),"reserved":torch.cuda.max_memory_reserved()}
 reports=[None]*world; dist.all_gather_object(reports,report)
 if rank==0: print(json.dumps({"d1_sp8":reports},indent=2),flush=True)
 dist.barrier(); dist.destroy_process_group()
if __name__=="__main__": main()
