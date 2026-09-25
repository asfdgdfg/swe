# SWE-Master origin_ck 进度交接（压缩版）

更新时间：2026-09-18（Asia/Shanghai）

## 目标

在原仓库 `origin_ck` 分支按 README 复现 SWE-Master 评测/训练结果。当前重点是保留原始 pipeline，区分正式复现、兼容性测试和历史结果；不把失败的环境运行计入模型性能。

## 连接与目录

- SSH：`ssh 11.11.18.2`，用户 `yyk11`，主机 `g48`。
- ROOT：`/home/yyk/yyk11/zhongtianyang/memory/SWE-Master`
- 分支：`origin_ck`；远端旧 Git 用 `git symbolic-ref --short HEAD`。
- 系统：CentOS 7，glibc 2.17，8×A800 80GB。
- 远端没有 `rg`/`ss`；用 `find`/`grep`/`curl`。
- 进度以本文件为准；重要状态变化先更新远端文件，再同步本地。

## 强制规则

1. 每次执行 README 中的新步骤前先向用户说明；遇到问题报告，不擅自扩大授权。
2. 不 `git reset --hard`、`git checkout --`，不删除无关 dirty changes、容器、模型、日志或实验产物。
3. 区分推理评测、RL 训练、兼容性测试；不同批次结果不可混算。
4. 不以 exit code、任务提交数、文件存在、API 可用单独判定成功；必须核对有效结果、reward、错误原因和唯一 instance IDs。
5. 不把环境错误、工具错误、上下文超限、协议解析错误直接归因模型能力。
6. 文中写“下一步/尚未执行”的命令，执行前仍需说明；不重复启动已有任务。
7. 不记录真实密钥；`OPENAI_API_KEY=xx` 只是本地占位值。

## 环境

- 数据下载：Conda base `/home/yyk/yyk11/miniconda3/bin/python`，曾使用 `HF_ENDPOINT=hf-mirror.com` 和系统 CA；五个数据集下载已完成。
- 推理环境：`ROOT/R2E-Gym/.venv/bin/python`，Python 3.11；评测依赖已按 glibc 2.17 做兼容固定，`uv pip check` 通过。不要直接 `uv sync`，旧 `uv.lock` 不是准确冻结环境；优先 `logs/readme_origin_ck_repro/eval_environment_freeze.txt`。
- RL 环境：`ROOT/DeepSWE_RL/rllm/.venv`，运行训练时须将其 `bin` 放在 PATH 首位，否则会误用 Conda base 并报 `No module named rllm`。
- vLLM：`/home/yyk/yyk11/miniconda3/envs/swe-master-rl/bin/vllm`，Python 3.11、vLLM 0.8.5.post1、Torch 2.6。
- TensorBoard 已在 RL 环境安装；数据预处理 `make_test_spec_for_rl.py` 已增加字符串/dict 类型保护并保留备份。

## 当前最新状态（优先于历史章节）

### 1. RL 训练（GPU0–3）

- 原始训练曾因 `max_token_len=14000 < max_seq_len=110592` 在 rollout 后、actor update 前失败，无 checkpoint。
- 调度器兼容测试已进入多轮 rollout，但 22 次 actor update 均触发同一断言；该测试已结束。
- 当前重启脚本：`swe_rl_grpo_maxtoken28672_20260918.sh`；日志：`logs/swe_rl_grpo_maxtoken28672_20260918.log`；实验名 `swe-agent-rl_docker_all_100K-timeout5400_2026-09-18_20-48-17`。
- 配置只把 `ppo_max_token_len_per_gpu` 从 14000 提到 28672（×Ulysses 4 = 114688），其余保持兼容测试配置；使用 ChatCompletionScheduler 替代缺失的定制路径。
- 21:35 状态：仍在 rollout，尚未到 actor update；无最终 reward、step 或 checkpoint。不能声称训练已跑通。

### 2. 7B LSP+memory 评测（GPU4，端口8001）

- 模型：Qwen2.5-Coder-7B-Instruct，vLLM served name `qwen25-coder-7b-instruct`，YaRN 4×、max-model-len 131072（服务侧扩展，未改原 config）。
- 20 条数据复用 `readme_available20_20260917.json`，4 并发、150 步、temperature 0.7、OpenHands 非函数调用、Docker、清华源。
- 开启 `--use_lsp True` 与 `--enable_compression True`；使用 `openhands_sp_non_fn_calling_memory.yaml`，memory 默认 `summary_window=10/keep_recent=5/compression_trigger_step=20`。LSP yaml 缺 summary prompt，未直接使用；LSP 仍由参数注入。
- launcher：55965；脚本：`swe_available20_qwen7b_lsp_memory_run.sh`；日志：`eval_available20_qwen7b_lsp_memory_20260918.log`；结果：`R2E-Gym/results/readme-available20-qwen7b-lsp-memory-20260918`。
- 启动初期容器正在安装 LSP 的 node 依赖，任务仍存活；无最终结果。对照：同组 7B 无 LSP/memory 为 0/20，32B-SFT 为 8/20。

### 3. 镜像

- Verified 精确 tag 已全部准备：原有 1 + 中转下载 499 = 500/500；下载日志末尾 `DONE 499`。
- 数据中另有 3 个训练样本使用内部 Harbor 镜像，当前网络不可达；不能用 Verified 清单替代它们，也不能擅自换镜像。

## 已确认的历史结果（不可与新批次混算）

- 32B-SFT：可用镜像同组 20 条，`8/20`。
- 7B 无 LSP/memory：同组 20 条，`0/20`；16 条达到绝对步数上限，4 条 agent 退出，13 条 patch 为空；存在函数调用协议失败，不能简单归因模型能力。
- 工具 Python 修复后的 32B-SFT：可用镜像 20 条，`8/20`（Astropy 4/10、Django 4/10）。结果：`R2E-Gym/results/readme-available20-toolpython-20260917`。
- 32B-RL：102 条，`55/102`；exit 原因 agent54、llm_query_error29、abs_step_limit19；部分 query error 已确认是超过 131072 上下文，不应把 29 条全部视为修复失败。
- 4B-RL：只有 65 条，`13/65`，不是完整 102 条；exitcode 缺失，停止原因未完全确认。
- 两条 smoke（SFT）：2/2 流程完成但 reward `0/2`，用于验证流程，不用于性能估计。

## 关键已完成工作

- 五个数据集下载完成；500 条 test-spec 生成并完成 runtime JSON 字符串兼容转换。
- 32B-SFT 权重补齐并核验 14 分片、771 tensors；4B/7B 权重也做过分片/index 核验。
- R2E-Gym 评测环境、依赖兼容、清华 pip/uv 源配置完成。
- Docker runtime 已按授权修复工具 Python 混用：容器内项目 Python 保持不变，编辑/执行工具使用 conda Python 3.11 独立 venv `/root/.swe-tools`。
- 已知 tracked dirty changes 要保留：`download_swe_datasets.sh` 清华源改动；`runtime/docker.py` Unix socket/DockerClient 改动及清华源/工具相关改动。所有备份在 `ROOT/logs/readme_origin_ck_repro/`。

## 关键已知问题

- 超长轨迹（`max_prompt_length=8192 + max_response_length=102400 = 110592`）与 seqlen balancing/micro-batch 上限曾不匹配。
- README 定制调度器路径 `verl.schedulers.completions_scheduler.CompletionsScheduler` 在当前环境缺失；兼容测试使用公开 `verl.workers.rollout.async_server.ChatCompletionScheduler`，不等价于严格原版。
- `make_test_spec` 曾发生 JSON 字符串二次编码；当前类型保护已修正，数据需验证一次 `json.loads` 后为 dict。
- 评测日志可能混入模型生成脚本的 SyntaxError、普通测试 ERROR、工具命令退出 1；分析时必须区分框架错误和 agent 行为。
- vLLM API 可用只证明服务启动；必须做实际请求和完整评测核验。

## 后续建议（均需先向用户说明）

1. 先只读检查当前 RL 进程、GPU、日志和 7B 评测结果，不重复启动。
2. 重点确认 `swe_rl_grpo_maxtoken28672_20260918.log` 是否到达首次 actor update；若仍失败，记录精确断言后再决定 dynamic batch、序列长度或其他参数。
3. 检查 7B LSP+memory 评测的 completed 数、exitcode、有效 JSONL 和 reward；单独记录 LSP/node 依赖错误。
4. 如需继续训练/评测，使用新日志/结果目录；保留当前批次和原始历史，不覆盖不混算。

## 重要路径

- 训练日志：`ROOT/logs/`，尤其 `readme_origin_ck_repro/`。
- 评测结果：`ROOT/R2E-Gym/results/`。
- 模型：`ROOT/models/`。
- 原始完整交接备份：由覆盖前自动生成，文件名含 `.before_compact_`。
## 2026-09-22 最新进度（离线数据恢复与 7B RL）

- 未重新下载 Hugging Face 数据；使用远端已有 cache 离线恢复 RL 数据。
- 从本地 Arrow cache 合并生成全量训练 parquet，共 4,578 条样本，字段为 prompt / reward_model / extra_info，文件大小约 900 MB。
- 训练数据：DeepSWE_RL/rllm/rllm/data/datasets/SWE_FULL_4578/train_00000_verl.parquet。
- 当前 7B RL 训练使用 origin_ck 分支、Qwen2.5-Coder-7B-Instruct、data.max_response_length=32768，8 卡启动。
- 训练会话：tmux swe7b_full。
- 训练日志：/home/yyk/yyk11/zhongtianyang/memory/origin_ck_7b_full4578_len32768.log。
- 训练刚启动时 tmux 会话仍存活，尚未据此宣称训练已产生有效 reward 或 checkpoint；后续需检查首次 rollout、actor update、显存和最终结果。
- 之前的 Hugging Face 下载任务因 aria2c CA 证书错误失败，不影响本次离线恢复；不要把该下载失败误判为数据不存在。

## 2026-09-23 SWE-smith 训练修复记录

- 已按仓库 README 的环境方式使用 `DeepSWE_RL/rllm/.venv/bin/python` 启动 SWE-smith 7B 训练；训练数据为 `SWE_SMITH_172_IMAGES/train_verl.parquet`，验证集为 `val_verl.parquet`。
- 发现并修复 SWE-smith reward 边界类型错误：`R2E-Gym/src/r2egym/agenthub/runtime/docker.py` 在无解析结果时原先无条件返回 `(0.0, output)`，而 `rllm/engine/agent_execution_engine.py` 在普通 reward 路径执行 `reward > 0`，触发 `TypeError: tuple and int`。
- 修复后仅在 `get_test_output=True` 时返回 `(reward, output)`，普通训练路径返回标量 `float`；原文件备份为 `docker.py.before_reward_fix_20260923`。
- 修复前的 `swe7b_swesmith_venv` 进程已停止，待修复后的最小导入/运行核验通过后再启动新实验；旧日志 `swesmith_7b_172_venv.log` 保留，不将修复前结果计入性能。
- 尚未解决的问题：部分 SWE-smith 容器测试文件 reset 报 `Exit code 123`，以及并发清理时偶发 Docker 404；这些与 reward tuple 修复分开核查，不能宣称已解决。

## 2026-09-23 7B case 记录训练

- 为查看完整训练 case，启用了训练器已有的 chat-completions 持久化逻辑；每个训练 step 写入 chat_completions/<step>.jsonl。
- 每条记录包含完整输入消息、assistant 输出、工具调用/环境反馈、traj_score、termination_reason 和轨迹指标。
- 新实验使用 origin_ck 分支、Qwen2.5-Coder-7B-Instruct、全量 SWE_FULL_4578 数据（4,578 条），max_response_length=32768。
- 为避免 Ray 临时目录清理，设置了持久化 default_local_dir：
  /home/yyk/yyk11/zhongtianyang/memory/SWE-Master/DeepSWE_RL/rllm/checkpoints/swe_master/swesmith_7b_cases_20260923
- 训练日志：
  /home/yyk/yyk11/zhongtianyang/memory/swesmith_7b_cases_20260923.log
- tmux 会话：swe7b_cases。
- 启动时确认使用 RL 环境 Python 3.11.13：
  /home/yyk/yyk11/zhongtianyang/memory/SWE-Master/DeepSWE_RL/rllm/.venv/bin/python
- 启动初期尚未生成第一个 chat_completions 文件；需等首个训练 step 完成后核验 case 文件、reward 和训练状态。
## 2026-09-25 Qwen3-4B SWE-smith 长度扩展训练

- 已停止旧的 `qwen3_4b_len32768_20260925` 进程，并启动新进程 PID `13546`。
- 新实验名：`qwen3_4b_len51200_20260925`；日志：`/home/yyk/yyk11/zhongtianyang/memory/qwen3_4b_len51200_20260925.log`。
- 新配置：`data.max_response_length=51200`；actor/ref/rollout token 上限、`max_model_len` 和 `max_num_batched_tokens` 均为 `57344`。
- 数据集仍为 `SWE_SMITH_FULL_NONEMPTY_VERIFIED/train_verl.parquet`，模型仍为本地 `Qwen3-4B`，`train_batch_size=8`、`rollout.n=4`，每个全局训练 step 预计收集 32 条 trajectory。
- 截至本次记录，训练已进入 rollout，单个样本最高推进到约 agent step 195；已完成 trajectory 约 5 条，尚未完成第一批 32 条，因此尚未发生第一次 actor 参数更新。
- GPU 显存约 47GB/卡，利用率约 29%–76%；进程和 Ray/vLLM 正常，暂未发现 Traceback、CUDA OOM 或启动错误。
- `SWE-SMITH TEST RATIO` 仅用于记录单样本测试通过比例；训练实际 reward 仍由 resolution status 判定为 0.0/1.0。

## 2026-09-23 Qwen3-4B RL 训练状态

- 仓库/分支：`/home/yyk/yyk11/zhongtianyang/memory/SWE-Master`，`origin_ck`。
- 模型：官方本地权重 `/home/yyk/yyk11/zhongtianyang/memory/models/Qwen3-4B`，约 4.02B 参数。
- 启动脚本：`/tmp/run_qwen3_4b.sh`；tmux session：`qwen3_rl`；主进程 PID：`72703`。
- 实验名：`qwen3_4b_verified_20260923`；输出目录：`DeepSWE_RL/rllm/checkpoints/swe_master/qwen3_4b_verified_20260923`。
- 数据：`SWE_SMITH_FULL_NONEMPTY_VERIFIED/train_verl.parquet`，41103 条；`train_batch_size=8`、`rollout.n=4`，因此每个训练 step 为 32 条 rollout；总训练 step 为 5137。
- 长度：`max_prompt_length=8192`、`max_response_length=24576`、`max_model_len=32768`、actor/ref/rollout token 上限为 32768。
- 工具协议已回退到 init commit `fb864a7` 的原始 `<function=...><parameter=...>` 解析和原始 SWE-Agent prompt；恢复支持 `execute_bash`、`str_replace_editor`、`submit` 等原始工具。保留 `problem_statement` 占位符兼容修复。
- 此前失败的 Qwen3 进程使用后改的 `<tool_call>` JSON parser，并错误地只允许 `execute_bash`，导致 `str_replace_editor` 被拒绝、JSONDecodeError 及空轨迹断言；该旧进程已停止。
- 当前新训练已完成 Qwen3 checkpoint、FSDP、Ray/vLLM 初始化并进入 `epoch 0, step 1`；8 张 GPU 均约占用 48GB。
- 截至同步时，第 1 个 batch 已完成 31/32 条 rollout；尚未完成的是 `Trajectory 14`，因此还未进入 step 2。
- 已完成 31 条的 reward 分布：reward 0.0 为 29 条，reward 1.0 为 2 条（Trajectory 10、15）。大量轨迹因 `TRUNCATION` 在 24576 response tokens 上限结束。
- 新进程截至同步时未再出现 `Unsupported tool: str_replace_editor`、JSONDecodeError 或 `Trajectory cannot complete`。
