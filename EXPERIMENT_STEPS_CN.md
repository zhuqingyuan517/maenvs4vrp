# DTOPTW 实验操作说明（uv + 6×A100 40 GB）

本文档对应当前实际实验方案：

- 机器配置：6 张 40 GB A100，其中 GPU 0 与 GPU 3、GPU 4 与 GPU 5 为 NVLink 对，
  GPU 1、2 为无 NVLink 的单卡；
- 6.3 Full-graph Critic：500-50 使用单卡，1000-100 使用一对 NVLink；
- 6.4 Global-action 消融：两张 A100 数据并行；
- 其余训练方法：每个任务一张 A100；
- 所有正式评估：`batch_size=1`、100 个固定种子。

所有命令默认在项目的根目录执行。

## 1. 六卡分工

当前机器实际的 NVLink 配对为 `0-3` 和 `4-5`，GPU 1、2 是无 NVLink
连接的单卡。后文命令按此拓扑编写。不要使用物理编号的相邻关系推断
NVLink；双卡任务必须把同一对中的两张卡写入 `CUDA_VISIBLE_DEVICES`。

500-50 的建议分配如下：

| 训练任务 | GPU | 进程数 | 全局 batch | 每卡 batch |
| --- | ---: | ---: | ---: | ---: |
| Proposed 500-50 | 0 | 1 | 16 | 16 |
| Full-graph 500-50（6.3） | 1 | 1 | 16 | 16 |
| MARDAM 500-50 | 2 | 1 | 16 | 16 |
| Global-action 500-50（6.4） | 4、5 | 2 | 16 | 8 |
| 500-50 评估 | 0-5（训练完成后） | 1 | 1 | 1 |

1000-100 的建议分配如下：

| 训练任务 | GPU | 进程数 | 全局 batch | 每卡 batch |
| --- | ---: | ---: | ---: | ---: |
| Full-graph 1000-100 | 0、3 | 2 | 16 | 8 |
| Global-action 1000-100 | 4、5 | 2 | 16 | 8 |
| Proposed 1000-100 | 1 | 1 | 16 | 16 |
| MARDAM 1000-100 | 2、3 | 2 | 16 | 8 |

这样安排的原因是 Full-graph 和 Global-action 的梯度同步分别完全走一对
NVLink；Proposed 使用单卡 GPU 1。MARDAM 使用 GPU 2、3，其中 GPU 3 与
Full-graph 的 rank 1 共享显存。GPU 2、3 之间没有 NVLink，MARDAM 的梯度
all-reduce 走 PCIe；其模型梯度很小，主要瓶颈仍是环境与 actor/critic 计算。
不要把双卡任务拆到两个不同的 NVLink 对，也不要在正式训练中直接把
Full-graph 或 Global-action 扩成四卡；跨对通信会经过 PCIe/NUMA，而且会
挤占其他实验。

6.3 和 6.4 使用的是同一个训练脚本：

```text
maenvs4vrp/learning/madyam/train_madyam_abs.py
```

两者只是参数不同：

- 6.3：`--vrp_env dtoptw_abs --model_variant abs`，普通 Python 单进程启动；
- 6.4：`--vrp_env dtoptw_abs_full --model_variant third`，使用 `torchrun`
  启动两个进程。

当前脚本已经支持双卡梯度平均，不需要再写第二份训练代码。

实验按两个完整阶段执行：

1. 完成 500-50 训练和正式评估，确认 checkpoint 和 CSV 检查通过；
2. 等所有 500-50 训练进程退出、6 张卡全部空闲后，再启动 1000-100；
3. 完成 1000-100 训练后，再统一执行正式评估。

500-50 和 1000-100 均从随机初始化开始训练，1000-100 不加载 500-50 的
checkpoint。两个阶段使用不同的 `log_path`，不会覆盖结果。

六卡满载不等于一个任务使用六卡。当前协议用四个独立训练任务并行占满六卡；
不要为了追求 GPU 利用率而随意增加全局 batch 或修改更新次数。

## 2. 同步最新代码

如果服务器上已有项目，只同步最新训练脚本也可以：

```bash
rsync -avP \
  maenvs4vrp/learning/madyam/train_madyam_abs.py \
  <user>@<server>:/path/to/maenvs4vrp/maenvs4vrp/learning/madyam/

rsync -avP \
  EXPERIMENT_STEPS_CN.md \
  <user>@<server>:/path/to/maenvs4vrp/
```

更稳妥的方式是同步整个工作目录，因为实验代码目前还包括其他未提交文件。
不要只同步 `maenvs4vrp/` 而遗漏 `setup.py`、`tests/` 和实验文档。

到服务器后确认：

```bash
cd /path/to/maenvs4vrp
grep -n "setup_distributed\|all_reduce_mean_gradients" \
  maenvs4vrp/learning/madyam/train_madyam_abs.py
```

应至少看到这两个函数。若没有，说明服务器上的训练脚本仍是旧版本。

## 3. 激活 uv 环境

```bash
cd /path/to/maenvs4vrp
source .venv/bin/activate

which python
which uv
which torchrun

python --version
uv --version
torchrun --version
```

重新注册当前目录中的可编辑项目并检查依赖：

```bash
uv pip install -e .
uv pip check
```

如果虚拟环境不在 `.venv/`，替换成实际激活路径。

## 4. 检查 6 张 A100 与 NVLink 拓扑

```bash
nvidia-smi -L

nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

nvidia-smi topo -m
nvidia-smi nvlink --status

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(i, props.name, f"{props.total_memory / 1024**3:.1f} GiB")
PY
```

必须得到：

```text
CUDA available: True
GPU count: 6
```

记录每张卡是 40 GB 还是 80 GB。1000-100 的 Full-graph 单卡约需
42.6 GB，不能放在 40 GB A100 上。

本文档对应 6 张 40 GB A100。在 `nvidia-smi topo -m` 的矩阵中，
标为 `NV#` 表示两张卡通过 NVLink 相连。本机实际拓扑应核对为：

```text
Pair A: 0-3
Pair B: 4-5
Singleton: 1
Singleton: 2
```

`CUDA_VISIBLE_DEVICES=0,3` 是完全合法的非连续编号：第一个 rank 映射到
物理 GPU 0，第二个 rank 映射到物理 GPU 3，两张卡之间的 all-reduce 走
NVLink。不要改成 `0,1`，因为 GPU 0 和 GPU 1 没有 NVLink 连接。

检查 CPU 与 GPU 的 NUMA 距离：

```bash
lscpu
numactl -H
```

环境生成主要运行在 CPU 上。六个训练 rank 同时运行时，应根据物理核数为
每个进程限制线程数，避免所有进程都抢占全部 CPU。先从下面的设置开始测试，
再根据实际 CPU/GPU 利用率调整：

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## 5. 建立目录和保存环境信息

```bash
mkdir -p runs/{500_50,1000_100}
mkdir -p results/{500_50,1000_100}
mkdir -p logs/{500_50,1000_100}

{
  date
  uname -a
  nvidia-smi
  nvidia-smi --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader
  nvidia-smi topo -m
  nvidia-smi nvlink --status
  python --version
  uv --version
  uv pip freeze
} > logs/environment.txt
```

把实际执行的完整命令写入：

```text
logs/commands.txt
```

不要把正式的 `runs/`、`results/` 和 `logs/` 放在高延迟网络文件系统上。
条件允许时先写入本地 NVMe，实验结束后再同步到长期存储；多个进程同时写
checkpoint 时，网络文件系统会明显拉长 epoch 尾部等待。

## 6. 先做冒烟测试

### 6.1 Full-graph 单卡检查

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export MPLCONFIGDIR="/tmp/matplotlib-cache-1"
mkdir -p "$MPLCONFIGDIR"

python -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs \
  --model_variant abs \
  --num_nodes 11 \
  --num_agents 2 \
  --batch_size 1 \
  --local_k_nodes 5 \
  --max_steps 8 \
  --done_feasible_check_freq 1 \
  --embed_dim 32 \
  --n_heads 4 \
  --n_layers 1 \
  --ff_hidden 64 \
  --epoch_count 1 \
  --iter_count 1 \
  --eval_every 0 \
  --device cuda:0 \
  --log_path /tmp/full_graph_smoke \
  --run_name full_graph_smoke \
  --save_models false
```

成功时应输出 `[ep 000] ...`。

### 6.2 Global-action 双卡检查

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=4,5
export MPLCONFIGDIR="/tmp/matplotlib-cache-4-5"
mkdir -p "$MPLCONFIGDIR"

torchrun --standalone --nproc_per_node=2 \
  -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs_full \
  --model_variant third \
  --num_nodes 11 \
  --num_agents 2 \
  --batch_size 2 \
  --critic_inducing_points 4 \
  --max_steps 8 \
  --done_feasible_check_freq 1 \
  --embed_dim 32 \
  --n_heads 4 \
  --n_layers 1 \
  --ff_hidden 64 \
  --epoch_count 1 \
  --iter_count 1 \
  --eval_every 0 \
  --device cuda:0 \
  --log_path /tmp/global_action_dp2_smoke \
  --run_name global_action_dp2_smoke \
  --save_models false \
  --grad_checkpoint true
```

这里 `--batch_size 2` 是全局 batch，每卡实际处理 1 个实例。
启动时可以临时设置 `NCCL_DEBUG=INFO`，确认日志中的通信拓扑使用了 NVLink
或 GPU P2P。不要设置 `NCCL_P2P_DISABLE=1`，否则会禁用 GPU 间直接通信。

### 6.3 1000-100 显存准入检查

正式启动完整训练前，必须在实际 1000-100 规模和正式模型结构上完成至少一次
前向、反向传播和 optimizer step。仅使用 `--profile_steps` 只能统计环境与
模型的前向耗时，不能替代显存准入检查。

对第 9 节中四个正式训练命令分别临时追加：

```text
--epoch_count 1
--iter_count 1
--eval_every 0
--save_models false
--log_path /tmp/dtoptw_1000_preflight
```

每个任务使用不同的 `--run_name`。同时保留正式的全局 batch、模型结构、
`--grad_checkpoint` 和 `CUDA_VISIBLE_DEVICES`。检查命令至少运行到输出
`[ep 000]`，并用 `nvidia-smi` 记录峰值显存：

```bash
nvidia-smi \
  --query-gpu=index,utilization.gpu,memory.used,memory.total,utilization.memory,power.draw \
  --format=csv -l 2
```

建议总显存不要长期超过 38 GB。Full-graph、Global-action 或 MARDAM 任一任务
在准入检查中 OOM，都不能直接开始完整的 100 epochs，应先按第 12 节调整。

## 7. 500-50 正式训练

建议用独立的 `tmux` 会话，避免 SSH 断开后训练终止。例如：

```bash
# 在 tmux 外执行，一次创建四个后台会话
tmux new-session -d -s proposed
tmux new-session -d -s full_graph
tmux new-session -d -s global_action_dp2
tmux new-session -d -s mardam

# 查看会话
tmux ls
```

进入某一个会话后，在会话内粘贴对应的训练命令：

```bash
tmux attach -t proposed
tmux attach -t full_graph
tmux attach -t global_action_dp2
tmux attach -t mardam
```

按 `Ctrl+b` 后再按 `d` 退出当前会话，但不会终止训练。每次只能进入一个
`tmux` 会话，执行完一个会话的 `attach` 后应先 detach，再进入下一个。

### 7.1 Full-graph：单卡 GPU 1

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export MPLCONFIGDIR="/tmp/matplotlib-cache-1"
mkdir -p "$MPLCONFIGDIR"
set -o pipefail

nvidia-smi \
  --id=1 \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader \
  -l 1 > logs/500_50/full_graph_gpu.csv &
export GPU_MONITOR_PID=$!

/usr/bin/time -v \
  -o logs/500_50/full_graph_train.time.txt \
  python -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs \
  --model_variant abs \
  --num_nodes 500 \
  --num_agents 50 \
  --local_k_nodes 30 \
  --epoch_count 100 \
  --iter_count 100 \
  --batch_size 16 \
  --eval_every 10 \
  --eval_episodes 10 \
  --eval_batch_size 4 \
  --device cuda:0 \
  --log_path runs/500_50 \
  --run_name full_graph_500_50 \
  2>&1 | tee logs/500_50/full_graph_train.log

kill "$GPU_MONITOR_PID" 2>/dev/null || true
wait "$GPU_MONITOR_PID" 2>/dev/null || true
```

训练结束后检查：

```bash
ls -lh runs/500_50/models/best_model_full_graph_500_50.zip
ls -lh runs/500_50/models/latest_model_full_graph_500_50.zip
```

### 7.2 Global-action：NVLink GPU 4、5

`CUDA_VISIBLE_DEVICES=4,5` 必须在 `torchrun` 前设置。两个进程分别使用
`cuda:0` 和 `cuda:1`，对应物理 GPU 4 和 5，因此 `--device cuda:0`
仍保留在命令里。

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=4,5
export MPLCONFIGDIR="/tmp/matplotlib-cache-4-5"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$MPLCONFIGDIR"
set -o pipefail

nvidia-smi --id=4,5 \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv

nvidia-smi \
  --id=4,5 \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader \
  -l 1 > logs/500_50/global_action_gpu_dp2.csv &
export GPU_MONITOR_PID=$!

/usr/bin/time -v \
  -o logs/500_50/global_action_train.time.txt \
  torchrun --standalone --nproc_per_node=2 \
  -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs_full \
  --model_variant third \
  --num_nodes 500 \
  --num_agents 50 \
  --critic_inducing_points 32 \
  --epoch_count 100 \
  --iter_count 100 \
  --batch_size 16 \
  --eval_every 10 \
  --eval_episodes 10 \
  --eval_batch_size 4 \
  --grad_checkpoint true \
  --device cuda:0 \
  --log_path runs/500_50 \
  --run_name global_action_500_50 \
  2>&1 | tee logs/500_50/global_action_train_dp2.log

kill "$GPU_MONITOR_PID" 2>/dev/null || true
wait "$GPU_MONITOR_PID" 2>/dev/null || true
```

说明：

- `--batch_size 16` 是全局 batch；每张卡实际处理 8 个实例；
- 每次 optimizer step 前会对两张卡的梯度求平均；
- `--grad_checkpoint true` 进一步降低单卡激活显存，但训练会更慢；
- 只有 rank 0 写 TensorBoard 和 checkpoint；
- 三张卡无法均分全局 batch 16。使用三卡必须显式改成全局 batch 15/18，
  不能在保持 batch 16 时声称三卡均分。

检查 checkpoint：

```bash
ls -lh runs/500_50/models/best_model_global_action_500_50.zip
ls -lh runs/500_50/models/latest_model_global_action_500_50.zip
```

### 7.3 Proposed：单卡 GPU 0

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR="/tmp/matplotlib-cache-0"
mkdir -p "$MPLCONFIGDIR"
set -o pipefail

nvidia-smi \
  --id=0 \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader \
  -l 1 > logs/500_50/proposed_gpu.csv &
export GPU_MONITOR_PID=$!

/usr/bin/time -v \
  -o logs/500_50/proposed_train.time.txt \
  python -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs \
  --model_variant third \
  --num_nodes 500 \
  --num_agents 50 \
  --local_k_nodes 30 \
  --critic_inducing_points 32 \
  --epoch_count 100 \
  --iter_count 100 \
  --batch_size 16 \
  --eval_every 10 \
  --eval_episodes 10 \
  --eval_batch_size 4 \
  --device cuda:0 \
  --log_path runs/500_50 \
  --run_name proposed_500_50 \
  2>&1 | tee logs/500_50/proposed_train.log

kill "$GPU_MONITOR_PID" 2>/dev/null || true
wait "$GPU_MONITOR_PID" 2>/dev/null || true
```

检查 checkpoint：

```bash
ls -lh runs/500_50/models/best_model_proposed_500_50.zip
ls -lh runs/500_50/models/latest_model_proposed_500_50.zip
```

### 7.4 MARDAM：单卡 GPU 2

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2
export MPLCONFIGDIR="/tmp/matplotlib-cache-2"
mkdir -p "$MPLCONFIGDIR"
set -o pipefail

nvidia-smi \
  --id=2 \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader \
  -l 1 > logs/500_50/mardam_gpu.csv &
export GPU_MONITOR_PID=$!

/usr/bin/time -v \
  -o logs/500_50/mardam_train.time.txt \
  python -m maenvs4vrp.learning.mardam.train_mardam_dtoptw \
  --vrp_env dtoptw_full \
  --num_nodes 500 \
  --num_agents 50 \
  --epoch_count 100 \
  --iter_count 200 \
  --batch_size 16 \
  --eval_every 1 \
  --eval_episodes 10 \
  --device cuda:0 \
  --log_path runs/500_50 \
  --run_name mardam_500_50 \
  2>&1 | tee logs/500_50/mardam_train.log

kill "$GPU_MONITOR_PID" 2>/dev/null || true
wait "$GPU_MONITOR_PID" 2>/dev/null || true
```

检查 checkpoint：

```bash
ls -lh runs/500_50/models/best_model_mardam_500_50.zip
ls -lh runs/500_50/models/latest_model_mardam_500_50.zip
```

500-50 的 MARDAM 继续使用单卡命令；1000-100 的双 rank 入口见第 9.4 节。
`--eval_every 1` 会每个 epoch 执行 10 个验证 batch，是训练吞吐的主要开销
之一。正式结果如果需要保持原有 checkpoint 选择方式，就保留该值；如果最终
评估独立进行，且接受更稀疏的 checkpoint 选择，可以先跑一次对照，再把
`--eval_every` 调整为 10。改变后必须记录为性能设置差异，不能与
`--eval_every 1` 的曲线混用。

六卡并行时，不要在没有显存实测的情况下把另一个 1000-100 重型训练任务叠加
到 Proposed 或 MARDAM 的进程中。Proposed 1000-100 约需 15.8 GB，理论上仍有
余量，但 Global-action 和 MARDAM 的 1000-100 峰值必须实测。MARDAM 1000-100
的 rank 1 放在 GPU 3 是基于后者已有剩余显存；启动后若 GPU 3 OOM，其他任务
可能受影响。适合共享的是低显存评估任务；所有任务必须使用独立日志、
checkpoint 和 CSV 路径。

## 8. 500-50 统一评估

正式评估统一使用：

```text
eval_episodes=100
batch_size=1
seed=12345
```

对应种子 `12345` 至 `12444`。

训练全部完成后，正式评估的 `batch_size=1`，显存占用通常低于训练。可以把
不同方法的评估放到不同 GPU 上并行执行；建议先按下面顺序使用：

```text
Proposed -> GPU 0
Full-graph -> GPU 1
Global-action -> GPU 4
MARDAM -> GPU 2
Greedy -> GPU 3
Random -> GPU 5
RHC-ALNS -> 任一张已空闲的 GPU
```

每个评估任务仍必须使用独立的 `--save_csv` 和日志文件，最终汇总相同的
100 个测试种子。

### 8.1 Proposed

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

python -m maenvs4vrp.learning.madyam.eval_policy_model \
  --vrp_env dtoptw_abs \
  --model_variant third \
  --checkpoint runs/500_50/models/best_model_proposed_500_50.zip \
  --num_nodes 500 \
  --num_agents 50 \
  --local_k_nodes 30 \
  --critic_inducing_points 32 \
  --embed_dim 128 \
  --n_heads 8 \
  --n_layers 3 \
  --ff_hidden 512 \
  --batch_size 1 \
  --eval_episodes 100 \
  --seed 12345 \
  --device cuda:0 \
  --save_csv results/500_50/proposed.csv \
  2>&1 | tee logs/500_50/proposed_eval.log
```

### 8.2 Full-graph

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1

python -m maenvs4vrp.learning.madyam.eval_policy_model \
  --vrp_env dtoptw_abs \
  --model_variant abs \
  --checkpoint runs/500_50/models/best_model_full_graph_500_50.zip \
  --num_nodes 500 \
  --num_agents 50 \
  --local_k_nodes 30 \
  --embed_dim 128 \
  --n_heads 8 \
  --n_layers 3 \
  --ff_hidden 512 \
  --batch_size 1 \
  --eval_episodes 100 \
  --seed 12345 \
  --device cuda:0 \
  --save_csv results/500_50/full_graph.csv \
  2>&1 | tee logs/500_50/full_graph_eval.log
```

### 8.3 Global-action

训练使用双卡，评估仍使用单卡和 `batch_size=1`：

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=4

python -m maenvs4vrp.learning.madyam.eval_policy_model \
  --vrp_env dtoptw_abs_full \
  --model_variant third \
  --checkpoint runs/500_50/models/best_model_global_action_500_50.zip \
  --num_nodes 500 \
  --num_agents 50 \
  --critic_inducing_points 32 \
  --embed_dim 128 \
  --n_heads 8 \
  --n_layers 3 \
  --ff_hidden 512 \
  --batch_size 1 \
  --eval_episodes 100 \
  --seed 12345 \
  --device cuda:0 \
  --save_csv results/500_50/global_action.csv \
  2>&1 | tee logs/500_50/global_action_eval.log
```

### 8.4 MARDAM

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2

python -m maenvs4vrp.learning.mardam.eval_mardam_dtoptw \
  --checkpoint runs/500_50/models/best_model_mardam_500_50.zip \
  --vrp_env dtoptw_full \
  --num_nodes 500 \
  --num_agents 50 \
  --batch_size 1 \
  --eval_episodes 100 \
  --seed 12345 \
  --device cuda:0 \
  --save_csv results/500_50/mardam.csv \
  2>&1 | tee logs/500_50/mardam_eval.log
```

### 8.5 Greedy

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=3

python -m maenvs4vrp.learning.madyam.eval_policy_baselines \
  --vrp_env dtoptw_abs \
  --policy_type greedy \
  --num_nodes 500 \
  --num_agents 50 \
  --local_k_nodes 30 \
  --batch_size 1 \
  --eval_episodes 100 \
  --num_workers 1 \
  --seed 12345 \
  --device cuda:0 \
  --save_csv results/500_50/greedy.csv \
  2>&1 | tee logs/500_50/greedy_eval.log
```

### 8.6 Random

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=5

python -m maenvs4vrp.learning.madyam.eval_policy_baselines \
  --vrp_env dtoptw_abs \
  --policy_type random \
  --num_nodes 500 \
  --num_agents 50 \
  --local_k_nodes 30 \
  --batch_size 1 \
  --eval_episodes 100 \
  --num_workers 1 \
  --seed 12345 \
  --device cuda:0 \
  --save_csv results/500_50/random.csv \
  2>&1 | tee logs/500_50/random_eval.log
```

### 8.7 RHC-ALNS

该评估应在 Proposed 评估结束、GPU 0 空闲后启动：

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

python -m maenvs4vrp.learning.madyam.alns_rhc_dtoptw_abs \
  --vrp_env dtoptw_full \
  --num_nodes 500 \
  --num_agents 50 \
  --batch_size 1 \
  --eval_episodes 100 \
  --seed 12345 \
  --alns_route_length 12 \
  --alns_iterations 40 \
  --alns_destroy_max 3 \
  --alns_candidate_pool 120 \
  --device cuda:0 \
  --save_csv results/500_50/alns.csv \
  2>&1 | tee logs/500_50/alns_eval.log
```

### 8.8 500-50 结果文件

全部评估完成后应得到：

```text
results/500_50/
  proposed.csv
  full_graph.csv
  global_action.csv
  mardam.csv
  greedy.csv
  random.csv
  alns.csv
```

每个 CSV 应有 100 条数据行，种子范围均为 `12345` 至 `12444`。正式进入
1000-100 训练前，应先执行第 10.1 和 10.2 节的 CSV 检查。

## 9. 1000-100

1000-100 的训练分配如下。MARDAM 的两个 rank 分别使用 GPU 2、3；GPU 3
需要与 Full-graph 的第二个 rank 共享显存：

| 训练任务 | 物理 GPU | 进程数 | 全局/每卡 batch |
| --- | ---: | ---: | ---: |
| Full-graph | 0、3 | 2 | 16/8 |
| Global-action | 4、5 | 2 | 16/8 |
| Proposed | 1 | 1 | 16/16 |
| MARDAM | 2、3 | 2 | 16/8 |

6.3 的 Full-graph 在 500-50 可以单卡运行；1000-100 单卡约需 42.6 GB，
40 GB A100 会 OOM，因此下面给出的是双卡完整命令。如果服务器有 80 GB A100，
也可以把 Full-graph 改成单卡运行，但最终论文必须写明实际使用的配置。

本机只有 `0-3` 和 `4-5` 两组 NVLink。不要因为编号相邻就启动以下错误配置：

```text
Full-graph:  GPU 0,1
Global-action: GPU 2,3
```

这些组合都没有利用 NVLink。`torchrun` 的 rank 编号
不改变物理链路；真正决定通信路径的是 `CUDA_VISIBLE_DEVICES` 中的物理卡。
只有在第 6.3 节准入检查通过后，才启动完整的 1000-100 训练。

### 9.0 启动六卡训练监测

在启动任何 1000-100 正式训练命令之前，先创建一个独立监测会话。该会话持续
记录六张卡的显存、GPU 利用率、显存控制器利用率、功耗和温度：

```bash
tmux new-session -d -s gpu_monitor_1000

tmux send-keys -t gpu_monitor_1000 \
  "nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
  --format=csv,noheader,nounits \
  -l 1 | tee logs/1000_100/all_gpus_train.csv" C-m
```

确认监测已经开始：

```bash
tmux ls
sleep 2
tail -n 6 logs/1000_100/all_gpus_train.csv
```

随后再分别启动 9.1 至 9.4 的四个训练任务。不要为每个训练任务分别启动重复
的六卡监控进程。记录当前训练进程：

```bash
nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

六卡监测应在四个训练任务全部结束后再停止：

```bash
tmux kill-session -t gpu_monitor_1000
```

`nvidia-smi` 按秒采样适合判断显存余量和 OOM 风险，但可能漏掉非常短暂的
峰值。论文中的精确峰值应使用训练脚本内 PyTorch 的
`torch.cuda.max_memory_allocated()` 和
`torch.cuda.max_memory_reserved()`，并在相同指标定义下重新测量所有方法。

### 9.1 Proposed 1000-100 训练：GPU 1

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export MPLCONFIGDIR="/tmp/matplotlib-cache-1"
mkdir -p "$MPLCONFIGDIR"
set -o pipefail

/usr/bin/time -v \
  -o logs/1000_100/proposed_train.time.txt \
  python -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs \
  --model_variant third \
  --num_nodes 1000 \
  --num_agents 100 \
  --local_k_nodes 30 \
  --critic_inducing_points 32 \
  --epoch_count 100 \
  --iter_count 100 \
  --batch_size 16 \
  --eval_every 10 \
  --eval_episodes 10 \
  --eval_batch_size 1 \
  --device cuda:0 \
  --log_path runs/1000_100 \
  --run_name proposed_1000_100 \
  2>&1 | tee logs/1000_100/proposed_train.log
```

### 9.2 Full-graph 1000-100 双卡训练：NVLink GPU 0、3

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,3
export MPLCONFIGDIR="/tmp/matplotlib-cache-0-3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$MPLCONFIGDIR"
set -o pipefail

/usr/bin/time -v \
  -o logs/1000_100/full_graph_train.time.txt \
  torchrun --standalone --nproc_per_node=2 \
  -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs \
  --model_variant abs \
  --num_nodes 1000 \
  --num_agents 100 \
  --local_k_nodes 30 \
  --epoch_count 100 \
  --iter_count 100 \
  --batch_size 16 \
  --eval_every 10 \
  --eval_episodes 10 \
  --eval_batch_size 1 \
  --grad_checkpoint true \
  --device cuda:0 \
  --log_path runs/1000_100 \
  --run_name full_graph_1000_100 \
  2>&1 | tee logs/1000_100/full_graph_train.log
```

### 9.3 Global-action 1000-100 双卡训练：NVLink GPU 4、5

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=4,5
export MPLCONFIGDIR="/tmp/matplotlib-cache-4-5"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$MPLCONFIGDIR"
set -o pipefail

/usr/bin/time -v \
  -o logs/1000_100/global_action_train.time.txt \
  torchrun --standalone --nproc_per_node=2 \
  -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs_full \
  --model_variant third \
  --num_nodes 1000 \
  --num_agents 100 \
  --critic_inducing_points 32 \
  --epoch_count 100 \
  --iter_count 100 \
  --batch_size 16 \
  --eval_every 10 \
  --eval_episodes 10 \
  --eval_batch_size 1 \
  --grad_checkpoint true \
  --device cuda:0 \
  --log_path runs/1000_100 \
  --run_name global_action_1000_100 \
  2>&1 | tee logs/1000_100/global_action_train.log
```

### 9.4 MARDAM 1000-100 训练：GPU 2、3

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2,3
export MPLCONFIGDIR="/tmp/matplotlib-cache-2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$MPLCONFIGDIR"
set -o pipefail

/usr/bin/time -v \
  -o logs/1000_100/mardam_train.time.txt \
  torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m maenvs4vrp.learning.mardam.train_mardam_dtoptw \
  --vrp_env dtoptw_full \
  --num_nodes 1000 \
  --num_agents 100 \
  --epoch_count 100 \
  --iter_count 200 \
  --batch_size 16 \
  --eval_every 1 \
  --eval_episodes 10 \
  --grad_checkpoint true \
  --log_path runs/1000_100 \
  --run_name mardam_1000_100 \
  2>&1 | tee logs/1000_100/mardam_train_dp2.log
```

`torchrun` 会启动两个 rank：rank 0 使用物理 GPU 2，rank 1 使用物理 GPU 3。
脚本把 `--batch_size 16` 解释为全局 batch，每个 rank 实际处理 8 个实例，
然后对 learner 和 critic 的梯度做 all-reduce。只有 rank 0 写 TensorBoard
和 checkpoint。首次运行建议同时记录 GPU 2、3 的显存峰值，并确认 GPU 3 上
Full-graph 进程没有因共享显存而 OOM。

### 9.5 检查四个 checkpoint

```bash
ls -lh runs/1000_100/models/best_model_proposed_1000_100.zip
ls -lh runs/1000_100/models/best_model_full_graph_1000_100.zip
ls -lh runs/1000_100/models/best_model_global_action_1000_100.zip
ls -lh runs/1000_100/models/best_model_mardam_1000_100.zip
```

四个 checkpoint 都存在并且对应训练进程已经退出后，停止六卡监测并汇总采样峰值：

```bash
tmux kill-session -t gpu_monitor_1000

python - <<'PY'
import csv

path = "logs/1000_100/all_gpus_train.csv"
stats = {}

with open(path, newline="", encoding="utf-8") as f:
    for row in csv.reader(f):
        if len(row) < 8:
            continue
        try:
            gpu = int(row[1])
            memory = float(row[2])
            gpu_util = float(row[4])
            memory_util = float(row[5])
            power = float(row[6])
            temperature = float(row[7])
        except ValueError:
            continue

        item = stats.setdefault(
            gpu,
            {
                "peak_memory_mib": memory,
                "peak_gpu_util": gpu_util,
                "peak_memory_util": memory_util,
                "peak_power_w": power,
                "peak_temperature_c": temperature,
            },
        )
        item["peak_memory_mib"] = max(item["peak_memory_mib"], memory)
        item["peak_gpu_util"] = max(item["peak_gpu_util"], gpu_util)
        item["peak_memory_util"] = max(item["peak_memory_util"], memory_util)
        item["peak_power_w"] = max(item["peak_power_w"], power)
        item["peak_temperature_c"] = max(item["peak_temperature_c"], temperature)

for gpu in sorted(stats):
    item = stats[gpu]
    print(
        f"GPU {gpu}: "
        f"peak_memory={item['peak_memory_mib']:.0f} MiB "
        f"peak_gpu_util={item['peak_gpu_util']:.0f}% "
        f"peak_memory_util={item['peak_memory_util']:.0f}% "
        f"peak_power={item['peak_power_w']:.0f} W "
        f"peak_temperature={item['peak_temperature_c']:.0f} C"
    )
PY
```

这个汇总记录的是整张 GPU 的占用。当前正式分配没有把两个训练任务放在同一张
卡上，因此可直接用于判断 40 GB 显存余量；论文中的方法峰值仍需按同一指标
使用 PyTorch 的 allocated/reserved peak 单独测量。

1000-100 训练完成后，下面前六个评估可以分别放在 GPU 0 至 GPU 5 上并行。
RHC-ALNS 在第 9.6 节 Proposed 评估结束后再启动。若任一评估出现 OOM，再
退回单卡串行，不要修改正式的 `batch_size=1` 和 seed 范围。

### 9.6 Proposed 1000-100 评估：GPU 0

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

/usr/bin/time -v \
  -o results/1000_100/proposed_eval.time.txt \
  python -m maenvs4vrp.learning.madyam.eval_policy_model \
  --vrp_env dtoptw_abs \
  --model_variant third \
  --checkpoint runs/1000_100/models/best_model_proposed_1000_100.zip \
  --num_nodes 1000 \
  --num_agents 100 \
  --local_k_nodes 30 \
  --critic_inducing_points 32 \
  --embed_dim 128 \
  --n_heads 8 \
  --n_layers 3 \
  --ff_hidden 512 \
  --batch_size 1 \
  --eval_episodes 100 \
  --seed 12345 \
  --device cuda:0 \
  --save_csv results/1000_100/proposed.csv \
  2>&1 | tee logs/1000_100/proposed_eval.log
```

### 9.7 Full-graph 1000-100 评估：GPU 1

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1

/usr/bin/time -v \
  -o results/1000_100/full_graph_eval.time.txt \
  python -m maenvs4vrp.learning.madyam.eval_policy_model \
  --vrp_env dtoptw_abs \
  --model_variant abs \
  --checkpoint runs/1000_100/models/best_model_full_graph_1000_100.zip \
  --num_nodes 1000 \
  --num_agents 100 \
  --local_k_nodes 30 \
  --embed_dim 128 \
  --n_heads 8 \
  --n_layers 3 \
  --ff_hidden 512 \
  --batch_size 1 \
  --eval_episodes 100 \
  --seed 12345 \
  --device cuda:0 \
  --save_csv results/1000_100/full_graph.csv \
  2>&1 | tee logs/1000_100/full_graph_eval.log
```

### 9.8 Global-action 1000-100 评估：GPU 2

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2

/usr/bin/time -v \
  -o results/1000_100/global_action_eval.time.txt \
  python -m maenvs4vrp.learning.madyam.eval_policy_model \
  --vrp_env dtoptw_abs_full \
  --model_variant third \
  --checkpoint runs/1000_100/models/best_model_global_action_1000_100.zip \
  --num_nodes 1000 \
  --num_agents 100 \
  --critic_inducing_points 32 \
  --embed_dim 128 \
  --n_heads 8 \
  --n_layers 3 \
  --ff_hidden 512 \
  --batch_size 1 \
  --eval_episodes 100 \
  --seed 12345 \
  --device cuda:0 \
  --save_csv results/1000_100/global_action.csv \
  2>&1 | tee logs/1000_100/global_action_eval.log
```

### 9.9 MARDAM 1000-100 评估：GPU 3

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=3

/usr/bin/time -v \
  -o results/1000_100/mardam_eval.time.txt \
  python -m maenvs4vrp.learning.mardam.eval_mardam_dtoptw \
  --checkpoint runs/1000_100/models/best_model_mardam_1000_100.zip \
  --vrp_env dtoptw_full \
  --num_nodes 1000 \
  --num_agents 100 \
  --batch_size 1 \
  --eval_episodes 100 \
  --seed 12345 \
  --device cuda:0 \
  --save_csv results/1000_100/mardam.csv \
  2>&1 | tee logs/1000_100/mardam_eval.log
```

### 9.10 Greedy 1000-100 评估：GPU 4

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=4

/usr/bin/time -v \
  -o results/1000_100/greedy_eval.time.txt \
  python -m maenvs4vrp.learning.madyam.eval_policy_baselines \
  --vrp_env dtoptw_abs \
  --policy_type greedy \
  --num_nodes 1000 \
  --num_agents 100 \
  --local_k_nodes 30 \
  --batch_size 1 \
  --eval_episodes 100 \
  --num_workers 1 \
  --seed 12345 \
  --device cuda:0 \
  --save_csv results/1000_100/greedy.csv \
  2>&1 | tee logs/1000_100/greedy_eval.log
```

### 9.11 Random 1000-100 评估：GPU 5

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=5

/usr/bin/time -v \
  -o results/1000_100/random_eval.time.txt \
  python -m maenvs4vrp.learning.madyam.eval_policy_baselines \
  --vrp_env dtoptw_abs \
  --policy_type random \
  --num_nodes 1000 \
  --num_agents 100 \
  --local_k_nodes 30 \
  --batch_size 1 \
  --eval_episodes 100 \
  --num_workers 1 \
  --seed 12345 \
  --device cuda:0 \
  --save_csv results/1000_100/random.csv \
  2>&1 | tee logs/1000_100/random_eval.log
```

### 9.12 RHC-ALNS 1000-100 评估：GPU 0（待 Proposed 评估结束）

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

/usr/bin/time -v \
  -o results/1000_100/alns_eval.time.txt \
  python -m maenvs4vrp.learning.madyam.alns_rhc_dtoptw_abs \
  --vrp_env dtoptw_full \
  --num_nodes 1000 \
  --num_agents 100 \
  --batch_size 1 \
  --eval_episodes 100 \
  --seed 12345 \
  --alns_route_length 12 \
  --alns_iterations 40 \
  --alns_destroy_max 3 \
  --alns_candidate_pool 120 \
  --device cuda:0 \
  --save_csv results/1000_100/alns.csv \
  2>&1 | tee logs/1000_100/alns_eval.log
```

### 9.13 1000-100 结果文件

完成后应得到：

```text
results/1000_100/
  proposed.csv
  full_graph.csv
  global_action.csv
  mardam.csv
  greedy.csv
  random.csv
  alns.csv
```

每个 CSV 应有 100 条数据行，种子范围均为 `12345` 至 `12444`。

## 10. 每次运行后的检查

### 10.1 CSV 应有 100 行数据

```bash
for f in results/500_50/*.csv results/1000_100/*.csv; do
  echo "$f $(($(wc -l < "$f") - 1)) rows"
done
```

每个正式 CSV 应显示 `100 rows`。

### 10.2 检查种子范围

```bash
python - <<'PY'
import csv
from pathlib import Path

for path in sorted(Path("results").glob("*/*.csv")):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "seed" not in rows[0]:
        continue
    seeds = [int(row["seed"]) for row in rows]
    print(path, len(rows), min(seeds), max(seeds))
PY
```

正式结果应为：

```text
100 12345 12444
```

### 10.3 检查 checkpoint 和日志

```bash
ls -lh runs/500_50/models/
ls -lh runs/1000_100/models/

tail -n 30 logs/500_50/full_graph_train.log
tail -n 30 logs/500_50/global_action_train_dp2.log
```

## 11. 与论文表述保持一致

当前 `main.tex` 写的是单张 RTX 4090，并声明不做同硬件 wall-clock 比较。
如果最终结果改由 A100 产生，需要在论文中据实写明：

- 机器为 6 张 40 GB A100，其中每两张通过 NVLink 相连；
- Proposed 使用单张 A100；
- Full-graph 在 500-50 使用单张 A100，在 1000-100 使用两张 A100，
  全局 batch 16、每卡 batch 8；
- Global-action 使用两张 A100，全局 batch 16、每卡 batch 8；
- MARDAM 的 500-50 使用单张 A100；1000-100 使用 GPU 2、3 两张 A100，
  全局 batch 16、每卡 batch 8，两个 rank 通过 PCIe 做梯度同步；
- 显存数值必须重新测量，不能直接把旧 RTX 4090 的 6.8/14.0/42.6 GB
  写成 A100 实测值；
- 双卡训练中每张卡的 BatchNorm 统计量来自本卡 8 个实例，因此双卡训练曲线
  与旧单卡曲线不应当作逐位相同复现；
- 训练时间、GPU 利用率和显存均按独占或共享资源的实际运行方式记录；
- 所有需要相互比较的方法应尽量使用相同训练方式，或在论文中明确差异。

## 12. 常见问题

### `torchrun: command not found`

```bash
source .venv/bin/activate
which torchrun
uv pip show torch
```

### 双卡任务看起来只用了 GPU 0

检查启动环境：

```bash
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

双卡任务必须在同一个终端中设置 `CUDA_VISIBLE_DEVICES=0,3` 或
`CUDA_VISIBLE_DEVICES=4,5`，然后只启动一条 `torchrun` 命令，不要再分别
启动两个 Python 进程。

### 三卡运行时报 batch 不可分

全局 batch 16 不能被 3 整除。推荐使用双卡并保留第三张卡给其他任务。
如果必须三卡，显式使用 `--batch_size 18`（每卡 6）或 `--batch_size 15`
（每卡 5），并记录为新的训练预算。

### 增大 batch 能不能加速

不能直接加速同一个正式实验。脚本中一个 epoch 的样本数是
`batch_size × iter_count`，optimizer step 数是 `iter_count`。保持
`iter_count=100` 而把 batch 从 16 增加到 32，会让每个 epoch 多处理一倍
样本，但不会减少更新次数；每个 step 通常还会更慢并增加显存。

如果同时把 `iter_count` 降到 50，样本数不变，但 Adam 更新次数少了一半，
训练动力学已经改变。除非所有比较方法都重跑，并同步修改论文中的 batch、
iteration 和训练预算，否则不要这样做。

### GPU 利用率只有约 30%，能否共享显卡

可以额外启动低显存评估任务，但 30% 的 GPU 利用率不代表还有 70% 的可用
算力。这个指标可能受环境步进、CPU 等待、同步和显存带宽限制。先观察
`utilization.gpu`、`utilization.memory`、功耗和 CPU 占用，再做短时间共享
测试。

适合共享的是 Greedy、Random 等低显存评估；不适合在 40 GB 卡上强行叠加
另一个未知峰值的 1000-100 重型训练任务。共享后算法结果通常不变，但单个
任务的 wall-clock、GPU 利用率和总显存统计会变化，不能把共享运行时间当作
独占 GPU 时间。

### NVLink 是否真的被使用

先运行：

```bash
nvidia-smi topo -m
nvidia-smi nvlink --status
```

双卡任务的 `CUDA_VISIBLE_DEVICES` 必须只包含同一个 `NV#` 对。短时间启动时
可以设置 `NCCL_DEBUG=INFO` 检查初始化日志。不要为了提速设置
`NCCL_P2P_DISABLE=1`。

### 能否让一个任务使用四张卡

技术上 `train_madyam_abs.py` 在全局 batch 16 下支持四卡，因为 16 可以被
4 整除。但四卡会跨两个 NVLink 对，通信经过 PCIe/NUMA，而且会使其他任务
无法并行。当前两对 NVLink、四个训练任务的总体吞吐优先方案仍然是
`2+2+1+1`，不是 `4+2`。

### Global-action 双卡仍 OOM

先检查 Global-action 所使用的 NVLink 对是否还有旧进程，确认
`--grad_checkpoint true` 和 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
已生效。若双卡在 40 GB 上仍 OOM，不要静默降低 batch：可以暂停其他任务并
测试四卡、全局 batch 16（每卡 batch 4）；或者降低全局 batch 并同步调整所有
需要比较的方法及论文训练预算。不能只改 Global-action 的 batch 后仍声称与
其他方法保持相同训练配置。

### MARDAM 在 40 GB 单卡上 OOM

`train_mardam_dtoptw.py` 已支持通过 `torchrun` 做双 rank 数据并行。它会把
全局 batch 平均分给两个 rank，对 learner 和 critic 的梯度做 all-reduce 后
统一执行 optimizer step。1000-100 可使用 GPU 2、3，其中 GPU 3 与已有任务
共享剩余显存：

```bash
CUDA_VISIBLE_DEVICES=2,3 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m maenvs4vrp.learning.mardam.train_mardam_dtoptw \
  --batch_size 16 ...
```

启动前先确认 GPU 2 空闲、GPU 3 的剩余显存足够容纳另一个 batch 8 训练
进程。若 GPU 3 仍 OOM，需要等待 Full-graph 结束、降低全局 batch 并同步
记录新的训练预算，或换用显存更大的 GPU。两个 rank 的 BatchNorm 统计量
分别来自本卡的 8 个实例，因此结果不应与旧单卡 batch 16 声称逐位相同。

### `CUDA error: invalid device ordinal`

设置 `CUDA_VISIBLE_DEVICES=0,3` 时，两个进程内部只能看到 `cuda:0` 和
`cuda:1`。即使物理 GPU 是 0 和 3，也不要使用 `cuda:3`。

### 找不到 checkpoint

训练脚本保存到：

```text
<log_path>/models/
```

不是 `<log_path>/<run_name>/models/`。

## 13. 结果打包

```bash
tar \
  --exclude='.venv' \
  --exclude='.idea' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.egg-info' \
  -czf dtoptw_a100_results.tar.gz \
  maenvs4vrp \
  tests \
  results \
  runs \
  logs \
  setup.py \
  conftest.py \
  README.rst \
  main.tex \
  requirements.txt \
  BASELINES.md \
  GPU_RUNBOOK.md \
  EXPERIMENT_STEPS_CN.md

sha256sum dtoptw_a100_results.tar.gz > dtoptw_a100_results.sha256
```

至少保留并返回：

```text
results/
runs/
logs/
```

其中 `results/` 用于论文统计，`runs/` 保存训练曲线和模型，
`logs/` 保存训练时间、显存和完整终端输出。
