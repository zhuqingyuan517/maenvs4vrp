# GPU migration and 500-50 / 1000-100 runbook

This runbook moves the working DTOPTW experiment code to another machine and
runs the paper-scale training and evaluation.

For the detailed post-transfer checklist in Chinese, including logging,
100-instance evaluation, output validation, and packaging instructions, see
`EXPERIMENT_STEPS_CN.md`.

## 1. Recommended GPU memory

| Workload | Suggested GPU memory |
| --- | --- |
| Proposed local actor + inducing-point critic, 500-50 | 24 GB or more |
| Proposed local actor + inducing-point critic, 1000-100 | 24 GB or more, preferably 32 GB |
| Full-graph critic, 500-50 | 24 GB or more |
| Full-graph critic, 1000-100 | 48 GB or more; the thesis reports about 42.6 GB |
| MARDAM and global-action ablation, 1000-100 | 48 GB preferred; exact peak must be measured |

The code currently supports single-GPU execution. `--device cuda:0` should be
passed explicitly because `train_madyam_abs.py` otherwise defaults to
`cuda:1` when CUDA is available.

## 2. Transfer the working tree

Run this on the current machine:

```bash
rsync -avP \
  --exclude='.idea/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.egg-info/' \
  --exclude='runs/' \
  /home/zqy/maenvs4vrp/ <gpu-user>@<gpu-host>:/data/<gpu-user>/
```

Do not copy only `maenvs4vrp/` while omitting `setup.py`, `conftest.py` and
`tests/`. The new DTOPTW and baseline modules are currently untracked files in
the source repository, so copying with `git clone` alone will not include them.

An archive alternative is:

```bash
tar \
  --exclude='.git' \
  --exclude='.idea' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.egg-info' \
  --exclude='runs' \
  -czf /tmp/maenvs4vrp-gpu.tar.gz \
  /home/zqy/maenvs4vrp
```

## 3. Create the GPU environment

On the GPU machine:

```bash
cd /data/<gpu-user>/maenvs4vrp
conda create -n maenvs4vrp-gpu python=3.11 -y
conda activate maenvs4vrp-gpu
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
```

If the installed NVIDIA driver does not support the CUDA 12.6 wheel, select a
compatible official PyTorch 2.7 wheel instead. Verify:

```bash
nvidia-smi
python -c "import torch, tensordict; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q tests/integration/test_dtoptw_baseline_smoke.py --device cuda:0
```

The DTOPTW experiments generate synthetic instances, so no benchmark download
is needed for Chapter 4.

## 4. Train 500-50

Run commands from the repository root and create result directories first:

```bash
mkdir -p runs results
```

Proposed local-perception model:

```bash
python -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs --model_variant third \
  --num_nodes 500 --num_agents 50 \
  --local_k_nodes 30 --critic_inducing_points 32 \
  --epoch_count 100 --iter_count 100 --batch_size 16 \
  --eval_every 10 --eval_episodes 10 --eval_batch_size 16 \
  --device cuda:0 --log_path runs/500_50 \
  --run_name proposed_500_50
```

Full-attention critic:

```bash
python -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs --model_variant abs \
  --num_nodes 500 --num_agents 50 --local_k_nodes 30 \
  --epoch_count 100 --iter_count 100 --batch_size 16 \
  --eval_every 10 --eval_episodes 10 --eval_batch_size 16 \
  --device cuda:0 --log_path runs/500_50 \
  --run_name full_graph_500_50
```

Global-action ablation with the inducing-point critic:

```bash
python -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs_full --model_variant third \
  --num_nodes 500 --num_agents 50 --critic_inducing_points 32 \
  --epoch_count 100 --iter_count 100 --batch_size 16 \
  --eval_every 10 --eval_episodes 10 --eval_batch_size 16 \
  --device cuda:0 --log_path runs/500_50 \
  --run_name global_action_500_50
```

MARDAM:

```bash
python -m maenvs4vrp.learning.mardam.train_mardam_dtoptw \
  --vrp_env dtoptw_full --num_nodes 500 --num_agents 50 \
  --epoch_count 100 --iter_count 200 --batch_size 16 \
  --eval_every 1 --eval_episodes 10 \
  --device cuda:0 --log_path runs/500_50 \
  --run_name mardam_500_50
```

Checkpoints are written under:

```text
runs/500_50/models/{latest,best}_model_<run_name>.zip
```

## 5. Train 1000-100

Repeat the commands with:

```text
--num_nodes 1000 --num_agents 100 --log_path runs/1000_100
--run_name <method>_1000_100
```

The shared protocol automatically sets the maximum decision steps to 1200 for
1000 nodes and 600 for 500 nodes.

Do not claim a same-budget comparison if memory pressure forces a method to use
a smaller batch. Record batch size, peak allocated/reserved GPU memory, training
time, and out-of-memory events. The full-graph 1000-100 baseline must be run on
a GPU with enough memory, or reported as OOM under the tested hardware.

## 6. Evaluate all methods on the same 100 seeds

Use `--seed 12345 --eval_episodes 100` for every method. This creates the same
test seeds `12345` through `12444`.

Proposed model:

```bash
python -m maenvs4vrp.learning.madyam.eval_policy_model \
  --vrp_env dtoptw_abs --model_variant third \
  --checkpoint runs/500_50/models/best_model_proposed_500_50.zip \
  --num_nodes 500 --num_agents 50 \
  --local_k_nodes 30 --critic_inducing_points 32 \
  --embed_dim 128 --n_heads 8 --n_layers 3 --ff_hidden 512 \
  --batch_size 1 --eval_episodes 100 --seed 12345 \
  --device cuda:0 --save_csv results/proposed_500_50.csv
```

Full-graph critic:

```bash
python -m maenvs4vrp.learning.madyam.eval_policy_model \
  --vrp_env dtoptw_abs --model_variant abs \
  --checkpoint runs/500_50/models/best_model_full_graph_500_50.zip \
  --num_nodes 500 --num_agents 50 --local_k_nodes 30 \
  --embed_dim 128 --n_heads 8 --n_layers 3 --ff_hidden 512 \
  --batch_size 1 --eval_episodes 100 --seed 12345 \
  --device cuda:0 --save_csv results/full_graph_500_50.csv
```

Global-action ablation:

```bash
python -m maenvs4vrp.learning.madyam.eval_policy_model \
  --vrp_env dtoptw_abs_full --model_variant third \
  --checkpoint runs/500_50/models/best_model_global_action_500_50.zip \
  --num_nodes 500 --num_agents 50 --critic_inducing_points 32 \
  --embed_dim 128 --n_heads 8 --n_layers 3 --ff_hidden 512 \
  --batch_size 1 --eval_episodes 100 --seed 12345 \
  --device cuda:0 --save_csv results/global_action_500_50.csv
```

MARDAM:

```bash
python -m maenvs4vrp.learning.mardam.eval_mardam_dtoptw \
  --checkpoint runs/500_50/models/best_model_mardam_500_50.zip \
  --vrp_env dtoptw_full --num_nodes 500 --num_agents 50 \
  --batch_size 1 --eval_episodes 100 --seed 12345 \
  --device cuda:0 --save_csv results/mardam_500_50.csv
```

Greedy and random:

```bash
python -m maenvs4vrp.learning.madyam.eval_policy_baselines \
  --vrp_env dtoptw_abs --policy_type greedy \
  --num_nodes 500 --num_agents 50 --local_k_nodes 30 \
  --batch_size 1 --eval_episodes 100 --num_workers 1 \
  --seed 12345 --device cuda:0 --save_csv results/greedy_500_50.csv

python -m maenvs4vrp.learning.madyam.eval_policy_baselines \
  --vrp_env dtoptw_abs --policy_type random \
  --num_nodes 500 --num_agents 50 --local_k_nodes 30 \
  --batch_size 1 --eval_episodes 100 --num_workers 1 \
  --seed 12345 --device cuda:0 --save_csv results/random_500_50.csv
```

RHC-ALNS:

```bash
python -m maenvs4vrp.learning.madyam.alns_rhc_dtoptw_abs \
  --vrp_env dtoptw_full --num_nodes 500 --num_agents 50 \
  --batch_size 1 --eval_episodes 100 --seed 12345 \
  --alns_route_length 12 --alns_iterations 40 \
  --alns_destroy_max 3 --alns_candidate_pool 120 \
  --device cuda:0 --save_csv results/alns_500_50.csv
```

For 1000-100, change the scale and checkpoint path in the same evaluation
commands.

## 7. Before reporting results

- Keep all seeds and environment settings identical across methods.
- Use a validation seed range to tune ALNS weights and neural hyperparameters.
- Report mean, standard deviation, and confidence intervals over the 100 test
  instances.
- Report online decision time separately from offline training time.
- Record peak GPU memory and failures explicitly.
- Do not use the CPU smoke-test numbers as paper results.
