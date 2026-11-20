# DTOPTW baseline experiments

This repository now contains the DTOPTW environment and the baseline code used
by Chapter 4 of the thesis.  The additional baselines address two different
reviewer questions:

- **MARDAM** provides an independent learned multi-agent routing comparison.
- **Route-level RHC-ALNS** provides a traditional online optimisation
  comparison that respects dynamic arrivals, time windows, capacity, flight
  range, and depot resupply.

The existing full-graph critic remains the architecture comparison.  Greedy and
random policies remain lower-bound sanity checks.

The included ALNS implementation is route-level: it optimises the acting UAV's
short-horizon route and replans after executing the first action.  This keeps
the baseline executable in the asynchronous environment.  A full fleet-level
ALNS that jointly re-routes every UAV is a stronger but slower extension and is
not part of this implementation.

## Important modules

| Module | Purpose |
| --- | --- |
| `maenvs4vrp/learning/baseline_protocol.py` | Shared `K=30`, `M=32`, horizon, and 600/1200 decision-step settings |
| `maenvs4vrp/environments/dtoptw*` | Chapter 4 environment, absolute-feature observations, local/global actions |
| `maenvs4vrp/learning/madyam/policy_net_ma_ac_abs.py` | Local-policy and full-attention-critic model |
| `maenvs4vrp/learning/madyam/policy_net_ma_ac_abs_third.py` | Inducing-point privileged critic |
| `maenvs4vrp/learning/madyam/alns_rhc_dtoptw_abs.py` | Receding-horizon ALNS baseline |
| `maenvs4vrp/learning/mardam/train_mardam_dtoptw.py` | MARDAM training on DTOPTW |
| `maenvs4vrp/learning/mardam/eval_mardam_dtoptw.py` | Fixed-seed MARDAM checkpoint evaluation |
| `maenvs4vrp/learning/madyam/eval_policy_baselines.py` | Greedy and random baseline evaluation |
| `tests/integration/test_dtoptw_baseline_smoke.py` | Fast CPU integration checks |

## Paper-scale commands

Use the conda environment and a CUDA device for the full experiments:

```bash
conda run -n maenvs4vrp python -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs --model_variant third \
  --num_nodes 500 --num_agents 50 --local_k_nodes 30 \
  --critic_inducing_points 32 --device cuda

conda run -n maenvs4vrp python -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs --model_variant abs \
  --num_nodes 500 --num_agents 50 --local_k_nodes 30 --device cuda

conda run -n maenvs4vrp python -m maenvs4vrp.learning.madyam.train_madyam_abs \
  --vrp_env dtoptw_abs_full --model_variant third \
  --num_nodes 500 --num_agents 50 --critic_inducing_points 32 --device cuda

conda run -n maenvs4vrp python -m maenvs4vrp.learning.madyam.alns_rhc_dtoptw_abs \
  --vrp_env dtoptw_full --num_nodes 500 --num_agents 50 \
  --eval_episodes 100 --seed 12345 --device cuda:0 \
  --save_csv results/alns_500_50.csv

conda run -n maenvs4vrp python -m maenvs4vrp.learning.mardam.train_mardam_dtoptw \
  --vrp_env dtoptw_full --num_nodes 500 --num_agents 50 --device cuda
```

For 1000 nodes and 100 UAVs, replace the scale arguments with
`--num_nodes 1000 --num_agents 100`.  The shared protocol resolves the maximum
decision steps to 600 for the smaller scale and 1200 for the larger scale.

The global-action command above is an action-space ablation: it keeps the
inducing-point critic but removes the KNN actor restriction.  Comparing it with
the local proposed model isolates the contribution of local perception.

## Evaluate MARDAM

Training writes checkpoints under
`<log_path>/models/{latest,best}_model_<run_name>.zip`.

```bash
conda run -n maenvs4vrp python -m maenvs4vrp.learning.mardam.eval_mardam_dtoptw \
  --checkpoint runs/models/best_model_<run_name>.zip \
  --vrp_env dtoptw_full --num_nodes 500 --num_agents 50 \
  --eval_episodes 100 --seed 12345 --device cuda
```

Use the same test seed range for every learned method.

## Greedy and random baselines

Both scripts now use the paper protocol by default (`K=30`, no forced depot
self-loop, and scale-dependent decision-step budgets):

```bash
conda run -n maenvs4vrp python -m maenvs4vrp.learning.madyam.eval_policy_baselines \
  --vrp_env dtoptw_abs --policy_type greedy \
  --num_nodes 500 --num_agents 50 --eval_episodes 100 --device cuda

conda run -n maenvs4vrp python -m maenvs4vrp.learning.madyam.eval_policy_baselines \
  --vrp_env dtoptw_abs --policy_type random \
  --num_nodes 500 --num_agents 50 --eval_episodes 100 --device cuda
```

## CPU smoke test

```bash
conda run -n maenvs4vrp pytest -q tests/integration/test_dtoptw_baseline_smoke.py
```

The smoke test is only a wiring check.  It does not produce results suitable
for the paper.
