# Code for the paper (EAAI submission)

**Paper:** *Scalable Task Planning for Multiple Unmanned Aerial Vehicles in
Large-Scale Continuous Delivery Using Local Perception and Inducing
Representation* — under review at *Engineering Applications of Artificial
Intelligence*.

This repository is a fork of [MAEnvs4VRP](https://github.com/ricgama/maenvs4vrp)
(Gama et al., MIT License). Everything outside the paths listed below is
upstream code and is unchanged except for the four files noted in the last
table.

Branch: `paper-eaai-uav-delivery`.

## What this fork adds — paper item → file

| Paper item | File(s) |
|---|---|
| DT-OPTW-ABS environment (problem model, agent selector, reward, observations, instance generator) | `maenvs4vrp/environments/dtoptw_abs/` |
| Variant environments used for ablations / reference scale | `maenvs4vrp/environments/dtoptw/`, `dtoptw_abs_full/`, `dtoptw_abs_second/`, `dtoptw_full/` |
| Local-perception actor: KNN candidate set + feasibility mask + masked pointer decoding (Sec. 3.2) | `maenvs4vrp/learning/madyam/policy_net_ma_ac_abs.py` |
| Privileged critic with learnable inducing points (Sec. 3.3) | `maenvs4vrp/learning/mardam/baselines/_critic.py`, `maenvs4vrp/learning/mardam/layers/_mha.py` |
| CTDE training loop (Sec. 3.4) | `maenvs4vrp/learning/madyam/train_madyam_abs.py` |
| Decentralized inference / evaluation of the proposed planner | `maenvs4vrp/learning/madyam/eval_policy_model.py` |
| Receding-horizon re-planning wrapper | `maenvs4vrp/learning/madyam/replan_rhc_dtoptw_abs.py` |
| Baselines: MARDAM | `maenvs4vrp/learning/mardam/train_mardam_dtoptw.py`, `eval_mardam_dtoptw.py` |
| Baselines: greedy nearest / random | `maenvs4vrp/learning/madyam/greedy_nearest_dtoptw_abs.py`, `eval_policy_baselines.py` |
| Baseline: RHC-ALNS (route-level receding-horizon search) | `maenvs4vrp/learning/madyam/alns_rhc_dtoptw_abs.py` |
| Shared evaluation protocol (identical seeds / instances across methods) | `maenvs4vrp/learning/baseline_protocol.py` |
| Numerical helpers | `maenvs4vrp/learning/mardam/numerics.py` |
| Figure generation (rollouts, Gantt, attention scope) | `maenvs4vrp/learning/madyam/visualize_policy_*.py` |
| Per-seed results behind the reported tables | `results/500_50/*.csv`, `results/1000_100/*.csv` |
| Unit / integration tests | `tests/unit/test_mardam_numerics.py`, `tests/integration/test_dtoptw_baseline_smoke.py` |

Upstream files modified (method changes, not new files):

| File | Change |
|---|---|
| `maenvs4vrp/learning/mardam/_learner.py` | training loop hooks for the new actor/critic |
| `maenvs4vrp/learning/mardam/baselines/_critic.py` | inducing-point critic |
| `maenvs4vrp/learning/mardam/layers/_loss.py` | loss terms |
| `maenvs4vrp/learning/mardam/layers/_mha.py` | attention interface (M×N) |

## Setup

```bash
python -m pip install -e .        # installs the package in editable mode
```

`requirements.txt` is a full freeze of the environment used for the reported
experiments (Linux, Python 3.13, PyTorch 2.7.0 + CUDA 12.6).

## Training and evaluation (examples)

```bash
# train the proposed planner on 500 nodes / 50 UAVs
python -m maenvs4vrp.learning.madyam.train_madyam_abs \
    --vrp_env dtoptw_abs --num_agents 50 --num_nodes 500 \
    --local_k_nodes 30 --device cuda --seed 0

# evaluate a checkpoint over the fixed evaluation seeds
python -m maenvs4vrp.learning.madyam.eval_policy_model \
    --vrp_env dtoptw_abs --model_variant abs --checkpoint runs/<run>/model.pt \
    --num_agents 50 --num_nodes 500 --local_k_nodes 30 --eval_episodes 100

# baselines under the same protocol
python -m maenvs4vrp.learning.madyam.eval_policy_baselines --vrp_env dtoptw_abs --policy_type greedy
python -m maenvs4vrp.learning.madyam.alns_rhc_dtoptw_abs --help
```

Run any script with `--help` for the authoritative argument list; the commands
above are entry-point examples, not a verbatim record of the cluster jobs.

## Reported results

`results/500_50/` and `results/1000_100/` contain the per-run CSV output of
each method (`proposed`, `global_action`, `full_graph`, `mardam`, `greedy`,
`random`, `alns`) together with evaluation wall-time files. Mean profit
reported in the paper is computed over these files.

## License

MIT, inherited from upstream. See `LICENSE`. The upstream copyright notice
(`Copyright (c) 2024 maenvs4vrp`) is retained.
