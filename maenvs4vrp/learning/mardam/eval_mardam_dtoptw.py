"""Evaluate a trained MARDAM checkpoint on fixed DTOPTW test seeds."""

from __future__ import annotations

import argparse
import csv
import math
import time

import torch

from maenvs4vrp.learning.baseline_protocol import resolve_max_steps
from maenvs4vrp.learning.mardam._learner import AttentionLearner
from maenvs4vrp.learning.mardam.train_mardam_dtoptw import build_env, reset_dtoptw_env, set_random_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--vrp_env", type=str, default="dtoptw_full")
    p.add_argument("--num_agents", type=int, default=100)
    p.add_argument("--num_nodes", type=int, default=1000)
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--profits", type=str, default="constant", choices=["distance", "constant", "uniform"])
    p.add_argument("--horizon", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=0, help="0 resolves to 600/1200 by node scale")
    p.add_argument("--max_flight_dist", type=float, default=300.0)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(str(x).lower()), default=True)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(str(x).lower()), default=True)
    p.add_argument("--done_feasible_check_freq", type=int, default=100)
    p.add_argument("--reward", type=str, default="dense", choices=["dense", "sparse"])
    p.add_argument("--save_csv", type=str, default="")
    return p.parse_args()


def _summary(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return mean, std, min(values), max(values)


def main() -> None:
    args = parse_args()
    args.max_steps = resolve_max_steps(args.num_nodes, args.max_steps)
    args.device = torch.device(args.device)
    set_random_seed(args.seed, True)

    env = build_env(args)
    learner = AttentionLearner(
        cust_feat_size=env.obs_builder.get_nodes_static_feat_dim(),
        veh_state_size=env.obs_builder.get_other_agents_feat_dim(),
    ).to(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = checkpoint.get("learner", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    learner.load_state_dict(state_dict)
    learner.eval()

    rows: list[dict[str, float | int]] = []
    start_all = time.perf_counter()
    with torch.no_grad():
        for episode in range(int(args.eval_episodes)):
            seed = int(args.seed) + episode
            td = reset_dtoptw_env(env, args, seed=seed)
            start = time.perf_counter()
            _actions, _logps, rewards = learner(env, td, deterministic=True)
            elapsed = time.perf_counter() - start
            total_return = torch.stack(rewards).sum(dim=0).mean().item()
            not_visited = (
                env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1.0
            ).mean().item()
            used_agents = (
                env.td_state["agents"]["visited_nodes"].sum(-1).gt(1).sum(-1).float()
            ).mean().item()
            rows.append(
                {
                    "episode": episode,
                    "seed": seed,
                    "return": float(total_return),
                    "not_visited": float(not_visited),
                    "used_agents": float(used_agents),
                    "rollout_seconds": float(elapsed),
                }
            )
            print(
                f"[episode {episode:03d}] seed={seed} return={total_return:.2f} "
                f"not_visited={not_visited:.2f} used_agents={used_agents:.2f} "
                f"seconds={elapsed:.3f}"
            )

    ret_mean, ret_std, ret_min, ret_max = _summary([float(r["return"]) for r in rows])
    elapsed_mean, _, _, _ = _summary([float(r["rollout_seconds"]) for r in rows])
    print("\n=== MARDAM Evaluation Summary ===")
    print(
        f"vrp_env={args.vrp_env} nodes={args.num_nodes} agents={args.num_agents} "
        f"episodes={args.eval_episodes} checkpoint={args.checkpoint}"
    )
    print(
        f"return_mean={ret_mean:.4f} return_std={ret_std:.4f} "
        f"return_min={ret_min:.4f} return_max={ret_max:.4f} "
        f"rollout_seconds_mean={elapsed_mean:.3f} "
        f"wall_seconds_total={time.perf_counter() - start_all:.3f}"
    )

    if args.save_csv:
        with open(args.save_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
