"""
Batch evaluation for baseline policies on dtoptw_abs / dtoptw_abs_second.

Usage examples:
  python -m maenvs4vrp.learning.madyam.eval_policy_baselines --vrp_env dtoptw_abs --policy_type random
  python -m maenvs4vrp.learning.madyam.eval_policy_baselines --vrp_env dtoptw_abs_second --policy_type greedy
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing as mp
from distutils.util import strtobool
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from maenvs4vrp.learning.baseline_protocol import resolve_max_steps
from maenvs4vrp.learning.madyam.visualize_policy_rollout import (
    build_action_generator,
    build_env,
    select_action,
    set_random_seed,
)


def prepare_output_path(path: str) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vrp_env", type=str, default="dtoptw_abs")
    p.add_argument("--policy_type", type=str, default="greedy", choices=["random", "greedy"])

    p.add_argument("--num_agents", type=int, default=100)
    p.add_argument("--num_nodes", type=int, default=1000)
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=2297)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=10, help="number of worker processes across episodes; 1 disables multiprocessing")
    p.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--local_k_nodes", type=int, default=30)
    p.add_argument("--local_view_mode", type=str, default="nearest", choices=["nearest", "random"])
    p.add_argument("--profits", type=str, default="constant", choices=["distance", "constant", "uniform"])
    p.add_argument("--horizon", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=0, help="0 resolves to 600/1200 by node scale")
    p.add_argument("--max_flight_dist", type=float, default=300)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--done_feasible_check_freq", type=int, default=100)

    p.add_argument("--save_csv", type=str, default="", help="optional csv output path")
    return p.parse_args()


def rollout_episode(args, env, seed: int) -> Dict[str, List[float] | int]:
    td = env.reset(
        num_agents=args.num_agents,
        num_nodes=args.num_nodes,
        profits=args.profits,
        seed=seed,
        sample_type="random",
        max_steps=args.max_steps,
        max_flight_dist=args.max_flight_dist,
        allow_depot_refuel=args.allow_depot_refuel,
        forbid_depot_self_loop=args.forbid_depot_self_loop,
        horizon=args.horizon,
        done_feasible_check_freq=args.done_feasible_check_freq,
    )

    total_ret = torch.zeros(td["reward"].shape[0], device=td.device, dtype=torch.float32)
    steps = 0
    learner = None
    model_cache_done = False
    action_rng = build_action_generator(seed, td.device)

    while not td["done"].all():
        action, model_cache_done = select_action(
            args,
            td,
            env,
            learner,
            model_cache_done=model_cache_done,
            action_rng=action_rng,
        )
        td["action"] = action.unsqueeze(1)
        td = env.step(td)

        step_r = td["reward"] + td["penalty"]
        if step_r.dim() == 2:
            step_r = step_r.squeeze(-1)
        total_ret += step_r
        steps += 1

    not_visited = env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1.0
    used_agents = env.td_state["agents"]["visited_nodes"].sum(-1).gt(1).sum(-1).float()

    return {
        "seed": seed,
        "steps": steps,
        "return": total_ret.detach().cpu().tolist(),
        "not_visited": not_visited.detach().cpu().tolist(),
        "used_agents": used_agents.detach().cpu().tolist(),
    }


def flatten_metric(episodes: List[Dict[str, List[float] | int]], key: str) -> List[float]:
    values: List[float] = []
    for ep in episodes:
        values.extend(float(v) for v in ep[key])  # type: ignore[index]
    return values


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(values),
        "max": max(values),
    }


def dump_csv(path: str, episodes: List[Dict[str, List[float] | int]]):
    output_path = prepare_output_path(path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("episode_idx,seed,batch_idx,steps,return,not_visited,used_agents\n")
        for ep_idx, ep in enumerate(episodes):
            seed = int(ep["seed"])  # type: ignore[arg-type]
            steps = int(ep["steps"])  # type: ignore[arg-type]
            returns = ep["return"]  # type: ignore[assignment]
            not_visited = ep["not_visited"]  # type: ignore[assignment]
            used_agents = ep["used_agents"]  # type: ignore[assignment]
            for batch_idx, (ret, nv, ua) in enumerate(zip(returns, not_visited, used_agents)):
                f.write(f"{ep_idx},{seed},{batch_idx},{steps},{float(ret):.6f},{float(nv):.6f},{float(ua):.6f}\n")


def _run_episode_worker(payload: Tuple[dict, int]) -> Dict[str, List[float] | int]:
    args_dict, seed = payload
    ns = argparse.Namespace(**args_dict)
    ns.device = torch.device(ns.device)
    set_random_seed(ns.seed, ns.torch_deterministic)
    env = build_env(ns)
    return rollout_episode(ns, env, seed=seed)


def evaluate_serial(args) -> List[Dict[str, List[float] | int]]:
    env = build_env(args)
    episodes: List[Dict[str, List[float] | int]] = []
    for ep_idx in range(args.eval_episodes):
        ep_seed = int(args.seed) + ep_idx
        result = rollout_episode(args, env, seed=ep_seed)
        episodes.append(result)
        ret_stats = summarize([float(v) for v in result["return"]])  # type: ignore[index]
        nv_stats = summarize([float(v) for v in result["not_visited"]])  # type: ignore[index]
        ua_stats = summarize([float(v) for v in result["used_agents"]])  # type: ignore[index]
        print(
            f"[episode {ep_idx:03d}] seed={ep_seed} steps={int(result['steps'])} "
            f"return_mean={ret_stats['mean']:.4f} not_visited_mean={nv_stats['mean']:.2f} used_agents_mean={ua_stats['mean']:.2f}"
        )
    return episodes


def evaluate_parallel(args) -> List[Dict[str, List[float] | int]]:
    args_dict = vars(args).copy()
    args_dict["device"] = str(args.device)
    payloads = [(args_dict, int(args.seed) + ep_idx) for ep_idx in range(args.eval_episodes)]
    episodes: List[Dict[str, List[float] | int]] = [None] * len(payloads)  # type: ignore[list-item]

    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(args.num_workers), mp_context=mp_context) as executor:
        for ep_idx, result in enumerate(executor.map(_run_episode_worker, payloads)):
            episodes[ep_idx] = result
            ep_seed = int(result["seed"])  # type: ignore[arg-type]
            ret_stats = summarize([float(v) for v in result["return"]])  # type: ignore[index]
            nv_stats = summarize([float(v) for v in result["not_visited"]])  # type: ignore[index]
            ua_stats = summarize([float(v) for v in result["used_agents"]])  # type: ignore[index]
            print(
                f"[episode {ep_idx:03d}] seed={ep_seed} steps={int(result['steps'])} "
                f"return_mean={ret_stats['mean']:.4f} not_visited_mean={nv_stats['mean']:.2f} used_agents_mean={ua_stats['mean']:.2f}"
            )
    return episodes


def main():
    args = parse_args()
    args.max_steps = resolve_max_steps(args.num_nodes, args.max_steps)
    args.device = torch.device(args.device)
    args.model_kind = "abs"
    set_random_seed(args.seed, args.torch_deterministic)

    try:
        if int(args.num_workers) <= 1:
            episodes = evaluate_serial(args)
        else:
            episodes = evaluate_parallel(args)
    except ModuleNotFoundError as e:
        if "tensordict" in str(e):
            raise SystemExit("Missing dependency: tensordict. Please install it in your runtime environment.") from e
        raise

    returns = flatten_metric(episodes, "return")
    not_visited = flatten_metric(episodes, "not_visited")
    used_agents = flatten_metric(episodes, "used_agents")
    steps = [float(ep["steps"]) for ep in episodes]

    ret_stats = summarize(returns)
    nv_stats = summarize(not_visited)
    ua_stats = summarize(used_agents)
    step_stats = summarize(steps)

    print("\n=== Baseline Eval Summary ===")
    print(
        f"policy_type={args.policy_type} vrp_env={args.vrp_env} "
        f"episodes={args.eval_episodes} batch_size={args.batch_size} num_workers={args.num_workers}"
    )
    print(
        f"return_mean={ret_stats['mean']:.4f} return_std={ret_stats['std']:.4f} "
        f"return_min={ret_stats['min']:.4f} return_max={ret_stats['max']:.4f}"
    )
    print(
        f"not_visited_mean={nv_stats['mean']:.4f} not_visited_std={nv_stats['std']:.4f} "
        f"used_agents_mean={ua_stats['mean']:.4f} used_agents_std={ua_stats['std']:.4f}"
    )
    print(
        f"steps_mean={step_stats['mean']:.2f} steps_std={step_stats['std']:.2f} "
        f"steps_min={step_stats['min']:.0f} steps_max={step_stats['max']:.0f}"
    )

    if args.save_csv:
        dump_csv(args.save_csv, episodes)
        print(f"saved csv: {prepare_output_path(args.save_csv)}")


if __name__ == "__main__":
    main()
