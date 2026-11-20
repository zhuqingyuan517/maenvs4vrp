"""
Greedy nearest-neighbor rollout on dtoptw_abs.

Usage:
  python -m maenvs4vrp.learning.madyam.greedy_nearest_dtoptw_abs
"""

from __future__ import annotations

import argparse
import importlib
import random
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Any, Dict, List

import torch

from maenvs4vrp.learning.baseline_protocol import resolve_max_steps


@dataclass
class StepRecord:
    step: int
    agent: int
    from_node: int
    to_node: int
    reward: float
    feasible_cnt: int
    local_mode: bool


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vrp_env", type=str, default="dtoptw_abs")
    p.add_argument("--num_agents", type=int, default=100)
    p.add_argument("--num_nodes", type=int, default=1000)
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--batch_idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=2297)
    p.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--local_k_nodes", type=int, default=20)
    p.add_argument("--local_view_mode", type=str, default="nearest", choices=["nearest", "random"])
    p.add_argument("--profits", type=str, default="constant", choices=["distance", "constant", "uniform"])
    p.add_argument("--horizon", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=0, help="0 resolves to 600/1200 by node scale")
    p.add_argument("--max_flight_dist", type=float, default=300)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--done_feasible_check_freq", type=int, default=100)
    return p.parse_args()


def set_random_seed(seed: int, torch_deterministic: bool):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def build_env(args):
    env_agent_selector_module_name = f"maenvs4vrp.environments.{args.vrp_env}.env_agent_selector"
    sel_mod = importlib.import_module(env_agent_selector_module_name)
    if args.selection == "rand":
        selector = sel_mod.RandomSelector()
    elif args.selection == "single":
        selector = sel_mod.AgentSelector()
    else:
        selector = sel_mod.SmallestTimeAgentSelector()

    observations_module_name = f"maenvs4vrp.environments.{args.vrp_env}.observations"
    obs_mod = importlib.import_module(observations_module_name)
    obs_kwargs: Dict[str, Any] = {}
    if args.vrp_env in ("dtoptw", "dtoptw_abs"):
        obs_kwargs["local_view"] = {"k_nodes": args.local_k_nodes, "mode": args.local_view_mode, "include_depot": True}
    observations = obs_mod.Observations(**obs_kwargs)

    generator_module_name = f"maenvs4vrp.environments.{args.vrp_env}.instances_generator"
    gen_mod = importlib.import_module(generator_module_name)
    generator = gen_mod.InstanceGenerator(device=args.device)

    env_reward_module_name = f"maenvs4vrp.environments.{args.vrp_env}.env_agent_reward"
    rew_mod = importlib.import_module(env_reward_module_name)
    reward_evaluator = rew_mod.DenseReward()

    environment_module_name = f"maenvs4vrp.environments.{args.vrp_env}.env"
    env_mod = importlib.import_module(environment_module_name)
    return env_mod.Environment(
        instance_generator_object=generator,
        obs_builder_object=observations,
        agent_selector_object=selector,
        reward_evaluator=reward_evaluator,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        max_steps=args.max_steps,
        max_flight_dist=args.max_flight_dist,
        allow_depot_refuel=args.allow_depot_refuel,
        done_feasible_check_freq=args.done_feasible_check_freq,
    )


def select_greedy_nearest_action(td, env) -> torch.Tensor:
    obs = td["observations"]
    action_mask = obs["action_mask"]  # [B, K] or [B, N], True feasible
    bsz = action_mask.shape[0]

    if (~action_mask.any(dim=-1)).any():
        action_mask = action_mask.clone()
        bad = ~action_mask.any(dim=-1)
        action_mask[bad, 0] = True

    coords = env.td_state["coords"]  # [B, N, 2]
    cur_node = env.td_state["cur_agent"]["cur_node"].squeeze(-1)  # [B]
    batch_idx = torch.arange(bsz, device=coords.device)
    cur_xy = coords[batch_idx, cur_node]  # [B, 2]

    if "local_node_idx" in obs.keys():
        cand_nodes = obs["local_node_idx"]  # [B, K]
    else:
        n = coords.shape[1]
        cand_nodes = torch.arange(n, device=coords.device).unsqueeze(0).expand(bsz, n)

    # Avoid self-loop when any other feasible action exists.
    non_self_feasible = action_mask & cand_nodes.ne(cur_node.unsqueeze(1))
    has_non_self = non_self_feasible.any(dim=-1, keepdim=True)
    action_mask = action_mask & (~cand_nodes.eq(cur_node.unsqueeze(1)) | ~has_non_self)

    cand_xy = coords.gather(1, cand_nodes.unsqueeze(-1).expand(-1, -1, 2))  # [B, A, 2]
    dists = torch.norm(cand_xy - cur_xy.unsqueeze(1), p=2, dim=-1)  # [B, A]
    dists = dists.masked_fill(~action_mask, float("inf"))
    return torch.argmin(dists, dim=-1)


def rollout_once(args, env):
    td = env.reset(
        num_agents=args.num_agents,
        num_nodes=args.num_nodes,
        profits=args.profits,
        seed=args.seed,
        sample_type="random",
        max_steps=args.max_steps,
        max_flight_dist=args.max_flight_dist,
        allow_depot_refuel=args.allow_depot_refuel,
        forbid_depot_self_loop=args.forbid_depot_self_loop,
        horizon=args.horizon,
        done_feasible_check_freq=args.done_feasible_check_freq,
    )
    b = int(args.batch_idx)
    if b < 0 or b >= int(td["done"].shape[0]):
        raise ValueError(f"batch_idx={b} out of range for batch_size={int(td['done'].shape[0])}")

    total_ret = torch.zeros(td["reward"].shape[0], device=td.device, dtype=torch.float32)
    history: List[StepRecord] = []
    step_i = 0

    while not td["done"].all():
        cur_agent = int(env.td_state["cur_agent_idx"][b, 0].item())
        cur_node = int(env.td_state["cur_agent"]["cur_node"][b, 0].item())
        global_feasible_cnt = int(env.td_state["cur_agent"]["action_mask"][b].sum().item())

        action = select_greedy_nearest_action(td, env)
        action_raw_b = int(action[b].item())
        obs = td["observations"]
        if "local_node_idx" in obs.keys():
            to_node = int(obs["local_node_idx"][b, action_raw_b].item())
            local_mode = True
        else:
            to_node = action_raw_b
            local_mode = False

        td["action"] = action.unsqueeze(1)
        td = env.step(td)

        step_r = td["reward"] + td["penalty"]
        if step_r.dim() == 2:
            step_r = step_r.squeeze(-1)
        total_ret += step_r

        history.append(
            StepRecord(
                step=step_i,
                agent=cur_agent,
                from_node=cur_node,
                to_node=to_node,
                reward=float(step_r[b].item()),
                feasible_cnt=global_feasible_cnt,
                local_mode=local_mode,
            )
        )
        step_i += 1

    return history, total_ret


def main():
    args = parse_args()
    args.max_steps = resolve_max_steps(args.num_nodes, args.max_steps)
    args.device = torch.device(args.device)
    set_random_seed(args.seed, args.torch_deterministic)

    try:
        env = build_env(args)
    except ModuleNotFoundError as e:
        if "tensordict" in str(e):
            raise SystemExit("Missing dependency: tensordict. Please install it in your runtime environment.") from e
        raise
    history, total_ret = rollout_once(args, env)

    b = int(args.batch_idx)
    ret_b = float(total_ret[b].item())
    ret_mean = float(total_ret.mean().item())
    not_visited = float(env.td_state["nodes"]["active_nodes_mask"][b].sum().item() - 1)
    used_agents = float(env.td_state["agents"]["visited_nodes"][b].sum(-1).gt(1).sum().item())

    print("=== Greedy Nearest Rollout Summary ===")
    print(f"vrp_env={args.vrp_env} seed={args.seed} num_nodes={args.num_nodes} num_agents={args.num_agents}")
    print(f"steps={len(history)} return_batch[{b}]={ret_b:.4f} return_mean={ret_mean:.4f}")
    print(f"not_visited_nodes(batch={b})={not_visited:.1f} used_agents(batch={b})={used_agents:.1f}")

    print("\n=== First 20 Steps ===")
    for r in history[:20]:
        print(
            f"step={r.step:03d} agent={r.agent:03d} {r.from_node:04d}->{r.to_node:04d} "
            f"feasible={r.feasible_cnt:04d} reward={r.reward:+.4f} local_mode={int(r.local_mode)}"
        )


if __name__ == "__main__":
    main()
