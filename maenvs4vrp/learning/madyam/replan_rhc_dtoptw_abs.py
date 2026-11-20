"""
Receding-horizon replanning (RHC) rollout on dtoptw_abs.

At each decision step, evaluate each currently feasible action by simulating
an H-step greedy lookahead plan, execute the best first action, then replan.

Usage:
  python -m maenvs4vrp.learning.madyam.replan_rhc_dtoptw_abs
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
    p.add_argument("--num_agents", type=int, default=10)
    p.add_argument("--num_nodes", type=int, default=100)
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--batch_idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=2297)
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

    p.add_argument("--replan_depth", type=int, default=5, help="lookahead depth H for each replanning step")
    p.add_argument("--gamma", type=float, default=0.95, help="discount factor in lookahead score")
    p.add_argument("--dist_weight", type=float, default=0.01, help="distance penalty in greedy lookahead tie-breaking")
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


def _feasible_nodes_mask(
    *,
    node: int,
    cur_time: torch.Tensor,
    cur_load: torch.Tensor,
    rem_dist: torch.Tensor | None,
    active_nodes_mask: torch.Tensor,
    appear_time: torch.Tensor,
    tw_low: torch.Tensor,
    tw_high: torch.Tensor,
    service_time: torch.Tensor,
    time2depot: torch.Tensor,
    demands: torch.Tensor,
    end_time: torch.Tensor,
    dist_mat: torch.Tensor,
) -> torch.Tensor:
    mask = active_nodes_mask & (appear_time <= cur_time)
    d = dist_mat[node]
    arrive = cur_time + d
    service_start = torch.maximum(arrive, tw_low)
    c1 = service_start <= tw_high
    c2 = service_start + service_time + time2depot <= end_time
    c3 = demands <= cur_load
    mask = mask & c1 & c2 & c3
    if rem_dist is not None:
        c4 = d + time2depot <= rem_dist
        mask = mask & c4
    return mask


def _simulate_choice_value(
    *,
    first_node: int,
    depth: int,
    gamma: float,
    dist_weight: float,
    depot_idx: int,
    capacity: torch.Tensor,
    max_flight_dist: torch.Tensor | None,
    allow_depot_refuel: bool,
    node: int,
    cur_time: torch.Tensor,
    cur_load: torch.Tensor,
    rem_dist: torch.Tensor | None,
    active_nodes_mask: torch.Tensor,
    cur_profits: torch.Tensor,
    appear_time: torch.Tensor,
    tw_low: torch.Tensor,
    tw_high: torch.Tensor,
    service_time: torch.Tensor,
    time2depot: torch.Tensor,
    demands: torch.Tensor,
    end_time: torch.Tensor,
    dist_mat: torch.Tensor,
) -> float:
    sim_node = int(node)
    sim_time = cur_time.clone()
    sim_load = cur_load.clone()
    sim_rem = rem_dist.clone() if rem_dist is not None else None
    sim_active = active_nodes_mask.clone()

    score = 0.0
    disc = 1.0

    chosen = int(first_node)
    for h in range(depth):
        if h > 0:
            feas = _feasible_nodes_mask(
                node=sim_node,
                cur_time=sim_time,
                cur_load=sim_load,
                rem_dist=sim_rem,
                active_nodes_mask=sim_active,
                appear_time=appear_time,
                tw_low=tw_low,
                tw_high=tw_high,
                service_time=service_time,
                time2depot=time2depot,
                demands=demands,
                end_time=end_time,
                dist_mat=dist_mat,
            )
            non_self = feas & (~torch.arange(feas.shape[0], device=feas.device).eq(sim_node))
            if non_self.any():
                feas = non_self
            if not feas.any():
                break
            drow = dist_mat[sim_node]
            greedy_val = cur_profits - dist_weight * drow
            greedy_val = greedy_val.masked_fill(~feas, float("-inf"))
            chosen = int(torch.argmax(greedy_val).item())

        d = dist_mat[sim_node, chosen]
        arrive = sim_time + d
        start = torch.maximum(arrive, tw_low[chosen])
        sim_time = start + service_time[chosen]

        if bool(sim_active[chosen].item()) and chosen != depot_idx:
            score += disc * float(cur_profits[chosen].item())
            sim_active[chosen] = False
        disc *= gamma

        sim_load = sim_load - demands[chosen]
        if allow_depot_refuel and chosen == depot_idx:
            sim_load = capacity.clone()

        if sim_rem is not None:
            sim_rem = sim_rem - d
            if chosen == depot_idx and max_flight_dist is not None:
                sim_rem = max_flight_dist.clone()

        sim_node = chosen

    return score


def select_replan_action(args, td, env) -> torch.Tensor:
    obs = td["observations"]
    action_mask = obs["action_mask"]  # [B, A], True feasible
    bsz = action_mask.shape[0]

    if (~action_mask.any(dim=-1)).any():
        action_mask = action_mask.clone()
        bad = ~action_mask.any(dim=-1)
        action_mask[bad, 0] = True

    if "local_node_idx" in obs.keys():
        cand_nodes = obs["local_node_idx"]
    else:
        n = env.td_state["coords"].shape[1]
        cand_nodes = torch.arange(n, device=action_mask.device).unsqueeze(0).expand(bsz, n)

    actions = torch.zeros(bsz, dtype=torch.long, device=action_mask.device)
    all_nodes = torch.arange(cand_nodes.shape[1], device=action_mask.device)

    for b in range(bsz):
        cur_node = int(env.td_state["cur_agent"]["cur_node"][b, 0].item())
        cur_time = env.td_state["cur_agent"]["cur_time"][b, 0]
        cur_load = env.td_state["cur_agent"]["cur_load"][b, 0]
        rem_dist = env.td_state["cur_agent"]["rem_dist"][b, 0] if "rem_dist" in env.td_state["cur_agent"].keys() else None

        coords = env.td_state["coords"][b]
        dist_mat = torch.cdist(coords, coords, p=2)
        if getattr(env, "n_digits", None) is not None:
            n_digits = float(env.n_digits)
            dist_mat = torch.floor(n_digits * dist_mat) / n_digits

        active_nodes_mask = env.td_state["nodes"]["active_nodes_mask"][b].clone()
        cur_profits = env.td_state["nodes"]["cur_profits"][b]
        appear_time = env.td_state["appear_time"][b]
        tw_low = env.td_state["tw_low"][b]
        tw_high = env.td_state["tw_high"][b]
        service_time = env.td_state["service_time"][b]
        time2depot = env.td_state["time2depot"][b]
        demands = env.td_state["demands"][b]
        end_time = env.td_state["end_time"][b]
        cap_row = env.td_state["agents"]["capacity"][b]
        capacity = cap_row.reshape(-1)[0]
        max_flight_dist = env.td_state["max_flight_dist"][b, 0] if "max_flight_dist" in env.td_state.keys() else None
        depot_idx = int(env.td_state["depot_idx"][b, 0].item())

        feasible_raw = all_nodes[action_mask[b]]
        if feasible_raw.numel() == 0:
            actions[b] = 0
            continue

        # Avoid self-loop when alternatives exist.
        candidate_globals = cand_nodes[b, feasible_raw]
        non_self = candidate_globals.ne(cur_node)
        if non_self.any():
            feasible_raw = feasible_raw[non_self]
            candidate_globals = candidate_globals[non_self]

        best_raw = int(feasible_raw[0].item())
        best_score = float("-inf")
        best_dist = float("inf")

        for ridx, node_idx in zip(feasible_raw.tolist(), candidate_globals.tolist()):
            node_idx = int(node_idx)
            sim_score = _simulate_choice_value(
                first_node=node_idx,
                depth=max(1, int(args.replan_depth)),
                gamma=float(args.gamma),
                dist_weight=float(args.dist_weight),
                depot_idx=depot_idx,
                capacity=capacity,
                max_flight_dist=max_flight_dist,
                allow_depot_refuel=bool(args.allow_depot_refuel),
                node=cur_node,
                cur_time=cur_time,
                cur_load=cur_load,
                rem_dist=rem_dist,
                active_nodes_mask=active_nodes_mask,
                cur_profits=cur_profits,
                appear_time=appear_time,
                tw_low=tw_low,
                tw_high=tw_high,
                service_time=service_time,
                time2depot=time2depot,
                demands=demands,
                end_time=end_time,
                dist_mat=dist_mat,
            )
            d = float(dist_mat[cur_node, node_idx].item())
            if (sim_score > best_score) or (sim_score == best_score and d < best_dist):
                best_score = sim_score
                best_dist = d
                best_raw = int(ridx)

        actions[b] = best_raw

    return actions


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

        action = select_replan_action(args, td, env)
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

    print("=== Replan RHC Rollout Summary ===")
    print(
        f"vrp_env={args.vrp_env} seed={args.seed} num_nodes={args.num_nodes} num_agents={args.num_agents} "
        f"depth={args.replan_depth} gamma={args.gamma}"
    )
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
