"""Receding-horizon ALNS baseline for DTOPTW-style UAV delivery.

At every environment decision, the planner:

1. snapshots the acting UAV, visible task pool, and remaining constraints;
2. constructs a short route from the currently feasible actions;
3. applies adaptive large-neighbourhood destroy/repair and local search;
4. executes only the first action and replans at the next decision.

The implementation deliberately uses no future arrivals and no private UAV
state.  It is therefore suitable as an online traditional-optimization
baseline rather than an offline oracle.

Usage:
  python -m maenvs4vrp.learning.madyam.alns_rhc_dtoptw_abs \
    --vrp_env dtoptw_full --num_nodes 1000 --num_agents 100 \
    --device cuda --seed 2297
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from maenvs4vrp.learning.baseline_protocol import resolve_max_steps
from maenvs4vrp.learning.madyam.replan_rhc_dtoptw_abs import (
    StepRecord,
    build_env,
    set_random_seed,
)


@dataclass(frozen=True)
class PlanningSnapshot:
    """CPU-side snapshot used by the pure-Python ALNS planner."""

    coords: tuple[tuple[float, float], ...]
    tw_low: tuple[float, ...]
    tw_high: tuple[float, ...]
    service_time: tuple[float, ...]
    appear_time: tuple[float, ...]
    demand: tuple[float, ...]
    profit: tuple[float, ...]
    active: tuple[bool, ...]
    visible_nodes: tuple[int, ...]
    depot_idx: int
    current_node: int
    current_time: float
    current_load: float
    capacity: float
    remaining_distance: float | None
    max_flight_distance: float | None
    end_time: float
    n_digits: int | None

    @property
    def remaining_distance_or_inf(self) -> float:
        return math.inf if self.remaining_distance is None else self.remaining_distance


@dataclass(frozen=True)
class PlanResult:
    route: tuple[int, ...]
    served: tuple[int, ...]
    score: float
    profit: float
    travel: float
    wait: float
    depot_resets: int
    final_node: int
    final_time: float
    final_load: float
    final_remaining_distance: float | None
    feasible: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vrp_env", type=str, default="dtoptw_full")
    p.add_argument("--num_agents", type=int, default=100)
    p.add_argument("--num_nodes", type=int, default=1000)
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--batch_idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=2297)
    p.add_argument("--eval_episodes", type=int, default=1)
    p.add_argument("--torch_deterministic", type=lambda x: bool(str(x).lower()), default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--local_k_nodes", type=int, default=30)
    p.add_argument("--local_view_mode", type=str, default="nearest", choices=["nearest", "random"])
    p.add_argument("--profits", type=str, default="constant", choices=["distance", "constant", "uniform"])
    p.add_argument("--horizon", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=0, help="0 resolves to 600/1200 by node scale")
    p.add_argument("--max_flight_dist", type=float, default=300.0)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(str(x).lower()), default=True)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(str(x).lower()), default=True)
    p.add_argument("--done_feasible_check_freq", type=int, default=100)

    p.add_argument(
        "--alns_route_length",
        type=int,
        default=12,
        help="maximum number of actions in each receding-horizon plan",
    )
    p.add_argument("--alns_iterations", type=int, default=40)
    p.add_argument("--alns_destroy_max", type=int, default=3)
    p.add_argument(
        "--alns_candidate_pool",
        type=int,
        default=120,
        help="nearest visible tasks considered per repair; 0 uses all visible tasks",
    )
    p.add_argument("--alns_dist_weight", type=float, default=0.01)
    p.add_argument("--alns_wait_weight", type=float, default=0.001)
    p.add_argument("--alns_depot_penalty", type=float, default=0.05)
    p.add_argument("--save_csv", type=str, default="")
    return p.parse_args()


def _distance(snapshot: PlanningSnapshot, a: int, b: int) -> float:
    ax, ay = snapshot.coords[a]
    bx, by = snapshot.coords[b]
    d = math.hypot(ax - bx, ay - by)
    if snapshot.n_digits is not None:
        d = math.floor(float(snapshot.n_digits) * d) / float(snapshot.n_digits)
    return float(d)


def _infeasible_plan(snapshot: PlanningSnapshot) -> PlanResult:
    return PlanResult(
        route=(),
        served=(),
        score=float("-inf"),
        profit=0.0,
        travel=0.0,
        wait=0.0,
        depot_resets=0,
        final_node=snapshot.current_node,
        final_time=snapshot.current_time,
        final_load=snapshot.current_load,
        final_remaining_distance=snapshot.remaining_distance,
        feasible=False,
    )


def evaluate_plan(
    snapshot: PlanningSnapshot,
    route: Sequence[int],
    *,
    dist_weight: float = 0.01,
    wait_weight: float = 0.001,
    depot_penalty: float = 0.05,
) -> PlanResult:
    """Evaluate one route from the current state without mutating the env."""
    node = snapshot.current_node
    now = snapshot.current_time
    load = snapshot.current_load
    rem_dist = snapshot.remaining_distance
    served: set[int] = set()
    travel = 0.0
    wait = 0.0
    profit = 0.0
    resets = 0

    for nxt in route:
        nxt = int(nxt)
        if nxt == node:
            return _infeasible_plan(snapshot)
        d = _distance(snapshot, node, nxt)
        arrive = now + d
        travel += d

        if nxt == snapshot.depot_idx:
            if arrive > snapshot.end_time + 1e-6:
                return _infeasible_plan(snapshot)
            if rem_dist is not None and d > rem_dist + 1e-6:
                return _infeasible_plan(snapshot)
            now = arrive
            load = snapshot.capacity
            if snapshot.max_flight_distance is not None:
                rem_dist = snapshot.max_flight_distance
            resets += 1
        else:
            if (
                not snapshot.active[nxt]
                or nxt in served
                or snapshot.profit[nxt] <= 0.0
                or snapshot.appear_time[nxt] > now + 1e-6
            ):
                return _infeasible_plan(snapshot)

            service_start = max(arrive, snapshot.tw_low[nxt], snapshot.appear_time[nxt])
            wait += max(0.0, service_start - arrive)
            if service_start > snapshot.tw_high[nxt] + 1e-6:
                return _infeasible_plan(snapshot)
            depot_distance = _distance(snapshot, nxt, snapshot.depot_idx)
            if (
                service_start + snapshot.service_time[nxt] + depot_distance
                > snapshot.end_time + 1e-6
            ):
                return _infeasible_plan(snapshot)
            if snapshot.demand[nxt] > load + 1e-6:
                return _infeasible_plan(snapshot)
            if rem_dist is not None and d + depot_distance > rem_dist + 1e-6:
                return _infeasible_plan(snapshot)

            now = service_start + snapshot.service_time[nxt]
            load -= snapshot.demand[nxt]
            if rem_dist is not None:
                rem_dist -= d
            served.add(nxt)
            profit += snapshot.profit[nxt]

        node = nxt

    score = profit - dist_weight * travel - wait_weight * wait - depot_penalty * resets
    return PlanResult(
        route=tuple(int(n) for n in route),
        served=tuple(sorted(served)),
        score=float(score),
        profit=float(profit),
        travel=float(travel),
        wait=float(wait),
        depot_resets=int(resets),
        final_node=node,
        final_time=float(now),
        final_load=float(load),
        final_remaining_distance=None if rem_dist is None else float(rem_dist),
        feasible=True,
    )


def _feasible_next(
    snapshot: PlanningSnapshot,
    state: PlanResult,
    nodes: Iterable[int],
) -> list[int]:
    feasible: list[int] = []
    served = set(state.served)
    for nxt in nodes:
        if nxt in served or int(nxt) == state.final_node:
            continue
        trial = evaluate_plan(snapshot, state.route + (int(nxt),))
        if trial.feasible:
            feasible.append(int(nxt))
    return feasible


def _depot_action_possible(snapshot: PlanningSnapshot, state: PlanResult) -> bool:
    if state.final_node == snapshot.depot_idx:
        return False
    return evaluate_plan(snapshot, state.route + (snapshot.depot_idx,)).feasible


def _greedy_extend(
    snapshot: PlanningSnapshot,
    initial_route: Sequence[int],
    *,
    route_length: int,
    dist_weight: float,
    wait_weight: float,
    depot_penalty: float,
) -> PlanResult:
    state = evaluate_plan(
        snapshot,
        initial_route,
        dist_weight=dist_weight,
        wait_weight=wait_weight,
        depot_penalty=depot_penalty,
    )
    if not state.feasible:
        return state

    for _ in range(max(0, route_length - len(state.route))):
        options = _feasible_next(snapshot, state, snapshot.visible_nodes)
        best_state = None
        best_key = None
        for nxt in options:
            trial = evaluate_plan(
                snapshot,
                state.route + (nxt,),
                dist_weight=dist_weight,
                wait_weight=wait_weight,
                depot_penalty=depot_penalty,
            )
            dist = _distance(snapshot, state.final_node, nxt)
            key = (
                trial.score,
                snapshot.profit[nxt] - dist_weight * dist,
                -dist,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_state = trial

        if best_state is None:
            if _depot_action_possible(snapshot, state):
                trial = evaluate_plan(
                    snapshot,
                    state.route + (snapshot.depot_idx,),
                    dist_weight=dist_weight,
                    wait_weight=wait_weight,
                    depot_penalty=depot_penalty,
                )
                if trial.feasible:
                    state = trial
                    continue
            break
        state = best_state
    return state


def _initial_states(
    snapshot: PlanningSnapshot,
    *,
    route_length: int,
    beam_width: int,
    dist_weight: float,
    wait_weight: float,
    depot_penalty: float,
) -> list[PlanResult]:
    singles: list[tuple[float, int]] = []
    for nxt in snapshot.visible_nodes:
        trial = evaluate_plan(
            snapshot,
            (nxt,),
            dist_weight=dist_weight,
            wait_weight=wait_weight,
            depot_penalty=depot_penalty,
        )
        if trial.feasible:
            d = _distance(snapshot, snapshot.current_node, nxt)
            singles.append((trial.score - 0.05 * d, nxt))
    singles.sort(reverse=True)

    states: list[PlanResult] = []
    for _, nxt in singles[: max(1, beam_width)]:
        state = _greedy_extend(
            snapshot,
            (nxt,),
            route_length=route_length,
            dist_weight=dist_weight,
            wait_weight=wait_weight,
            depot_penalty=depot_penalty,
        )
        if state.feasible:
            states.append(state)

    if not states and _depot_action_possible(
        snapshot,
        evaluate_plan(snapshot, (), dist_weight=dist_weight, wait_weight=wait_weight, depot_penalty=depot_penalty),
    ):
        depot_state = _greedy_extend(
            snapshot,
            (snapshot.depot_idx,),
            route_length=route_length,
            dist_weight=dist_weight,
            wait_weight=wait_weight,
            depot_penalty=depot_penalty,
        )
        if depot_state.feasible:
            states.append(depot_state)
    return states


def _repair(
    snapshot: PlanningSnapshot,
    partial_route: Sequence[int],
    candidates: Sequence[int],
    *,
    route_length: int,
    dist_weight: float,
    wait_weight: float,
    depot_penalty: float,
) -> PlanResult:
    state = evaluate_plan(
        snapshot,
        partial_route,
        dist_weight=dist_weight,
        wait_weight=wait_weight,
        depot_penalty=depot_penalty,
    )
    if not state.feasible:
        return state

    for nxt in candidates:
        nxt = int(nxt)
        if nxt in state.route or nxt in state.served or len(state.route) >= route_length:
            continue

        options: list[PlanResult] = []
        for pos in range(len(state.route) + 1):
            trial_route = state.route[:pos] + (nxt,) + state.route[pos:]
            trial = evaluate_plan(
                snapshot,
                trial_route,
                dist_weight=dist_weight,
                wait_weight=wait_weight,
                depot_penalty=depot_penalty,
            )
            if trial.feasible:
                options.append(trial)

            if pos == 0 or state.route[pos - 1] != snapshot.depot_idx:
                via_route = state.route[:pos] + (snapshot.depot_idx, nxt) + state.route[pos:]
                via = evaluate_plan(
                    snapshot,
                    via_route,
                    dist_weight=dist_weight,
                    wait_weight=wait_weight,
                    depot_penalty=depot_penalty,
                )
                if via.feasible:
                    options.append(via)

        if options:
            best = max(options, key=lambda r: (r.score, -r.travel))
            if best.score > state.score + 1e-9:
                state = best
    return state


def _local_search(
    snapshot: PlanningSnapshot,
    state: PlanResult,
    *,
    dist_weight: float,
    wait_weight: float,
    depot_penalty: float,
) -> PlanResult:
    if not state.feasible:
        return state
    best = state
    route = list(state.route)
    task_positions = [i for i, n in enumerate(route) if n != snapshot.depot_idx]
    for i_idx, i in enumerate(task_positions):
        for j in task_positions[i_idx + 1 :]:
            trial_route = route.copy()
            trial_route[i], trial_route[j] = trial_route[j], trial_route[i]
            trial = evaluate_plan(
                snapshot,
                trial_route,
                dist_weight=dist_weight,
                wait_weight=wait_weight,
                depot_penalty=depot_penalty,
            )
            if trial.feasible and (trial.score, -trial.travel) > (best.score, -best.travel):
                best = trial
    return best


def optimize_route(
    snapshot: PlanningSnapshot,
    *,
    rng: random.Random,
    route_length: int,
    iterations: int,
    destroy_max: int,
    candidate_pool: int,
    dist_weight: float,
    wait_weight: float,
    depot_penalty: float,
) -> PlanResult:
    """Apply constructive ALNS to one acting UAV's short-horizon route."""
    initial = _initial_states(
        snapshot,
        route_length=route_length,
        beam_width=4,
        dist_weight=dist_weight,
        wait_weight=wait_weight,
        depot_penalty=depot_penalty,
    )
    empty = evaluate_plan(
        snapshot,
        (),
        dist_weight=dist_weight,
        wait_weight=wait_weight,
        depot_penalty=depot_penalty,
    )
    if not initial:
        return empty

    best = max(initial, key=lambda r: (r.score, -r.travel))
    incumbent = best
    visible = list(snapshot.visible_nodes)
    pool_size = len(visible) if int(candidate_pool) <= 0 else min(len(visible), int(candidate_pool))
    visible.sort(key=lambda n: _distance(snapshot, snapshot.current_node, n))
    candidate_pool_nodes = visible[:pool_size]

    for iteration in range(max(0, int(iterations))):
        if iteration > 0 and iteration % 12 == 0:
            restarts = _initial_states(
                snapshot,
                route_length=route_length,
                beam_width=2,
                dist_weight=dist_weight,
                wait_weight=wait_weight,
                depot_penalty=depot_penalty,
            )
            if restarts:
                incumbent = rng.choice(restarts)
        else:
            incumbent = best if rng.random() < 0.7 else incumbent

        route = list(incumbent.route)
        task_positions = [i for i, n in enumerate(route) if n != snapshot.depot_idx]
        if not task_positions:
            break
        destroy_count = rng.randint(
            1,
            min(len(task_positions), max(1, int(destroy_max))),
        )
        removed_positions = set(rng.sample(task_positions, destroy_count))
        removed = [route[i] for i in sorted(removed_positions)]
        partial = [n for i, n in enumerate(route) if i not in removed_positions]

        candidates = removed + rng.sample(
            candidate_pool_nodes,
            k=min(len(candidate_pool_nodes), max(8, destroy_count * 3)),
        )
        rng.shuffle(candidates)
        candidate = _repair(
            snapshot,
            partial,
            candidates,
            route_length=route_length,
            dist_weight=dist_weight,
            wait_weight=wait_weight,
            depot_penalty=depot_penalty,
        )
        if not candidate.feasible:
            continue

        # Mild diversification acceptance prevents a small repair loss from
        # freezing the incumbent while preserving the best route separately.
        if candidate.score >= incumbent.score or rng.random() < 0.1:
            incumbent = candidate
        if (candidate.score, -candidate.travel) > (best.score, -best.travel):
            best = candidate

    best = _local_search(
        snapshot,
        best,
        dist_weight=dist_weight,
        wait_weight=wait_weight,
        depot_penalty=depot_penalty,
    )
    return best


def _build_snapshot(env, batch_idx: int, allowed_nodes: Sequence[int] | None = None) -> PlanningSnapshot:
    state = env.td_state
    coords = state["coords"][batch_idx].detach().cpu().tolist()
    tw_low = state["tw_low"][batch_idx].detach().cpu().tolist()
    tw_high = state["tw_high"][batch_idx].detach().cpu().tolist()
    service = state["service_time"][batch_idx].detach().cpu().tolist()
    appear = state["appear_time"][batch_idx].detach().cpu().tolist()
    demand = state["demands"][batch_idx].detach().cpu().tolist()
    profit = state["nodes"]["cur_profits"][batch_idx].detach().cpu().tolist()
    active = state["nodes"]["active_nodes_mask"][batch_idx].detach().cpu().tolist()
    depot_idx = int(state["depot_idx"][batch_idx, 0].item())
    current_node = int(state["cur_agent"]["cur_node"][batch_idx, 0].item())
    current_time = float(state["cur_agent"]["cur_time"][batch_idx, 0].item())
    current_load = float(state["cur_agent"]["cur_load"][batch_idx, 0].item())
    capacity = float(state["capacity"][batch_idx].reshape(-1)[0].item())
    current_end_time = float(state["end_time"][batch_idx].item())

    rem_dist = None
    max_flight = None
    if "rem_dist" in state["cur_agent"].keys():
        rem_dist = float(state["cur_agent"]["rem_dist"][batch_idx, 0].item())
    if "max_flight_dist" in state.keys():
        max_flight = float(state["max_flight_dist"][batch_idx, 0].item())

    if allowed_nodes is None:
        allowed_nodes = range(len(coords))
    visible = tuple(
        dict.fromkeys(
            int(n)
            for n in allowed_nodes
            if int(n) != depot_idx
            and bool(active[int(n)])
            and profit[int(n)] > 0.0
            and appear[int(n)] <= current_time + 1e-6
        )
    )

    return PlanningSnapshot(
        coords=tuple((float(x), float(y)) for x, y in coords),
        tw_low=tuple(float(v) for v in tw_low),
        tw_high=tuple(float(v) for v in tw_high),
        service_time=tuple(float(v) for v in service),
        appear_time=tuple(float(v) for v in appear),
        demand=tuple(float(v) for v in demand),
        profit=tuple(float(v) for v in profit),
        active=tuple(bool(v) for v in active),
        visible_nodes=visible,
        depot_idx=depot_idx,
        current_node=current_node,
        current_time=current_time,
        current_load=current_load,
        capacity=capacity,
        remaining_distance=rem_dist,
        max_flight_distance=max_flight,
        end_time=current_end_time,
        n_digits=getattr(env, "n_digits", None),
    )


def select_alns_action(
    args: argparse.Namespace,
    td,
    env,
    *,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, PlanResult]:
    """Select one action for every batch row by route-level RHC-ALNS."""
    obs = td["observations"]
    action_mask = obs["action_mask"].bool()
    bsz = action_mask.shape[0]
    actions = torch.zeros(bsz, dtype=torch.long, device=action_mask.device)
    plans: list[PlanResult] = []
    rng = rng or random.Random(int(args.seed))

    local_nodes = obs.get("local_node_idx", None)
    for b in range(bsz):
        allowed = None
        if local_nodes is not None:
            allowed = local_nodes[b].detach().cpu().tolist()
        snapshot = _build_snapshot(env, b, allowed_nodes=allowed)
        plan = optimize_route(
            snapshot,
            rng=rng,
            route_length=int(args.alns_route_length),
            iterations=int(args.alns_iterations),
            destroy_max=int(args.alns_destroy_max),
            candidate_pool=int(args.alns_candidate_pool),
            dist_weight=float(args.alns_dist_weight),
            wait_weight=float(args.alns_wait_weight),
            depot_penalty=float(args.alns_depot_penalty),
        )
        plans.append(plan)

        raw_action = None
        if plan.feasible and plan.route:
            target = int(plan.route[0])
            if local_nodes is None:
                raw_action = target
            else:
                matches = (local_nodes[b] == target).nonzero(as_tuple=False).flatten()
                if matches.numel() > 0:
                    raw_action = int(matches[0].item())
            if raw_action is not None and bool(action_mask[b, raw_action].item()):
                actions[b] = raw_action
                continue

        legal = action_mask[b].nonzero(as_tuple=False).flatten()
        if legal.numel() > 0:
            actions[b] = int(legal[0].item())
        else:
            actions[b] = 0
            action_mask[b, 0] = True
        plans[-1] = _infeasible_plan(snapshot)
    return actions, plans[-1] if plans else _infeasible_plan(_build_snapshot(env, 0))


def rollout_once(args: argparse.Namespace, env):
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

    rng = random.Random(int(args.seed))
    total_ret = torch.zeros(td["reward"].shape[0], device=td.device, dtype=torch.float32)
    history: list[StepRecord] = []
    decision_times: list[float] = []
    step_i = 0

    while not td["done"].all():
        cur_agent = int(env.td_state["cur_agent_idx"][b, 0].item())
        cur_node = int(env.td_state["cur_agent"]["cur_node"][b, 0].item())
        global_feasible_cnt = int(env.td_state["cur_agent"]["action_mask"][b].sum().item())

        start = time.perf_counter()
        action, _plan = select_alns_action(args, td, env, rng=rng)
        decision_times.append(time.perf_counter() - start)

        obs = td["observations"]
        action_raw_b = int(action[b].item())
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

    return history, total_ret, decision_times


def main() -> None:
    args = parse_args()
    args.max_steps = resolve_max_steps(args.num_nodes, args.max_steps)
    args.device = torch.device(args.device)
    base_seed = int(args.seed)
    b = int(args.batch_idx)
    rows: list[dict[str, float | int]] = []
    wall_start = time.perf_counter()
    for episode in range(int(args.eval_episodes)):
        args.seed = base_seed + episode
        set_random_seed(args.seed, args.torch_deterministic)
        env = build_env(args)
        history, total_ret, decision_times = rollout_once(args, env)

        ret_b = float(total_ret[b].item())
        ret_mean = float(total_ret.mean().item())
        not_visited = float(env.td_state["nodes"]["active_nodes_mask"][b].sum().item() - 1)
        used_agents = float(env.td_state["agents"]["visited_nodes"][b].sum(-1).gt(1).sum().item())
        mean_ms = 1000.0 * sum(decision_times) / max(len(decision_times), 1)
        rows.append(
            {
                "episode": episode,
                "seed": args.seed,
                "steps": len(history),
                "return": ret_b,
                "return_batch_mean": ret_mean,
                "not_visited": not_visited,
                "used_agents": used_agents,
                "decision_ms_mean": mean_ms,
            }
        )
        print(
            f"[episode {episode:03d}] seed={args.seed} steps={len(history)} "
            f"return={ret_b:.4f} not_visited={not_visited:.1f} "
            f"used_agents={used_agents:.1f} decision_ms_mean={mean_ms:.2f}"
        )

    returns = [float(row["return"]) for row in rows]
    ret_mean = sum(returns) / len(returns)
    ret_std = math.sqrt(sum((v - ret_mean) ** 2 for v in returns) / len(returns))
    seconds_mean = sum(float(row["decision_ms_mean"]) for row in rows) / len(rows)

    print("\n=== RHC-ALNS Evaluation Summary ===")
    print(
        f"vrp_env={args.vrp_env} nodes={args.num_nodes} agents={args.num_agents} "
        f"episodes={args.eval_episodes} seed={base_seed} "
        f"route_length={args.alns_route_length} iterations={args.alns_iterations}"
    )
    print(
        f"return_mean={ret_mean:.4f} return_std={ret_std:.4f} "
        f"return_min={min(returns):.4f} return_max={max(returns):.4f} "
        f"decision_ms_mean={seconds_mean:.2f} "
        f"wall_seconds_total={time.perf_counter() - wall_start:.3f}"
    )

    if args.save_csv:
        with open(args.save_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
