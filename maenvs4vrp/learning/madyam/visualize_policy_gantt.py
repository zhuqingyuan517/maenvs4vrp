"""
Draw a per-agent Gantt chart for one rollout on dtoptw_abs / dtoptw_abs_second.

Usage examples:
  1) Random policy on dtoptw_abs:
     python -m maenvs4vrp.learning.madyam.visualize_policy_gantt --vrp_env dtoptw_abs --policy_type random

  2) ABS model policy with checkpoint:
     python -m maenvs4vrp.learning.madyam.visualize_policy_gantt --vrp_env dtoptw_abs --policy_type model --model_kind abs --checkpoint runs/xxx.pt

  3) ABS-second model policy with checkpoint:
     python -m maenvs4vrp.learning.madyam.visualize_policy_gantt --vrp_env dtoptw_abs_second --policy_type model --model_kind abs --checkpoint runs/xxx.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from distutils.util import strtobool
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from maenvs4vrp.learning.madyam.visualize_policy_rollout import (
    build_env,
    build_model,
    configure_chinese_font,
    dedupe_local_action_mask,
    prefer_non_depot_action_mask,
    select_action,
    set_random_seed,
)


configure_chinese_font()


@dataclass
class GanttRecord:
    step: int
    agent: int
    from_node: int
    to_node: int
    depart_time: float
    arrival_time: float
    wait_start: float
    service_start: float
    service_end: float
    travel_time: float
    wait_time: float
    service_time: float
    reward: float
    is_depot: bool


def prepare_output_path(path: str) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vrp_env", type=str, default="dtoptw_abs")
    p.add_argument("--policy_type", type=str, default="model", choices=["random", "model", "greedy"])
    p.add_argument("--model_kind", type=str, default="abs", choices=["abs", "obs"], help="used only when policy_type=model")
    p.add_argument("--checkpoint", type=str, default="/home/sjm/PycharmProjects/maenvs4vrp_yc/maenvs4vrp/learning/madyam/1000runs/models/latest_model_abs_ac_dtoptw_abs_stime_1000n_100a_2026_03_09_15h38m.zip", help="optional model checkpoint (.pt/.pth)")
    p.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=True)

    p.add_argument("--num_agents", type=int, default=100)
    p.add_argument("--num_nodes", type=int, default=1000)
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--batch_idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=2297)
    p.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--local_k_nodes", type=int, default=50)
    p.add_argument("--local_view_mode", type=str, default="nearest", choices=["nearest", "random"])
    p.add_argument("--profits", type=str, default="constant", choices=["distance", "constant", "uniform"])
    p.add_argument("--horizon", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--max_flight_dist", type=float, default=300)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--done_feasible_check_freq", type=int, default=10)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--ff_hidden", type=int, default=512)

    p.add_argument("--fig_width", type=float, default=14.0)
    p.add_argument("--agent_row_height", type=float, default=0.55)
    p.add_argument("--annotate_nodes", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--annotate_step", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--annotate_reward", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--hide_idle_agents", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--save_fig", type=str, default="gantt/ceshi.png", help="optional png path")
    p.add_argument("--save_csv", type=str, default="", help="optional csv path")
    p.add_argument("--no_show", type=lambda x: bool(strtobool(x)), default=False)
    return p.parse_args()


def select_greedy_nearest_action(td, env) -> torch.Tensor:
    obs = td["observations"]
    action_mask = dedupe_local_action_mask(obs, obs["action_mask"])
    action_mask = prefer_non_depot_action_mask(obs, env, action_mask)
    bsz = action_mask.shape[0]

    if (~action_mask.any(dim=-1)).any():
        action_mask = action_mask.clone()
        bad = ~action_mask.any(dim=-1)
        action_mask[bad, 0] = True

    coords = env.td_state["coords"]
    cur_node = env.td_state["cur_agent"]["cur_node"].squeeze(-1)
    batch_idx = torch.arange(bsz, device=coords.device)
    cur_xy = coords[batch_idx, cur_node]

    if "local_node_idx" in obs.keys():
        cand_nodes = obs["local_node_idx"]
    else:
        n = coords.shape[1]
        cand_nodes = torch.arange(n, device=coords.device).unsqueeze(0).expand(bsz, n)

    non_self_feasible = action_mask & cand_nodes.ne(cur_node.unsqueeze(1))
    has_non_self = non_self_feasible.any(dim=-1, keepdim=True)
    action_mask = action_mask & (~cand_nodes.eq(cur_node.unsqueeze(1)) | ~has_non_self)

    cand_xy = coords.gather(1, cand_nodes.unsqueeze(-1).expand(-1, -1, 2))
    dists = torch.norm(cand_xy - cur_xy.unsqueeze(1), p=2, dim=-1)
    dists = dists.masked_fill(~action_mask, float("inf"))
    return torch.argmin(dists, dim=-1)


def rollout_once(args, env, learner):
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
    history: List[GanttRecord] = []
    model_cache_done = False
    step_i = 0

    while not td["done"].all():
        cur_agent = int(env.td_state["cur_agent_idx"][b, 0].item())
        cur_node = int(env.td_state["cur_agent"]["cur_node"][b, 0].item())
        depart_time = float(env.td_state["cur_agent"]["cur_time"][b, 0].item())

        if args.policy_type == "greedy":
            action = select_greedy_nearest_action(td, env)
        else:
            action, model_cache_done = select_action(args, td, env, learner, model_cache_done=model_cache_done)
        action_b = action[b : b + 1].unsqueeze(1)

        if env.use_local_view and getattr(env, "_local_node_idx", None) is not None:
            action_global_b = env._local_node_idx[b : b + 1].gather(1, action_b)
        else:
            action_global_b = action_b

        to_node = int(action_global_b[0, 0].item())

        time2j = float(
            torch.pairwise_distance(
                env.td_state["coords"][b : b + 1, cur_node : cur_node + 1],
                env.td_state["coords"][b : b + 1, to_node : to_node + 1],
                eps=0,
                keepdim=False,
            )[0].item()
        )
        if env.n_digits is not None:
            time2j = float(torch.floor(env.n_digits * torch.tensor(time2j)).item() / env.n_digits)

        tw_low = float(env.td_state["tw_low"][b, to_node].item())
        service_time = float(env.td_state["service_time"][b, to_node].item())
        arrival_time = depart_time + time2j
        wait_time = max(0.0, tw_low - arrival_time)
        service_start = arrival_time + wait_time
        service_end = service_start + service_time

        td["action"] = action.unsqueeze(1)
        td = env.step(td)

        step_r = td["reward"] + td["penalty"]
        if step_r.dim() == 2:
            step_r = step_r.squeeze(-1)
        total_ret += step_r

        history.append(
            GanttRecord(
                step=step_i,
                agent=cur_agent,
                from_node=cur_node,
                to_node=to_node,
                depart_time=depart_time,
                arrival_time=arrival_time,
                wait_start=arrival_time,
                service_start=service_start,
                service_end=service_end,
                travel_time=time2j,
                wait_time=wait_time,
                service_time=service_time,
                reward=float(step_r[b].item()),
                is_depot=bool(env.td_state["is_depot"][b, to_node].item()),
            )
        )
        step_i += 1

    return td, history, total_ret


def _rect(ax, y, start, duration, color, height, label=None, edgecolor="white", alpha=0.95):
    if duration <= 0:
        return
    ax.barh(y=y, width=duration, left=start, height=height, color=color, edgecolor=edgecolor, linewidth=0.8, alpha=alpha, label=label)


def draw_gantt(env, history: List[GanttRecord], batch_idx: int, args):
    n_agents = int(env.td_state["agents"]["cur_time"].shape[1])
    end_time = float(env.td_state["end_time"][batch_idx].item())
    used_agents = sorted({rec.agent for rec in history})
    agent_ids = used_agents if args.hide_idle_agents else list(range(n_agents))

    fig_height = max(4.6, 1.4 + 0.62 * max(1, len(agent_ids)))
    fig, ax = plt.subplots(figsize=(args.fig_width, fig_height))
    fig.patch.set_facecolor("#f7f5ef")
    ax.set_facecolor("#fffdf8")

    colors = {
        "travel": "#2a6f97",
        "wait": "#e9a03b",
        "service": "#5b8c5a",
        "depot": "#c44536",
    }

    y_positions = {agent: idx for idx, agent in enumerate(agent_ids)}
    seen_labels = set()

    for rec in history:
        if rec.agent not in y_positions:
            continue
        y = y_positions[rec.agent]
        travel_label = "行驶" if "travel" not in seen_labels else None
        wait_label = "等待" if "wait" not in seen_labels else None
        service_label = ("仓库停留" if rec.is_depot else "服务") if ("depot" if rec.is_depot else "service") not in seen_labels else None

        _rect(ax, y, rec.depart_time, rec.travel_time, colors["travel"], args.agent_row_height, travel_label)
        seen_labels.add("travel")
        _rect(ax, y, rec.wait_start, rec.wait_time, colors["wait"], args.agent_row_height, wait_label)
        if rec.wait_time > 0:
            seen_labels.add("wait")
        service_color = colors["depot"] if rec.is_depot else colors["service"]
        _rect(ax, y, rec.service_start, rec.service_time, service_color, args.agent_row_height, service_label)
        if rec.service_time > 0:
            seen_labels.add("depot" if rec.is_depot else "service")

        if args.annotate_nodes:
            label = f"{rec.from_node}->{rec.to_node}"
            if args.annotate_step:
                label = f"s{rec.step} {label}"
            if args.annotate_reward:
                label = f"{label} r={rec.reward:.1f}"
            text_x = rec.depart_time + max(rec.travel_time + rec.wait_time + rec.service_time, 1.0) * 0.5
            ax.text(
                text_x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="#1f1f1f",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.72),
            )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f"智能体 {agent}" for agent in agent_ids])
    ax.set_xlabel("时间")
    ax.set_ylabel("智能体")
    ax.set_xlim(0, end_time)
    total_wait = sum(rec.wait_time for rec in history)
    total_service = sum(rec.service_time for rec in history)
    total_travel = sum(rec.travel_time for rec in history)
    ax.set_title(
        f"策略甘特图 | 环境={args.vrp_env} | batch={batch_idx}\n"
        f"片段数={len(history)} | 行驶={total_travel:.1f} | 等待={total_wait:.1f} | 服务={total_service:.1f}",
        fontsize=13,
        pad=12,
    )
    ax.grid(axis="x", color="#d9d3c7", linestyle="--", linewidth=0.7, alpha=0.55)
    ax.axvline(end_time, color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(end_time, -0.65, f"时域上限={end_time:.1f}", ha="right", va="bottom", fontsize=8, color="#444444")
    ax.invert_yaxis()

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#b9b2a5")
    ax.spines["bottom"].set_color("#b9b2a5")

    legend_handles = [
        mpatches.Patch(color=colors["travel"], label="行驶"),
        mpatches.Patch(color=colors["wait"], label="等待"),
        mpatches.Patch(color=colors["service"], label="服务"),
        mpatches.Patch(color=colors["depot"], label="仓库停留"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, facecolor="white", edgecolor="#d8d2c4")
    fig.tight_layout()
    return fig


def dump_csv(path: str, records: List[GanttRecord]):
    output_path = prepare_output_path(path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "step,agent,from_node,to_node,depart_time,arrival_time,service_start,service_end,"
            "travel_time,wait_time,service_time,reward,is_depot\n"
        )
        for r in records:
            f.write(
                f"{r.step},{r.agent},{r.from_node},{r.to_node},{r.depart_time:.6f},{r.arrival_time:.6f},"
                f"{r.service_start:.6f},{r.service_end:.6f},{r.travel_time:.6f},{r.wait_time:.6f},"
                f"{r.service_time:.6f},{r.reward:.6f},{int(r.is_depot)}\n"
            )


def main():
    args = parse_args()
    args.device = torch.device(args.device)
    set_random_seed(args.seed, args.torch_deterministic)

    try:
        env = build_env(args)
    except ModuleNotFoundError as e:
        if "tensordict" in str(e):
            raise SystemExit("Missing dependency: tensordict. Please install it in your runtime environment.") from e
        raise
    learner = build_model(args, env)

    _td_final, history, total_ret = rollout_once(args, env, learner)
    b = int(args.batch_idx)
    ret_b = float(total_ret[b].item())
    not_visited = float(env.td_state["nodes"]["active_nodes_mask"][b].sum().item() - 1)
    used_agents = float(env.td_state["agents"]["visited_nodes"][b].sum(-1).gt(1).sum().item())

    print("=== Gantt Rollout Summary ===")
    print(f"policy_type={args.policy_type} model_kind={args.model_kind} deterministic={args.deterministic}")
    print(f"vrp_env={args.vrp_env} seed={args.seed} num_nodes={args.num_nodes} num_agents={args.num_agents}")
    print(f"steps={len(history)} return_batch[{b}]={ret_b:.4f}")
    print(f"not_visited_nodes(batch={b})={not_visited:.1f} used_agents(batch={b})={used_agents:.1f}")

    print("\n=== First 20 Schedule Segments ===")
    for r in history[:20]:
        print(
            f"step={r.step:03d} agent={r.agent:03d} {r.from_node:04d}->{r.to_node:04d} "
            f"depart={r.depart_time:7.1f} arrive={r.arrival_time:7.1f} "
            f"start={r.service_start:7.1f} end={r.service_end:7.1f}"
        )

    fig = draw_gantt(env=env, history=history, batch_idx=b, args=args)

    if args.save_fig:
        fig_path = prepare_output_path(args.save_fig)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"saved figure: {fig_path}")
    if args.save_csv:
        dump_csv(args.save_csv, history)
        print(f"saved csv: {prepare_output_path(args.save_csv)}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
