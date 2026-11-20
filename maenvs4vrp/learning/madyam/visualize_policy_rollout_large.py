"""
Lightweight rollout visualization for large VRP instances.

This variant keeps the rollout/data path identical to visualize_policy_rollout,
but uses thinner primitives and faster matplotlib collections so that scenes like
1000 nodes / 100 agents remain readable.

Usage example:
  python -m maenvs4vrp.learning.madyam.visualize_policy_rollout_large --vrp_env dtoptw_abs_second --policy_type model --num_nodes 1000 --num_agents 100 --checkpoint runs/xxx.pt --save_fig rollout_large.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import torch

from maenvs4vrp.learning.madyam.visualize_policy_rollout import (
    build_env,
    build_model,
    configure_chinese_font,
    dump_step_log,
    select_action,
    prepare_output_path,
    set_random_seed,
)


configure_chinese_font()


@dataclass
class LargeStepRecord:
    step: int
    agent: int
    from_node: int
    to_node: int
    action_raw: int
    reward: float
    feasible_cnt: int
    local_mode: bool
    depart_time: float
    arrival_time: float
    service_start: float
    service_end: float
    is_depot: bool


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vrp_env", type=str, default="dtoptw_abs")
    p.add_argument("--policy_type", type=str, default="random", choices=["random", "model", "greedy"])
    p.add_argument("--model_kind", type=str, default="third", choices=["abs", "third", "obs"], help="used only when policy_type=model")
    p.add_argument("--checkpoint", type=str, default="/home/sjm/PycharmProjects/maenvs4vrp_yc/maenvs4vrp/learning/madyam/1000runs/models/latest_model_third_ac_dtoptw_abs_stime_1000n_100a_2026_03_09_20h02m.zip", help="optional model checkpoint (.pt/.pth)")
    p.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=True)

    p.add_argument("--num_agents", type=int, default=100)
    p.add_argument("--num_nodes", type=int, default=1000)
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--batch_idx", type=int, default=1, help="which batch item to visualize")
    p.add_argument("--seed", type=int, default=2297)
    p.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--local_k_nodes", type=int, default=30)
    p.add_argument("--local_view_mode", type=str, default="nearest", choices=["nearest", "random"])
    p.add_argument("--profits", type=str, default="constant", choices=["distance", "constant", "uniform"])
    p.add_argument("--horizon", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=1200)
    p.add_argument("--max_flight_dist", type=float, default=300)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--done_feasible_check_freq", type=int, default=10)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--ff_hidden", type=int, default=512)
    p.add_argument("--critic_inducing_points", type=int, default=16)

    p.add_argument("--max_plot_steps", type=int, default=1500, help="limit edges drawn on figure")
    p.add_argument("--time_cutoff", type=float, default=480, help="only visualize/measure rollout up to this time; <0 means full horizon")
    p.add_argument("--time_cutoffs", type=float, nargs="*", default=[120, 240, 360, 480], help="optional list of cutoffs; generates one figure per cutoff")
    p.add_argument("--coverage_count_mode", type=str, default="service_start", choices=["arrival", "service_start", "service_end"], help="when a node counts as covered under time_cutoff")
    p.add_argument("--highlight_new_nodes", type=lambda x: bool(strtobool(x)), default=True, help="highlight nodes whose appear_time falls in the current cutoff interval")
    p.add_argument("--new_node_size", type=float, default=22.0)
    p.add_argument("--new_node_edgewidth", type=float, default=0.9)
    p.add_argument("--show_step_labels", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--show_node_labels", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--show_agent_legend", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--legend_agents", type=int, default=12, help="max agents shown in legend")
    p.add_argument("--service_node_size", type=float, default=8.0)
    p.add_argument("--depot_node_size", type=float, default=42.0)
    p.add_argument("--route_linewidth", type=float, default=0.8)
    p.add_argument("--route_alpha", type=float, default=0.42)
    p.add_argument("--node_alpha", type=float, default=0.82)
    p.add_argument("--fig_width", type=float, default=11.0)
    p.add_argument("--fig_height", type=float, default=11.0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--rasterized", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--save_fig", type=str, default="", help="optional png path")
    p.add_argument("--save_step_log", type=str, default="", help="optional txt path")
    p.add_argument("--no_show", type=lambda x: bool(strtobool(x)), default=False)
    return p.parse_args()


def _resolve_depot_mask(td, batch_idx: int, n_nodes: int) -> torch.Tensor:
    depot_mask = torch.zeros(n_nodes, dtype=torch.bool)
    if "is_depot" in td.keys():
        depot_mask = td["is_depot"][batch_idx].detach().cpu().to(torch.bool)
    elif "depot_idx" in td.keys():
        depot_idx = td["depot_idx"][batch_idx]
        if depot_idx.dim() == 0:
            depot_mask[int(depot_idx.item())] = True
        else:
            for d in depot_idx.view(-1).tolist():
                d = int(d)
                if 0 <= d < n_nodes:
                    depot_mask[d] = True
    else:
        depot_mask[0] = True
    return depot_mask


def rollout_once_large(args, env, learner):
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
    history: List[LargeStepRecord] = []
    model_cache_done = False
    step_i = 0

    while not td["done"].all():
        cur_agent = int(env.td_state["cur_agent_idx"][b, 0].item())
        cur_node = int(env.td_state["cur_agent"]["cur_node"][b, 0].item())
        global_feasible_cnt = int(env.td_state["cur_agent"]["action_mask"][b].sum().item())
        depart_time = float(env.td_state["cur_agent"]["cur_time"][b, 0].item())

        action, model_cache_done = select_action(args, td, env, learner, model_cache_done=model_cache_done)
        action_raw_b = int(action[b].item())

        obs = td["observations"]
        if "local_node_idx" in obs.keys():
            to_node = int(obs["local_node_idx"][b, action_raw_b].item())
            local_mode = True
        else:
            to_node = action_raw_b
            local_mode = False

        from_xy = env.td_state["coords"][b, cur_node]
        to_xy = env.td_state["coords"][b, to_node]
        travel_time = float(torch.norm(to_xy - from_xy, p=2).item())
        if getattr(env, "n_digits", None) is not None:
            travel_time = float(torch.floor(env.n_digits * torch.tensor(travel_time)).item() / env.n_digits)

        tw_low = float(env.td_state["tw_low"][b, to_node].item())
        service_time = float(env.td_state["service_time"][b, to_node].item())
        arrival_time = depart_time + travel_time
        service_start = max(arrival_time, tw_low)
        service_end = service_start + service_time

        td["action"] = action.unsqueeze(1)
        td = env.step(td)

        step_r = td["reward"] + td["penalty"]
        if step_r.dim() == 2:
            step_r = step_r.squeeze(-1)
        total_ret += step_r

        history.append(
            LargeStepRecord(
                step=step_i,
                agent=cur_agent,
                from_node=cur_node,
                to_node=to_node,
                action_raw=action_raw_b,
                reward=float(step_r[b].item()),
                feasible_cnt=global_feasible_cnt,
                local_mode=local_mode,
                depart_time=depart_time,
                arrival_time=arrival_time,
                service_start=service_start,
                service_end=service_end,
                is_depot=bool(env.td_state["is_depot"][b, to_node].item()) if "is_depot" in env.td_state.keys() else (to_node == 0),
            )
        )
        step_i += 1

    return td, history, total_ret


def filter_history_by_time(history: List[LargeStepRecord], time_cutoff: float) -> List[LargeStepRecord]:
    if time_cutoff < 0:
        return history
    return [rec for rec in history if rec.depart_time <= time_cutoff]


def compute_coverage(history: List[LargeStepRecord], time_cutoff: float, count_mode: str) -> tuple[int, int]:
    if time_cutoff < 0:
        cutoff_records = history
    elif count_mode == "arrival":
        cutoff_records = [rec for rec in history if rec.arrival_time <= time_cutoff]
    elif count_mode == "service_end":
        cutoff_records = [rec for rec in history if rec.service_end <= time_cutoff]
    else:
        cutoff_records = [rec for rec in history if rec.service_start <= time_cutoff]

    visited_service_nodes = {rec.to_node for rec in cutoff_records if not rec.is_depot}
    used_agents = {rec.agent for rec in cutoff_records}
    return len(visited_service_nodes), len(used_agents)


def resolve_cutoff_specs(args) -> List[tuple[float, float]]:
    if args.time_cutoffs:
        clean = sorted(float(x) for x in args.time_cutoffs)
        specs: List[tuple[float, float]] = []
        prev = 0.0
        for cutoff in clean:
            specs.append((prev, cutoff))
            prev = cutoff
        return specs
    if args.time_cutoff >= 0:
        return [(0.0, float(args.time_cutoff))]
    return [(-1.0, -1.0)]


def get_new_node_mask(td, batch_idx: int, n_nodes: int, prev_cutoff: float, time_cutoff: float) -> torch.Tensor:
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    if prev_cutoff < 0 or time_cutoff < 0:
        return mask
    if "appear_time" not in td.keys():
        return mask

    appear_time = td["appear_time"][batch_idx].detach().cpu().float().view(-1)
    valid = appear_time.shape[0] >= n_nodes
    if not valid:
        return mask
    return (appear_time[:n_nodes] > prev_cutoff) & (appear_time[:n_nodes] <= time_cutoff)


def draw_rollout_large(
    env,
    history: List[LargeStepRecord],
    batch_idx: int,
    prev_time_cutoff: float,
    max_plot_steps: int,
    time_cutoff: float,
    coverage_count_mode: str,
    highlight_new_nodes: bool,
    new_node_size: float,
    new_node_edgewidth: float,
    show_step_labels: bool,
    show_node_labels: bool,
    show_agent_legend: bool,
    legend_agents: int,
    service_node_size: float,
    depot_node_size: float,
    route_linewidth: float,
    route_alpha: float,
    node_alpha: float,
    fig_width: float,
    fig_height: float,
    rasterized: bool,
):
    legend_fontsize = 11
    td = env.td_state
    coords = td["coords"][batch_idx].detach().cpu()
    n_nodes = coords.shape[0]
    depot_mask = _resolve_depot_mask(td, batch_idx=batch_idx, n_nodes=n_nodes)
    service_mask = ~depot_mask
    new_node_mask = get_new_node_mask(td, batch_idx=batch_idx, n_nodes=n_nodes, prev_cutoff=prev_time_cutoff, time_cutoff=time_cutoff)
    new_service_mask = new_node_mask & service_mask
    old_service_mask = service_mask & (~new_service_mask)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if old_service_mask.any():
        ax.scatter(
            coords[old_service_mask, 0],
            coords[old_service_mask, 1],
            c="#4C78A8",
            s=service_node_size,
            alpha=node_alpha,
            linewidths=0.0,
            label="service nodes",
            zorder=2,
            rasterized=rasterized,
        )
    if highlight_new_nodes and new_service_mask.any():
        ax.scatter(
            coords[new_service_mask, 0],
            coords[new_service_mask, 1],
            c="#F58518",
            s=new_node_size,
            alpha=0.98,
            linewidths=0.0,
            label="new nodes in interval",
            zorder=4,
            rasterized=rasterized,
        )
        ax.scatter(
            coords[new_service_mask, 0],
            coords[new_service_mask, 1],
            facecolors="none",
            edgecolors="#9C3D00",
            s=new_node_size * 1.9,
            alpha=0.95,
            linewidths=new_node_edgewidth,
            zorder=5,
            rasterized=rasterized,
        )
    if depot_mask.any():
        ax.scatter(
            coords[depot_mask, 0],
            coords[depot_mask, 1],
            c="#E45756",
            s=depot_node_size,
            alpha=0.95,
            marker="s",
            linewidths=0.0,
            label="depots",
            zorder=3,
            rasterized=rasterized,
        )

    time_history = filter_history_by_time(history, time_cutoff=time_cutoff)
    plot_history = time_history[: max(0, int(max_plot_steps))]
    segments_by_agent: Dict[int, List[List[List[float]]]] = defaultdict(list)
    used_agents = sorted({r.agent for r in plot_history})
    cmap = plt.get_cmap("tab20")
    for rec in plot_history:
        u, v = rec.from_node, rec.to_node
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            continue
        segments_by_agent[rec.agent].append(
            [
                [float(coords[u, 0]), float(coords[u, 1])],
                [float(coords[v, 0]), float(coords[v, 1])],
            ]
        )

    for agent, segments in segments_by_agent.items():
        lc = LineCollection(
            segments,
            colors=[cmap(agent % 20)],
            linewidths=route_linewidth,
            alpha=route_alpha,
            zorder=1,
            rasterized=rasterized,
        )
        ax.add_collection(lc)

    # if show_step_labels:
    #     label_every = max(1, len(plot_history) // 200)
    #     for rec in plot_history[::label_every]:
    #         u, v = rec.from_node, rec.to_node
    #         if not (0 <= u < n_nodes and 0 <= v < n_nodes):
    #             continue
    #         mx = (float(coords[u, 0]) + float(coords[v, 0])) * 0.5
    #         my = (float(coords[u, 1]) + float(coords[v, 1])) * 0.5
    #         ax.text(mx, my, str(rec.step), fontsize=5, color=cmap(rec.agent % 20), alpha=0.8, zorder=4)

    # if show_node_labels:
    #     label_every = max(1, n_nodes // 250)
    #     for i in range(0, n_nodes, label_every):
    #         ax.annotate(
    #             str(i),
    #             (float(coords[i, 0]), float(coords[i, 1])),
    #             textcoords="offset points",
    #             xytext=(2, 2),
    #             fontsize=5,
    #             color="#333333",
    #             alpha=0.8,
    #             zorder=5,
    #         )

    # title = f"Large Rollout | env={env.env_name} | batch={batch_idx}"
    # ax.set_title(title, fontsize=11, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.02)
    ax.grid(color="#d8d8d8", linestyle=":", linewidth=0.35, alpha=0.35)
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")

    legend_handles = []
    if old_service_mask.any():
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="None", markersize=7, markerfacecolor="#4C78A8", markeredgewidth=0.0, label="service nodes")
        )
    if highlight_new_nodes and new_service_mask.any():
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="None", markersize=8, markerfacecolor="#F58518", markeredgecolor="#9C3D00", markeredgewidth=new_node_edgewidth, label="new nodes")
        )
    if depot_mask.any():
        legend_handles.append(
            Line2D([0], [0], marker="s", linestyle="None", markersize=8, markerfacecolor="#E45756", markeredgewidth=0.0, label="depots")
        )

    if show_agent_legend and used_agents:
        legend_handles.extend([
            Line2D([0], [0], color=cmap(agent % 20), lw=1.2, label=f"agent {agent}")
            for agent in used_agents[: max(0, int(legend_agents))]
        ])
        if len(used_agents) > legend_agents:
            legend_handles.append(Line2D([0], [0], color="#666666", lw=0.0, label=f"... +{len(used_agents) - legend_agents} more"))

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
            frameon=True,
            facecolor="white",
            edgecolor="#dddddd",
            fontsize=legend_fontsize,
            ncol=len(legend_handles),
            columnspacing=1.2,
            handletextpad=0.5,
        )

    fig.subplots_adjust(top=0.92)

    return fig


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

    _td_final, history, total_ret = rollout_once_large(args, env, learner)
    b = int(args.batch_idx)
    ret_b = float(total_ret[b].item())
    ret_mean = float(total_ret.mean().item())
    not_visited = float(env.td_state["nodes"]["active_nodes_mask"][b].sum().item() - 1)
    used_agents = float(env.td_state["agents"]["visited_nodes"][b].sum(-1).gt(1).sum().item())
    covered_nodes, covered_agents = compute_coverage(history, time_cutoff=args.time_cutoff, count_mode=args.coverage_count_mode)

    print("=== Rollout Summary ===")
    print(f"policy_type={args.policy_type} model_kind={args.model_kind} deterministic={args.deterministic}")
    print(f"vrp_env={args.vrp_env} seed={args.seed} num_nodes={args.num_nodes} num_agents={args.num_agents}")
    print(f"steps={len(history)} return_batch[{b}]={ret_b:.4f} return_mean={ret_mean:.4f}")
    print(f"not_visited_nodes(batch={b})={not_visited:.1f} used_agents(batch={b})={used_agents:.1f}")
    if args.time_cutoff >= 0:
        print(
            f"time_cutoff={args.time_cutoff:g} coverage_mode={args.coverage_count_mode} "
            f"covered_nodes={covered_nodes} covered_agents={covered_agents}"
        )
    print(f"plotted_edges(batch={b})={min(len(history), max(0, int(args.max_plot_steps)))}")

    cutoff_specs = resolve_cutoff_specs(args)
    multi_cutoff = len(cutoff_specs) > 1
    figures = []
    for prev_cutoff, time_cutoff in cutoff_specs:
        current_covered_nodes, current_covered_agents = compute_coverage(
            history, time_cutoff=time_cutoff, count_mode=args.coverage_count_mode
        )
        new_node_mask = get_new_node_mask(
            env.td_state,
            batch_idx=b,
            n_nodes=int(env.td_state["coords"][b].shape[0]),
            prev_cutoff=prev_cutoff,
            time_cutoff=time_cutoff,
        )
        new_node_mask = new_node_mask & (~_resolve_depot_mask(env.td_state, batch_idx=b, n_nodes=int(env.td_state["coords"][b].shape[0])))
        if time_cutoff >= 0:
            print(
                f"snapshot time_cutoff={time_cutoff:g} prev_cutoff={prev_cutoff:g} "
                f"covered_nodes={current_covered_nodes} covered_agents={current_covered_agents} "
                f"new_nodes_in_interval={int(new_node_mask.sum().item())}"
            )

        fig = draw_rollout_large(
            env=env,
            history=history,
            batch_idx=b,
            prev_time_cutoff=prev_cutoff,
            max_plot_steps=args.max_plot_steps,
            time_cutoff=time_cutoff,
            coverage_count_mode=args.coverage_count_mode,
            highlight_new_nodes=args.highlight_new_nodes,
            new_node_size=args.new_node_size,
            new_node_edgewidth=args.new_node_edgewidth,
            show_step_labels=args.show_step_labels,
            show_node_labels=args.show_node_labels,
            show_agent_legend=args.show_agent_legend,
            legend_agents=args.legend_agents,
            service_node_size=args.service_node_size,
            depot_node_size=args.depot_node_size,
            route_linewidth=args.route_linewidth,
            route_alpha=args.route_alpha,
            node_alpha=args.node_alpha,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            rasterized=args.rasterized,
        )
        figures.append(fig)

        if args.save_fig:
            fig_path = prepare_output_path(args.save_fig)
            if multi_cutoff:
                suffix = "full" if time_cutoff < 0 else f"t{int(time_cutoff) if float(time_cutoff).is_integer() else str(time_cutoff).replace('.', 'p')}"
                fig_path = fig_path.with_name(f"{fig_path.stem}_{suffix}{fig_path.suffix}")
            fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
            print(f"saved figure: {fig_path}")

    if args.save_step_log:
        dump_step_log(args.save_step_log, history)
        print(f"saved step log: {prepare_output_path(args.save_step_log)}")

    if not args.no_show:
        plt.show()
    for fig in figures:
        plt.close(fig)


if __name__ == "__main__":
    main()
