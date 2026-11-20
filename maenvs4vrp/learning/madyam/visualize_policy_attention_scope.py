"""
Visualize one decision step with two probability heatmaps:

1) full-graph actor distribution: run the same actor on all currently decodable nodes
2) local-K actor distribution: run the actor on the actual local K-node observation

This makes it easy to check whether the local-K policy is already concentrating most
of its mass on nearby nodes that also dominate the full-graph distribution.

Example:
  python -m maenvs4vrp.learning.madyam.visualize_policy_attention_scope ^
    --vrp_env dtoptw_abs_second --policy_type model --model_kind abs ^
    --checkpoint runs/xxx.pt --target_step 20 --save_fig attention_scope.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from distutils.util import strtobool
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch

from maenvs4vrp.learning.madyam.visualize_policy_rollout import (
    build_env,
    build_model,
    configure_chinese_font,
    prepare_output_path,
    select_action,
    set_random_seed,
)


configure_chinese_font()


@dataclass
class DecisionSnapshot:
    step: int
    batch_idx: int
    agent_idx: int
    cur_node: int
    local_action: int
    local_to_node: int
    local_probs: torch.Tensor
    local_node_idx: torch.Tensor
    global_probs: torch.Tensor
    global_mask: torch.Tensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vrp_env", type=str, default="dtoptw_abs")
    p.add_argument("--policy_type", type=str, default="model", choices=["model"])
    p.add_argument("--model_kind", type=str, default="third", choices=["abs", "third"], help="obs is not supported here")
    p.add_argument("--checkpoint", type=str, default="/home/sjm/PycharmProjects/maenvs4vrp_yc/maenvs4vrp/learning/madyam/1000runs/models/latest_model_third_ac_dtoptw_abs_stime_1000n_100a_2026_03_09_20h02m.zip", help="required model checkpoint (.pt/.pth)")
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
    p.add_argument("--max_steps", type=int, default=35)
    p.add_argument("--max_flight_dist", type=float, default=300)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--done_feasible_check_freq", type=int, default=10)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--ff_hidden", type=int, default=512)
    p.add_argument("--critic_inducing_points", type=int, default=16)

    p.add_argument("--target_step", type=int, default=0, help="decision step to analyze")
    p.add_argument("--top_nodes", type=int, default=10, help="how many highest-probability nodes to print")
    p.add_argument("--point_size", type=float, default=60.0)
    p.add_argument("--save_fig", type=str, default="", help="optional png path")
    p.add_argument("--no_show", type=lambda x: bool(strtobool(x)), default=False)
    return p.parse_args()


def _masked_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _compute_actor_logits(args, learner, obs, use_global_view: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if use_global_view:
        node_static_obs = obs["critic_node_static_obs"]
        action_mask = obs["critic_nodes_mask"]
    else:
        node_static_obs = obs["node_static_obs"]
        action_mask = obs["action_mask"]

    if args.vrp_env == "dtoptw_abs_second":
        logits, _ = learner.actor(
            node_static_obs=node_static_obs,
            agent_obs=obs["agent_obs"],
            action_mask=action_mask,
            other_agents_obs=obs.get("other_agents_obs", None),
            agents_mask=obs.get("agents_mask", None),
        )
    else:
        logits, _ = learner.actor(
            node_static_obs=node_static_obs,
            agent_obs=obs["agent_obs"],
            action_mask=action_mask,
        )
    return logits, action_mask


def _scatter_local_probs_to_global(local_probs: torch.Tensor, local_node_idx: torch.Tensor, n_nodes: int) -> torch.Tensor:
    full = torch.zeros(n_nodes, dtype=local_probs.dtype, device=local_probs.device)
    full.scatter_add_(0, local_node_idx.to(torch.long), local_probs)
    return full


def _top_nodes(probs: torch.Tensor, count: int) -> List[tuple[int, float]]:
    k = min(max(1, int(count)), int(probs.numel()))
    top_p, top_idx = torch.topk(probs, k=k)
    return [(int(i.item()), float(p.item())) for p, i in zip(top_p, top_idx)]


def capture_snapshot(args, env, learner) -> DecisionSnapshot:
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

    model_cache_done = False
    step_i = 0
    while True:
        if td["done"].all():
            raise ValueError(f"Episode finished at step={step_i}, target_step={args.target_step} is out of range.")

        if args.model_kind in ("abs", "third") and not model_cache_done:
            learner.make_critic_cache_(td["observations"]["critic_node_static_obs"], with_grad=False)
            model_cache_done = True

        if step_i == int(args.target_step):
            obs = td["observations"]
            local_logits, _local_mask = _compute_actor_logits(args, learner, obs, use_global_view=False)
            global_logits, global_mask = _compute_actor_logits(args, learner, obs, use_global_view=True)

            local_probs = _masked_probs(local_logits[b])
            global_probs = _masked_probs(global_logits[b])
            action, _ = select_action(args, td, env, learner, model_cache_done=model_cache_done)

            local_action = int(action[b].item())
            local_node_idx = obs["local_node_idx"][b].detach().clone()
            local_to_node = int(local_node_idx[local_action].item())

            return DecisionSnapshot(
                step=step_i,
                batch_idx=b,
                agent_idx=int(env.td_state["cur_agent_idx"][b, 0].item()),
                cur_node=int(env.td_state["cur_agent"]["cur_node"][b, 0].item()),
                local_action=local_action,
                local_to_node=local_to_node,
                local_probs=local_probs.detach().cpu(),
                local_node_idx=local_node_idx.detach().cpu(),
                global_probs=global_probs.detach().cpu(),
                global_mask=global_mask[b].detach().cpu(),
            )

        action, model_cache_done = select_action(args, td, env, learner, model_cache_done=model_cache_done)
        td["action"] = action.unsqueeze(1)
        td = env.step(td)
        step_i += 1


def _draw_prob_panel(
    ax,
    coords: torch.Tensor,
    probs: torch.Tensor,
    title: str,
    cur_node: int,
    local_nodes: torch.Tensor,
    chosen_node: int,
    norm: Normalize,
    point_size: float,
):
    local_unique = torch.unique(local_nodes.to(torch.long))
    local_mask = torch.zeros(coords.shape[0], dtype=torch.bool)
    local_mask[local_unique] = True

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=probs.numpy(),
        s=point_size,
        cmap="YlOrRd",
        norm=norm,
        alpha=0.9,
        linewidths=0.0,
        zorder=2,
    )
    ax.scatter(
        coords[local_mask, 0],
        coords[local_mask, 1],
        facecolors="none",
        edgecolors="#205072",
        s=point_size * 1.7,
        linewidths=1.0,
        zorder=3,
    )
    ax.scatter(
        [float(coords[cur_node, 0])],
        [float(coords[cur_node, 1])],
        c="#2a9d8f",
        s=point_size * 2.4,
        marker="P",
        linewidths=0.0,
        zorder=4,
    )
    ax.scatter(
        [float(coords[chosen_node, 0])],
        [float(coords[chosen_node, 1])],
        c="#264653",
        s=point_size * 2.2,
        marker="X",
        linewidths=0.0,
        zorder=5,
    )
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#d9d9d9", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_snapshot(env, snapshot: DecisionSnapshot, point_size: float):
    coords = env.td_state["coords"][snapshot.batch_idx].detach().cpu()
    n_nodes = coords.shape[0]
    local_full_probs = _scatter_local_probs_to_global(snapshot.local_probs, snapshot.local_node_idx, n_nodes).cpu()

    vmax = float(max(snapshot.global_probs.max().item(), local_full_probs.max().item()))
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-8))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    fig.patch.set_facecolor("white")

    _draw_prob_panel(
        axes[0],
        coords=coords,
        probs=snapshot.global_probs,
        title="Full-Graph Actor Probability",
        cur_node=snapshot.cur_node,
        local_nodes=snapshot.local_node_idx,
        chosen_node=snapshot.local_to_node,
        norm=norm,
        point_size=point_size,
    )
    _draw_prob_panel(
        axes[1],
        coords=coords,
        probs=local_full_probs,
        title=f"Local-K Actor Probability (K={len(snapshot.local_node_idx)})",
        cur_node=snapshot.cur_node,
        local_nodes=snapshot.local_node_idx,
        chosen_node=snapshot.local_to_node,
        norm=norm,
        point_size=point_size,
    )

    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label("action probability")

    fig.subplots_adjust(wspace=0.08, right=0.9)
    return fig, local_full_probs


def main():
    args = parse_args()
    args.device = torch.device(args.device)
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for this visualization.")

    set_random_seed(args.seed, args.torch_deterministic)

    env = build_env(args)
    learner = build_model(args, env)
    snapshot = capture_snapshot(args, env, learner)

    fig, local_full_probs = draw_snapshot(env, snapshot, point_size=args.point_size)

    local_unique = torch.unique(snapshot.local_node_idx.to(torch.long))
    local_mass = float(snapshot.global_probs[local_unique].sum().item())
    top_full = _top_nodes(snapshot.global_probs, args.top_nodes)
    top_local = _top_nodes(local_full_probs, args.top_nodes)
    top_full_in_local = sum(1 for idx, _ in top_full if idx in set(int(x.item()) for x in local_unique))

    print("=== Attention Scope Snapshot ===")
    print(
        f"step={snapshot.step} batch={snapshot.batch_idx} agent={snapshot.agent_idx} "
        f"cur_node={snapshot.cur_node} chosen_local_idx={snapshot.local_action} chosen_node={snapshot.local_to_node}"
    )
    print(f"local_k_unique={len(local_unique)} global_mass_on_local_k={local_mass:.4f}")
    print(f"top_full_nodes_in_local_k={top_full_in_local}/{len(top_full)}")
    print("top full-graph nodes:")
    for idx, prob in top_full:
        print(f"  node={idx:04d} prob={prob:.6f}")
    print("top local-K nodes:")
    for idx, prob in top_local:
        print(f"  node={idx:04d} prob={prob:.6f}")

    if args.save_fig:
        fig_path = prepare_output_path(args.save_fig)
        fig.savefig(fig_path, dpi=180, bbox_inches="tight")
        print(f"saved figure: {fig_path}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
