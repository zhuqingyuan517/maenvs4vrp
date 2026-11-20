"""
Visualize one rollout for random/model policy to debug environment + policy behavior.

Usage examples:
  1) Random policy on dtoptw_abs:
     python -m maenvs4vrp.learning.madyam.visualize_policy_rollout --vrp_env dtoptw_abs --policy_type random

  2) ABS model policy with checkpoint:
     python -m maenvs4vrp.learning.madyam.visualize_policy_rollout --vrp_env dtoptw_abs --policy_type model --model_kind abs --checkpoint runs/xxx.pt

  3) OLD OBS model policy on dtoptw_full:
     python -m maenvs4vrp.learning.madyam.visualize_policy_rollout --vrp_env dtoptw_full --policy_type model --model_kind obs --checkpoint runs/xxx.pt
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import random
import warnings
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import torch


def configure_chinese_font() -> Optional[str]:
    font_paths = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    font_names = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "PingFang SC",
        "Arial Unicode MS",
    ]

    selected_name = None
    for path in font_paths:
        if path.exists():
            try:
                fm.fontManager.addfont(str(path))
                selected_name = fm.FontProperties(fname=str(path)).get_name()
                break
            except Exception:
                continue

    if selected_name is None:
        available = {f.name for f in fm.fontManager.ttflist}
        for name in font_names:
            if name in available:
                selected_name = name
                break

    if selected_name is None:
        warnings.warn(
            "No Chinese-capable font found. Install Noto Sans CJK SC or WenQuanYi on the Linux server.",
            RuntimeWarning,
            stacklevel=2,
        )
        selected_name = "DejaVu Sans"

    plt.rcParams["font.family"] = selected_name
    plt.rcParams["font.sans-serif"] = [selected_name, *font_names, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected_name


configure_chinese_font()


@dataclass
class StepRecord:
    step: int
    agent: int
    from_node: int
    to_node: int
    action_raw: int
    reward: float
    feasible_cnt: int
    local_mode: bool


def prepare_output_path(path: str) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


def set_random_seed(seed: int, torch_deterministic: bool):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vrp_env", type=str, default="dtoptw_abs_second")
    p.add_argument("--policy_type", type=str, default="random", choices=["random", "model", "greedy"])
    p.add_argument("--model_kind", type=str, default="abs", choices=["abs", "third", "obs"], help="used only when policy_type=model")
    p.add_argument("--checkpoint", type=str, default="/home/sjm/PycharmProjects/maenvs4vrp_yc/50-8runs/models/best_model_ata_ac_dtoptw_abs_stime_51n_8a_2026_03_02_10h20m.zip", help="optional model checkpoint (.pt/.pth)")
    p.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=True)

    p.add_argument("--num_agents", type=int, default=8)
    p.add_argument("--num_nodes", type=int, default=51)
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--batch_idx", type=int, default=0, help="which batch item to visualize")
    p.add_argument("--seed", type=int, default=2297)
    p.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--local_k_nodes", type=int, default=30)
    p.add_argument("--local_view_mode", type=str, default="nearest", choices=["nearest", "random"])
    p.add_argument("--profits", type=str, default="constant", choices=["distance", "constant", "uniform"])
    p.add_argument("--horizon", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=70)
    p.add_argument("--max_flight_dist", type=float, default=300)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--done_feasible_check_freq", type=int, default=10)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--ff_hidden", type=int, default=512)
    p.add_argument("--critic_inducing_points", type=int, default=16)

    p.add_argument("--max_plot_steps", type=int, default=200, help="limit edges drawn on figure")
    p.add_argument("--show_step_labels", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--save_fig", type=str, default="", help="optional png path")
    p.add_argument("--save_step_log", type=str, default="", help="optional txt path")
    p.add_argument("--no_show", type=lambda x: bool(strtobool(x)), default=False)
    return p.parse_args()


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
    if args.vrp_env in ("dtoptw", "dtoptw_abs", "dtoptw_abs_second"):
        obs_kwargs["local_view"] = {"k_nodes": args.local_k_nodes, "mode": args.local_view_mode, "include_depot": True}
    if args.policy_type == "model" and args.model_kind == "abs":
        obs_kwargs["return_critic_obs"] = True
    try:
        observations = obs_mod.Observations(**obs_kwargs)
    except TypeError:
        observations = obs_mod.Observations()

    generator_module_name = f"maenvs4vrp.environments.{args.vrp_env}.instances_generator"
    gen_mod = importlib.import_module(generator_module_name)
    generator = gen_mod.InstanceGenerator(device=args.device)

    env_reward_module_name = f"maenvs4vrp.environments.{args.vrp_env}.env_agent_reward"
    rew_mod = importlib.import_module(env_reward_module_name)
    reward_evaluator = rew_mod.DenseReward()

    environment_module_name = f"maenvs4vrp.environments.{args.vrp_env}.env"
    env_mod = importlib.import_module(environment_module_name)
    env = env_mod.Environment(
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
    return env


def build_model(args, env):
    if args.policy_type != "model":
        return None

    if args.model_kind in ("abs", "third"):
        if args.vrp_env == "dtoptw_abs_second":
            from maenvs4vrp.learning.madyam.policy_net_ma_ac_abs_ata import Learner as AtaLearner

            learner = AtaLearner(
                nodes_stat_dim=env.obs_builder.get_nodes_static_feat_dim(),
                agent_dim=env.obs_builder.get_agent_feat_dim(),
                other_agents_dim=env.obs_builder.get_other_agents_feat_dim(),
                embed_dim=args.embed_dim,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                feed_forward_hidden=args.ff_hidden,
            ).to(args.device)
        else:
            if args.model_kind == "third":
                from maenvs4vrp.learning.madyam.policy_net_ma_ac_abs_third import Learner as ThirdLearner

                learner = ThirdLearner(
                    nodes_stat_dim=env.obs_builder.get_nodes_static_feat_dim(),
                    agent_dim=env.obs_builder.get_agent_feat_dim(),
                    embed_dim=args.embed_dim,
                    n_heads=args.n_heads,
                    n_layers=args.n_layers,
                    feed_forward_hidden=args.ff_hidden,
                    n_inducing_points=args.critic_inducing_points,
                ).to(args.device)
            else:
                from maenvs4vrp.learning.madyam.policy_net_ma_ac_abs import Learner as AbsLearner

                learner = AbsLearner(
                    nodes_stat_dim=env.obs_builder.get_nodes_static_feat_dim(),
                    agent_dim=env.obs_builder.get_agent_feat_dim(),
                    embed_dim=args.embed_dim,
                    n_heads=args.n_heads,
                    n_layers=args.n_layers,
                    feed_forward_hidden=args.ff_hidden,
                ).to(args.device)
    else:
        from maenvs4vrp.learning.madyam.policy_net_ma_ac_old import Learner as ObsLearner

        learner = ObsLearner(
            nodes_stat_obs_dim=env.obs_builder.get_nodes_static_feat_dim(),
            nodes_dyn_obs_dim=env.obs_builder.get_nodes_dynamic_feat_dim(),
            agent_obs_dim=env.obs_builder.get_agent_feat_dim(),
            agents_obs_dim=env.obs_builder.get_other_agents_feat_dim(),
            global_obs_dim=env.obs_builder.get_global_feat_dim(),
            embed_dim=args.embed_dim,
        ).to(args.device)

    if args.checkpoint:
        raw = torch.load(args.checkpoint, map_location=args.device)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        missing, unexpected = learner.load_state_dict(raw, strict=False)
        print(f"[model] loaded checkpoint={args.checkpoint}")
        print(f"[model] missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[model] checkpoint not provided, using random-initialized network")

    learner.eval()
    return learner


def build_action_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def pick_random_action(action_mask: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    # action_mask: [B, K] or [B, N], True means feasible
    if (~action_mask.any(dim=-1)).any():
        action_mask = action_mask.clone()
        bad = ~action_mask.any(dim=-1)
        action_mask[bad, 0] = True
    return torch.multinomial(action_mask.float(), 1, generator=rng).squeeze(1)


def dedupe_local_action_mask(obs, action_mask: torch.Tensor) -> torch.Tensor:
    if "local_node_idx" not in obs.keys():
        return action_mask

    local_node_idx = obs["local_node_idx"]
    dedup_mask = action_mask.clone()
    bsz, width = local_node_idx.shape
    for b in range(bsz):
        seen = set()
        for j in range(width):
            if not bool(action_mask[b, j].item()):
                continue
            node = int(local_node_idx[b, j].item())
            if node in seen:
                dedup_mask[b, j] = False
            else:
                seen.add(node)

    if (~dedup_mask.any(dim=-1)).any():
        return action_mask
    return dedup_mask


def prefer_non_depot_action_mask(obs, env, action_mask: torch.Tensor) -> torch.Tensor:
    if "local_node_idx" in obs.keys():
        cand_nodes = obs["local_node_idx"]
    else:
        n = action_mask.shape[1]
        cand_nodes = torch.arange(n, device=action_mask.device).unsqueeze(0).expand(action_mask.shape[0], n)

    if "is_depot" in env.td_state.keys():
        depot_mask = env.td_state["is_depot"].gather(1, cand_nodes).to(torch.bool)
    else:
        depot_mask = cand_nodes.eq(0)

    non_depot_feasible = action_mask & (~depot_mask)
    has_non_depot = non_depot_feasible.any(dim=-1, keepdim=True)
    filtered_mask = torch.where(has_non_depot, non_depot_feasible, action_mask)

    if (~filtered_mask.any(dim=-1)).any():
        return action_mask
    return filtered_mask


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


def select_action(args, td, env, learner, model_cache_done: bool, action_rng: Optional[torch.Generator] = None):
    obs = td["observations"]

    if args.policy_type == "random":
        action_mask = dedupe_local_action_mask(obs, obs["action_mask"])
        action_mask = prefer_non_depot_action_mask(obs, env, action_mask)
        if action_rng is None:
            raise ValueError("Random policy requires a seeded action generator for reproducible sampling.")
        action = pick_random_action(action_mask, action_rng)
        return action, model_cache_done
    if args.policy_type == "greedy":
        action = select_greedy_nearest_action(td, env)
        return action, model_cache_done

    if args.model_kind in ("abs", "third"):
        if args.vrp_env == "dtoptw_abs_second":
            if not model_cache_done:
                learner.make_critic_cache_(obs["critic_node_static_obs"], with_grad=False)
                model_cache_done = True
            action, _logp, _ent, _value = learner.get_action_and_value(
                node_static_obs=obs["node_static_obs"],
                agent_obs=obs["agent_obs"],
                action_mask=obs["action_mask"],
                critic_node_static_obs=obs["critic_node_static_obs"],
                critic_nodes_mask=obs["critic_nodes_mask"],
                other_agents_obs=obs.get("other_agents_obs", None),
                agents_mask=obs.get("agents_mask", None),
                deterministic=args.deterministic,
            )
            return action, model_cache_done

        if not model_cache_done:
            learner.make_critic_cache_(obs["critic_node_static_obs"], with_grad=False)
            model_cache_done = True
        action, _logp, _ent, _value = learner.get_action_and_value(
            node_static_obs=obs["node_static_obs"],
            agent_obs=obs["agent_obs"],
            action_mask=obs["action_mask"],
            critic_node_static_obs=obs["critic_node_static_obs"],
            critic_nodes_mask=obs["critic_nodes_mask"],
            deterministic=args.deterministic,
        )
        return action, model_cache_done

    # model_kind == "obs"
    if not model_cache_done:
        learner.policy.make_cache_(nodes_obs=obs["node_static_obs"])
        model_cache_done = True
    action, _logp, _ent, _val = learner.get_action_and_logs(
        nodes_obs=obs["node_dynamic_obs"],
        self_obs=obs["agent_obs"],
        agents_obs=obs["other_agents_obs"],
        agents_mask=obs["agents_mask"],
        global_obs=obs["global_obs"],
        action_mask=obs["action_mask"],
        deterministic=args.deterministic,
    )
    return action, model_cache_done


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
    history: List[StepRecord] = []
    model_cache_done = False
    action_rng = build_action_generator(args.seed, args.device)
    step_i = 0

    while not td["done"].all():
        cur_agent = int(env.td_state["cur_agent_idx"][b, 0].item())
        cur_node = int(env.td_state["cur_agent"]["cur_node"][b, 0].item())
        global_feasible_cnt = int(env.td_state["cur_agent"]["action_mask"][b].sum().item())

        action, model_cache_done = select_action(
            args,
            td,
            env,
            learner,
            model_cache_done=model_cache_done,
            action_rng=action_rng,
        )
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
                action_raw=action_raw_b,
                reward=float(step_r[b].item()),
                feasible_cnt=global_feasible_cnt,
                local_mode=local_mode,
            )
        )
        step_i += 1

    return td, history, total_ret


def draw_rollout(env, history: List[StepRecord], batch_idx: int, max_plot_steps: int, show_step_labels: bool):
    td = env.td_state
    coords = td["coords"][batch_idx].detach().cpu()
    n_nodes = coords.shape[0]

    depot_mask = torch.zeros(n_nodes, dtype=torch.bool)
    if "is_depot" in td.keys():
        depot_mask = td["is_depot"][batch_idx].detach().cpu().to(torch.bool)
    elif "depot_idx" in td.keys():
        di = td["depot_idx"][batch_idx]
        if di.dim() == 0:
            depot_mask[int(di.item())] = True
        else:
            for d in di.view(-1).tolist():
                if 0 <= int(d) < n_nodes:
                    depot_mask[int(d)] = True
    else:
        depot_mask[0] = True

    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    fig.patch.set_facecolor("#f7f5ef")
    ax.set_facecolor("#fffdf8")
    service_mask = ~depot_mask
    if service_mask.any():
        ax.scatter(
            coords[service_mask, 0],
            coords[service_mask, 1],
            c="#2a6f97",
            s=34,
            edgecolors="white",
            linewidths=0.5,
            label="服务节点",
            zorder=3,
        )
    if depot_mask.any():
        ax.scatter(
            coords[depot_mask, 0],
            coords[depot_mask, 1],
            c="#c44536",
            s=120,
            marker="s",
            edgecolors="white",
            linewidths=0.8,
            label="仓库",
            zorder=4,
        )

    max_agent = max((r.agent for r in history), default=0)
    used_agents = sorted({r.agent for r in history[: max(0, int(max_plot_steps))]})
    cmap = plt.get_cmap("tab20")
    for rec in history[: max(0, int(max_plot_steps))]:
        u, v = rec.from_node, rec.to_node
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            continue
        col = cmap(rec.agent % 20)
        x0, y0 = float(coords[u, 0]), float(coords[u, 1])
        x1, y1 = float(coords[v, 0]), float(coords[v, 1])
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.7, alpha=0.9, mutation_scale=11.0),
        )
        if show_step_labels:
            mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            ax.text(
                mx,
                my,
                str(rec.step),
                color=col,
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7),
                zorder=5,
            )

    for i in range(n_nodes):
        ax.annotate(
            str(i),
            (float(coords[i, 0]), float(coords[i, 1])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#2f2f2f",
            zorder=6,
        )

    title = f"策略轨迹图 | 环境={env.env_name} | batch={batch_idx}"
    subtitle = f"步骤数={len(history)} | 启用智能体={len(used_agents)} | 最大智能体编号={max_agent}"
    # ax.set_title(f"{title}\n{subtitle}", fontsize=13, pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#d9d3c7", linestyle="--", linewidth=0.7, alpha=0.55)
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")

    fig.subplots_adjust(right=0.78, bottom=0.16)

    node_legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        edgecolor="#d8d2c4",
    )
    agent_handles = [
        Line2D([0], [0], color=cmap(agent % 20), lw=2.0, label=f"智能体 {agent}")
        for agent in used_agents[:10]
    ]
    if len(used_agents) > 10:
        agent_handles.append(Line2D([0], [0], color="#666666", lw=0, label=f"... 另有 {len(used_agents) - 10} 个"))
    if agent_handles:
        agent_legend = ax.legend(
            handles=agent_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            borderaxespad=0.0,
            frameon=True,
            facecolor="white",
            edgecolor="#d8d2c4",
            ncol=len(agent_handles),
            columnspacing=1.2,
            handlelength=2.2,
            title="路径颜色",
        )
        ax.add_artist(node_legend)
        ax.add_artist(agent_legend)
    else:
        ax.add_artist(node_legend)
    return fig


def dump_step_log(path: str, records: List[StepRecord]):
    output_path = prepare_output_path(path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("step,agent,from_node,to_node,action_raw,reward,feasible_cnt,local_mode\n")
        for r in records:
            f.write(
                f"{r.step},{r.agent},{r.from_node},{r.to_node},{r.action_raw},{r.reward:.6f},"
                f"{r.feasible_cnt},{int(r.local_mode)}\n"
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

    td_final, history, total_ret = rollout_once(args, env, learner)
    b = int(args.batch_idx)
    ret_b = float(total_ret[b].item())
    ret_mean = float(total_ret.mean().item())
    not_visited = float(env.td_state["nodes"]["active_nodes_mask"][b].sum().item() - 1)
    used_agents = float(env.td_state["agents"]["visited_nodes"][b].sum(-1).gt(1).sum().item())

    print("=== Rollout Summary ===")
    print(f"policy_type={args.policy_type} model_kind={args.model_kind} deterministic={args.deterministic}")
    print(f"vrp_env={args.vrp_env} seed={args.seed} num_nodes={args.num_nodes} num_agents={args.num_agents}")
    print(f"steps={len(history)} return_batch[{b}]={ret_b:.4f} return_mean={ret_mean:.4f}")
    print(f"not_visited_nodes(batch={b})={not_visited:.1f} used_agents(batch={b})={used_agents:.1f}")

    print("\n=== First 20 Steps ===")
    for r in history[:20]:
        act_repr = f"local_idx={r.action_raw}" if r.local_mode else f"global_idx={r.action_raw}"
        print(
            f"step={r.step:03d} agent={r.agent:03d} {r.from_node:04d}->{r.to_node:04d} "
            f"{act_repr} feasible={r.feasible_cnt:04d} reward={r.reward:+.4f}"
        )

    print(f"\nrollout_edges(batch={b})={len(history)}")

    fig = draw_rollout(
        env=env,
        history=history,
        batch_idx=b,
        max_plot_steps=args.max_plot_steps,
        show_step_labels=args.show_step_labels,
    )

    if args.save_fig:
        fig_path = prepare_output_path(args.save_fig)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"saved figure: {fig_path}")
    if args.save_step_log:
        dump_step_log(args.save_step_log, history)
        print(f"saved step log: {prepare_output_path(args.save_step_log)}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
