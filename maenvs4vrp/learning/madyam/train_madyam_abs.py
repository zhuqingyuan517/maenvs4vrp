"""
Train dtoptw_abs with an absolute-feature actor-critic (encoder-decoder glimpse).

This script is intentionally independent from the existing train_madyam.py and does not
modify existing training logic.
"""

import argparse
import importlib
import os
import random
import time
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.checkpoint import checkpoint
from tqdm import trange
from distutils.util import strtobool
from time import perf_counter

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # optional dependency
    SummaryWriter = None

from maenvs4vrp.learning.madyam.layers._loss import reinforce_mask_loss
from maenvs4vrp.learning.baseline_protocol import resolve_max_steps
from maenvs4vrp.learning.madyam.policy_net_ma_ac_abs import Learner as AbsLearner
from maenvs4vrp.learning.madyam.policy_net_ma_ac_abs_ata import Learner as AtaLearner
from maenvs4vrp.learning.madyam.policy_net_ma_ac_abs_third import Learner as ThirdLearner


class _NullWriter:
    def add_text(self, *_args, **_kwargs):
        return None

    def add_scalar(self, *_args, **_kwargs):
        return None

    def close(self):
        return None


def set_random_seed(seed: int, torch_deterministic: bool):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def setup_distributed(device_arg: str):
    """Initialize optional torchrun data parallelism without changing single-GPU use."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return False, rank, world_size, torch.device(device_arg)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        dist.init_process_group(backend="gloo")
        device = torch.device("cpu")
    return True, dist.get_rank(), dist.get_world_size(), device


def all_reduce_mean_gradients(model, world_size: int):
    """Average per-rank mean gradients to match a single global mini-batch."""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vrp_env", type=str, default="dtoptw_abs")
    p.add_argument(
        "--model_variant",
        type=str,
        default="third",
        choices=["abs", "ata", "third"],
        help="policy model variant: abs (baseline), ata (agent-to-agents attention actor), or third (inducing-point critic)",
    )
    p.add_argument("--num_agents", type=int, default=50)
    p.add_argument("--num_nodes", type=int, default=500)
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="global training batch size; under torchrun it is divided evenly across ranks",
    )
    p.add_argument("--seed", type=int, default=2297)
    p.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--tf32",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="enable TF32 matmul on CUDA (faster, small numeric drift)",
    )

    p.add_argument("--local_k_nodes", type=int, default=30)
    p.add_argument("--local_view_mode", type=str, default="nearest", choices=["nearest", "random"])
    p.add_argument(
        "--critic_mask_unappeared",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="if True, critic decode mask excludes not-yet-appeared nodes; if False, critic can attend all active nodes",
    )
    p.add_argument(
        "--critic_encode_unappeared",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="if True, critic encoder sees all nodes; if False, not-yet-appeared node features are zeroed before critic cache build",
    )
    p.add_argument("--profits", type=str, default="constant", choices=["distance", "constant", "uniform"])
    p.add_argument("--horizon", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=0, help="0 resolves to 600/1200 by node scale")
    p.add_argument("--max_flight_dist", type=float, default=300)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--done_feasible_check_freq", type=int, default=100, help="frequency for expensive global-feasible termination check (higher is faster)")

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--ff_hidden", type=int, default=512)
    p.add_argument("--critic_inducing_points", type=int, default=32)
    p.add_argument("--epoch_count", type=int, default=100)
    p.add_argument("--iter_count", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=2.0)
    p.add_argument(
        "--critic_cache_with_grad",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="if True, build critic cache with autograd so critic encoder is trained (higher memory)",
    )
    p.add_argument(
        "--grad_checkpoint",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="recompute each decision-step actor/critic graph during backward to reduce activation memory",
    )

    # Eval: no Solomon/benchmark dependency; evaluate on fixed random seeds.
    p.add_argument("--eval_every", type=int, default=10, help="run eval every N epochs (0 disables)")
    p.add_argument("--eval_episodes", type=int, default=10, help="number of random eval episodes")
    p.add_argument("--eval_seed", type=int, default=12345, help="base seed for random eval instances")
    p.add_argument("--eval_batch_size", type=int, default=16, help="batch size for eval env")
    p.add_argument("--profile_steps", type=int, default=0, help="if >0, print per-section timings for first N env steps and exit")
    p.add_argument("--log_path", type=str, default="500runs", help="tensorboard log root")
    p.add_argument("--run_name", type=str, default="", help="override run name (default is auto)")
    p.add_argument("--save_models", type=lambda x: bool(strtobool(x)), default=True, help="save latest and best checkpoints")
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
    obs_kwargs = dict(
        local_view={"k_nodes": args.local_k_nodes, "mode": args.local_view_mode, "include_depot": True},
        return_critic_obs=True,
        critic_mask_unappeared=args.critic_mask_unappeared,
        critic_encode_unappeared=args.critic_encode_unappeared,
    )
    try:
        observations = obs_mod.Observations(**obs_kwargs)
    except TypeError:
        # Global-action observation variants expose only node_static_obs and
        # use the full action mask, so they do not accept local-view options.
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


def build_eval_env(args):
    class _EvalArgs:
        pass

    ea = _EvalArgs()
    for k, v in vars(args).items():
        setattr(ea, k, v)
    ea.batch_size = int(getattr(args, "eval_batch_size", args.batch_size))
    return build_env(ea)


def _policy_value_forward(
    learner,
    node_static_obs,
    agent_obs,
    action_mask,
    critic_node_static_obs,
    critic_nodes_mask,
    use_agent_context,
    other_agents_obs,
    agents_mask,
):
    if use_agent_context:
        logits, _glimpse = learner.actor(
            node_static_obs,
            agent_obs,
            action_mask,
            other_agents_obs=other_agents_obs,
            agents_mask=agents_mask,
        )
    else:
        logits, _glimpse = learner.actor(node_static_obs, agent_obs, action_mask)
    value = learner.critic(critic_node_static_obs, agent_obs, critic_nodes_mask)
    return logits, value


def _get_action_and_value(args, learner, obs, deterministic: bool):
    critic_node_static_obs = obs.get("critic_node_static_obs", obs["node_static_obs"])
    # The full-attention and inducing-point critics both summarize the complete
    # static instance.  Using identical all-node visibility isolates the critic
    # architecture from active/future-node masking differences.
    critic_nodes_mask = torch.ones(
        critic_node_static_obs.shape[:2],
        dtype=torch.bool,
        device=critic_node_static_obs.device,
    )
    use_agent_context = str(getattr(args, "model_variant", "abs")).lower() == "ata"
    other_agents_obs = None
    agents_mask = None
    if use_agent_context:
        keys = set(obs.keys())
        other_agents_obs = obs["other_agents_obs"] if "other_agents_obs" in keys else None
        agents_mask = obs["agents_mask"] if "agents_mask" in keys else None

    forward_args = (
        learner,
        obs["node_static_obs"],
        obs["agent_obs"],
        obs["action_mask"],
        critic_node_static_obs,
        critic_nodes_mask,
        use_agent_context,
        other_agents_obs,
        agents_mask,
    )
    if bool(getattr(args, "grad_checkpoint", False)) and torch.is_grad_enabled():
        logits, value = checkpoint(
            _policy_value_forward,
            *forward_args,
            use_reentrant=False,
        )
    else:
        logits, value = _policy_value_forward(*forward_args)

    dist = torch.distributions.Categorical(logits=logits)
    action = dist.mode if deterministic else dist.sample()
    logp = dist.log_prob(action)
    ent = dist.entropy()
    return action, logp, ent, value


def eval_epoch(args, env, learner, ep: int, writer: Any):
    learner.eval()

    total_reward = []
    not_visited_nodes = []
    number_used_agents = []

    with torch.no_grad():
        for k in range(int(getattr(args, "eval_episodes", 10))):
            td = env.reset(
                num_agents=args.num_agents,
                num_nodes=args.num_nodes,
                profits=args.profits,
                seed=int(getattr(args, "eval_seed", 0)) + ep * 1000 + k,
                sample_type="random",
                max_steps=args.max_steps,
                max_flight_dist=args.max_flight_dist,
                allow_depot_refuel=args.allow_depot_refuel,
                forbid_depot_self_loop=args.forbid_depot_self_loop,
                horizon=args.horizon,
                done_feasible_check_freq=args.done_feasible_check_freq,
            )

            obs = td["observations"]
            critic_node_static_obs = obs.get("critic_node_static_obs", obs["node_static_obs"])
            learner.make_critic_cache_(critic_node_static_obs, with_grad=False)

            f_reward = []
            while not td["done"].all():
                obs = td["observations"]
                action, _logp, _ent, _value = _get_action_and_value(args, learner, obs, deterministic=True)
                td["action"] = action.unsqueeze(1)  # local index (will be mapped by env.step)
                td = env.step(td)
                f_reward.append(td["reward"] + td["penalty"])

            total_reward.append(torch.cat(f_reward, dim=1).sum(-1))
            not_visited_nodes.append(env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1)
            number_used_agents.append(env.td_state["agents"]["visited_nodes"].sum(-1).gt(1).sum(-1).float())

        total_reward = torch.cat(total_reward).mean()
        not_visited_nodes = torch.cat(not_visited_nodes).mean()
        number_used_agents = torch.cat(number_used_agents).mean()

    writer.add_scalar("eval/episodic_return", float(total_reward.item()), ep)
    writer.add_scalar("eval/episodic_not_visited_nodes", float(not_visited_nodes.item()), ep)
    writer.add_scalar("eval/episodic_number_used_agents", float(number_used_agents.item()), ep)

    return total_reward.item(), not_visited_nodes.item(), number_used_agents.item()


def train_epoch(args, env, learner, optim, ep: int, writer: Any, global_step_base: int, rank: int = 0):
    learner.train()
    ep_loss = 0.0
    ep_ret = 0.0

    for it in trange(args.iter_count, desc=f"Ep {ep+1}/{args.epoch_count}", disable=rank != 0):
        seed = args.seed + (
            (ep * args.iter_count + it) * int(getattr(args, "world_size", 1)) + rank
        )
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

        # Cache critic encoder projections once per episode (absolute node features).
        critic_node_static_obs = td["observations"].get(
            "critic_node_static_obs",
            td["observations"]["node_static_obs"],
        )
        learner.make_critic_cache_(
            critic_node_static_obs,
            with_grad=bool(getattr(args, "critic_cache_with_grad", True)),
        )

        logps = []
        rewards = []
        values = []
        step_mask = []

        n_profile = int(getattr(args, "profile_steps", 0) or 0)
        t_obs = 0.0
        t_model = 0.0
        t_step = 0.0
        step_i = 0
        while not td["done"].all():
            if n_profile and ep == 0 and it == 0 and step_i >= n_profile:
                break

            t0 = perf_counter()
            obs = td["observations"]
            t_obs += perf_counter() - t0

            t0 = perf_counter()
            action, logp, ent, value = _get_action_and_value(args, learner, obs, deterministic=False)
            t_model += perf_counter() - t0

            if n_profile and ep == 0 and it == 0 and step_i == 0:
                # Actor action space is local K, not global N.
                print(
                    f"[profile] actor_K={obs['node_static_obs'].shape[1]} "
                    f"critic_N={obs.get('critic_node_static_obs', obs['node_static_obs']).shape[1]} "
                    f"action_max={int(action.max().item())} (should be <K)"
                )

            td["action"] = action.unsqueeze(1)  # local index (will be mapped by env.step)
            t0 = perf_counter()
            with torch.no_grad():
                td = env.step(td)
            t_step += perf_counter() - t0

            logps.append(logp.unsqueeze(1))
            values.append(value)
            step_r = td["reward"] + td["penalty"]
            if step_r.dim() == 1:
                step_r = step_r.unsqueeze(1)
            rewards.append(step_r)
            step_mask.append((~td["done"]).unsqueeze(1))
            step_i += 1

        if n_profile and ep == 0 and it == 0:
            denom = max(1, step_i)
            print(
                f"[profile] steps={step_i} "
                f"obs={t_obs:.3f}s ({t_obs/denom*1000:.2f}ms/step) "
                f"model={t_model:.3f}s ({t_model/denom*1000:.2f}ms/step) "
                f"env_step={t_step:.3f}s ({t_step/denom*1000:.2f}ms/step)"
            )
            return 0.0, 0.0

        loss = reinforce_mask_loss(logps, rewards, baseline=values, step_mask=step_mask)

        optim.zero_grad()
        loss.backward()
        if bool(getattr(args, "distributed", False)):
            all_reduce_mean_gradients(learner, int(args.world_size))
        if args.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(learner.parameters(), args.max_grad_norm)
        optim.step()

        # Avoid keeping a stale graph (when critic_cache_with_grad=True) and free memory early.
        learner.critic.clear_cache_()

        with torch.no_grad():
            ret = torch.stack(rewards).sum(dim=0).mean().item()
            v = torch.stack(values)  # [T,B,1]
            v_mean = v.mean().item()
            v_std = v.std(unbiased=False).item() if v.numel() > 1 else 0.0
            v0 = values[0].mean().item() if values else 0.0

            # Advantage stats (discount=1.0, mask padded done-steps)
            r = torch.stack(rewards)  # [T,B,1]
            m = torch.stack(step_mask).float()  # [T,B,1]
            cumul = torch.zeros_like(r[0])
            rets = []
            for rr in reversed(r):
                cumul = rr + cumul
                rets.append(cumul)
            rets.reverse()
            rets = torch.stack(rets)  # [T,B,1]
            adv = (rets - v.detach()) * m
            denom = m.sum().clamp_min(1.0)
            adv_mean = (adv.sum() / denom).item()
            # masked std
            adv2 = (adv - adv_mean) * m
            adv_std = torch.sqrt((adv2.pow(2).sum() / denom).clamp_min(0.0)).item()

        gs = global_step_base + it
        writer.add_scalar("loss/episodic_loss", float(loss.item()), gs)
        writer.add_scalar("train/episodic_return", float(ret), gs)
        writer.add_scalar("critic/value_mean", float(v_mean), gs)
        writer.add_scalar("critic/value_std", float(v_std), gs)
        writer.add_scalar("critic/value_t0", float(v0), gs)
        writer.add_scalar("critic/adv_mean", float(adv_mean), gs)
        writer.add_scalar("critic/adv_std", float(adv_std), gs)
        ep_loss += float(loss.item())
        ep_ret += ret

    return ep_loss / args.iter_count, ep_ret / args.iter_count


def save_model_state_dict(save_path: str, model: nn.Module):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    orig_device = next(model.parameters()).device
    state_dict = model.to("cpu").state_dict()
    torch.save(state_dict, save_path)
    model.to(orig_device)


def main():
    args = parse_args()
    args.distributed, args.rank, args.world_size, args.device = setup_distributed(args.device)
    if args.distributed and args.batch_size % args.world_size != 0:
        raise ValueError(
            f"global batch_size={args.batch_size} must be divisible by world_size={args.world_size}; "
            "use two ranks with batch_size=16, or explicitly choose a new global batch size."
        )
    args.global_batch_size = int(args.batch_size)
    args.local_batch_size = args.global_batch_size // args.world_size
    args.batch_size = args.local_batch_size
    args.max_steps = resolve_max_steps(args.num_nodes, args.max_steps)
    set_random_seed(args.seed, args.torch_deterministic)

    if args.device.type == "cuda" and bool(getattr(args, "tf32", False)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if not args.run_name:
        args.run_name = (
            f"{args.model_variant}_ac_{args.vrp_env}_{args.selection}_"
            f"{args.num_nodes}n_{args.num_agents}a_{time.strftime('%Y_%m_%d_%Hh%Mm')}"
        )

    if SummaryWriter is None or args.rank != 0:
        writer = _NullWriter()
    else:
        writer = SummaryWriter(f"{args.log_path}/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    env = build_env(args)
    eval_env = None
    if args.rank == 0 and int(getattr(args, "eval_every", 0) or 0) > 0:
        eval_env = build_eval_env(args)
    nodes_stat_dim = env.obs_builder.get_nodes_static_feat_dim()
    agent_dim = env.obs_builder.get_agent_feat_dim()
    other_agents_dim = env.obs_builder.get_other_agents_feat_dim()
    model_variant = str(getattr(args, "model_variant", "abs")).lower()

    if model_variant == "ata":
        learner = AtaLearner(
            nodes_stat_dim=nodes_stat_dim,
            agent_dim=agent_dim,
            other_agents_dim=other_agents_dim,
            embed_dim=args.embed_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            feed_forward_hidden=args.ff_hidden,
        ).to(args.device)
    elif model_variant == "third":
        learner = ThirdLearner(
            nodes_stat_dim=nodes_stat_dim,
            agent_dim=agent_dim,
            embed_dim=args.embed_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            feed_forward_hidden=args.ff_hidden,
            n_inducing_points=args.critic_inducing_points,
        ).to(args.device)
    else:
        learner = AbsLearner(
            nodes_stat_dim=nodes_stat_dim,
            agent_dim=agent_dim,
            embed_dim=args.embed_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            feed_forward_hidden=args.ff_hidden,
        ).to(args.device)
    if args.distributed:
        for param in learner.parameters():
            dist.broadcast(param.data, src=0)
        set_random_seed(args.seed + args.rank * 100003, args.torch_deterministic)
    optim = Adam(learner.parameters(), lr=args.learning_rate)

    if args.rank == 0:
        print(f"train env={args.vrp_env} model={model_variant}:", vars(args))
    t0 = time.time()
    best_metric = float("-inf")
    model_dir = os.path.join(args.log_path, "models")
    latest_model_path = os.path.join(model_dir, f"latest_model_{args.run_name}.zip")
    best_model_path = os.path.join(model_dir, f"best_model_{args.run_name}.zip")

    for ep in range(args.epoch_count):
        global_step_base = ep * args.iter_count
        loss, ret = train_epoch(
            args,
            env,
            learner,
            optim,
            ep,
            writer=writer,
            global_step_base=global_step_base,
            rank=args.rank,
        )

        if args.rank == 0:
            print(f"[ep {ep:03d}] loss={loss:.4f} return={ret:.2f} elapsed={time.time()-t0:.1f}s")

            # Epoch-level logging.
            writer.add_scalar("epoch/mean_loss", float(loss), ep)
            writer.add_scalar("epoch/mean_return", float(ret), ep)

            if eval_env is not None and (ep % int(getattr(args, "eval_every", 1)) == 0):
                er, env_left, e_agents = eval_epoch(args, eval_env, learner, ep, writer)
                print(f"[ep {ep:03d}] eval_return={er:.2f} not_visited={env_left:.2f} used_agents={e_agents:.2f}")
                model_metric = float(er)
            else:
                model_metric = float(ret)

            if bool(getattr(args, "save_models", True)):
                save_model_state_dict(latest_model_path, learner)
                if model_metric > best_metric:
                    best_metric = model_metric
                    save_model_state_dict(best_model_path, learner)
                    print(f"[ep {ep:03d}] new best metric={best_metric:.4f} -> saved {best_model_path}")

        if int(getattr(args, "profile_steps", 0) or 0) > 0:
            # Profiling mode exits after first epoch.
            break
        if args.distributed:
            dist.barrier()

    if args.rank == 0 and bool(getattr(args, "save_models", True)):
        save_model_state_dict(latest_model_path, learner)
        print(f"saved latest model: {latest_model_path}")
        if best_metric == float("-inf"):
            # No epoch updated best (e.g., edge case). Keep best aligned with latest.
            save_model_state_dict(best_model_path, learner)
            print(f"saved fallback best model: {best_model_path}")

    writer.close()
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
