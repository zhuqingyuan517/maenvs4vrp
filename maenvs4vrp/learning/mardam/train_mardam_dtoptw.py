"""
Train mardam on dtoptw-style environments with dtoptw-specific runtime args.

Recommended env:
- dtoptw_full (global action space, compatible with mardam vehicle-state input)
"""

import argparse
import importlib
import math
import os
import random
import time
from distutils.util import strtobool
from itertools import chain

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from maenvs4vrp.learning.mardam._learner import AttentionLearner
from maenvs4vrp.learning.mardam.baselines import CriticBaseline
from maenvs4vrp.learning.mardam.layers import reinforce_loss, reinforce_mask_loss
from maenvs4vrp.learning.mardam.numerics import (
    NonFiniteTrainingError,
    describe_nonfinite,
)
from maenvs4vrp.learning.baseline_protocol import resolve_max_steps


def set_random_seed(seed: int, torch_deterministic: bool):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def init_distributed(args) -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.distributed = world_size > 1
    args.rank = 0
    args.world_size = world_size
    args.local_batch_size = args.batch_size
    if not args.distributed:
        return False

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    local_rank = int(os.environ["LOCAL_RANK"])
    if args.device.type == "cuda":
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible"
            )
        args.device = torch.device("cuda", local_rank)
        torch.cuda.set_device(args.device)
        backend = "nccl"
    else:
        args.device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    if args.batch_size % args.world_size != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be divisible by world_size ({args.world_size})"
        )
    args.local_batch_size = args.batch_size // args.world_size
    return True


def broadcast_module_state(module) -> None:
    buffers = module.buffers() if hasattr(module, "buffers") else ()
    for tensor in chain(module.parameters(), buffers):
        dist.broadcast(tensor.data, src=0)


def average_gradients(modules, world_size: int) -> None:
    for module in modules:
        for param in module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def average_float_buffers(module, world_size: int) -> None:
    for buffer in module.buffers():
        if buffer.is_floating_point():
            dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            buffer.div_(world_size)


def average_metrics(values, args):
    if not args.distributed:
        return values
    metrics = torch.tensor(values, dtype=torch.float32, device=args.device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    metrics.div_(args.world_size)
    return metrics.tolist()


def synchronize_bad_update(is_bad, args):
    if not args.distributed:
        return bool(is_bad)
    flag = torch.tensor(
        [int(bool(is_bad))],
        dtype=torch.int32,
        device=args.device,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def gradients_are_finite(optim):
    for group in optim.param_groups:
        for param in group["params"]:
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return False
    return True


def module_state_is_finite(module):
    buffers = module.buffers() if hasattr(module, "buffers") else ()
    tensors = chain(module.parameters(), buffers)
    return all(
        not tensor.is_floating_point() or torch.isfinite(tensor).all()
        for tensor in tensors
    )


def skip_update(args, optim, ep, it, seed, reason):
    optim.zero_grad(set_to_none=True)
    print(
        f"[skip] rank={args.rank} ep={ep:03d} iter={it:03d} "
        f"seed={seed} reason={reason}",
        flush=True,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vrp_env", type=str, default="dtoptw_full")
    p.add_argument("--num_agents", type=int, default=50)
    p.add_argument("--num_nodes", type=int, default=500, help="includes depot")
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Global batch per optimizer step; split evenly across torchrun ranks",
    )
    p.add_argument("--seed", type=int, default=2297)
    p.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # dtoptw-specific args
    p.add_argument("--profits", type=str, default="constant", choices=["distance", "constant", "uniform"])
    p.add_argument("--horizon", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=0, help="0 resolves to 600/1200 by node scale")
    p.add_argument("--max_flight_dist", type=float, default=300.0)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--done_feasible_check_freq", type=int, default=10)
    p.add_argument("--reward", type=str, default="dense", choices=["dense", "sparse"])

    # optimization
    p.add_argument("--epoch_count", type=int, default=100)
    p.add_argument("--iter_count", type=int, default=200)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--critic_rate", type=float, default=1e-3)
    p.add_argument("--max_grad_norm", type=float, default=2.0)
    p.add_argument("--use_step_mask", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument(
        "--normalize_step_loss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Average the masked REINFORCE loss over valid decision steps.",
    )
    p.add_argument(
        "--debug_numerics",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Run detailed finite-value checks inside every policy decision.",
    )
    p.add_argument(
        "--grad_checkpoint",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help=(
            "recompute each decision-step vehicle attention during backward to reduce "
            "activation memory (enabled by default for long DTOPTW rollouts)"
        ),
    )

    # eval / logging
    p.add_argument("--eval_every", type=int, default=1, help="0 disables eval")
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=12345)
    p.add_argument("--log_path", type=str, default="runs")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--save_models", type=lambda x: bool(strtobool(x)), default=True)
    return p.parse_args()


def build_env(args):
    sel_mod = importlib.import_module(f"maenvs4vrp.environments.{args.vrp_env}.env_agent_selector")
    if args.selection == "rand":
        selector = sel_mod.RandomSelector()
    elif args.selection == "single":
        selector = sel_mod.AgentSelector()
    else:
        selector = sel_mod.SmallestTimeAgentSelector()

    obs_mod = importlib.import_module(f"maenvs4vrp.environments.{args.vrp_env}.observations")
    observations = obs_mod.Observations()

    gen_mod = importlib.import_module(f"maenvs4vrp.environments.{args.vrp_env}.instances_generator")
    generator = gen_mod.InstanceGenerator(device=args.device)

    rew_mod = importlib.import_module(f"maenvs4vrp.environments.{args.vrp_env}.env_agent_reward")
    reward_evaluator = rew_mod.DenseReward() if args.reward == "dense" else rew_mod.SparseReward()

    env_mod = importlib.import_module(f"maenvs4vrp.environments.{args.vrp_env}.env")
    local_batch_size = getattr(args, "local_batch_size", args.batch_size)
    env = env_mod.Environment(
        instance_generator_object=generator,
        obs_builder_object=observations,
        agent_selector_object=selector,
        reward_evaluator=reward_evaluator,
        device=args.device,
        batch_size=local_batch_size,
        seed=args.seed,
        max_steps=args.max_steps,
        max_flight_dist=args.max_flight_dist,
        allow_depot_refuel=args.allow_depot_refuel,
        forbid_depot_self_loop=args.forbid_depot_self_loop,
        done_feasible_check_freq=args.done_feasible_check_freq,
    )
    return env


def reset_dtoptw_env(env, args, seed: int):
    return env.reset(
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


def train_epoch(args, env, baseline, optim, ep: int, writer: SummaryWriter):
    baseline.learner.train()
    ep_loss = 0.0
    ep_ret = 0.0
    ep_bl = 0.0

    for it in trange(
        args.iter_count,
        desc=f"Ep {ep+1}/{args.epoch_count}",
        disable=args.rank != 0,
    ):
        base_seed = args.seed + ep * args.iter_count + it
        local_seed = base_seed * args.world_size + args.rank
        td = reset_dtoptw_env(env, args, seed=local_seed)
        rollout_error = None
        try:
            _, logps, rewards, bl_vals, step_mask = baseline(env, td)
        except NonFiniteTrainingError as exc:
            rollout_error = str(exc)
            logps = rewards = bl_vals = step_mask = None

        if synchronize_bad_update(rollout_error is not None, args):
            skip_update(args, optim, ep, it, local_seed, rollout_error or "peer rank failed")
            continue

        if args.use_step_mask:
            loss = reinforce_mask_loss(
                logps,
                rewards,
                baseline=bl_vals,
                step_mask=step_mask,
                normalize_step_loss=args.normalize_step_loss,
            )
        else:
            loss = reinforce_loss(logps, rewards, baseline=bl_vals)

        optim.zero_grad()
        nonfinite_loss = not torch.isfinite(loss).all()
        if synchronize_bad_update(nonfinite_loss, args):
            skip_update(args, optim, ep, it, local_seed, "non-finite loss")
            continue
        loss.backward()
        nonfinite_gradients = not gradients_are_finite(optim)
        if synchronize_bad_update(nonfinite_gradients, args):
            skip_update(args, optim, ep, it, local_seed, "non-finite gradient")
            continue
        if args.distributed:
            average_gradients((baseline.learner, baseline), args.world_size)
        grad_norm = 0.0
        if args.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                chain.from_iterable(grp["params"] for grp in optim.param_groups), args.max_grad_norm
            )
        nonfinite_grad_norm = math.isnan(float(grad_norm)) or math.isinf(float(grad_norm))
        if synchronize_bad_update(nonfinite_grad_norm, args):
            skip_update(args, optim, ep, it, local_seed, "non-finite gradient norm")
            continue
        optim.step()
        if args.distributed:
            average_float_buffers(baseline.learner, args.world_size)
            broadcast_module_state(baseline.learner)
            broadcast_module_state(baseline)
        nonfinite_state = (
            not module_state_is_finite(baseline.learner)
            or not module_state_is_finite(baseline)
        )
        if synchronize_bad_update(nonfinite_state, args):
            raise NonFiniteTrainingError(
                f"non-finite module state after optimizer step "
                f"rank={args.rank} ep={ep:03d} iter={it:03d} seed={local_seed}"
            )

        with torch.no_grad():
            metrics = average_metrics(
                [
                    loss.item(),
                    torch.stack(rewards).sum(dim=0).mean().item(),
                    bl_vals[0].mean().item(),
                    (env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1.0).mean().item(),
                    env.td_state["agents"]["visited_nodes"].sum(-1).gt(1).sum(-1).float().mean().item(),
                ],
                args,
            )

        gs = ep * args.iter_count + it
        if writer is not None:
            writer.add_scalar("loss/episodic_loss", float(metrics[0]), gs)
            writer.add_scalar("train/episodic_return", float(metrics[1]), gs)
            writer.add_scalar("train/episodic_baseline", float(metrics[2]), gs)
            writer.add_scalar("train/episodic_not_visited_nodes", float(metrics[3]), gs)
            writer.add_scalar("train/episodic_number_used_agents", float(metrics[4]), gs)
            writer.add_scalar("train/grad_norm", float(grad_norm), gs)

        ep_loss += float(metrics[0])
        ep_ret += float(metrics[1])
        ep_bl += float(metrics[2])

    return ep_loss / args.iter_count, ep_ret / args.iter_count, ep_bl / args.iter_count


@torch.no_grad()
def eval_epoch(args, env, learner, ep: int, writer: SummaryWriter):
    learner.eval()
    rets = []
    not_visited_nodes = []
    used_agents = []

    for k in range(args.eval_batch_count):
        td = reset_dtoptw_env(env, args, seed=args.eval_seed + ep * 1000 + k)
        _, _, rewards = learner(env, td, deterministic=True)
        rets.append(torch.stack(rewards).sum(dim=0).mean())
        not_visited_nodes.append((env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1.0).mean())
        used_agents.append(env.td_state["agents"]["visited_nodes"].sum(-1).gt(1).sum(-1).float().mean())

    ret = torch.stack(rets).mean().item()
    not_visited = torch.stack(not_visited_nodes).mean().item()
    n_agents = torch.stack(used_agents).mean().item()
    writer.add_scalar("eval/return", float(ret), ep)
    writer.add_scalar("eval/not_visited_nodes", float(not_visited), ep)
    writer.add_scalar("eval/number_used_agents", float(n_agents), ep)
    return ret


def save_checkpoint(path: str, learner, baseline, args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    learner_cpu = {k: v.detach().cpu() for k, v in learner.state_dict().items()}
    baseline_cpu = {k: v.detach().cpu() for k, v in baseline.state_dict().items()}
    torch.save(
        {
            "learner": learner_cpu,
            "baseline": baseline_cpu,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    args.max_steps = resolve_max_steps(args.num_nodes, args.max_steps)
    args.device = torch.device(args.device)
    init_distributed(args)
    args.eval_batch_count = args.eval_episodes * args.world_size
    set_random_seed(args.seed, args.torch_deterministic)

    if not args.run_name:
        args.run_name = (
            f"mardam_{args.vrp_env}_{args.selection}_{args.num_nodes}n_{args.num_agents}a_"
            f"{time.strftime('%Y_%m_%d_%Hh%Mm')}"
        )

    writer = SummaryWriter(f"{args.log_path}/{args.run_name}") if args.rank == 0 else None
    if writer is not None:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
        )

    env = build_env(args)
    eval_env = build_env(args) if args.rank == 0 else None

    nodes_static_feat_dim = env.obs_builder.get_nodes_static_feat_dim()
    agents_feat_dim = env.obs_builder.get_other_agents_feat_dim()
    if agents_feat_dim <= 0:
        raise ValueError(
            f"{args.vrp_env} provides other_agents_feat_dim={agents_feat_dim}. "
            "mardam requires positive vehicle-state features; use dtoptw_full/dtoptw."
        )

    learner = AttentionLearner(cust_feat_size=nodes_static_feat_dim, veh_state_size=agents_feat_dim).to(args.device)
    baseline = CriticBaseline(
        learner,
        args.num_nodes - 1,
        grad_checkpoint=args.grad_checkpoint,
        debug_numerics=args.debug_numerics,
    )
    baseline.to(args.device)
    if args.distributed:
        broadcast_module_state(learner)
        broadcast_module_state(baseline)
        set_random_seed(args.seed + args.rank, args.torch_deterministic)

    optim = Adam(
        [
            {"params": learner.parameters(), "lr": args.learning_rate},
            {"params": baseline.parameters(), "lr": args.critic_rate},
        ]
    )

    if args.rank == 0:
        print("train_mardam_dtoptw args:", vars(args))
    best_eval = -1e18
    model_dir = os.path.join(args.log_path, "models")
    latest_model_path = os.path.join(model_dir, f"latest_model_{args.run_name}.zip")
    best_model_path = os.path.join(model_dir, f"best_model_{args.run_name}.zip")
    t0 = time.time()
    for ep in range(args.epoch_count):
        loss, ret, bl = train_epoch(args, env, baseline, optim, ep, writer)
        if writer is not None:
            writer.add_scalar("epoch/loss", float(loss), ep)
            writer.add_scalar("epoch/return", float(ret), ep)
            writer.add_scalar("epoch/baseline", float(bl), ep)

        msg = f"[ep {ep:03d}] loss={loss:.4f} return={ret:.2f} baseline={bl:.2f}"
        if args.rank == 0 and args.eval_every > 0 and (ep % args.eval_every == 0):
            e = eval_epoch(args, eval_env, learner, ep, writer)
            if e > best_eval:
                best_eval = e
                if args.save_models:
                    save_checkpoint(best_model_path, learner, baseline, args)
            writer.add_scalar("eval/best_return", float(best_eval), ep)
            msg += f" eval={e:.2f} best_eval={best_eval:.2f}"

        if args.rank == 0 and args.save_models:
            save_checkpoint(latest_model_path, learner, baseline, args)

        if writer is not None:
            writer.add_scalar("charts/SPS", int((ep + 1) / max(time.time() - t0, 1e-6)), ep)
            print(msg)
        if args.distributed:
            dist.barrier()

    if writer is not None:
        writer.close()
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
