import argparse
import copy
import importlib
import os
import random
import time
from distutils.util import strtobool
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

from maenvs4vrp.learning.madyam.policy_net_ma_am_abs_second import AttentionModel
from maenvs4vrp.learning.madyam.layers._loss import reinforce_mask_loss


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vrp_env", type=str, default="dtoptw_abs_second")
    p.add_argument("--num_agents", type=int, default=10)
    p.add_argument("--num_nodes", type=int, default=100)
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=2297)
    p.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tf32", type=lambda x: bool(strtobool(x)), default=True)

    p.add_argument("--local_k_nodes", type=int, default=30)
    p.add_argument("--local_view_mode", type=str, default="nearest", choices=["nearest", "random"])
    p.add_argument("--profits", type=str, default="constant", choices=["distance", "constant", "uniform"])
    p.add_argument("--horizon", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--max_flight_dist", type=float, default=300.0)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--done_feasible_check_freq", type=int, default=10)

    p.add_argument("--model_size", type=int, default=128)
    p.add_argument("--layer_count", type=int, default=3)
    p.add_argument("--head_count", type=int, default=8)
    p.add_argument("--ff_size", type=int, default=512)
    p.add_argument("--tanh_clipping", type=float, default=10.0)

    p.add_argument("--epoch_count", type=int, default=100)
    p.add_argument("--iter_count", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--entropy_coef", type=float, default=1e-3)
    p.add_argument("--baseline_type", type=str, default="critic", choices=["critic", "rollout"])

    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=12345)
    p.add_argument("--eval_batch_size", type=int, default=16)
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
    observations = obs_mod.Observations(
        local_view={"k_nodes": args.local_k_nodes, "mode": args.local_view_mode, "include_depot": True},
        return_critic_obs=False,
    )

    gen_mod = importlib.import_module(f"maenvs4vrp.environments.{args.vrp_env}.instances_generator")
    generator = gen_mod.InstanceGenerator(device=args.device)

    rew_mod = importlib.import_module(f"maenvs4vrp.environments.{args.vrp_env}.env_agent_reward")
    reward_evaluator = rew_mod.DenseReward()

    env_mod = importlib.import_module(f"maenvs4vrp.environments.{args.vrp_env}.env")
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


def reset_env(env, args, seed: int):
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


@torch.no_grad()
def eval_epoch(args, env, model, ep: int, writer: Any):
    model.eval()
    total_reward = []
    not_visited_nodes = []
    used_agents = []

    for k in range(args.eval_episodes):
        td = reset_env(env, args, seed=int(args.eval_seed) + ep * 1000 + k)
        _actions, _logps, rewards, _entropies, _values, masks = model.rollout(env, td, deterministic=True)
        rewards_t = torch.stack(rewards)
        masks_t = torch.stack(masks).to(rewards_t.device)
        total_reward.append((rewards_t * masks_t).sum(dim=0))
        not_visited_nodes.append(env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1.0)
        used_agents.append(env.td_state["agents"]["visited_nodes"].sum(-1).gt(1).sum(-1).float())

    ret = torch.cat(total_reward).mean().item()
    not_visited = torch.cat(not_visited_nodes).mean().item()
    n_agents = torch.cat(used_agents).mean().item()

    writer.add_scalar("eval/episodic_return", float(ret), ep)
    writer.add_scalar("eval/episodic_not_visited_nodes", float(not_visited), ep)
    writer.add_scalar("eval/episodic_number_used_agents", float(n_agents), ep)
    return ret, not_visited, n_agents


def _rewards_to_returns(rewards, discount: float = 1.0):
    cumul = torch.zeros_like(rewards[0])
    vals = []
    for reward in reversed(rewards):
        cumul = reward + discount * cumul
        vals.append(cumul)
    vals.reverse()
    return vals


@torch.no_grad()
def _greedy_rollout_baseline(env, model_bl, td_init, gamma: float):
    _actions, _logps, rewards_bl, _entropies, _values, _masks = model_bl.rollout(env, td_init, deterministic=True)
    return _rewards_to_returns(rewards_bl, discount=gamma)


def train_epoch(args, env, model, optim, ep: int, writer: Any, global_step_base: int):
    model.train()
    ep_loss = 0.0
    ep_ret = 0.0
    ep_v = 0.0
    use_rollout_baseline = str(getattr(args, "baseline_type", "critic")).lower() == "rollout"
    model_bl = None
    if use_rollout_baseline:
        model_bl = copy.deepcopy(model)
        model_bl.eval()
        for p in model_bl.parameters():
            p.requires_grad_(False)

    for it in trange(args.iter_count, desc=f"Ep {ep+1}/{args.epoch_count}"):
        seed_i = args.seed + ep * args.iter_count + it
        td = reset_env(env, args, seed=seed_i)
        _actions, logps, rewards, entropies, values, active_masks = model.rollout(env, td, deterministic=False)

        step_mask = [m.unsqueeze(1) for m in active_masks]
        logps_in = [lp.unsqueeze(1) for lp in logps]
        rewards_in = [rw.unsqueeze(1) for rw in rewards]
        if use_rollout_baseline:
            with torch.no_grad():
                td_bl = reset_env(env, args, seed=seed_i)
                baseline_vals = _greedy_rollout_baseline(env, model_bl, td_bl, gamma=args.gamma)
                T = len(logps)
                if len(baseline_vals) < T:
                    pad = baseline_vals[-1]
                    baseline_vals = baseline_vals + [pad] * (T - len(baseline_vals))
                elif len(baseline_vals) > T:
                    baseline_vals = baseline_vals[:T]
            baseline_in = [bl.unsqueeze(1) for bl in baseline_vals]
        else:
            baseline_vals = values
            baseline_in = [v.unsqueeze(1) for v in values]
        loss = reinforce_mask_loss(
            logps_in,
            rewards_in,
            baseline=baseline_in,
            step_mask=step_mask,
            discount=args.gamma,
        )

        ent_t = torch.stack(entropies)
        mask_t = torch.stack(active_masks).to(ent_t.device)
        entropy = float(((ent_t * mask_t).sum() / mask_t.sum().clamp_min(1.0)).item())
        loss = loss - args.entropy_coef * ((ent_t * mask_t).sum() / mask_t.sum().clamp_min(1.0))
        ret = (torch.stack(rewards) * mask_t).sum(dim=0).mean().item()
        baseline_stack = torch.stack(baseline_vals)
        v0 = baseline_stack[0].mean().item()
        v_mean = baseline_stack.mean().item()

        optim.zero_grad()
        loss.backward()
        grad_norm = 0.0
        if args.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()

        gs = global_step_base + it
        writer.add_scalar("loss/episodic_loss", float(loss.item()), gs)
        writer.add_scalar("train/episodic_return", float(ret), gs)
        writer.add_scalar(f"train/{args.baseline_type}_t0", float(v0), gs)
        writer.add_scalar(f"train/{args.baseline_type}_mean", float(v_mean), gs)
        writer.add_scalar("train/entropy", float(entropy), gs)
        writer.add_scalar("train/grad_norm", float(grad_norm), gs)

        ep_loss += float(loss.item())
        ep_ret += float(ret)
        ep_v += float(v_mean)

    return ep_loss / args.iter_count, ep_ret / args.iter_count, ep_v / args.iter_count


def save_model_state_dict(save_path: str, model: nn.Module):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    orig_device = next(model.parameters()).device
    state_dict = model.to("cpu").state_dict()
    torch.save(state_dict, save_path)
    model.to(orig_device)


def main():
    args = parse_args()
    args.device = torch.device(args.device)
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
            f"am_{args.vrp_env}_{args.selection}_{args.num_nodes}n_{args.num_agents}a_"
            f"{time.strftime('%Y_%m_%d_%Hh%Mm')}"
        )

    writer = _NullWriter() if SummaryWriter is None else SummaryWriter(f"{args.log_path}/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    env = build_env(args)
    eval_env = build_eval_env(args) if int(getattr(args, "eval_every", 0) or 0) > 0 else None

    model = AttentionModel(
        nodes_stat_dim=env.obs_builder.get_nodes_static_feat_dim(),
        agent_dim=env.obs_builder.get_agent_feat_dim(),
        other_agents_dim=env.obs_builder.get_other_agents_feat_dim(),
        model_size=args.model_size,
        layer_count=args.layer_count,
        head_count=args.head_count,
        ff_size=args.ff_size,
        tanh_clipping=args.tanh_clipping,
    ).to(args.device)
    optim = Adam(model.parameters(), lr=args.learning_rate)

    print(f"train env={args.vrp_env} model=attention_model:", vars(args))
    best_metric = float("-inf")
    t0 = time.time()
    model_dir = os.path.join(args.log_path, "models")
    latest_model_path = os.path.join(model_dir, f"latest_model_{args.run_name}.pt")
    best_model_path = os.path.join(model_dir, f"best_model_{args.run_name}.pt")

    for ep in range(args.epoch_count):
        global_step_base = ep * args.iter_count
        loss, ret, v = train_epoch(args, env, model, optim, ep, writer, global_step_base=global_step_base)
        print(f"[ep {ep:03d}] loss={loss:.4f} return={ret:.2f} {args.baseline_type}={v:.2f} elapsed={time.time()-t0:.1f}s")
        writer.add_scalar("epoch/mean_loss", float(loss), ep)
        writer.add_scalar("epoch/mean_return", float(ret), ep)
        writer.add_scalar(f"epoch/mean_{args.baseline_type}", float(v), ep)

        model_metric = float(ret)
        if eval_env is not None and (ep % int(getattr(args, "eval_every", 1)) == 0):
            er, env_left, e_agents = eval_epoch(args, eval_env, model, ep, writer)
            model_metric = float(er)
            print(f"[ep {ep:03d}] eval_return={er:.2f} not_visited={env_left:.2f} used_agents={e_agents:.2f}")

        if bool(getattr(args, "save_models", True)):
            save_model_state_dict(latest_model_path, model)
            if model_metric > best_metric:
                best_metric = model_metric
                save_model_state_dict(best_model_path, model)
                print(f"[ep {ep:03d}] new best metric={best_metric:.4f} -> saved {best_model_path}")

    if bool(getattr(args, "save_models", True)):
        save_model_state_dict(latest_model_path, model)
        if best_metric == float("-inf"):
            save_model_state_dict(best_model_path, model)

    writer.close()


if __name__ == "__main__":
    main()
