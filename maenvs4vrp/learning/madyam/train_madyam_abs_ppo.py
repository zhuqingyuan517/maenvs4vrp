"""
Train dtoptw_abs with the memory-efficient abs_third learner using PPO.
"""

import argparse
import importlib
import os
import random
import time
from distutils.util import strtobool
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vrp_env", type=str, default="dtoptw_abs")
    p.add_argument("--num_agents", type=int, default=50)
    p.add_argument("--num_nodes", type=int, default=500)
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=2297)
    p.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tf32", type=lambda x: bool(strtobool(x)), default=True)

    p.add_argument("--local_k_nodes", type=int, default=30)
    p.add_argument("--local_view_mode", type=str, default="nearest", choices=["nearest", "random"])
    p.add_argument("--critic_mask_unappeared", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--critic_encode_unappeared", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--profits", type=str, default="constant", choices=["distance", "constant", "uniform"])
    p.add_argument("--horizon", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--max_flight_dist", type=float, default=300)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--done_feasible_check_freq", type=int, default=10)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--ff_hidden", type=int, default=512)
    p.add_argument("--critic_inducing_points", type=int, default=16)

    p.add_argument("--total_updates", type=int, default=200)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--anneal_lr", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--mini_batch_size", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=2.0)
    p.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--target_kl", type=float, default=0.03)

    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=12345)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--log_path", type=str, default="1000runs")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--save_models", type=lambda x: bool(strtobool(x)), default=True)
    return p.parse_args()


def build_env(args, *, batch_size: int | None = None, seed: int | None = None):
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
    observations = obs_mod.Observations(
        local_view={"k_nodes": args.local_k_nodes, "mode": args.local_view_mode, "include_depot": True},
        return_critic_obs=True,
        critic_mask_unappeared=args.critic_mask_unappeared,
        critic_encode_unappeared=args.critic_encode_unappeared,
    )

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
        batch_size=args.batch_size if batch_size is None else batch_size,
        seed=args.seed if seed is None else seed,
        max_steps=args.max_steps,
        max_flight_dist=args.max_flight_dist,
        allow_depot_refuel=args.allow_depot_refuel,
        done_feasible_check_freq=args.done_feasible_check_freq,
    )
    return env


def rollout_episode(args, env, learner, update_idx: int):
    td = env.reset(
        num_agents=args.num_agents,
        num_nodes=args.num_nodes,
        profits=args.profits,
        seed=args.seed + update_idx,
        sample_type="random",
        max_steps=args.max_steps,
        max_flight_dist=args.max_flight_dist,
        allow_depot_refuel=args.allow_depot_refuel,
        forbid_depot_self_loop=args.forbid_depot_self_loop,
        horizon=args.horizon,
        done_feasible_check_freq=args.done_feasible_check_freq,
    )

    critic_node_static_obs = td["observations"]["critic_node_static_obs"]
    learner.make_critic_cache_(critic_node_static_obs, with_grad=False)

    storage = {
        "node_static_obs": [],
        "agent_obs": [],
        "action_mask": [],
        "critic_node_static_obs": [],
        "critic_nodes_mask": [],
        "actions": [],
        "logprobs": [],
        "values": [],
        "rewards": [],
        "step_mask": [],
        "returns": None,
        "advantages": None,
    }

    while not td["done"].all():
        obs = td["observations"]
        with torch.no_grad():
            action, logp, ent, value = learner.get_action_and_value(
                node_static_obs=obs["node_static_obs"],
                agent_obs=obs["agent_obs"],
                action_mask=obs["action_mask"],
                critic_node_static_obs=obs["critic_node_static_obs"],
                critic_nodes_mask=obs["critic_nodes_mask"],
                deterministic=False,
            )

        storage["node_static_obs"].append(obs["node_static_obs"].detach().clone())
        storage["agent_obs"].append(obs["agent_obs"].detach().clone())
        storage["action_mask"].append(obs["action_mask"].detach().clone())
        storage["critic_node_static_obs"].append(obs["critic_node_static_obs"].detach().clone())
        storage["critic_nodes_mask"].append(obs["critic_nodes_mask"].detach().clone())
        storage["actions"].append(action.detach().clone())
        storage["logprobs"].append(logp.detach().clone())
        storage["values"].append(value.squeeze(-1).detach().clone())
        storage["step_mask"].append((~td["done"]).detach().clone())

        td["action"] = action.unsqueeze(1)
        td = env.step(td)
        reward = td["reward"] + td["penalty"]
        storage["rewards"].append(reward.squeeze(-1).detach().clone())

    rewards = torch.stack(storage["rewards"])
    values = torch.stack(storage["values"])
    step_mask = torch.stack(storage["step_mask"]).float()

    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(rewards[0])
    next_nonterminal = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.size(0))):
        delta = rewards[t] + args.gamma * next_value * next_nonterminal - values[t]
        lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
        next_value = values[t]
        next_nonterminal = step_mask[t]
    returns = advantages + values

    storage["advantages"] = advantages
    storage["returns"] = returns

    episode_return = rewards.sum(dim=0).mean().item()
    not_visited = (env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1.0).mean().item()
    used_agents = env.td_state["agents"]["visited_nodes"].sum(-1).gt(1).sum(-1).float().mean().item()
    return storage, episode_return, not_visited, used_agents


def flatten_rollout(storage):
    flat = {}
    for key, value in storage.items():
        if value is None:
            continue
        if isinstance(value, list):
            value = torch.stack(value)
        if key in {"returns", "advantages"}:
            value = value
        flat[key] = value
    step_mask = flat["step_mask"].bool()
    data = {}
    for key, value in flat.items():
        if key == "step_mask":
            data[key] = value
        else:
            data[key] = value[step_mask]
    return data


def evaluate(args, env, learner, update_idx: int, writer: Any):
    learner.eval()
    total_reward = []
    not_visited_nodes = []
    used_agents = []

    with torch.no_grad():
        for k in range(args.eval_episodes):
            td = env.reset(
                num_agents=args.num_agents,
                num_nodes=args.num_nodes,
                profits=args.profits,
                seed=args.eval_seed + update_idx * 1000 + k,
                sample_type="random",
                max_steps=args.max_steps,
                max_flight_dist=args.max_flight_dist,
                allow_depot_refuel=args.allow_depot_refuel,
                forbid_depot_self_loop=args.forbid_depot_self_loop,
                horizon=args.horizon,
                done_feasible_check_freq=args.done_feasible_check_freq,
            )
            learner.make_critic_cache_(td["observations"]["critic_node_static_obs"], with_grad=False)
            ep_rewards = []
            while not td["done"].all():
                obs = td["observations"]
                action, _, _, _ = learner.get_action_and_value(
                    node_static_obs=obs["node_static_obs"],
                    agent_obs=obs["agent_obs"],
                    action_mask=obs["action_mask"],
                    critic_node_static_obs=obs["critic_node_static_obs"],
                    critic_nodes_mask=obs["critic_nodes_mask"],
                    deterministic=True,
                )
                td["action"] = action.unsqueeze(1)
                td = env.step(td)
                ep_rewards.append((td["reward"] + td["penalty"]).squeeze(-1))

            total_reward.append(torch.stack(ep_rewards).sum(0))
            not_visited_nodes.append(env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1.0)
            used_agents.append(env.td_state["agents"]["visited_nodes"].sum(-1).gt(1).sum(-1).float())

    ret = torch.cat(total_reward).mean().item()
    nv = torch.cat(not_visited_nodes).mean().item()
    ua = torch.cat(used_agents).mean().item()
    writer.add_scalar("eval/episodic_return", ret, update_idx)
    writer.add_scalar("eval/not_visited", nv, update_idx)
    writer.add_scalar("eval/used_agents", ua, update_idx)
    return ret, nv, ua


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

    if args.device.type == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if not args.run_name:
        args.run_name = (
            f"third_ppo_{args.vrp_env}_{args.selection}_"
            f"{args.num_nodes}n_{args.num_agents}a_{time.strftime('%Y_%m_%d_%Hh%Mm')}"
        )

    writer = _NullWriter() if SummaryWriter is None else SummaryWriter(f"1000runs/{args.run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])))

    env = build_env(args)
    eval_env = build_env(args, batch_size=args.eval_batch_size, seed=args.eval_seed)

    learner = ThirdLearner(
        nodes_stat_dim=env.obs_builder.get_nodes_static_feat_dim(),
        agent_dim=env.obs_builder.get_agent_feat_dim(),
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        feed_forward_hidden=args.ff_hidden,
        n_inducing_points=args.critic_inducing_points,
    ).to(args.device)
    optimizer = Adam(learner.parameters(), lr=args.learning_rate, eps=1e-5)

    best_metric = float("-inf")
    model_dir = os.path.join(args.log_path, "models")
    latest_model_path = os.path.join(model_dir, f"latest_model_{args.run_name}.zip")
    best_model_path = os.path.join(model_dir, f"best_model_{args.run_name}.zip")
    start_time = time.time()

    for update in range(args.total_updates):
        if args.anneal_lr:
            frac = 1.0 - update / max(1, args.total_updates)
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        learner.eval()
        storage, ep_return, nv, ua = rollout_episode(args, env, learner, update)
        batch = flatten_rollout(storage)

        advantages = batch["advantages"]
        if args.norm_adv and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        sample_count = advantages.size(0)
        if sample_count < args.mini_batch_size:
            raise ValueError(
                f"Collected {sample_count} valid transitions, which is smaller than mini_batch_size={args.mini_batch_size}."
            )

        b_index = np.arange(sample_count)
        clip_fracs = []
        approx_kl = torch.tensor(0.0, device=args.device)
        policy_loss = torch.tensor(0.0, device=args.device)
        value_loss = torch.tensor(0.0, device=args.device)
        entropy_loss = torch.tensor(0.0, device=args.device)

        learner.train()
        for _ in range(args.update_epochs):
            np.random.shuffle(b_index)
            for start in range(0, sample_count, args.mini_batch_size):
                mb_idx = b_index[start : start + args.mini_batch_size]
                if len(mb_idx) == 0:
                    continue

                _, newlogprob, entropy, value = learner.evaluate_actions(
                    node_static_obs=batch["node_static_obs"][mb_idx],
                    agent_obs=batch["agent_obs"][mb_idx],
                    action_mask=batch["action_mask"][mb_idx],
                    critic_node_static_obs=batch["critic_node_static_obs"][mb_idx],
                    critic_nodes_mask=batch["critic_nodes_mask"][mb_idx],
                    action=batch["actions"][mb_idx],
                )
                value = value.squeeze(-1)
                logratio = newlogprob - batch["logprobs"][mb_idx]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clip_fracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = advantages[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                old_values = batch["values"][mb_idx]
                returns = batch["returns"][mb_idx]
                v_loss_unclipped = (value - returns) ** 2
                v_clipped = old_values + torch.clamp(value - old_values, -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(learner.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl > 0 and approx_kl.item() > args.target_kl:
                break

        writer.add_scalar("train/episodic_return", ep_return, update)
        writer.add_scalar("train/not_visited", nv, update)
        writer.add_scalar("train/used_agents", ua, update)
        writer.add_scalar("loss/policy", policy_loss.item(), update)
        writer.add_scalar("loss/value", value_loss.item(), update)
        writer.add_scalar("loss/entropy", entropy_loss.item(), update)
        writer.add_scalar("loss/approx_kl", approx_kl.item(), update)
        writer.add_scalar("loss/clipfrac", float(np.mean(clip_fracs)) if clip_fracs else 0.0, update)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], update)
        writer.add_scalar("charts/SPS", int((update + 1) / max(1e-6, time.time() - start_time)), update)

        if args.eval_every > 0 and update % args.eval_every == 0:
            er, env_left, e_agents = evaluate(args, eval_env, learner, update, writer)
            model_metric = er
            print(
                f"[update {update:03d}] return={ep_return:.2f} eval_return={er:.2f} "
                f"not_visited={env_left:.2f} used_agents={e_agents:.2f}"
            )
        else:
            model_metric = ep_return
            print(f"[update {update:03d}] return={ep_return:.2f} not_visited={nv:.2f} used_agents={ua:.2f}")

        if args.save_models:
            save_model_state_dict(latest_model_path, learner)
            if model_metric > best_metric:
                best_metric = model_metric
                save_model_state_dict(best_model_path, learner)

    if args.save_models:
        save_model_state_dict(latest_model_path, learner)
        if best_metric == float("-inf"):
            save_model_state_dict(best_model_path, learner)
    writer.close()


if __name__ == "__main__":
    main()
