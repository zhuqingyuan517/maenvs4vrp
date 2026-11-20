"""
Train dtoptw_full (global action space) with the original policy_net_ma_ac_old model (unchanged).

We do NOT modify policy_net_ma_ac_old.py. The `dtoptw_full` environment provides:
- Full node-set action space (global indices in [0, N)).
- Rich node dynamic features + global features (compatible with the old model signature).
"""

import argparse
import importlib
import random
import time
from itertools import chain

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool
from tqdm import trange

from maenvs4vrp.learning.madyam.layers._loss import reinforce_loss, reinforce_mask_loss
from maenvs4vrp.learning.madyam.policy_net_ma_ac_old import Learner


def set_random_seed(seed: int, torch_deterministic: bool):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vrp_env", type=str, default="dtoptw_full")
    p.add_argument("--num_agents", type=int, default=5)
    p.add_argument("--num_nodes", type=int, default=41)
    p.add_argument("--selection", type=str, default="stime", choices=["rand", "single", "stime"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=2297)
    p.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Global action space: no local view settings here.
    p.add_argument("--profits", type=str, default="constant", choices=["distance", "constant", "uniform"])
    p.add_argument("--horizon", type=int, default=480)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--max_flight_dist", type=float, default=300)
    p.add_argument("--allow_depot_refuel", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--forbid_depot_self_loop", type=lambda x: bool(strtobool(x)), default=False)
    p.add_argument("--done_feasible_check_freq", type=int, default=10)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--epoch_count", type=int, default=100)
    p.add_argument("--iter_count", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=2.0)
    p.add_argument("--use_step_mask", type=lambda x: bool(strtobool(x)), default=True, help="use reinforce_mask_loss to ignore done-padded steps")

    p.add_argument("--log_path", type=str, default="runs")
    p.add_argument("--run_name", type=str, default="")
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


def train_epoch(args, env, learner, optim, ep: int, writer: SummaryWriter):
    learner.train()

    ep_loss = 0.0
    ep_ret = 0.0

    for it in trange(args.iter_count, desc=f"Ep {ep+1}/{args.epoch_count}"):
        td = env.reset(
            num_agents=args.num_agents,
            num_nodes=args.num_nodes,
            profits=args.profits,
            seed=args.seed + ep * args.iter_count + it,
            sample_type="random",
            max_steps=args.max_steps,
            max_flight_dist=args.max_flight_dist,
            allow_depot_refuel=args.allow_depot_refuel,
            forbid_depot_self_loop=args.forbid_depot_self_loop,
            horizon=args.horizon,
            done_feasible_check_freq=args.done_feasible_check_freq,
        )

        logps = []
        entropies = []
        values = []
        rewards = []
        step_mask = []

        while not td["done"].all():
            obs = td["observations"]
            node_stat = obs["node_static_obs"]         # [B,N,F]
            node_dyn = obs["node_dynamic_obs"]         # [B,N,0]
            action_mask = obs["action_mask"]           # [B,N] (True allowed)
            self_obs = obs["agent_obs"]                # [B,A]
            global_obs = obs["global_obs"]             # [B,0]

            # Static encoder cache is valid for the whole episode (global nodes are fixed).
            if len(logps) == 0:
                learner.policy.make_cache_(nodes_obs=node_stat)

            # Use the (possibly dummy) other-agents tensors provided by the env observations.
            agents_obs = obs["other_agents_obs"]
            agents_mask = obs["agents_mask"]

            # Safety: ensure at least one feasible action per row (depot fallback).
            if (~action_mask.any(dim=1)).any():
                action_mask = action_mask.clone()
                bad = ~action_mask.any(dim=1)
                action_mask[bad, 0] = True

            action, logp, ent, val = learner.get_action_and_logs(
                nodes_obs=node_dyn,
                self_obs=self_obs,
                agents_obs=agents_obs,
                agents_mask=agents_mask,
                global_obs=global_obs,
                action_mask=action_mask,
                deterministic=False,
            )

            td["action"] = action.unsqueeze(1)  # global index
            td = env.step(td)

            logps.append(logp.unsqueeze(1))
            entropies.append(ent.unsqueeze(1))
            values.append(val)

            step_r = td["reward"] + td["penalty"]
            if step_r.dim() == 1:
                step_r = step_r.unsqueeze(1)
            rewards.append(step_r)
            step_mask.append((~td["done"]).unsqueeze(1))

        if args.use_step_mask:
            loss = reinforce_mask_loss(logps, rewards, baseline=values, step_mask=step_mask)
        else:
            loss = reinforce_loss(logps, rewards, baseline=values)

        optim.zero_grad()
        loss.backward()
        if args.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(chain.from_iterable(grp["params"] for grp in optim.param_groups), args.max_grad_norm)
        optim.step()

        with torch.no_grad():
            ret = torch.stack(rewards).sum(dim=0).mean().item()
            ent_m = torch.stack(entropies).mean().item() if entropies else 0.0
            v = torch.stack(values)  # [T,B,1]
            v_mean = v.mean().item()
            v_std = v.std(unbiased=False).item() if v.numel() > 1 else 0.0
            v0 = values[0].mean().item() if values else 0.0

            r = torch.stack(rewards)
            m = torch.stack(step_mask).float()
            cumul = torch.zeros_like(r[0])
            rets = []
            for rr in reversed(r):
                cumul = rr + cumul
                rets.append(cumul)
            rets.reverse()
            rets = torch.stack(rets)
            adv = (rets - v.detach()) * m
            denom = m.sum().clamp_min(1.0)
            adv_mean = (adv.sum() / denom).item()
            adv2 = (adv - adv_mean) * m
            adv_std = torch.sqrt((adv2.pow(2).sum() / denom).clamp_min(0.0)).item()

        gs = ep * args.iter_count + it
        writer.add_scalar("loss/episodic_loss", float(loss.item()), gs)
        writer.add_scalar("train/episodic_return", float(ret), gs)
        writer.add_scalar("train/entropy", float(ent_m), gs)
        writer.add_scalar("critic/value_mean", float(v_mean), gs)
        writer.add_scalar("critic/value_std", float(v_std), gs)
        writer.add_scalar("critic/value_t0", float(v0), gs)
        writer.add_scalar("critic/adv_mean", float(adv_mean), gs)
        writer.add_scalar("critic/adv_std", float(adv_std), gs)

        ep_loss += float(loss.item())
        ep_ret += float(ret)

    return ep_loss / args.iter_count, ep_ret / args.iter_count


def main():
    args = parse_args()
    args.device = torch.device(args.device)
    set_random_seed(args.seed, args.torch_deterministic)

    if not args.run_name:
        args.run_name = f"old_dtoptw_full_{args.selection}_{args.num_nodes}n_{args.num_agents}a_{time.strftime('%Y_%m_%d_%Hh%Mm')}"

    writer = SummaryWriter(f"{args.log_path}/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    env = build_env(args)
    nodes_stat_dim = env.obs_builder.get_nodes_static_feat_dim()
    nodes_dyn_dim = env.obs_builder.get_nodes_dynamic_feat_dim()
    agent_dim = env.obs_builder.get_agent_feat_dim()
    agents_dim = env.obs_builder.get_other_agents_feat_dim()
    global_dim = env.obs_builder.get_global_feat_dim()

    learner = Learner(
        nodes_stat_obs_dim=nodes_stat_dim,
        nodes_dyn_obs_dim=nodes_dyn_dim,
        agent_obs_dim=agent_dim,
        agents_obs_dim=agents_dim,
        global_obs_dim=global_dim,
        embed_dim=args.embed_dim,
    ).to(args.device)

    optim = Adam(learner.parameters(), lr=args.learning_rate)

    print("train_madyam_obs (old model) args:", vars(args))
    t0 = time.time()
    for ep in range(args.epoch_count):
        loss, ret = train_epoch(args, env, learner, optim, ep, writer)
        print(f"[ep {ep:03d}] loss={loss:.4f} return={ret:.2f} elapsed={time.time()-t0:.1f}s")
        writer.add_scalar("epoch/mean_loss", float(loss), ep)
        writer.add_scalar("epoch/mean_return", float(ret), ep)

    writer.close()


if __name__ == "__main__":
    main()
