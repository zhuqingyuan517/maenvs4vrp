from __future__ import annotations

import argparse
import random

import torch

from maenvs4vrp.learning.madyam.alns_rhc_dtoptw_abs import select_alns_action
from maenvs4vrp.learning.madyam.policy_net_ma_ac_abs_third import Learner as InducedLearner
from maenvs4vrp.learning.madyam.replan_rhc_dtoptw_abs import build_env as build_alns_env
from maenvs4vrp.learning.madyam.train_madyam_abs import (
    _get_action_and_value,
    build_env as build_global_env,
)
from maenvs4vrp.learning.mardam._learner import AttentionLearner
from maenvs4vrp.learning.mardam.baselines import CriticBaseline
from maenvs4vrp.learning.mardam.train_mardam_dtoptw import (
    build_env as build_mardam_env,
    reset_dtoptw_env as reset_mardam_env,
)


def _alns_args(device: str) -> argparse.Namespace:
    return argparse.Namespace(
        vrp_env="dtoptw_full",
        num_agents=2,
        num_nodes=11,
        selection="stime",
        batch_size=1,
        batch_idx=0,
        seed=17,
        torch_deterministic=True,
        device=torch.device(device),
        local_k_nodes=5,
        local_view_mode="nearest",
        profits="constant",
        horizon=100,
        max_steps=8,
        max_flight_dist=300.0,
        allow_depot_refuel=True,
        forbid_depot_self_loop=True,
        done_feasible_check_freq=1,
        alns_route_length=4,
        alns_iterations=4,
        alns_destroy_max=2,
        alns_candidate_pool=11,
        alns_dist_weight=0.01,
        alns_wait_weight=0.001,
        alns_depot_penalty=0.05,
    )


def test_alns_selects_and_executes_a_legal_global_action(device: str) -> None:
    args = _alns_args(device)
    env = build_alns_env(args)
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

    action, plan = select_alns_action(args, td, env, rng=random.Random(args.seed))
    assert action.shape == (1,)
    assert bool(td["observations"]["action_mask"][0, int(action.item())].item())
    assert plan.feasible or not bool(td["observations"]["action_mask"].any().item())

    td["action"] = action.unsqueeze(1)
    td = env.step(td)
    assert td["reward"].shape[0] == 1


def test_mardam_baseline_completes_a_short_episode(device: str) -> None:
    args = argparse.Namespace(
        vrp_env="dtoptw_full",
        num_agents=2,
        num_nodes=11,
        selection="stime",
        batch_size=1,
        seed=19,
        torch_deterministic=True,
        device=torch.device(device),
        profits="constant",
        horizon=100,
        max_steps=6,
        max_flight_dist=300.0,
        allow_depot_refuel=True,
        forbid_depot_self_loop=True,
        done_feasible_check_freq=1,
        reward="dense",
    )
    env = build_mardam_env(args)
    td = reset_mardam_env(env, args, seed=args.seed)
    learner = AttentionLearner(
        cust_feat_size=env.obs_builder.get_nodes_static_feat_dim(),
        veh_state_size=env.obs_builder.get_other_agents_feat_dim(),
        model_size=32,
        layer_count=1,
        head_count=4,
        ff_size=64,
    ).to(args.device)
    baseline = CriticBaseline(learner, args.num_nodes - 1)
    baseline.to(args.device)

    _actions, logps, rewards, baselines, step_mask = baseline(env, td)
    assert logps
    assert rewards
    assert baselines
    assert step_mask
    assert torch.stack(rewards).squeeze(-1).shape[0] == len(logps)


def test_global_action_ablation_executes_a_legal_action(device: str) -> None:
    args = argparse.Namespace(
        vrp_env="dtoptw_abs_full",
        model_variant="third",
        num_agents=2,
        num_nodes=11,
        selection="stime",
        batch_size=1,
        seed=29,
        torch_deterministic=True,
        device=torch.device(device),
        tf32=False,
        local_k_nodes=5,
        local_view_mode="nearest",
        critic_mask_unappeared=False,
        critic_encode_unappeared=True,
        profits="constant",
        horizon=100,
        max_steps=6,
        max_flight_dist=300.0,
        allow_depot_refuel=True,
        forbid_depot_self_loop=False,
        done_feasible_check_freq=1,
        embed_dim=32,
        n_heads=4,
        n_layers=1,
        ff_hidden=64,
        critic_inducing_points=4,
    )
    env = build_global_env(args)
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
    obs = td["observations"]
    learner = InducedLearner(
        nodes_stat_dim=env.obs_builder.get_nodes_static_feat_dim(),
        agent_dim=env.obs_builder.get_agent_feat_dim(),
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        feed_forward_hidden=args.ff_hidden,
        n_inducing_points=args.critic_inducing_points,
    ).to(args.device)
    learner.make_critic_cache_(obs["node_static_obs"], with_grad=False)
    action, _logp, _entropy, _value = _get_action_and_value(
        args,
        learner,
        obs,
        deterministic=True,
    )
    assert action.shape == (1,)
    assert bool(obs["action_mask"][0, int(action.item())].item())
    td["action"] = action.unsqueeze(1)
    td = env.step(td)
    assert td["reward"].shape[0] == 1
