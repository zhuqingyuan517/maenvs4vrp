"""
Training script for madyam model


"""

import sys
sys.path.insert(0, '../')

import argparse
from distutils.util import strtobool

from tqdm import tqdm, trange
from itertools import chain
import numpy as np

import time
import random
import os.path as osp
import os
import torch
from tensordict import TensorDict

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torch.nn.functional as F

import importlib

import copy

from maenvs4vrp.learning.madyam.layers import reinforce_loss
from maenvs4vrp.learning.madyam.policy_net_ma_ac import Learner

def save_model_state_dict(save_path, model_policy):
    # save the policy state dict 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state_dict = model_policy.to("cpu").state_dict()
    torch.save(state_dict, save_path)
    

def set_random_seed(seed, torch_deterministic):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic



@torch.no_grad()
def _greedy_rollout_rewards(env, learner_bl, td_init):
    """
    Greedy rollout using baseline learner (deterministic=True).
    Returns: list of step rewards, each tensor shaped [B, 1] (same as td['reward']).
    """
    td = td_init
    rewards_bl = []

    node_stat_obs = td['observations']['node_static_obs']

    while not td["done"].all():
        obs = td['observations']
        action, _, _, _ = learner_bl.get_action_and_logs(
            nodes_stat_obs=obs.get('node_static_obs', node_stat_obs),
            nodes_dyn_obs=obs['node_dynamic_obs'],
            self_obs=obs['agent_obs'],
            agents_obs=obs['other_agents_obs'],
            agents_mask=obs['agents_mask'],
            global_obs=obs['global_obs'],
            action_mask=obs['action_mask'],
            compute_value=False,
            deterministic=True
        )
        td['action'] = action.unsqueeze(1)
        td = env.step(td)
        rewards_bl.append(td['reward'] + td['penalty'])

    return rewards_bl


def _rewards_to_returns(rewards, discount=1.0):
    """
    Convert list of step rewards -> list of reward-to-go returns (same length).
    """
    cumul = torch.zeros_like(rewards[0])
    vals = []
    for r in reversed(rewards):
        cumul = r + discount * cumul
        vals.append(cumul)
    vals.reverse()
    return vals


def train_epoch(args, env, learner, optim, ep, writer):
    """
    REINFORCE + Greedy Rollout Baseline (epoch-frozen baseline policy).
    - Sampling rollout: learner (stochastic)
    - Baseline rollout: learner_bl (deterministic greedy), fixed for the whole epoch
    """
    learner.train()

    # ---- build epoch-frozen baseline learner (copy current learner at epoch start) ----
    learner_bl = copy.deepcopy(learner)
    learner_bl.eval()
    # freeze baseline parameters (no grad)
    for p in learner_bl.parameters():
        p.requires_grad_(False)

    ep_loss = 0.0
    ep_prob = 0.0
    ep_val = 0.0
    ep_bl = 0.0
    ep_norm = 0.0

    with trange(args.iter_count, desc="Ep.#{: >3d}/{: <3d}".format(ep + 1, args.epoch_count)) as progress:
        for episode in progress:
            seed_i = args.seed + ep * args.iter_count + episode

            # ------------------- sampling rollout (train policy) -------------------
            td = env.reset(
                num_agents=args.num_agents,
                num_nodes=args.num_nodes,
                profits=getattr(args, "profits", "distance"),
                seed=seed_i,
                sample_type='random',
                max_steps=getattr(args, "max_steps", None),
                max_flight_dist=getattr(args, "max_flight_dist", None),
                allow_depot_refuel=getattr(args, "allow_depot_refuel", False),
                horizon=getattr(args, "horizon", None),
            )

            logps = []
            entropy = []
            rewards = []
            step_mask = []

            node_stat_obs = td['observations']['node_static_obs']

            while not td["done"].all():
                obs = td['observations']

                action, logprobs, ent, _values = learner.get_action_and_logs(
                    nodes_stat_obs=obs.get('node_static_obs', node_stat_obs),
                    nodes_dyn_obs=obs['node_dynamic_obs'],
                    self_obs=obs['agent_obs'],
                    agents_obs=obs['other_agents_obs'],
                    agents_mask=obs['agents_mask'],
                    global_obs=obs['global_obs'],
                    action_mask=obs['action_mask'],
                    critic_nodes_stat_obs=obs.get('critic_node_static_obs', None),
                    critic_nodes_dyn_obs=obs.get('critic_node_dynamic_obs', None),
                    critic_self_obs=obs.get('critic_agent_obs', None),
                    critic_agents_obs=obs.get('critic_other_agents_obs', None),
                    critic_agents_mask=obs.get('critic_agents_mask', None),
                    critic_global_obs=obs.get('critic_global_obs', None),
                    critic_nodes_mask=obs.get('critic_nodes_mask', None),
                    compute_value=False,
                )

                td['action'] = action.unsqueeze(1)
                td = env.step(td)

                logps.append(logprobs.unsqueeze(1))          # [B,1]
                entropy.append(ent.unsqueeze(1))             # [B,1] maybe
                rewards.append(td['reward'] + td['penalty']) # [B,1]
                step_mask.append(~td['done'].unsqueeze(1))   # [B,1]

            # ------------------- greedy rollout baseline (same instance, same seed) -------------------
            with torch.no_grad():
                td_bl = env.reset(
                    num_agents=args.num_agents,
                    num_nodes=args.num_nodes,
                    profits=getattr(args, "profits", "distance"),
                    seed=seed_i,
                    sample_type='random',
                    max_steps=getattr(args, "max_steps", None),
                    max_flight_dist=getattr(args, "max_flight_dist", None),
                    allow_depot_refuel=getattr(args, "allow_depot_refuel", False),
                    horizon=getattr(args, "horizon", None),
                )
                rewards_bl = _greedy_rollout_rewards(env, learner_bl, td_bl)  # list of [B,1]
                baseline_vals = _rewards_to_returns(rewards_bl, discount=1.0) # list of [B,1]

                # Make baseline length match the sampled trajectory length (robust padding/truncation)
                T = len(logps)
                if len(baseline_vals) < T:
                    pad = baseline_vals[-1]
                    baseline_vals = baseline_vals + [pad] * (T - len(baseline_vals))
                elif len(baseline_vals) > T:
                    baseline_vals = baseline_vals[:T]

            # ------------------- loss & update -------------------
            loss = reinforce_loss(logps, rewards, baseline_vals)

            prob = torch.stack(logps).sum(0).exp().mean()
            val = torch.stack(rewards).sum(0).mean()          # episodic return (sum of step rewards)
            bl = baseline_vals[0].mean()                      # baseline at t=0 (return-to-go)

            optim.zero_grad()
            loss.backward()

            grad_norm = 0.0
            if args.max_grad_norm is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    chain.from_iterable(grp["params"] for grp in optim.param_groups),
                    args.max_grad_norm
                ).item()

            optim.step()

            progress.set_postfix_str(
                "l={:.4g} p={:9.4g} val={:6.4g} bl={:6.4g} |g|={:.4g}".format(
                    float(loss), float(prob), float(val), float(bl), float(grad_norm)
                )
            )

            # ------------------- logging (keep your original stats) -------------------
            not_visited_nodes = env.td_state['nodes']['active_nodes_mask'].sum(-1).float() - 1
            number_used_agents = env.td_state['agents']['visited_nodes'].sum(-1).gt(1).sum(-1).float()

            writer.add_scalar("losses/episodic_loss", float(loss), episode)
            writer.add_scalar("losses/episodic_prob", float(prob), episode)

            writer.add_scalar("train/episodic_return", float(val), episode)
            writer.add_scalar("train/episodic_baseline", float(bl), episode)

            writer.add_scalar("train/episodic_not_visited_nodes", not_visited_nodes.mean().item(), episode)
            writer.add_scalar("train/episodic_number_used_agents", number_used_agents.mean().item(), episode)

            ep_loss += float(loss)
            ep_prob += float(prob)
            ep_val += float(val)
            ep_bl += float(bl)
            ep_norm += float(grad_norm)

    writer.add_scalar("losses/epoch_loss", ep_loss / args.iter_count, ep)
    writer.add_scalar("losses/epoch_prob", ep_prob / args.iter_count, ep)

    writer.add_scalar("train/epoch_return", ep_val / args.iter_count, ep)
    writer.add_scalar("train/epoch_baseline", ep_bl / args.iter_count, ep)

    return tuple(stat / args.iter_count for stat in (ep_loss, ep_prob, ep_val, ep_bl, ep_norm))



# xxx
# def test_epoch
def eval_epoch(args, test_env, learner, ep, writer):
    learner.eval()

    total_reward = []
    not_visited_nodes = []
    number_used_agents = []

    instances = getattr(test_env.inst_generator, "set_of_instances", None)

    with torch.no_grad():
        # --------- 分支1：有验证集（saved instances）---------
        if instances is not None:
            for instance_name in instances:
                td = test_env.reset(num_agents=args.num_agents,
                                    num_nodes=args.num_nodes,
                                    sample_type='saved',
                                    instance_name=instance_name,
                                    max_steps=getattr(args, "max_steps", None),
                                    max_flight_dist=getattr(args, "max_flight_dist", None),
                                    allow_depot_refuel=getattr(args, "allow_depot_refuel", False),
                                    horizon=getattr(args, "horizon", None),
                                    seed=0)
                f_reward = []
                node_stat_obs = td['observations']['node_static_obs']

                while not td["done"].all():
                    node_dyn_obs = td['observations']['node_dynamic_obs']
                    action_mask = td['observations']['action_mask']
                    self_obs = td['observations']['agent_obs']
                    global_obs = td['observations']['global_obs']
                    agents_mask = td['observations']['agents_mask']
                    agents_obs = td['observations']['other_agents_obs']
                    node_stat_obs = td['observations'].get('node_static_obs', node_stat_obs)
                    obs = td['observations']

                    action, _, _, _ = learner.get_action_and_logs(
                        nodes_stat_obs=node_stat_obs,
                        nodes_dyn_obs=node_dyn_obs,
                        self_obs=self_obs,
                        agents_obs=agents_obs,
                        agents_mask=agents_mask,
                        global_obs=global_obs,
                        action_mask=action_mask,
                        critic_nodes_stat_obs=obs.get('critic_node_static_obs', None),
                        critic_nodes_dyn_obs=obs.get('critic_node_dynamic_obs', None),
                        critic_self_obs=obs.get('critic_agent_obs', None),
                        critic_agents_obs=obs.get('critic_other_agents_obs', None),
                        critic_agents_mask=obs.get('critic_agents_mask', None),
                        critic_global_obs=obs.get('critic_global_obs', None),
                        critic_nodes_mask=obs.get('critic_nodes_mask', None),
                        compute_value=False,
                        deterministic=True
                    )

                    td['action'] = action.unsqueeze(1)
                    td = test_env.step(td)
                    f_reward.append(td['reward'] + td['penalty'])

                total_reward.append(torch.cat(f_reward, dim=1).sum(-1))
                not_visited_nodes.append(test_env.td_state['nodes']['active_nodes_mask'].sum(-1).float() - 1)
                number_used_agents.append(test_env.td_state['agents']['visited_nodes'].sum(-1).gt(1).sum(-1).float())

        # --------- 分支2：没有验证集（random eval）---------
        else:
            # 用 eval_batch_size 跑若干个随机 episode 当作验证
            n_eval_episodes = 20  # 你想评估多少个 episode，可调
            for k in range(n_eval_episodes):
                td = test_env.reset(num_agents=args.num_agents,
                                    num_nodes=args.num_nodes,
                                    sample_type='random',
                                    seed=args.eval_seed + ep * 1000 + k,
                                    max_steps=getattr(args, "max_steps", None),
                                    max_flight_dist=getattr(args, "max_flight_dist", None),
                                    allow_depot_refuel=getattr(args, "allow_depot_refuel", False),
                                    horizon=getattr(args, "horizon", None),
                                    )

                f_reward = []
                node_stat_obs = td['observations']['node_static_obs']

                while not td["done"].all():
                    node_dyn_obs = td['observations']['node_dynamic_obs']
                    action_mask = td['observations']['action_mask']
                    self_obs = td['observations']['agent_obs']
                    global_obs = td['observations']['global_obs']
                    agents_mask = td['observations']['agents_mask']
                    agents_obs = td['observations']['other_agents_obs']
                    node_stat_obs = td['observations'].get('node_static_obs', node_stat_obs)
                    obs = td['observations']

                    action, _, _, _ = learner.get_action_and_logs(
                        nodes_stat_obs=node_stat_obs,
                        nodes_dyn_obs=node_dyn_obs,
                        self_obs=self_obs,
                        agents_obs=agents_obs,
                        agents_mask=agents_mask,
                        global_obs=global_obs,
                        action_mask=action_mask,
                        critic_nodes_stat_obs=obs.get('critic_node_static_obs', None),
                        critic_nodes_dyn_obs=obs.get('critic_node_dynamic_obs', None),
                        critic_self_obs=obs.get('critic_agent_obs', None),
                        critic_agents_obs=obs.get('critic_other_agents_obs', None),
                        critic_agents_mask=obs.get('critic_agents_mask', None),
                        critic_global_obs=obs.get('critic_global_obs', None),
                        critic_nodes_mask=obs.get('critic_nodes_mask', None),
                        compute_value=False,
                        deterministic=True
                    )

                    td['action'] = action.unsqueeze(1)
                    td = test_env.step(td)
                    f_reward.append(td['reward'] + td['penalty'])

                total_reward.append(torch.cat(f_reward, dim=1).sum(-1))
                not_visited_nodes.append(test_env.td_state['nodes']['active_nodes_mask'].sum(-1).float() - 1)
                number_used_agents.append(test_env.td_state['agents']['visited_nodes'].sum(-1).gt(1).sum(-1).float())

        total_reward = torch.cat(total_reward).mean()
        not_visited_nodes = torch.cat(not_visited_nodes).mean()
        number_used_agents = torch.cat(number_used_agents).mean()

        writer.add_scalar("eval/episodic_return", total_reward, ep)
        writer.add_scalar("eval/episodic_not_visited_nodes", not_visited_nodes, ep)
        writer.add_scalar("eval/episodic_number_used_agents", number_used_agents, ep)

    print("Reward on eval: {:5.2f}".format(total_reward))
    return total_reward, not_visited_nodes, number_used_agents


def train(args, writer):

    """ ENV SETUP """

    if args.selection == 'rand':
        env_agent_selector_module_name = f'maenvs4vrp.environments.{args.vrp_env}.env_agent_selector'
        env_agent_selector = importlib.import_module(env_agent_selector_module_name).RandomSelector()
    elif args.selection == 'single':
        env_agent_selector_module_name = f'maenvs4vrp.environments.{args.vrp_env}.env_agent_selector'
        env_agent_selector = importlib.import_module(env_agent_selector_module_name).AgentSelector()
    elif args.selection == 'stime':
        env_agent_selector_module_name = f'maenvs4vrp.environments.{args.vrp_env}.env_agent_selector'
        env_agent_selector = importlib.import_module(env_agent_selector_module_name).SmallestTimeAgentSelector()

    observations_module_name = f'maenvs4vrp.environments.{args.vrp_env}.observations'
    local_view = None
    if getattr(args, "use_local_view", False):
        local_view = {
            "k_nodes": getattr(args, "local_k_nodes", 30),
            "k_agents": getattr(args, "local_k_agents", 5),
            "include_depot": True,
            "mode": getattr(args, "local_view_mode", "nearest"),
        }
    obs_mod = importlib.import_module(observations_module_name)
    try:
        observations = obs_mod.Observations(local_view=local_view, return_critic_obs=getattr(args, "use_global_critic", False))
    except TypeError:
        observations = obs_mod.Observations()

    generator_module_name = f'maenvs4vrp.environments.{args.vrp_env}.instances_generator'
    generator = importlib.import_module(generator_module_name).InstanceGenerator(device=args.device)

    environment_module_name = f'maenvs4vrp.environments.{args.vrp_env}.env'
    environment_module = importlib.import_module(environment_module_name)

    env_agent_reward_module_name = f'maenvs4vrp.environments.{args.vrp_env}.env_agent_reward'
    # reward_evaluator = importlib.import_module(env_agent_reward_module_name).DenseReward()
    reward_evaluator = importlib.import_module(env_agent_reward_module_name).SparseReward()


    env = environment_module.Environment(instance_generator_object=generator,  
                    obs_builder_object=observations,
                    agent_selector_object=env_agent_selector,
                    reward_evaluator=reward_evaluator,
                    device=args.device,
                    batch_size = args.batch_size,
                    seed=args.seed)    
    
   
    if args.val_set == 'None':
        eval_generator = importlib.import_module(generator_module_name).InstanceGenerator(device=args.device) 
    else:
        set_of_instances = set(generator.get_list_of_benchmark_instances()[args.val_set]['validation'])
        eval_generator = importlib.import_module(generator_module_name).InstanceGenerator(set_of_instances=set_of_instances,
                                                                                          device=args.device) 
        args.eval_batch_size = None   

    eval_env = environment_module.Environment(instance_generator_object=eval_generator,  
                        obs_builder_object=observations,
                        agent_selector_object=env_agent_selector,
                        reward_evaluator=reward_evaluator,
                        device=args.device,
                        batch_size = args.eval_batch_size,
                        seed=args.eval_seed)
    

    nodes_static_obs_dim = env.obs_builder.get_nodes_static_feat_dim()
    nodes_dynamic_obs_dim = env.obs_builder.get_nodes_dynamic_feat_dim()
    agent_obs_dim = env.obs_builder.get_agent_feat_dim()
    agents_obs_dim = env.obs_builder.get_other_agents_feat_dim()
    global_obs_dim = env.obs_builder.get_global_feat_dim()

    """ ALGO LOGIC: EPISODE STORAGE"""
    start_time = time.time()

    print("Initializing attention model...")
    learner = Learner(nodes_stat_obs_dim=nodes_static_obs_dim, 
                     nodes_dyn_obs_dim=nodes_dynamic_obs_dim,  
                     agent_obs_dim=agent_obs_dim, 
                     agents_obs_dim=agents_obs_dim, 
                     global_obs_dim=global_obs_dim,
                     embed_dim=128)

    learner.to(args.device)


    # OPTIMIZER AND LR SCHEDULER
    print("Initializing Adam optimizer...")

    lr_sched = None
    optim = Adam([
        {"params": learner.policy.parameters(), "lr": args.learning_rate},
        {"params": learner.global_critic.parameters(), "lr": args.critic_rate}
        ])

    best_lb_total_return = -10000000

    """ TRAINING LOGIC """
    print("Running...")
    train_stats = []
    test_stats = []
    for ep in range(0, args.epoch_count):
        train_stats.append( train_epoch(args, env, learner, optim, ep, writer) )
        # test_metrics =  test_epoch(args, eval_env, learner, ep, writer)
        test_metrics =  eval_epoch(args, eval_env, learner, ep, writer)
        test_stats.append(test_metrics)
        latest_episodic_return = test_metrics[0]

        writer.add_scalar("charts/SPS", int(ep / (time.time() - start_time)), ep)

        print("\n-------------------------------------------\n")
        print ('Saving latest model...')
        save_model_state_dict(osp.join(args.log_path, "models/latest_model_"+args.run_name+".zip"), learner)
        learner.to(args.device)
        print ('done')

        if latest_episodic_return > best_lb_total_return:
            print ('Old best model: {: .2f}'.format(best_lb_total_return))
            best_lb_total_return = latest_episodic_return
            print ('New best model: {: .2f}'.format(latest_episodic_return))
            print ('Saving new best model')
            save_model_state_dict(osp.join(args.log_path, "models/best_model_"+args.run_name+".zip"), learner)
            learner.to(args.device)
            print ('done')
        else:
            print ('No improvement')
            print (f'Latest model: {latest_episodic_return}')
            print (f'Current best model: {best_lb_total_return}')

        writer.add_scalar("eval/best_model_lb_total_reward", best_lb_total_return, ep)

        print("\n-------------------------------------------\n")
         
    print ('saving latest model')
    save_model_state_dict(osp.join(args.log_path, "models/latest_model_"+args.run_name+".zip"), learner)
    learner.to(args.device)
    print ('done')
    writer.close()

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrp_env", type=str, default="dist_toptw", help="select the vrp environment to train on")
    parser.add_argument("--num_agents", type=int, default=3, help="number of agents")
    parser.add_argument("--num_nodes", type=int, default=51, help="number of nodes")
    parser.add_argument("--selection", type=str, default="rand", choices=['rand', 'single', 'stime'], help="next agent selection strategy")
    parser.add_argument("--val_set", type=str, default='None', help="validation set")
    parser.add_argument("--use_local_view", type=lambda x: bool(strtobool(x)), default=False, help="use local observations for the actor")
    parser.add_argument("--local_k_nodes", type=int, default=30, help="local nodes per decision")
    parser.add_argument("--local_k_agents", type=int, default=5, help="local agents per decision")
    parser.add_argument("--local_view_mode", type=str, default="nearest", help="local node selection mode: nearest|random")
    parser.add_argument("--use_global_critic", type=lambda x: bool(strtobool(x)), default=True, help="use an independent global critic")
    parser.add_argument("--max_steps", type=int, default=None, help="max environment steps per episode")
    parser.add_argument("--max_flight_dist", type=float, default=None, help="max flight distance per agent")
    parser.add_argument("--allow_depot_refuel", type=lambda x: bool(strtobool(x)), default=False, help="allow depot refuel without terminating agent")
    parser.add_argument("--horizon", type=int, default=None, help="planning horizon for dynamic instances")
    parser.add_argument("--profits", type=str, default="distance", choices=["distance", "constant", "uniform"], help="profit sampling mode")
    args = parser.parse_args()
    return args


def get_args():
    args = parse_args()
    args.model_name = 'our_net'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.learning_rate = 0.0001
    args.critic_rate = 0.001
    args.epoch_count = 100
    args.iter_count = 2500
    args.batch_size = 64
    args.eval_batch_size = 64
    args.max_grad_norm = 2
    args.torch_deterministic = True
    args.seed = 2297
    args.log_path = 'runs'
    args.eval_seed = 0
    args.time = time.strftime("%Y_%m_%d_%Hh%Mm")
    args.run_name = f"{args.model_name}_{args.vrp_env}_{args.selection}_{args.num_nodes}n_{args.num_agents}a_{args.time}"
    return args


def main(args):
    print("Training with args", args)

    if args.seed != None:
        set_random_seed(args.seed, args.torch_deterministic)

    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    train(args, writer)

if __name__ == "__main__":
    main(get_args())
