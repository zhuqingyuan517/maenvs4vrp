from maenvs4vrp.learning.mardam.baselines._base import Baseline
from maenvs4vrp.learning.mardam.numerics import (
    NonFiniteTrainingError,
    assert_finite,
    describe_nonfinite,
)

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CriticBaseline(Baseline):
    def __init__(self, learner, cust_count, use_qval = False, use_cumul_reward = False,
            grad_checkpoint = False, debug_numerics = False):
        super().__init__(learner, use_cumul_reward)
        self.use_qval = use_qval
        self.grad_checkpoint = grad_checkpoint
        self.debug_numerics = debug_numerics
        self.project = nn.Linear(cust_count+1, cust_count+1 if use_qval else 1, bias = False)

    def _check_finite(self, name, tensor, context=None):
        if self.debug_numerics:
            assert_finite(name, tensor, context=context)

    @staticmethod
    def _rollout_context(env, action_mask, attention_mask):
        empty_actions = (~action_mask.any(dim=-1)).sum().item()
        empty_attention = attention_mask.all(dim=-1).sum().item()
        return (
            f"env_steps={getattr(env, 'env_nsteps', None)}, "
            f"empty_action_rows={empty_actions}, "
            f"empty_attention_rows={empty_attention}"
        )

    def _repr_vehicle(self, agents_obs, cur_veh_idx, mask):
        if self.grad_checkpoint and torch.is_grad_enabled():
            return checkpoint(
                self.learner._repr_vehicle,
                agents_obs,
                cur_veh_idx,
                mask,
                use_reentrant=False,
            )
        return self.learner._repr_vehicle(agents_obs, cur_veh_idx, mask)

    def eval_step(self, cur_veh_mask, learner_compat, cust_idx):
        compat = learner_compat.clone()
        compat.masked_fill_(cur_veh_mask==True, 0)
        val = self.project(compat)
        if self.use_qval:
            val = val.gather(2, cust_idx.unsqueeze(1).expand(-1,1,-1))
        return val.squeeze(1)

    def __call__(self, env, td):

        node_stat_obs = td['observations']['node_static_obs']   

        actions, logps, rewards, bl_vals, step_mask = [], [], [], [], []
        
        self._check_finite("node_static_obs", node_stat_obs)
        self.learner._encode_customers(node_stat_obs)
        self._check_finite("customer_encoder_output", self.learner.cust_enc)
        self._check_finite("customer_representation", self.learner.cust_repr)

        while not td["done"].all():

            # rollover the observations
            node_dyn_obs = td['observations']['node_dynamic_obs']
            action_mask = td['observations']['action_mask']
            #self_obs = td['observations']['agent_obs']
            #global_obs = td['observations']['global_obs']
            agents_mask = td['observations']['agents_mask']
            agents_obs = td['observations']['other_agents_obs']
            node_obs = torch.cat((node_stat_obs, node_dyn_obs), dim=2)

            cur_veh_idx = td['cur_agent_idx']
            mask = ~env.td_state['agents']['feasible_nodes'].clone()
            context = (
                self._rollout_context(env, action_mask, mask)
                if self.debug_numerics
                else None
            )

            self._check_finite("other_agents_obs", agents_obs, context=context)
            veh_repr = self._repr_vehicle(agents_obs, cur_veh_idx, mask)
            self._check_finite("vehicle_representation", veh_repr, context=context)
            compat = self.learner._score_customers(veh_repr)
            self._check_finite("compatibility_scores", compat, context=context)
            logp = self.learner._get_logp(
                compat,
                ~action_mask.unsqueeze(1),
                debug=self.debug_numerics,
            )

            probabilities = logp.exp()
            probability_sums = probabilities.sum(dim=-1)
            if (
                not torch.isfinite(probabilities).all()
                or not torch.isfinite(probability_sums).all()
                or (probability_sums <= 0).any()
            ):
                details = describe_nonfinite(
                    "action_probabilities",
                    probabilities,
                    context=f"{context}, logp={describe_nonfinite('logp', logp)}",
                )
                raise NonFiniteTrainingError(details)
            cust_idx = probabilities.multinomial(1)
            if not(self.use_cumul and bl_vals):
                baseline_value = self.eval_step(~action_mask.unsqueeze(1), compat, cust_idx)
                self._check_finite("baseline_value", baseline_value, context=context)
                bl_vals.append(baseline_value)

            actions.append( (cur_veh_idx, cust_idx) )
            logps.append( logp.gather(1, cust_idx) )
            
            td['action'] = cust_idx
            td = env.step(td)
            reward = td['reward'] + td['penalty']
            self._check_finite("reward", reward, context=context)
            rewards.append(reward)
            step_mask.append( ~td['done'].unsqueeze(1) )

        if self.use_cumul:
            rewards = torch.stack(rewards).sum(dim = 0)
            bl_vals = bl_vals[0]
        return actions, logps, rewards, bl_vals, step_mask

    def parameters(self):
        return self.project.parameters()

    def state_dict(self):
        return self.project.state_dict()

    def load_state_dict(self, state_dict):
        return self.project.load_state_dict(state_dict)

    def to(self, device):
        self.project.to(device = device)
