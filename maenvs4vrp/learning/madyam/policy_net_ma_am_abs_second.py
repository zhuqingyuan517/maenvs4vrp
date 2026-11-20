import math

import torch
import torch.nn as nn

from maenvs4vrp.learning.mardam.layers import MultiHeadAttention, TransformerEncoder


class PointerScore(nn.Module):
    def __init__(self, model_size: int, tanh_clipping: float = 10.0):
        super().__init__()
        self.query_proj = nn.Linear(model_size, model_size, bias=False)
        self.key_proj = nn.Linear(model_size, model_size, bias=False)
        self.model_size = model_size
        self.tanh_clipping = tanh_clipping

    def forward(self, query, keys, mask=None):
        query = self.query_proj(query)
        keys = self.key_proj(keys)
        logits = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.model_size)
        if self.tanh_clipping is not None:
            logits = self.tanh_clipping * torch.tanh(logits)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        return logits


class AttentionModel(nn.Module):
    """
    Autoregressive attention policy for dtoptw_abs_second.

    This policy is decoder-style: at each environment decision, it encodes the current
    local candidate set and samples the next action from a pointer distribution.
    """

    def __init__(
        self,
        nodes_stat_dim: int,
        agent_dim: int,
        other_agents_dim: int = 0,
        model_size: int = 128,
        layer_count: int = 3,
        head_count: int = 8,
        ff_size: int = 512,
        tanh_clipping: float = 10.0,
    ):
        super().__init__()
        self.model_size = model_size
        self.other_agents_dim = int(other_agents_dim)

        self.node_embedding = nn.Linear(nodes_stat_dim, model_size, bias=False)
        self.agent_embedding = nn.Linear(agent_dim, model_size, bias=False)
        self.other_agents_embedding = (
            nn.Linear(other_agents_dim, model_size, bias=False) if self.other_agents_dim > 0 else None
        )

        self.node_encoder = TransformerEncoder(layer_count, head_count, model_size, ff_size)
        self.team_glimpse = MultiHeadAttention(head_count, model_size)
        self.decode_glimpse = MultiHeadAttention(head_count, model_size)

        # Standard AM-style decoder context:
        # [graph embedding, last selected node embedding, first selected node embedding].
        self.context_to_query = nn.Linear(3 * model_size, model_size, bias=False)
        self.node_key_proj = nn.Linear(model_size, model_size, bias=False)
        self.node_val_proj = nn.Linear(model_size, model_size, bias=False)
        self.query_norm = nn.LayerNorm(model_size)
        self.pointer = PointerScore(model_size=model_size, tanh_clipping=tanh_clipping)
        self.value_head = nn.Sequential(
            nn.Linear(3 * model_size, model_size, bias=False),
            nn.ReLU(),
            nn.Linear(model_size, 1, bias=False),
        )

        self.init_last_placeholder = nn.Parameter(torch.empty(model_size))
        self.init_first_placeholder = nn.Parameter(torch.empty(model_size))

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _masked_mean(self, x, mask):
        weights = mask.unsqueeze(-1).to(x.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def _encode_nodes(self, node_static_obs, action_mask):
        hidden_mask = ~action_mask.bool()
        node_emb = self.node_embedding(node_static_obs)
        node_enc = self.node_encoder(node_emb, hidden_mask)
        graph_ctx = self._masked_mean(node_enc, action_mask.bool())
        return node_enc, graph_ctx

    def _compute_team_context(self, query, other_agents_obs=None, agents_mask=None):
        if (
            self.other_agents_embedding is None
            or other_agents_obs is None
            or other_agents_obs.numel() == 0
            or other_agents_obs.size(1) == 0
        ):
            return torch.zeros_like(query)

        team_mask = agents_mask.bool() if agents_mask is not None else torch.ones(
            other_agents_obs.shape[:2], dtype=torch.bool, device=other_agents_obs.device
        )
        team_emb = self.other_agents_embedding(other_agents_obs)
        if not team_mask.any():
            return torch.zeros_like(query)

        # Avoid all-masked rows causing NaNs inside attention.
        has_any = team_mask.any(dim=1, keepdim=True)
        safe_mask = torch.where(has_any, team_mask, torch.ones_like(team_mask))
        return self.team_glimpse(query, team_emb, team_emb, ~safe_mask.unsqueeze(1))

    def _build_decoder_query(self, graph_ctx, first_node_embed=None, last_node_embed=None):
        batch_size = graph_ctx.size(0)
        if first_node_embed is None:
            first_node_embed = self.init_first_placeholder.unsqueeze(0).expand(batch_size, -1)
        if last_node_embed is None:
            last_node_embed = self.init_last_placeholder.unsqueeze(0).expand(batch_size, -1)

        context = torch.cat([graph_ctx, last_node_embed, first_node_embed], dim=-1)
        return self.context_to_query(context).unsqueeze(1)

    def get_action_and_value(self, obs, first_node_embed=None, last_node_embed=None, deterministic: bool = False):
        node_static_obs = obs["node_static_obs"]
        agent_obs = obs["agent_obs"]
        action_mask = obs["action_mask"].bool()
        other_agents_obs = obs.get("other_agents_obs", None)
        agents_mask = obs.get("agents_mask", None)

        node_enc, graph_ctx = self._encode_nodes(node_static_obs, action_mask)
        query = self._build_decoder_query(
            graph_ctx,
            first_node_embed=first_node_embed,
            last_node_embed=last_node_embed,
        )
        team_ctx = self._compute_team_context(query, other_agents_obs=other_agents_obs, agents_mask=agents_mask)
        agent_ctx = self.agent_embedding(agent_obs)
        query = self.query_norm(query + team_ctx + agent_ctx.unsqueeze(1))

        key = self.node_key_proj(node_enc)
        value = self.node_val_proj(node_enc)
        node_ctx = self.decode_glimpse(query, key, value, ~action_mask.unsqueeze(1))
        logits = self.pointer(node_ctx, key, mask=action_mask).squeeze(1)
        critic_state = torch.cat([graph_ctx, query.squeeze(1), node_ctx.squeeze(1)], dim=-1)
        state_value = self.value_head(critic_state).squeeze(-1)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        selected_embed = node_enc.gather(1, action.view(-1, 1, 1).expand(-1, 1, self.model_size)).squeeze(1)
        return action, logp, entropy, state_value, selected_embed

    def get_action(self, obs, first_node_embed=None, last_node_embed=None, deterministic: bool = False):
        action, logp, entropy, _value, selected_embed = self.get_action_and_value(
            obs,
            first_node_embed=first_node_embed,
            last_node_embed=last_node_embed,
            deterministic=deterministic,
        )
        return action, logp, entropy, selected_embed

    def rollout(self, env, td, deterministic: bool = False):
        actions = []
        logps = []
        rewards = []
        entropies = []
        values = []
        active_masks = []
        first_node_embed = None
        last_node_embed = None

        while not td["done"].all():
            active_masks.append((~td["done"]).float())
            obs = td["observations"]
            action, logp, entropy, state_value, selected_embed = self.get_action_and_value(
                obs,
                first_node_embed=first_node_embed,
                last_node_embed=last_node_embed,
                deterministic=deterministic,
            )

            td["action"] = action.unsqueeze(1)
            td = env.step(td)

            actions.append(action)
            logps.append(logp)
            entropies.append(entropy)
            values.append(state_value)
            step_reward = td["reward"] + td["penalty"]
            if step_reward.dim() > 1:
                step_reward = step_reward.squeeze(-1)
            rewards.append(step_reward)
            if first_node_embed is None:
                first_node_embed = selected_embed
            last_node_embed = selected_embed

        return actions, logps, rewards, entropies, values, active_masks
