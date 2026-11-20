import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Absolute-feature actor-critic for dtoptw_abs.

Third variant:
- Actor is kept identical to policy_net_ma_ac_abs.
- Critic replaces full-node self-attention with a two-stage inducing-point attention block:
  1) learned inducing tokens attend over all node embeddings with the current decode mask;
  2) the current agent query attends only to the induced summaries to produce V(s).

This avoids the O(N^2) full-node encoder inside the critic while keeping the external Learner
API compatible with the existing training / evaluation scripts.
"""


class AttentionScore(nn.Module):
    def __init__(self, use_tanh: bool = False, C: float = 10.0):
        super().__init__()
        self.use_tanh = use_tanh
        self.C = C

    def forward(self, query, key, mask=None):
        u = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        logits = torch.tanh(u) * self.C if self.use_tanh else u
        if mask is not None:
            m = mask
            if logits.dim() == 3:
                if m.dim() == 2:
                    m = m.unsqueeze(1)
            elif logits.dim() == 4:
                if m.dim() == 2:
                    m = m.unsqueeze(0).unsqueeze(2)
                elif m.dim() == 3:
                    m = m.unsqueeze(0)
            else:
                while m.dim() < logits.dim():
                    m = m.unsqueeze(0)
            logits = logits.masked_fill(m.expand_as(logits) == False, float("-inf"))
        return logits


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = 8):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.attn_score = AttentionScore()
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def _make_heads(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        num_heads = self.n_heads
        return x.view(batch_size, num_tokens, num_heads, embed_dim // num_heads).permute(2, 0, 1, 3)

    def _mask_to_sdpa_bias(self, mask, query, key):
        if mask is None:
            return None
        m = mask.to(device=query.device, dtype=torch.bool)
        if m.dim() == 2:
            m = m.unsqueeze(1).unsqueeze(2)
        elif m.dim() == 3:
            m = m.unsqueeze(1)
        elif m.dim() != 4:
            raise ValueError(f"Unsupported mask rank for SDPA: {m.dim()}")
        m = m.expand(query.size(1), -1, query.size(2), key.size(2)).permute(1, 0, 2, 3)
        bias = torch.zeros_like(m, dtype=query.dtype)
        bias = bias.masked_fill(~m, float("-inf"))
        return bias

    def _manual_attention(self, query, key, value, mask):
        qh = self._make_heads(query)
        kh = self._make_heads(key)
        vh = self._make_heads(value)

        compat = self.attn_score(qh, kh, mask)
        attn = torch.softmax(compat, dim=-1)
        out_h = torch.matmul(attn, vh)
        out = out_h.permute(1, 2, 0, 3).contiguous().view(query.size(0), query.size(1), -1)
        return self.project_out(out)

    def forward(self, query, key, value, mask):
        qh = self._make_heads(query).permute(1, 0, 2, 3)
        kh = self._make_heads(key).permute(1, 0, 2, 3)
        vh = self._make_heads(value).permute(1, 0, 2, 3)

        if hasattr(F, "scaled_dot_product_attention"):
            try:
                attn_bias = self._mask_to_sdpa_bias(mask, qh, kh)
                out = F.scaled_dot_product_attention(
                    qh,
                    kh,
                    vh,
                    attn_mask=attn_bias,
                    dropout_p=0.0,
                    is_causal=False,
                )
                out = out.permute(0, 2, 1, 3).contiguous().view(query.size(0), query.size(1), -1)
                return self.project_out(out)
            except RuntimeError:
                pass
            except NotImplementedError:
                pass

        return self._manual_attention(query, key, value, mask)


class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class MultiHeadAttentionProj(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = 8):
        super().__init__()
        self.queryEncoder = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keyEncoder = nn.Linear(embed_dim, embed_dim, bias=False)
        self.valueEncoder = nn.Linear(embed_dim, embed_dim, bias=False)
        self.MHA = MultiHeadAttention(embed_dim=embed_dim, n_heads=n_heads)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q
        query = self.queryEncoder(q)
        key = self.keyEncoder(h)
        value = self.valueEncoder(h)
        return self.MHA(query, key, value, mask)


class Normalization(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_dim, affine=True, track_running_stats=False)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        return self.norm(x.view(batch_size * num_tokens, embed_dim)).view(batch_size, num_tokens, embed_dim)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, feed_forward_hidden=512):
        super().__init__()
        self.attn = SkipConnection(MultiHeadAttentionProj(embed_dim=embed_dim, n_heads=n_heads))
        self.norm1 = Normalization(embed_dim)
        self.ff = SkipConnection(FeedForward(embed_dim, feed_forward_hidden))
        self.norm2 = Normalization(embed_dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.norm1(x)
        x = self.ff(x)
        x = self.norm2(x)
        return x


class GraphAttentionEncoder(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int, n_layers: int, feed_forward_hidden: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(n_heads=n_heads, embed_dim=embed_dim, feed_forward_hidden=feed_forward_hidden)
                for _ in range(n_layers)
            )
        )

    def forward(self, x):
        h = self.layers(x)
        return h, h.mean(dim=1)


class ActorNet(nn.Module):
    def __init__(
        self,
        nodes_stat_dim: int,
        agent_dim: int,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        feed_forward_hidden: int = 512,
    ):
        super().__init__()
        self.nodes_embedding = nn.Linear(nodes_stat_dim, embed_dim, bias=False)
        self.agent_embedding = nn.Linear(agent_dim, embed_dim, bias=False)
        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            feed_forward_hidden=feed_forward_hidden,
        )
        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.nodes_glimpse = MultiHeadAttention(embed_dim=embed_dim, n_heads=n_heads)
        self.pointer = AttentionScore(use_tanh=True, C=10.0)
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, node_static_obs, agent_obs, action_mask):
        node_emb = self.nodes_embedding(node_static_obs)
        node_enc, _ = self.encoder(node_emb)
        glimpse_K, glimpse_V, logit_K = self.project_node_embeddings(node_enc).chunk(3, dim=-1)

        query = self.agent_embedding(agent_obs).unsqueeze(1)
        glimpse = self.nodes_glimpse(query, glimpse_K, glimpse_V, action_mask)
        logits = self.pointer(glimpse, logit_K, action_mask).squeeze(1)
        return logits, glimpse.squeeze(1)


class InducedCriticNet(nn.Module):
    """
    Memory-efficient critic using inducing-point attention.

    Stage 1: learned inducing tokens attend over all node embeddings under the current decode mask.
    Stage 2: the current agent query attends over the induced summaries.
    """

    def __init__(
        self,
        nodes_stat_dim: int,
        agent_dim: int,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_inducing_points: int = 32,
        feed_forward_hidden: int = 512,
    ):
        super().__init__()
        self.n_inducing_points = int(n_inducing_points)
        self.nodes_embedding = nn.Linear(nodes_stat_dim, embed_dim, bias=False)
        self.agent_embedding = nn.Linear(agent_dim, embed_dim, bias=False)
        self.inducing_tokens = nn.Parameter(torch.empty(1, self.n_inducing_points, embed_dim))
        self.induce_attn = MultiHeadAttentionProj(embed_dim=embed_dim, n_heads=n_heads)
        self.induce_norm1 = Normalization(embed_dim)
        self.induce_ff = SkipConnection(FeedForward(embed_dim, feed_forward_hidden))
        self.induce_norm2 = Normalization(embed_dim)
        self.agent_to_induced = MultiHeadAttentionProj(embed_dim=embed_dim, n_heads=n_heads)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=False),
        )
        self._cached_induced = None
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_cache_(self, node_static_all):
        node_emb = self.nodes_embedding(node_static_all)
        batch_size = node_emb.size(0)
        inducing = self.inducing_tokens.expand(batch_size, -1, -1)
        induced = inducing + self.induce_attn(inducing, node_emb, mask=None)
        induced = self.induce_norm1(induced)
        induced = self.induce_ff(induced)
        induced = self.induce_norm2(induced)
        self._cached_induced = induced

    def clear_cache_(self):
        self._cached_induced = None

    def forward(self, node_static_all, agent_obs, decode_mask):
        if self._cached_induced is None:
            self.make_cache_(node_static_all)
        query = self.agent_embedding(agent_obs).unsqueeze(1)
        agent_ctx = self.agent_to_induced(query, self._cached_induced, mask=None)
        value = self.value_head(agent_ctx.squeeze(1))
        return value


class Learner(nn.Module):
    def __init__(
        self,
        nodes_stat_dim: int,
        agent_dim: int,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        feed_forward_hidden: int = 512,
        n_inducing_points: int = 16,
    ):
        super().__init__()
        self.actor = ActorNet(
            nodes_stat_dim=nodes_stat_dim,
            agent_dim=agent_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            feed_forward_hidden=feed_forward_hidden,
        )
        self.critic = InducedCriticNet(
            nodes_stat_dim=nodes_stat_dim,
            agent_dim=agent_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_inducing_points=n_inducing_points,
            feed_forward_hidden=feed_forward_hidden,
        )

    def make_critic_cache_(self, critic_node_static_obs, *, with_grad: bool = False):
        ctx = nullcontext() if with_grad else torch.no_grad()
        with ctx:
            self.critic.make_cache_(critic_node_static_obs)

    def get_action_and_value(
        self,
        node_static_obs,
        agent_obs,
        action_mask,
        critic_node_static_obs,
        critic_nodes_mask,
        action=None,
        deterministic: bool = False,
    ):
        logits, _glimpse = self.actor(node_static_obs, agent_obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.mode if deterministic else dist.sample()
        logp = dist.log_prob(action)
        ent = dist.entropy()

        value = self.critic(critic_node_static_obs, agent_obs, critic_nodes_mask)
        return action, logp, ent, value

    def evaluate_actions(
        self,
        node_static_obs,
        agent_obs,
        action_mask,
        critic_node_static_obs,
        critic_nodes_mask,
        action,
    ):
        self.critic.clear_cache_()
        return self.get_action_and_value(
            node_static_obs=node_static_obs,
            agent_obs=agent_obs,
            action_mask=action_mask,
            critic_node_static_obs=critic_node_static_obs,
            critic_nodes_mask=critic_nodes_mask,
            action=action,
            deterministic=False,
        )
