import math
import torch
import torch.nn as nn
from contextlib import nullcontext


"""
Absolute-feature actor-critic for dtoptw_abs.

Based on the encoder-decoder / glimpse-pointer style in policy_net_ma_ac_old.py, but:
- Actor and critic do NOT share parameters.
- Actor observes only K local nodes (default 30) and re-encodes each decision (no cache).
- Critic observes all nodes (including not-yet-appeared) and caches encoder projections
  (glimpse_K, glimpse_V, logit_K) since node features are absolute (no relative-to-agent features).
- No "nearby agents" branch.

ATA variant:
- Actor includes an agents-attention encoder:
  current decision agent embedding is used as Query; other agents embeddings are Key/Value.
  The actor dynamically allocates attention weights over other agents and fuses this context
  with node-glimpse context before pointer decoding.
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
            # Here mask=True means "allowed". Support broadcasting to multi-head shapes.
            m = mask
            if logits.dim() == 3:
                # logits: [B,1,N]
                if m.dim() == 2:
                    m = m.unsqueeze(1)  # [B,1,N]
            elif logits.dim() == 4:
                # logits: [H,B,1,N]
                if m.dim() == 2:
                    m = m.unsqueeze(0).unsqueeze(2)  # [1,B,1,N]
                elif m.dim() == 3:
                    m = m.unsqueeze(0)  # [1,B,1,N]
            else:
                # Generic: add singleton dims to the front until ranks match.
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
        # x: [B, T, D] -> [H, B, T, D/H]
        B, T, D = x.shape
        H = self.n_heads
        return x.view(B, T, H, D // H).permute(2, 0, 1, 3)

    def forward(self, query, key, value, mask):
        qh = self._make_heads(query)
        kh = self._make_heads(key)
        vh = self._make_heads(value)

        compat = self.attn_score(qh, kh, mask)  # [H,B,1,N]
        attn = torch.softmax(compat, dim=-1)
        out_h = torch.matmul(attn, vh)  # [H,B,1,D/H]
        out = out_h.permute(1, 2, 0, 3).contiguous().view(query.size(0), query.size(1), -1)
        return self.project_out(out)


class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class MultiHeadAttentionProj(nn.Module):
    """Multi-head attention with linear projections (self-attn if h is None)."""

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
        # x: [B,N,D] -> batchnorm over D
        B, N, D = x.shape
        return self.norm(x.view(B * N, D)).view(B, N, D)


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
    """Actor with agents-attention: encode local K nodes and attend to other agents each step."""

    def __init__(
        self,
        nodes_stat_dim: int,
        agent_dim: int,
        other_agents_dim: int = 0,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        feed_forward_hidden: int = 512,
    ):
        super().__init__()
        self.nodes_embedding = nn.Linear(nodes_stat_dim, embed_dim, bias=False)
        self.agent_embedding = nn.Linear(agent_dim, embed_dim, bias=False)
        self.other_agents_dim = int(other_agents_dim)
        self.other_agents_embedding = (
            nn.Linear(other_agents_dim, embed_dim, bias=False) if self.other_agents_dim > 0 else None
        )
        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            feed_forward_hidden=feed_forward_hidden,
        )
        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        self.nodes_glimpse = MultiHeadAttention(embed_dim=embed_dim, n_heads=n_heads)
        self.agent_glimpse = MultiHeadAttentionProj(embed_dim=embed_dim, n_heads=n_heads)
        
        # ---------------------------------------------------------
        # 修改点 1: 移除原来的 agent_fusion (MLP)，替换为 LayerNorm
        # ---------------------------------------------------------
        self.agent_fusion_norm = nn.LayerNorm(embed_dim)
        
        self.pointer = AttentionScore(use_tanh=True, C=10.0)

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _compute_agents_context(self, query, other_agents_obs=None, agents_mask=None):
        # query: [B,1,D]
        if (
            self.other_agents_embedding is None
            or other_agents_obs is None
            or other_agents_obs.numel() == 0
            or other_agents_obs.size(1) == 0
        ):
            return torch.zeros_like(query)

        agents_embed = self.other_agents_embedding(other_agents_obs)  # [B,M,D]
        B, M, _ = agents_embed.shape
        if agents_mask is None:
            agents_mask = torch.ones((B, M), dtype=torch.bool, device=agents_embed.device)
        else:
            agents_mask = agents_mask.to(dtype=torch.bool, device=agents_embed.device)

        # Some batch rows legitimately have no valid "other agents" (e.g. single-agent
        # setting, or only the current agent remains active). In that case, skip the
        # attention call entirely for those rows to avoid all-masked softmax -> NaN.
        ctx = torch.zeros_like(query)
        valid_rows = agents_mask.any(dim=1)  # [B]
        if valid_rows.any():
            ctx_valid = self.agent_glimpse(
                query[valid_rows],
                agents_embed[valid_rows],
                agents_mask[valid_rows],
            )
            ctx[valid_rows] = ctx_valid
        return ctx

    def forward(self, node_static_obs, agent_obs, action_mask, other_agents_obs=None, agents_mask=None):
        # node_static_obs: [B,K,F], agent_obs: [B,A], action_mask: [B,K] (True allowed)
        node_emb = self.nodes_embedding(node_static_obs)
        node_enc, _ = self.encoder(node_emb)
        glimpse_K, glimpse_V, logit_K = self.project_node_embeddings(node_enc).chunk(3, dim=-1)

        query = self.agent_embedding(agent_obs).unsqueeze(1)  # [B,1,D]

        # 1) Query first attends to the fleet (other agents) to get team context.
        agents_glimpse = self._compute_agents_context(query, other_agents_obs, agents_mask)
        
        # ---------------------------------------------------------
        # 修改点 2: 采用特征加法 (Residual) 并进行 LayerNorm 归一化
        # ---------------------------------------------------------
        fused_query = self.agent_fusion_norm(query + agents_glimpse)

        # 2) Use the fused query to attend candidate nodes.
        node_glimpse = self.nodes_glimpse(fused_query, glimpse_K, glimpse_V, action_mask)
        logits = self.pointer(node_glimpse, logit_K, action_mask).squeeze(1)  # [B,K]
        return logits, node_glimpse.squeeze(1)


class CriticNet(nn.Module):
    """Critic: same encoder-decoder logic, but outputs V(s). Encodes all nodes with cache."""

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
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=False),
        )

        self._cached = None
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_cache_(self, node_static_all):
        # node_static_all: [B,N,F]
        node_emb = self.nodes_embedding(node_static_all)
        node_enc, _ = self.encoder(node_emb)
        self._cached = self.project_node_embeddings(node_enc).chunk(3, dim=-1)

    def clear_cache_(self):
        self._cached = None

    def forward(self, node_static_all, agent_obs, decode_mask):
        # decode_mask follows the shared critic-visibility protocol.
        if self._cached is None:
            self.make_cache_(node_static_all)
        glimpse_K, glimpse_V, _logit_K = self._cached

        query = self.agent_embedding(agent_obs).unsqueeze(1)
        glimpse = self.nodes_glimpse(query, glimpse_K, glimpse_V, decode_mask)  # [B,1,D]
        value = self.value_head(glimpse.squeeze(1))  # [B,1]
        return value


class Learner(nn.Module):
    def __init__(
        self,
        nodes_stat_dim: int,
        agent_dim: int,
        other_agents_dim: int = 0,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        feed_forward_hidden: int = 512,
    ):
        super().__init__()
        self.actor = ActorNet(
            nodes_stat_dim=nodes_stat_dim,
            agent_dim=agent_dim,
            other_agents_dim=other_agents_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            feed_forward_hidden=feed_forward_hidden,
        )
        self.critic = CriticNet(
            nodes_stat_dim=nodes_stat_dim,
            agent_dim=agent_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            feed_forward_hidden=feed_forward_hidden,
        )

    def make_critic_cache_(self, critic_node_static_obs, *, with_grad: bool = False):
        """
        Cache critic encoder projections.

        Note:
        - Default `with_grad=False` keeps this cache detached to save memory/speed, but then the critic
          encoder (nodes_embedding/encoder/project_node_embeddings) will NOT receive gradients.
        - Set `with_grad=True` if you want to train the critic encoder through the cached projections.
        """
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
        other_agents_obs=None,
        agents_mask=None,
        deterministic: bool = False,
    ):
        logits, _glimpse = self.actor(
            node_static_obs,
            agent_obs,
            action_mask,
            other_agents_obs=other_agents_obs,
            agents_mask=agents_mask,
        )
        if not torch.isfinite(logits).all():
            bad_rows = (~torch.isfinite(logits).all(dim=1)).nonzero(as_tuple=False).squeeze(-1).tolist()
            agents_any = None
            if agents_mask is not None:
                agents_any = agents_mask.any(dim=1).detach().cpu().tolist()
            raise RuntimeError(
                f"Non-finite ATA actor logits detected. bad_rows={bad_rows}, "
                f"action_mask_any={action_mask.any(dim=1).detach().cpu().tolist()}, "
                f"agents_mask_any={agents_any}"
            )
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        logp = dist.log_prob(action)
        ent = dist.entropy()

        value = self.critic(critic_node_static_obs, agent_obs, critic_nodes_mask)
        return action, logp, ent, value
