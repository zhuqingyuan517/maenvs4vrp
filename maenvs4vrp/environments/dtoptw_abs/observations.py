import torch
from tensordict import TensorDict

from typing import Optional, Dict


class Observations:
    """Absolute-feature observation builder for dtoptw.

    Design goals (per user request):
    - No relative node features (no rel_x/rel_y, no ETA/slack relative-to-agent in node features).
    - Actor only observes K local nodes (default 30) + current agent features.
    - Critic observes all nodes (including not-yet-appeared) for encoding, but decoder-time mask
      excludes not-yet-appeared nodes.
    - No "nearby agents" observations.
    """

    def __init__(
        self,
        local_view: Optional[Dict] = None,
        return_critic_obs: bool = True,
        critic_mask_unappeared: bool = True,
        critic_encode_unappeared: bool = True,
        debug_cache: bool = False,
    ):
        self.env = None
        # Let dtoptw env know it must map local actions via env._local_node_idx.
        self.use_local_view = True
        self.return_critic_obs = bool(return_critic_obs)
        # If True, critic decode-time mask excludes not-yet-appeared nodes.
        # If False, critic can attend all active nodes (including future/unappeared).
        self.critic_mask_unappeared = bool(critic_mask_unappeared)
        # If False, hide not-yet-appeared nodes from critic encoder input by zeroing their static features.
        # Note: existing train_madyam_abs caches critic encoding once per episode, so this is applied at cache-build time.
        self.critic_encode_unappeared = bool(critic_encode_unappeared)
        self.debug_cache = bool(debug_cache)
        self.local_view = local_view or {"k_nodes": 30, "mode": "nearest", "include_depot": True}
        self.k_nodes = int(self.local_view.get("k_nodes", 30))
        self.mode = str(self.local_view.get("mode", "nearest"))
        self.include_depot = bool(self.local_view.get("include_depot", True))

        # Cache: absolute static node features don't depend on the current decision agent.
        # Recompute only on reset / when the underlying instance changes.
        self._cached_node_static_all = None
        self._cached_key = None  # (num_nodes, device, dtype)

        # Fixed feature layout (absolute only).
        # Node static: [x_norm, y_norm, tw_low_norm, tw_high_norm, service_norm, appear_norm, demand_norm, profit]
        self._node_static_dim = 8
        self._node_dynamic_dim = 0
        # Agent: [x_norm, y_norm, cur_time_norm, cur_load_norm, rem_dist_norm]
        self._agent_dim = 5
        self._global_dim = 0

    def set_env(self, env):
        self.env = env
        self._cached_node_static_all = None
        self._cached_key = None

    def _debug_check_node_static(self, node_static_all: torch.Tensor) -> None:
        """Sanity-check that cached node_static_all matches the current instance tensors."""
        td = self.env.td_state
        xy = self._norm_xy(td["coords"])
        # node_static_all: [x,y,...]
        max_xy_err = (node_static_all[..., 0:2] - xy).abs().amax().item()
        if max_xy_err > 1e-4:
            raise RuntimeError(f"dtoptw_abs cache mismatch: xy max_err={max_xy_err:.3g}")

    # API expected by existing training scripts
    def get_nodes_static_feat_dim(self):
        return self._node_static_dim

    def get_nodes_dynamic_feat_dim(self):
        return self._node_dynamic_dim

    def get_nodes_feat_dim(self):
        return self._node_static_dim + self._node_dynamic_dim

    def get_agent_feat_dim(self):
        return self._agent_dim

    def get_other_agents_feat_dim(self):
        return 0

    def get_global_feat_dim(self):
        return self._global_dim

    def _norm_time(self, t: torch.Tensor) -> torch.Tensor:
        # Normalize by end_time for scale stability.
        end = self.env.td_state["end_time"]  # [B]
        if t.dim() == 1:
            denom = end
        elif t.dim() == 2:
            denom = end.unsqueeze(1)
        elif t.dim() == 3:
            denom = end.unsqueeze(1).unsqueeze(2)
        else:
            # Fallback: expand with singleton dims.
            denom = end.view(end.shape[0], *([1] * (t.dim() - 1)))
        return t / denom.clamp_min(1.0)

    def _norm_xy(self, xy: torch.Tensor) -> torch.Tensor:
        # Random generator uses coords in [0,100] integer by default.
        # Keep it robust by inferring a scale from the data.
        # Avoid per-step recompute: coords are static so max can be cached by torch, but this is cheap.
        coords = self.env.td_state["coords"]
        scale = coords.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0)  # [B,1,1]
        return xy / scale

    def _compute_node_static_all(self) -> torch.Tensor:
        td = self.env.td_state
        coords = td["coords"]  # [B,N,2]
        xy = self._norm_xy(coords)

        tw_low = td["tw_low"].unsqueeze(-1)
        tw_high = td["tw_high"].unsqueeze(-1)
        service = td["service_time"].unsqueeze(-1)
        appear = td["appear_time"].unsqueeze(-1)

        tw_low_n = self._norm_time(tw_low)
        tw_high_n = self._norm_time(tw_high)
        service_n = self._norm_time(service)
        appear_n = self._norm_time(appear)

        # Demands/capacity are static in the sampled instance; served nodes are handled by masks.
        cap = td["capacity"].clamp_min(1.0)  # [B,1]
        demand_n = (td["demands"] / cap).unsqueeze(-1)

        profit = td.get("profits", torch.zeros_like(td["demands"])).unsqueeze(-1)

        return torch.cat([xy, tw_low_n, tw_high_n, service_n, appear_n, demand_n, profit], dim=-1)

    def _compute_agent_obs(self) -> torch.Tensor:
        td = self.env.td_state
        cur_node = td["cur_agent"]["cur_node"]  # [B,1]
        cur_xy = td["coords"].gather(1, cur_node[..., None].expand(-1, -1, 2))  # [B,1,2]
        cur_xy = self._norm_xy(cur_xy).squeeze(1)

        cur_time = self._norm_time(td["cur_agent"]["cur_time"]).squeeze(1)
        cur_load = (td["cur_agent"]["cur_load"] / td["capacity"].clamp_min(1.0)).squeeze(1)

        if "rem_dist" in td["cur_agent"].keys() and "max_flight_dist" in td.keys():
            rem = (td["cur_agent"]["rem_dist"] / td["max_flight_dist"].clamp_min(1.0)).squeeze(1)
        else:
            rem = torch.ones_like(cur_time)

        return torch.cat([cur_xy, cur_time.unsqueeze(-1), cur_load.unsqueeze(-1), rem.unsqueeze(-1)], dim=-1)

    def _select_local_nodes(self, node_static_all: torch.Tensor) -> torch.Tensor:
        """Return local_node_idx [B,K] in global index space."""
        td = self.env.td_state
        N = td["coords"].shape[1]

        global_mask = td["cur_agent"]["action_mask"]  # [B,N], True means feasible
        # Keep feasibility as-is; depot will still be included in slot 0, but may be masked off.

        # Build candidates from feasible set.
        # NOTE: if feasible nodes are fewer than K, we will pad by repeating depot (0).
        K = self.k_nodes
        out = torch.zeros((*td.batch_size, K), dtype=torch.long, device=td.device)
        out[:, 0] = 0  # slot 0 is depot for stability

        # If K==1, we're done.
        if K == 1:
            return out

        # Compute scores for selection.
        if self.mode == "random":
            # Sample from feasible nodes excluding depot to fill remaining slots.
            probs = global_mask.float()
            probs[:, 0] = 0.0
            # If no non-depot feasible, keep depot padding.
            has = probs.sum(dim=1) > 0
            if has.any():
                # multinomial expects non-negative probs
                probs2 = probs[has]
                # with replacement to allow K>feasible_count
                samp = torch.multinomial(probs2, K - 1, replacement=True)
                out[has, 1:] = samp
            return out

        # nearest: distance from current agent location to all nodes, then pick top-k among feasible
        cur_node = td["cur_agent"]["cur_node"]  # [B,1]
        cur_xy = td["coords"].gather(1, cur_node[..., None].expand(-1, -1, 2)).squeeze(1)  # [B,2]
        diff = td["coords"] - cur_xy[:, None, :]
        # sqrt is monotonic, so squared distances are sufficient for top-k nearest selection.
        dist = diff.pow(2).sum(dim=-1)  # [B,N]

        # Mask out infeasible and depot (we already reserved slot 0).
        dist = dist.masked_fill(~global_mask, float("inf"))
        dist[:, 0] = float("inf")

        # Top-k smallest distances.
        # If fewer than K-1 feasible, topk returns infs; we'll replace those with depot (0).
        vals, idx = torch.topk(dist, k=min(K - 1, N - 1), largest=False, dim=1)

        out[:, 1 : 1 + idx.shape[1]] = idx
        # Replace invalid (inf) with depot
        invalid = torch.isinf(vals)
        if invalid.any():
            out[:, 1 : 1 + idx.shape[1]][invalid] = 0
        return out

    def get_observations(self, is_reset: bool = False) -> TensorDict:
        td = self.env.td_state

        # Static node features cache (absolute features).
        # Use tensor storage pointers in the key so a new sampled instance (new tensors, same shapes)
        # always invalidates the cache even if is_reset is accidentally False somewhere upstream.
        coords = td["coords"]
        tw_low = td["tw_low"]
        profits = td.get("profits", None)
        profits_ptr = int(profits.data_ptr()) if profits is not None else -1
        key = (
            tuple(td.batch_size),
            coords.shape[1],
            td.device,
            coords.dtype,
            int(coords.data_ptr()),
            int(tw_low.data_ptr()),
            profits_ptr,
        )
        if is_reset or self._cached_node_static_all is None or self._cached_key != key:
            self._cached_node_static_all = self._compute_node_static_all()  # [B,N,F]
            self._cached_key = key
        node_static_all = self._cached_node_static_all
        if self.debug_cache and is_reset:
            self._debug_check_node_static(node_static_all)

        local_node_idx = self._select_local_nodes(node_static_all)  # [B,K]
        self.env._local_node_idx = local_node_idx  # used by env.step() mapping in dtoptw

        node_static_local = node_static_all.gather(
            1, local_node_idx[..., None].expand(-1, -1, node_static_all.size(-1))
        )  # [B,K,F]

        # Local action mask corresponding to selected indices.
        global_mask = td["cur_agent"]["action_mask"]  # [B,N]
        local_mask = global_mask.gather(1, local_node_idx)
        # Do NOT force depot feasible here: global feasibility may forbid depot->depot (e.g. forbid_depot_self_loop).

        agent_obs = self._compute_agent_obs()  # [B,A]

        obs = TensorDict(
            {
                "node_static_obs": node_static_local,
                "node_dynamic_obs": torch.zeros((*td.batch_size, self.k_nodes, 0), device=td.device),
                "agent_obs": agent_obs,
                "global_obs": torch.zeros((*td.batch_size, 0), device=td.device),
                "other_agents_obs": torch.zeros((*td.batch_size, 0, 0), device=td.device),
                "agents_mask": torch.zeros((*td.batch_size, 0), dtype=torch.bool, device=td.device),
                "action_mask": local_mask,
                "local_node_idx": local_node_idx,
            },
            batch_size=td.batch_size,
            device=td.device,
        )

        if self.return_critic_obs:
            # Critic encodes all nodes (including not-yet-appeared), but decoder-time mask excludes not-yet-appeared.
            cur_time = td["cur_agent"]["cur_time"]  # [B,1]
            appeared = td["appear_time"] <= cur_time  # [B,N]
            active = td["nodes"]["active_nodes_mask"]  # [B,N]
            if self.critic_mask_unappeared:
                critic_decode_mask = appeared & active
            else:
                critic_decode_mask = active
            # keep depot always allowed
            critic_decode_mask = critic_decode_mask.clone()
            critic_decode_mask[:, 0] = True

            critic_node_static_obs = node_static_all
            if not self.critic_encode_unappeared:
                # Hide not-yet-appeared nodes from encoder by zeroing features.
                visible = appeared.unsqueeze(-1).to(node_static_all.dtype)  # [B,N,1]
                critic_node_static_obs = node_static_all * visible
                critic_node_static_obs = critic_node_static_obs.clone()
                critic_node_static_obs[:, 0, :] = node_static_all[:, 0, :]

            obs.update(
                {
                    "critic_node_static_obs": critic_node_static_obs,
                    "critic_node_dynamic_obs": torch.zeros((*td.batch_size, td["coords"].shape[1], 0), device=td.device),
                    "critic_agent_obs": agent_obs,
                    "critic_global_obs": torch.zeros((*td.batch_size, 0), device=td.device),
                    "critic_nodes_mask": critic_decode_mask,
                }
            )

        return obs
