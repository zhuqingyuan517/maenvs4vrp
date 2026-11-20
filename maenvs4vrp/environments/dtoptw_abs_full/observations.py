import torch
from tensordict import TensorDict

from typing import Optional, Dict


class Observations:
    """Absolute-feature observation builder with full node set (global action space).

    - node_static_obs: [B, N, F] (absolute, cached per episode)
    - node_dynamic_obs: [B, N, 0] (kept for API compatibility)
    - action_mask: [B, N] (global feasible mask from env)

    This is intended for policy_net_ma_ac_old (global pointer net).
    """

    def __init__(self, feature_list: Optional[Dict] = None, debug_cache: bool = False):
        self.env = None
        self.use_local_view = False
        self.debug_cache = bool(debug_cache)

        # Cache: absolute node statics do not depend on the decision agent.
        self._cached_node_static_all = None
        self._cached_key = None  # (batch_size, num_nodes, device, dtype)

        # Node static: [x_norm, y_norm, tw_low_norm, tw_high_norm, service_norm, appear_norm, demand_norm, profit]
        self._node_static_dim = 8
        self._node_dynamic_dim = 0
        # Agent: [x_norm, y_norm, cur_time_norm, cur_load_norm, rem_dist_norm]
        self._agent_dim = 5
        # Keep "other agents" / "global" dims at 0 for strict old-model API compatibility without neighbor info.
        self._agents_dim = 0
        self._global_dim = 0

    def set_env(self, env):
        self.env = env
        self._cached_node_static_all = None
        self._cached_key = None

    def _debug_check_node_static(self, node_static_all: torch.Tensor) -> None:
        """Sanity-check that cached node_static_all matches the current instance tensors."""
        td = self.env.td_state
        xy = self._norm_xy(td["coords"])
        max_xy_err = (node_static_all[..., 0:2] - xy).abs().amax().item()
        if max_xy_err > 1e-4:
            raise RuntimeError(f"dtoptw_abs_full cache mismatch: xy max_err={max_xy_err:.3g}")

    def get_nodes_static_feat_dim(self):
        return self._node_static_dim

    def get_nodes_dynamic_feat_dim(self):
        return self._node_dynamic_dim

    def get_nodes_feat_dim(self):
        return self._node_static_dim + self._node_dynamic_dim

    def get_agent_feat_dim(self):
        return self._agent_dim

    def get_other_agents_feat_dim(self):
        return self._agents_dim

    def get_global_feat_dim(self):
        return self._global_dim

    def _norm_time(self, t: torch.Tensor) -> torch.Tensor:
        end = self.env.td_state["end_time"]  # [B]
        if t.dim() == 1:
            denom = end
        elif t.dim() == 2:
            denom = end.unsqueeze(1)
        elif t.dim() == 3:
            denom = end.unsqueeze(1).unsqueeze(2)
        else:
            denom = end.view(end.shape[0], *([1] * (t.dim() - 1)))
        return t / denom.clamp_min(1.0)

    def _norm_xy(self, xy: torch.Tensor) -> torch.Tensor:
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

    def _ensure_nonempty_action_mask(self, action_mask: torch.Tensor) -> torch.Tensor:
        """Keep every batch sample decodable even when its current agent has no action."""
        safe_mask = action_mask.clone()
        no_action = ~safe_mask.any(dim=-1)
        if no_action.any():
            bad_rows = no_action.nonzero(as_tuple=False).squeeze(-1)
            depot_idx = self.env.td_state["depot_idx"].reshape(-1)
            safe_mask[bad_rows, depot_idx[bad_rows]] = True
        return safe_mask

    def get_observations(self, is_reset: bool = False) -> TensorDict:
        td = self.env.td_state
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
            self._cached_node_static_all = self._compute_node_static_all()
            self._cached_key = key
        node_static_all = self._cached_node_static_all
        if self.debug_cache and is_reset:
            self._debug_check_node_static(node_static_all)
        action_mask = self._ensure_nonempty_action_mask(td["cur_agent"]["action_mask"])

        obs = TensorDict(
            {
                "node_static_obs": node_static_all,
                "node_dynamic_obs": torch.zeros((*td.batch_size, td["coords"].shape[1], 0), device=td.device),
                "agent_obs": self._compute_agent_obs(),
                "global_obs": torch.zeros((*td.batch_size, 0), device=td.device),
                "other_agents_obs": torch.zeros((*td.batch_size, 1, 0), device=td.device),
                "agents_mask": torch.ones((*td.batch_size, 1), dtype=torch.bool, device=td.device),
                "action_mask": action_mask,
            },
            batch_size=td.batch_size,
            device=td.device,
        )
        return obs
