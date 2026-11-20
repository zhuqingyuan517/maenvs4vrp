import torch
from tensordict import TensorDict

from typing import Optional, Dict

from maenvs4vrp.environments.dtoptw.observations import Observations as _BaseObs


class Observations(_BaseObs):
    """DTOPTW observations with full (global) node action space.

    Differences vs `maenvs4vrp.environments.dtoptw.observations.Observations`:
    - Always returns `node_static_obs` (cached), so training loops can read it every step.
    - Does NOT mask node static features by the current `action_mask`.
      This is important for DTOPTW where many nodes are infeasible early (e.g., not-yet-appeared);
      masking statics at reset would break pointer-network caching in `policy_net_ma_ac_old`.
    - Keeps the full node set: actions are global indices in `[0, N)`.
    """

    def __init__(
        self,
        feature_list: Optional[Dict] = None,
        return_critic_obs: bool = False,
    ):
        super().__init__(feature_list=feature_list, local_view=None, return_critic_obs=return_critic_obs)

    def get_observations(self, is_reset: bool = False) -> TensorDict:
        observations = TensorDict({}, batch_size=self.env.batch_size, device=self.env.device)
        self._reset_step_cache()

        if is_reset or self._cached_static is None:
            self._cached_static = self.compute_static_features()

        static_feat = self._cached_static
        observations["node_static_obs"] = static_feat

        action_mask = self.env.td_state["cur_agent"]["action_mask"].clone()

        if self.feature_list.get("nodes_dynamic"):
            dynamic_feat = self.compute_dynamic_features()
            observations["node_dynamic_obs"] = action_mask.unsqueeze(-1) * dynamic_feat
        else:
            observations["node_dynamic_obs"] = torch.zeros(
                (*self.env.batch_size, self.env.num_nodes, 0), device=self.env.device
            )

        if self.feature_list.get("agent"):
            observations["agent_obs"] = self.compute_agent_features()
        else:
            observations["agent_obs"] = torch.zeros((*self.env.batch_size, 0), device=self.env.device)

        if self.feature_list.get("other_agents"):
            agents_feat = self.compute_agents_features()
            agents_mask = self.env.td_state["agents"]["active_agents_mask"].clone()
            observations["other_agents_obs"] = agents_mask.unsqueeze(-1) * agents_feat
            observations["agents_mask"] = agents_mask
        else:
            # Keep API stable: provide tensors even if unused.
            agents_mask = self.env.td_state["agents"]["active_agents_mask"].clone()
            observations["other_agents_obs"] = torch.zeros((*self.env.batch_size, agents_mask.shape[1], 0), device=self.env.device)
            observations["agents_mask"] = agents_mask

        if self.feature_list.get("global"):
            observations["global_obs"] = self.compute_global_features()
        else:
            observations["global_obs"] = torch.zeros((*self.env.batch_size, 0), device=self.env.device)

        observations["action_mask"] = action_mask
        return observations

