import torch
from tensordict import TensorDict

from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env import AECEnv

from typing import Optional, Dict


class Observations(ObservationBuilder):
    """
    DTOPTW observations class.
    """

    POSSIBLE_NODES_STATIC_FEATURES = ['x_coordinate', 'y_coordinate', 'tw_low',
                                    'tw_high', 'profits', 'demand', 'service_time', 'tw_high_minus_tw_low_div_max_dur',
                                    'x_coordinate_min_max', 'y_coordinate_min_max', 'is_depot']

    POSSIBLE_NODES_DYNAMIC_FEATURES = [
        # absolute time-window style
        'time2open_div_end_time',
        'time2close_div_end_time',
        'arrive2node_div_end_time',
        'time2open_after_step_div_end_time',
        'time2close_after_step_div_end_time',
        'time2end_after_step_div_end_time',
        'fract_time_after_step_div_end_time',
        'reachable_frac_agents',
        # relative / MPC-friendly
        'rel_x',
        'rel_y',
        'dist2agent_div_end_time',
        'eta_div_end_time',
        'wait_div_end_time',
        'slack_div_end_time',
    ]

    POSSIBLE_AGENT_FEATURES = ['x_coordinate', 'y_coordinate','x_coordinate_min_max', 'y_coordinate_min_max', 'frac_current_time', 
                                'frac_current_load', 'frac_current_profit', 'arrivedepot_div_end_time', 
                                'frac_feasible_nodes']

    POSSIBLE_OTHER_AGENTS_FEATURES = [
        'x_coordinate',
        'y_coordinate',
        'x_coordinate_min_max',
        'y_coordinate_min_max',
        'frac_current_time',
        'frac_current_load',
        'frac_current_profit',
        'dist2depot_div_end_time',
        'dist2agent_div_end_time',
        'frac_feasible_nodes',
        'time_delta2agent_div_max_dur',
        'was_last',
        # relative to current decision agent
        'rel_x',
        'rel_y',
    ]

    POSSIBLE_GLOBAL_FEATURES = [ 'frac_demands', 'frac_fleet_load_capacity',
                                'frac_done_agents', 'frac_profits', 'frac_colect_profits']


    def __init__(self, feature_list:Dict = None, local_view: Optional[Dict] = None, return_critic_obs: bool = False):
        super().__init__()
        """
        Constructor.

        Args:
            feature_list(Dict): Dictionary containing observation features list to be available to the agent. Defaults to None.

        Returns:
            None.
        """

        self.default_feature_list = {'nodes_static': {'x_coordinate': {'feat': 'x_coordinate', 'norm': None},
                                        'y_coordinate': {'feat': 'y_coordinate', 'norm': None},
                                        'tw_low': {'feat': 'tw_low', 'norm': None},
                                        'tw_high': {'feat': 'tw_high', 'norm': None},
                                        'profits': {'feat': 'profits', 'norm': 'min_max'},
                                        'demand': {'feat': 'demand', 'norm': None},
                                        'service_time': {'feat': 'service_time', 'norm': 'min_max'},
                                        'is_depot': {'feat': 'is_depot', 'norm': None}},
                                    'nodes_dynamic': ['time2open_div_end_time',
                                        'time2close_div_end_time',
                                        'arrive2node_div_end_time',
                                        'eta_div_end_time',
                                        'wait_div_end_time',
                                        'slack_div_end_time',
                                        'rel_x',
                                        'rel_y',
                                        'time2open_after_step_div_end_time',
                                        'time2close_after_step_div_end_time',
                                        'time2end_after_step_div_end_time',
                                        'fract_time_after_step_div_end_time',
                                        'reachable_frac_agents'],
                                    'agent': ['x_coordinate',
                                        'y_coordinate',
                                        'frac_current_time',
                                        'frac_current_load',
                                        'arrivedepot_div_end_time',
                                        'frac_feasible_nodes'],
                                    'other_agents': ['x_coordinate',
                                        'y_coordinate',
                                        'frac_current_time',
                                        'frac_current_load',
                                        'frac_feasible_nodes',
                                        'dist2agent_div_end_time',
                                        'time_delta2agent_div_max_dur',
                                        'rel_x',
                                        'rel_y',
                                        'was_last'],
                                    'global': ['frac_demands', 'frac_fleet_load_capacity', 'frac_done_agents']}

        if feature_list is None:
            feature_list = self.default_feature_list

        self.feature_list = feature_list
        self.possible_nodes_static_features = self.POSSIBLE_NODES_STATIC_FEATURES
        self.possible_nodes_dynamic_features = self.POSSIBLE_NODES_DYNAMIC_FEATURES
        self.possible_agent_features = self.POSSIBLE_AGENT_FEATURES
        self.possible_agents_features = self.POSSIBLE_OTHER_AGENTS_FEATURES
        self.possible_global_features = self.POSSIBLE_GLOBAL_FEATURES

        self.local_view = local_view or {}
        self.use_local_view = bool(local_view)
        if self.use_local_view:
            self.local_k_nodes = int(self.local_view.get('k_nodes', 30))
            self.local_k_agents = int(self.local_view.get('k_agents', 5))
            self.include_depot_in_local = bool(self.local_view.get('include_depot', True))
            self.local_select_mode = str(self.local_view.get('mode', 'nearest'))
        else:
            self.local_k_nodes = None
            self.local_k_agents = None
            self.include_depot_in_local = False
            self.local_select_mode = 'nearest'

        self.return_critic_obs = bool(return_critic_obs or self.use_local_view)
        self._cached_static = None
        self._step_cache = {}

    def set_env(self, env:AECEnv):   

        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """

        super().set_env(env)
        self._step_cache = {}

    def _reset_step_cache(self):
        self._step_cache = {}

    def _get_end_time(self):
        # end_time is stored as [B] in dt_state, but we want [B, 1] for broadcasting.
        end_time = self.env.td_state['end_time']
        return end_time.unsqueeze(-1) if end_time.dim() == 1 else end_time

    def _build_node_step_cache(self):
        """
        Precompute per-node quantities that are reused by multiple dynamic features.
        This is the primary speedup when N is large (e.g., 2000).
        """
        if self._step_cache.get("node_cache_built", False):
            return

        coords = self.env.td_state["coords"]  # [B, N, 2]
        cur_node = self.env.td_state['cur_agent']['cur_node']  # [B, 1]
        cur_loc = coords.gather(1, cur_node[:, :, None].expand(-1, -1, 2))  # [B, 1, 2]

        # Distance/time from current agent position to every node: [B, N]
        time2j = torch.pairwise_distance(cur_loc, coords, eps=0, keepdim=False)
        n_digits = getattr(self.env, "n_digits", None)
        if n_digits is not None:
            time2j = torch.floor(n_digits * time2j) / n_digits

        ptime = self.env.td_state['cur_agent']['cur_time']  # [B, 1]
        arrivej = ptime + time2j  # [B, N]

        tw_low = self.env.td_state['tw_low']
        tw_high = self.env.td_state['tw_high']
        wait = torch.clamp(tw_low - arrivej, min=0.0)
        service_start = arrivej + wait  # max(arrivej, tw_low)
        slack = tw_high - service_start

        end_time = self._get_end_time()  # [B, 1]
        start_time = self.env.td_state['start_time']
        start_time = start_time.unsqueeze(-1) if start_time.dim() == 1 else start_time

        self._step_cache["coords"] = coords
        self._step_cache["cur_loc"] = cur_loc
        self._step_cache["time2j"] = time2j
        self._step_cache["arrivej"] = arrivej
        self._step_cache["wait"] = wait
        self._step_cache["service_start"] = service_start
        self._step_cache["slack"] = slack
        self._step_cache["end_time"] = end_time
        self._step_cache["start_time"] = start_time

        self._step_cache["node_cache_built"] = True

    
    def get_nodes_static_feat_dim(self):
        """
        Nodes static features dimensions.

        Args:
            n/a.

        Returns:
            int: Nodes static features dimensions.
        """
        return sum([self.feature_list.get('nodes_static', []).get(f).get('dim', 1) \
                    for f in self.feature_list.get('nodes_static')])

    def get_nodes_dynamic_feat_dim(self):
        """
        Nodes dynamic features dimensions.

        Args:
            n/a.

        Returns:
            int: Nodes dynamic features dimensions.
        """
        return len(self.feature_list.get('nodes_dynamic', []))

    def get_nodes_feat_dim(self):
        """
        Nodes features dimensions.

        Args:
            n/a.

        Returns:
            int: Nodes features dimensions.
        """
        return self.get_nodes_static_feat_dim()+self.get_nodes_dynamic_feat_dim()

    def get_agent_feat_dim(self):
        """
        Agent features dimensions.

        Args:
            n/a.

        Returns:
            int: Agent features dimensions.
        """
        return len(self.feature_list.get('agent', []))

    def get_other_agents_feat_dim(self):
        """
        Other agent features dimensions.

        Args:
            n/a.

        Returns:
            int: Other agent features dimensions.
        """
        return len(self.feature_list.get('other_agents', []))

    def get_global_feat_dim(self):
        """
        Global features dimensions.

        Args:
            n/a.

        Returns:
            int: Global features dimensions.
        """
        return len(self.feature_list.get('global', []))


    ## static features
    def get_feat_x_coordinate(self):
        """
        Instance nodes X coordinates.

        Args:
            n/a.

        Returns:
            torch.Tensor: Instance nodes X coordinates.
        """
        return self.env.td_state["coords"][:, :, 0]

    def get_feat_y_coordinate(self):
        """ 
        Instance nodes Y coordinates.

        Args:
            n/a.

        Returns:
            torch.Tensor: Instance nodes Y coordinates.
        """
        return self.env.td_state["coords"][:, :, 1]

    def get_feat_x_coordinate_min_max(self):
        """ 
        Min-max normalized X coordinates of instance nodes.

        Args:
            n/a.

        Returns:
            torch.Tensor: Min. and max. x coordinates of instance nodes.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        feat = ncoord[:,:, 0]
        return feat

    def get_feat_y_coordinate_min_max(self):
        """ 
        Min-max normalized Y coordinates of instance nodes.

        Args:
            n/a.

        Returns:
            torch.Tensor: Min-max normalized Y coordinates of instance nodes.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        feat = ncoord[:, :, 1]
        return feat

    def get_feat_tw_low(self):
        """ 
        Nodes time windows starting times.

        Args:
            n/a.

        Returns:
            torch.Tensor: Nodes time windows starting times.
        """
        return self.env.td_state['tw_low'] / self.env.td_state['max_tour_duration'].unsqueeze(dim=-1)

    def get_feat_tw_high(self):
        """ 
        Nodes time windows ending times.

        Args:
            n/a.

        Returns:
            torch.Tensor: Nodes time windows ending times.
        """
        return self.env.td_state['tw_high'] / self.env.td_state['max_tour_duration'].unsqueeze(dim=-1)

    def get_feat_demand(self):
        """
        Nodes demand.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes demand.
        """
        return self.env.td_state['demands'] / self.env.td_state['capacity']

    def get_feat_service_time(self):
        """
        Nodes service time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes service time.
        """
        return self.env.td_state['service_time']

    def get_feat_profits(self):
        """
        Nodes profits.

        Args:
            n/a.

        Returns:
            torch.Tensor: Nodes profits.
        """
        return self.env.td_state['profits']

    def get_feat_tw_high_minus_tw_low_div_max_dur(self):
        """ 
        Nodes time window amplitude divided by max tour duration.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes time window amplitude divided by max tour duration.
        """
        tw_high = self.get_feat_tw_high()
        tw_low = self.get_feat_tw_low()
        return (tw_high-tw_low) / self.env.td_state['max_tour_duration'].unsqueeze(dim=-1)

    def get_feat_is_depot(self):
        """ 
        Checks if node is depot.
        
        Args:
            n/a.

        Returns: 
            torch.Tensor: If the node is depot or not.
        """
        return self.env.td_state['is_depot']

    def _gather_nodes_features(self, feats: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        feat_dim = feats.shape[-1]
        return feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, feat_dim))

    def _select_local_nodes(self, candidate_mask: torch.Tensor):
        self._build_node_step_cache()
        dist = self._step_cache["time2j"]

        candidate_mask = candidate_mask.clone()
        if self.include_depot_in_local:
            depot_idx = self.env.td_state['depot_idx'].squeeze(-1)
            candidate_mask.scatter_(1, depot_idx.unsqueeze(1), True)

        if self.local_select_mode == 'random':
            cand = candidate_mask.float()
            idx = torch.multinomial(cand, num_samples=self.local_k_nodes, replacement=True)
            valid = candidate_mask.gather(1, idx)
        else:
            dist_masked = dist.masked_fill(~candidate_mask, float('inf'))
            topk = dist_masked.topk(self.local_k_nodes, largest=False)
            idx = topk.indices
            valid = torch.isfinite(topk.values) & candidate_mask.gather(1, idx)

        depot_idx = self.env.td_state['depot_idx'].squeeze(-1).unsqueeze(1).expand_as(idx)
        idx = torch.where(valid, idx, depot_idx)

        # Guarantee depot is present at least once to avoid "all actions masked" which causes NaNs in attention softmax.
        if self.include_depot_in_local:
            idx[:, 0] = self.env.td_state['depot_idx'].squeeze(-1)
            valid[:, 0] = True
        return idx, valid

    def _select_local_agents(self):
        coords = self.env.td_state["coords"]
        cur_agent_node = self.env.td_state['cur_agent']['cur_node']
        cur_loc = coords.gather(1, cur_agent_node[:, :, None].expand(-1, -1, 2))

        agent_nodes = self.env.td_state['agents']['cur_node']
        agent_locs = coords.gather(1, agent_nodes[:, :, None].expand(-1, -1, 2))
        dist = torch.pairwise_distance(cur_loc, agent_locs, eps=0, keepdim=False)

        agent_ids = torch.arange(self.env.num_agents, device=self.env.device).unsqueeze(0).expand_as(dist)
        cur_agent_idx = self.env.td_state['cur_agent_idx'].expand_as(dist)
        dist = dist.masked_fill(agent_ids.eq(cur_agent_idx), float('inf'))

        topk = dist.topk(self.local_k_agents, largest=False)
        idx = topk.indices
        valid = torch.isfinite(topk.values)

        cur_agent_idx = self.env.td_state['cur_agent_idx'].expand_as(idx)
        idx = torch.where(valid, idx, cur_agent_idx)
        return idx, valid
        

    ## dynamic features
    def get_feat_time2open_div_end_time(self):
        """ 
        Nodes time to open divided by end time.
        
        Args:
            n/a.

        Returns: 
            torch.Tensor: Nodes time to open divided by end time.
        """
        self._build_node_step_cache()
        feat = (self.env.td_state['tw_low'] - self.env.td_state['cur_agent']['cur_time']) / self._step_cache["end_time"]
        return feat

    def get_feat_time2close_div_end_time(self):
        """
        Nodes time to close divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes time to close divided by end time.
        """
        self._build_node_step_cache()
        feat = (self.env.td_state['tw_high'] - self.env.td_state['cur_agent']['cur_time']) / self._step_cache["end_time"]
        return feat

    def get_feat_arrive2node_div_end_time(self):
        """ 
        Agent arriving time to nodes divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agent arriving time to nodes divided by end time.
        """
        self._build_node_step_cache()
        return self._step_cache["arrivej"] / self._step_cache["end_time"]

    # --------------------------------------------------------------------------------------
    # Relative / MPC-friendly node features
    def get_feat_rel_x(self):
        """
        Node relative x (node_x - agent_x) scaled by 100 (coords are typically in [0,100]).
        """
        self._build_node_step_cache()
        dx = self._step_cache["coords"][:, :, 0] - self._step_cache["cur_loc"][:, :, 0]
        return dx / 100.0

    def get_feat_rel_y(self):
        """
        Node relative y (node_y - agent_y) scaled by 100 (coords are typically in [0,100]).
        """
        self._build_node_step_cache()
        dy = self._step_cache["coords"][:, :, 1] - self._step_cache["cur_loc"][:, :, 1]
        return dy / 100.0

    def get_feat_dist2agent_div_end_time(self):
        """
        Travel time (distance) from current agent location to each node, divided by end_time.
        """
        self._build_node_step_cache()
        return self._step_cache["time2j"] / self._step_cache["end_time"]

    def get_feat_eta_div_end_time(self):
        """
        ETA (cur_time + travel_time) to each node, divided by end_time.
        """
        # In this environment ETA == arrivej (travel time only; waiting handled separately).
        self._build_node_step_cache()
        return self._step_cache["arrivej"] / self._step_cache["end_time"]

    def get_feat_wait_div_end_time(self):
        """
        Waiting time at node: max(tw_low - ETA, 0), divided by end_time.
        """
        self._build_node_step_cache()
        return self._step_cache["wait"] / self._step_cache["end_time"]

    def get_feat_slack_div_end_time(self):
        """
        Slack at node: tw_high - service_start, divided by end_time.
        service_start = max(ETA, tw_low). Can be negative for infeasible nodes (useful signal).
        """
        self._build_node_step_cache()
        return self._step_cache["slack"] / self._step_cache["end_time"]
    
    def get_feat_time2open_after_step_div_end_time(self):
        """ 
        Nodes time to open, after agent step, divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes time to open, after agent step, divided by end time.
        """

        self._build_node_step_cache()
        feat = (self.env.td_state['tw_low'] - self._step_cache["arrivej"])
        return feat / self._step_cache["end_time"]
    
    def get_feat_time2close_after_step_div_end_time(self):
        """
        Nodes time to close, after agent step, divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes time to close, after agent step, divided by end time.
        """
        self._build_node_step_cache()
        feat = (self.env.td_state['tw_high'] - self._step_cache["arrivej"])
        return feat / self._step_cache["end_time"]

    def get_feat_time2end_after_step_div_end_time(self):
        """ 
        Time end, after agent step to node, divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Time end, after agent step to node, divided by end time.
        """
        self._build_node_step_cache()
        feat = (self._step_cache["end_time"] - self._step_cache["arrivej"])
        return feat / self._step_cache["end_time"]
       
    def get_feat_fract_time_after_step_div_end_time(self):
        """
        Fraction of time left, after agent step to node.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Fraction of time left, after agent step to node.
        """
        self._build_node_step_cache()
        feat = (self._step_cache["arrivej"] - self._step_cache["start_time"])
        return feat / self._step_cache["end_time"]
    
    def get_feat_reachable_frac_agents(self):
        """ 
        Feasible nodes per agent.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Feasible nodes per agent.
        """
        feat = self.env.td_state['agents']['feasible_nodes'].sum(dim=1)

        return feat / self.env.num_agents
    
    ## Agent features
    def get_feat_agent_x_coordinate(self):
        """
        Current agent X coordinate.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Current agent X coordinate.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat

    def get_feat_agent_y_coordinate(self):
        """ 
        Current agent Y coordinate.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Current agent Y coordinate.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat
    
    def get_feat_agent_x_coordinate_min_max(self):
        """ 
        Current agent min-max normalized X location.

        Args:
            n/a.

        Returns:
            torch.Tensor: Current agent min-max normalized X location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat

    def get_feat_agent_y_coordinate_min_max(self):
        """ 
        Current agent min-max normalized Y location.

        Args:
            n/a.

        Returns:
            torch.Tensor: Current agent min-max normalized Y location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat

    def get_feat_agent_frac_current_time(self):
        """ 
        Agent fraction of time elapsed.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agent fraction of time elapsed.
        """
        feat =  (self.env.td_state['cur_agent']['cur_time'] - self.env.td_state['start_time'].unsqueeze(1)) 
        return feat / self.env.td_state['max_tour_duration'].unsqueeze(1)

    def get_feat_agent_frac_current_load(self):
        """
        Agent fraction of used capacity.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agent fraction of used capacity.
        """
        feat =  (self.env.td_state['cur_agent']['cur_load'] - self.env.td_state['agents']['capacity'])  / self.env.td_state['agents']['capacity']
        return feat

    def get_feat_agent_frac_current_profit(self):
        """
        Fraction of profits.

        Args:
            n/a.

        Returns:
            torch.Tensor: Fraction of profits.
        """
        feat = self.env.td_state['cur_agent']['cum_profit'] / self.env.td_state['profits'].sum(dim=-1).unsqueeze(1)
        return feat

    def get_feat_agent_arrivedepot_div_end_time(self):
        """ 
        Agent time to depot divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agent time to depot divided by end time.
        """
        loc = self.env.td_state['coords'].gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        ptime = self.env.td_state['cur_agent']['cur_time'].clone()
        time2depot = torch.pairwise_distance(loc, self.env.td_state['depot_loc'], eps=0, keepdim = False)
        arrivej = ptime + time2depot

        feat = (arrivej - self.env.td_state['start_time'].unsqueeze(1))

        return feat / self.env.td_state['end_time'].unsqueeze(1)

    def get_feat_agent_frac_feasible_nodes(self):
        """ 
        Fraction of current agent feasible nodes, in order to the total number of instance nodes.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Fraction of current agent feasible nodes, in order to the total number of instance nodes.
        """
        feat = self.env.td_state['cur_agent']['action_mask'].sum(dim=1).unsqueeze(1)
        return feat / self.env.num_nodes

    def get_feat_agents_dist2depot_div_end_time(self):
        """
        Fraction of current agent distance to depot compared to its end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Fraction of current agent distance to depot compared to its end time.
        """
        locs = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = torch.pairwise_distance(self.env.td_state['depot_loc'], locs, eps=0, keepdim = False)
        return feat  / self.env.td_state['end_time'].unsqueeze(dim=-1)    

    ## Other agents features
    def get_feat_agents_x_coordinate(self):
        """ 
        Agents X coordinates.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents X coordinates.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat
    
    def get_feat_agents_y_coordinate(self):
        """ 
        Agents Y coordinates.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents Y coordinates.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat
    
    def get_feat_agents_x_coordinate_min_max(self):
        """ 
        Agents min-max normalized X location.

        Args:
            n/a.

        Returns:
            torch.Tensor: Agents min-max normalized X location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['agents']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat
    
    def get_feat_agents_y_coordinate_min_max(self):
        """
        Agents min-max normalized Y location.

        Args:
            n/a.

        Returns:
            torch.Tensor: Agents min-max normalized Y location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['agents']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat
        
    def get_feat_agents_frac_current_time(self):
        """ 
        Agents fraction of elapsed time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents fraction of elapsed time.
        """
        feats = self.env.td_state['agents']['cur_time'] / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return feats
    
    def get_feat_agents_frac_current_load(self):
        """ 
        Agents fraction of used capacity.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents fraction of used capacity.
        """
        feats = self.env.td_state['agents']['cur_load'] / self.env.td_state['agents']['capacity']
        return feats

    def get_feat_agents_frac_current_profit(self):
        """ 
        Agents fraction of profit.

        Args:
            n/a.

        Returns: 
            torch.Tensor: Agents fraction of profit.
        """
        feats = self.env.td_state['agents']['cum_profit'] / self.env.td_state['profits'].sum(dim=-1).unsqueeze(1)
        return feats


    def get_feat_agents_frac_feasible_nodes(self):
        """ 
        Fraction of agents feasible nodes, in order to the total number of instance nodes.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Fraction of agents feasible nodes, in order to the total number of instance nodes.
        """
        feat = self.env.td_state['agents']['feasible_nodes'].sum(dim=-1)
        return feat / self.env.num_nodes
    
    def get_feat_agents_dist2agent_div_end_time(self):
        """ 
        Agents distance to active agent divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents distance to active agent divided by end time.
        """
        locs = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node'][:,:,None].expand(-1, -1, 2))
        loc = self.env.td_state['coords'].gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        
        feat = torch.pairwise_distance(loc, locs, eps=0, keepdim = False)
        return feat  / self.env.td_state['end_time'].unsqueeze(dim=-1)

    def get_feat_agents_rel_x(self):
        """
        Other agents relative x (agent_x - cur_agent_x) scaled by 100.
        """
        cur_loc = self.env.td_state["coords"].gather(
            1, self.env.td_state['cur_agent']['cur_node'][:, :, None].expand(-1, -1, 2)
        )
        locs = self.env.td_state["coords"].gather(
            1, self.env.td_state['agents']['cur_node'][:, :, None].expand(-1, -1, 2)
        )
        dx = locs[:, :, 0] - cur_loc[:, :, 0]
        return dx / 100.0

    def get_feat_agents_rel_y(self):
        """
        Other agents relative y (agent_y - cur_agent_y) scaled by 100.
        """
        cur_loc = self.env.td_state["coords"].gather(
            1, self.env.td_state['cur_agent']['cur_node'][:, :, None].expand(-1, -1, 2)
        )
        locs = self.env.td_state["coords"].gather(
            1, self.env.td_state['agents']['cur_node'][:, :, None].expand(-1, -1, 2)
        )
        dy = locs[:, :, 1] - cur_loc[:, :, 1]
        return dy / 100.0
    
    def get_feat_agents_time_delta2agent_div_max_dur(self):
        """ 
        Difference between agents time and current agent time, divided by max. tour duration.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Difference between agents time and current agent time, divided by max. tour duration.
        """
        feats = (self.env.td_state['agents']['cur_time'] - self.env.td_state['cur_agent']['cur_time'] )/ self.env.td_state['max_tour_duration'].unsqueeze(dim=-1)
        return feats  

    def get_feat_agents_was_last(self):
        """
        Last agent performing an action.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Last agent performing an action.
        """
        feats = torch.zeros_like(self.env.td_state['agents']['active_agents_mask'], dtype=torch.long).scatter_(1, self.env.td_state['cur_agent_idx'], torch.ones_like(self.env.td_state['cur_agent_idx']))
        return feats   
        
    ## Global features

    def get_feat_global_frac_done_agents(self):
        """
        Fraction of done agents.

        Args:
            n/a.

        Returns: 
            torch.Tensor: Fraction of done agents.
        """
        feat = self.env.td_state['agents']['active_agents_mask'].sum(dim=1).unsqueeze(1)
        return 1 - (feat / self.env.num_agents)

    def get_feat_global_frac_demands(self):
        """
        Fraction of served demands.

        Args:
            n/a.

        Returns: 
            torch.Tensor: Fraction of served demands.
        """
        feat = self.env.td_state['nodes']['cur_demands'].sum(dim=-1).unsqueeze(1)
        return feat / self.env.td_state['demands'].sum(dim=-1).unsqueeze(1)

    def get_feat_global_frac_fleet_load_capacity(self):
        """
        Fraction of fleet load capacity.

        Args:
            n/a.

        Returns: 
            torch.Tensor: Fraction of fleet load capacity.
        """
        feat = self.env.td_state['agents']['cur_load'].sum(dim=-1).unsqueeze(1)
        capacity = self.env.td_state['agents']['capacity']
        return feat / (capacity * self.env.num_agents)

    def get_feat_global_frac_profits(self):
        """
        Fraction of profits.

        Args:
            n/a.

        Returns:
            torch.Tensor: Fraction of profits.
        """
        feat = self.env.td_state['nodes']['cur_profits'].sum(dim=-1).unsqueeze(1)
        return feat / self.env.td_state['profits'].sum(dim=-1).unsqueeze(1)

    def get_feat_global_frac_colect_profits(self):
        """
        Fraction of agents profits collection.

        Args:
            n/a.

        Returns:
            torch.Tensor: Fraction of agents profits collection.
        """
        feat = self.env.td_state['agents']['cum_profit'].sum(dim=-1).unsqueeze(1)
        return feat / self.env.td_state['profits'].sum(dim=-1).unsqueeze(1)
    
    # --------------------------------------------------------------------------------------
    def compute_static_features(self):
        """
        Compute nodes static features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Nodes static features.
        """
        features_static = self.feature_list.get('nodes_static')
        features_static_set = set([features_static.get(f).get('feat') for f in features_static])
        undefined_feat = features_static_set-set(self.possible_nodes_static_features)
        assert_msg = f'{undefined_feat} are not defined, choose from {str(self.possible_nodes_static_features)}'
        assert len(undefined_feat)==0, assert_msg

        features = list()
        for f in features_static:
            f_feat = features_static.get(f).get('feat')
            dim = features_static.get(f).get('dim')
            if dim:
                feature = getattr(self, f'get_feat_{f_feat}')(dim)
            else:
                feature = getattr(self, f'get_feat_{f_feat}')()
            f_norm = features_static.get(f).get('norm')
            norm_feature = self._normalize_feature(feature, f_norm)
            features.append(norm_feature)
        return self._concat_features(features)

    def compute_dynamic_features(self):
        """
        Compute nodes dynamic features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Nodes dynamic features.
        """
        features_dynamic = self.feature_list.get('nodes_dynamic')
        undefined_feat = set(features_dynamic)-set(self.possible_nodes_dynamic_features)
        assert_msg = f'{undefined_feat} are not defined, choose from {str(self.possible_nodes_dynamic_features)}'
        assert len(undefined_feat)==0, assert_msg
        features = list()
        for f in features_dynamic:
            features.append(getattr(self, f'get_feat_{f}')())
        return self._concat_features(features)

    def compute_agent_features(self):
        """
        Compute current agent features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Current agent features.
        """
        features_self = self.feature_list.get('agent')
        undefined_feat = set(features_self)-set(self.possible_agent_features)
        assert_msg = f'{undefined_feat} are not defined, choose from {str(self.possible_agent_features)}'
        assert len(undefined_feat)==0, assert_msg
        features = list()
        for f in features_self:
            features.append(getattr(self, f'get_feat_agent_{f}')())
        return self._concat_features(features).squeeze(1)
    
    def compute_agents_features(self):
        """
        Compute other agent features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Other agent features.
        """
        features_agents = self.feature_list.get('other_agents')
        undefined_feat = set(features_agents)-set(self.possible_agents_features)
        assert_msg = f'{undefined_feat} are not defined, choose from {str(self.possible_agents_features)}'
        assert len(undefined_feat)==0, assert_msg
        features = list()
        for f in features_agents:
            features.append(getattr(self, f'get_feat_agents_{f}')())
        return self._concat_features(features)

    def compute_global_features(self):
        """
        Compute global features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Global features.
        """
        features_global = self.feature_list.get('global')
        undefined_feat = set(features_global)-set(self.possible_global_features)
        assert_msg = f'{undefined_feat} are not defined, choose from {str(self.possible_global_features)}'
        assert len(undefined_feat)==0, assert_msg
        features = list()
        for f in features_global:
            features.append(getattr(self, f'get_feat_global_{f}')())
        return self._concat_features(features).squeeze(dim=1)


    def get_observations(self, is_reset=False)-> TensorDict:
        """
        Compute the environment.

        Args:
            is_reset(bool): If the environment is on reset. Defauts to False.

        Returns
            observations(TensorDict): Current environment observations and masks dictionary.
        """
        observations = TensorDict({}, batch_size=self.env.batch_size, device=self.env.device)
        # Reset per-step caches (they depend on the current decision agent state).
        self._reset_step_cache()

        if is_reset or self._cached_static is None:
            self._cached_static = self.compute_static_features()

        static_feat = self._cached_static

        if self.use_local_view:
            global_action_mask = self.env.td_state['cur_agent']['action_mask']
            # Local candidate set: sample only from globally feasible actions for the current decision agent.
            # This avoids the degenerate case where the local 30 nodes contain no feasible non-depot node,
            # which otherwise leads to many "depot-only" decisions and low total reward.
            candidate_mask = global_action_mask.clone()

            local_node_idx, local_slot_mask = self._select_local_nodes(candidate_mask)
            local_action_mask = global_action_mask.gather(1, local_node_idx) & local_slot_mask
            self.env._local_node_idx = local_node_idx

            # Safety: ensure at least one feasible action exists (depot fallback).
            if (~local_action_mask.any(dim=1)).any():
                depot = self.env.td_state['depot_idx'].squeeze(-1)
                bad = ~local_action_mask.any(dim=1)
                local_node_idx[bad, 0] = depot[bad]
                local_action_mask[bad, 0] = True
                local_slot_mask[bad, 0] = True

            local_static = self._gather_nodes_features(static_feat, local_node_idx)
            local_static = local_static * local_slot_mask.unsqueeze(-1)
            observations['node_static_obs'] = local_static

            if self.feature_list.get('nodes_dynamic'):
                dynamic_feat = self.compute_dynamic_features()
                local_dynamic = self._gather_nodes_features(dynamic_feat, local_node_idx)
                local_dynamic = local_dynamic * local_slot_mask.unsqueeze(-1)
                observations['node_dynamic_obs'] = local_dynamic

            if self.feature_list.get('agent'):
                agent_feat = self.compute_agent_features()
                observations['agent_obs'] = agent_feat

            if self.feature_list.get('other_agents'):
                agents_feat = self.compute_agents_features()
                local_agents_idx, local_agents_mask = self._select_local_agents()
                local_agents_feat = agents_feat.gather(1, local_agents_idx.unsqueeze(-1).expand(-1, -1, agents_feat.shape[-1]))
                local_agents_feat = local_agents_feat * local_agents_mask.unsqueeze(-1)
                observations['other_agents_obs'] = local_agents_feat
                observations['agents_mask'] = local_agents_mask
            else:
                observations['agents_mask'] = self.env.td_state['agents']['active_agents_mask'].clone()

            if self.feature_list.get('global'):
                global_feat = self.compute_global_features()
                observations['global_obs'] = global_feat

            observations['action_mask'] = local_action_mask
            observations['local_node_idx'] = local_node_idx

            if self.return_critic_obs:
                observations['critic_node_static_obs'] = static_feat
                if self.feature_list.get('nodes_dynamic'):
                    observations['critic_node_dynamic_obs'] = self.compute_dynamic_features()
                if self.feature_list.get('agent'):
                    observations['critic_agent_obs'] = self.compute_agent_features()
                if self.feature_list.get('other_agents'):
                    observations['critic_other_agents_obs'] = self.compute_agents_features()
                observations['critic_agents_mask'] = self.env.td_state['agents']['active_agents_mask'].clone()
                observations['critic_nodes_mask'] = self.env.td_state['nodes']['active_nodes_mask'].clone()
                if self.feature_list.get('global'):
                    observations['critic_global_obs'] = self.compute_global_features()
        else:
            if is_reset:
                mask_static_feat = self.env.td_state['cur_agent']['action_mask'].unsqueeze(dim=-1) * static_feat
                observations['node_static_obs'] =  mask_static_feat   

            if self.feature_list.get('nodes_dynamic'):
                dynamic_feat = self.compute_dynamic_features()
                mask_dynamic_feat = self.env.td_state['cur_agent']['action_mask'].unsqueeze(dim=-1) * dynamic_feat
                observations['node_dynamic_obs'] =  mask_dynamic_feat   
          
            if self.feature_list.get('agent'):
                agent_feat = self.compute_agent_features()
                observations['agent_obs'] = agent_feat

            if self.feature_list.get('other_agents'):
                agents_feat = self.compute_agents_features()
                mask_agents_feat = self.env.td_state['agents']['active_agents_mask'].unsqueeze(dim=-1) * agents_feat
                observations['other_agents_obs'] = mask_agents_feat

            if self.feature_list.get('global'):
                global_feat = self.compute_global_features()
                observations['global_obs'] = global_feat
            
        return observations

    @staticmethod
    def _concat_features(features):
        """
        Concatenate features.

        Args:
            features(list): Features to concatenate.

        Returns:
            torch.Tensor: Concatenated tensor.
        """

        return torch.cat(\
                [f.unsqueeze(dim=-1) if f.dim()==2 else f for f in features],
                              dim=-1)

    def _normalize_feature(self, x, norm):
        """
        Normalize features.

        Args:
            x(torch.Tensor): Tensor to be normalized.
            norm(str): Type of normalization. It can be 'min_max' or 'standardize'. If None, tensor is returned.

        Returns:
            torch.Tensor: Tensor normalized or default tensor if norm is invalid.
        """

        if norm == 'min_max':
            return self._min_max_normalization(x)
        elif norm == 'standardize':
            return self._standardize(x)
        elif norm == None:
            return x

    @staticmethod
    def _min_max_normalization(x):
        """
        Min. max. normalization.

        Args:
            x(torch.Tensor): Tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        max_x = torch.max(x, dim=1, keepdim=True)[0]
        min_x = torch.min(x, dim=1, keepdim=True)[0]
        return (x - min_x) / (max_x - min_x)

    @staticmethod
    def _min_max_normalization2d(x):
        """
        Min. max. normalization 2 dimensions.

        Args:
            x(torch.Tensor): Tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        max_x = torch.max(x)
        min_x = torch.min(x)
        return (x - min_x) / (max_x - min_x)

    @staticmethod
    def _standardize(x):
        """
        Tensor standardization.

        Args:
            x(torch.Tensor): Tensor to be normalized.
        
        Returns:
            torch.Tensor: Normalized tensor.
        """
        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True)
        return (x - means) / stds
