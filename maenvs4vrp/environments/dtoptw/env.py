import torch
from tensordict import TensorDict

from typing import Tuple, Optional

import warnings

from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env_agent_selector import BaseSelector
from maenvs4vrp.core.env_agent_reward import RewardFn
from maenvs4vrp.core.env import AECEnv
from maenvs4vrp.utils.utils import gather_by_index

class Environment(AECEnv):
    """
    DTOPTW environment generator class.
    """
    def __init__(self,
                instance_generator_object: InstanceBuilder,  
                obs_builder_object: ObservationBuilder,
                agent_selector_object: BaseSelector,
                reward_evaluator: RewardFn,
                seed=None,         
                device: Optional[str] = None,
                batch_size: torch.Size = None,
                max_steps: Optional[int] = None,
                max_flight_dist: Optional[float] = None,
                allow_depot_refuel: bool = False,
                forbid_depot_self_loop: bool = False,
                use_local_view: bool = False,
                done_feasible_check_freq: int = 10,
                validate_actions: bool = False):

        """
        Constructor.

        Args:
            instance_generator_object(InstanceBuilder): Generator instance.
            obs_builder_object(ObservationBuilder): Observations instance.
            agent_selector_object(BaseSelector): Agent selector instance
            reward_evaluator(RewardFn): Reward evaluator instance.
            seed(int): Random number generator seed. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to None.
            batch_size(torch.Size): Batch size. Defaults to None.
        """

        self.version = 'v0'
        self.env_name = 'dtoptw'
        self.max_steps = max_steps
        self.max_flight_dist = max_flight_dist
        self.allow_depot_refuel = allow_depot_refuel
        # If True, depot->depot is disallowed whenever there exists any feasible non-depot action.
        # This prevents event-based simulation from stalling at cur_time==0 while dynamic nodes never appear.
        self.forbid_depot_self_loop = bool(forbid_depot_self_loop)
        self.done_feasible_check_freq = int(done_feasible_check_freq) if done_feasible_check_freq is not None else 10
        self.use_local_view = bool(getattr(obs_builder_object, "use_local_view", False) or use_local_view)
        self._local_node_idx = None
        self._done_by_feasible_cache = None
        self._done_by_feasible_cache_step = -1
        self.validate_actions = bool(validate_actions)

        # seed the environment
        if seed is None:
            self._set_seed(self.DEFAULT_SEED)
        else:
            self._set_seed(seed)

        self.agent_selector = agent_selector_object
        self.inst_generator = instance_generator_object
        self.inst_generator._set_seed(self.seed)
        self.obs_builder = obs_builder_object
        self.obs_builder.set_env(self)
        self.reward_evaluator = reward_evaluator
        self.reward_evaluator.set_env(self)   
        self.env_nsteps = 0

        if device is None:
            self.device = self.inst_generator.device
        else:
            self.device = device
            self.inst_generator.device = device

        if batch_size is None:
            self.batch_size =  self.inst_generator.batch_size
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
            self.inst_generator.batch_size = torch.Size(batch_size)
            
        self.td_state = TensorDict({}, batch_size=self.batch_size, device=self.device)

        
    def observe(self, is_reset=False)-> TensorDict:
        """
        Retrieve agent environment observations.

        Args:
            is_reset(bool): If the environment is on reset. Defauts to False.

        Returns
            td_observations(TensorDict): Current agent observaions and masks dictionary.
        """

        self._update_feasibility()
        td_observations = self.obs_builder.get_observations(is_reset=is_reset)
        if 'action_mask' not in td_observations:
            td_observations['action_mask'] = self.td_state['cur_agent']['action_mask'].clone()
        if 'agents_mask' not in td_observations:
            td_observations['agents_mask'] = self.td_state['agents']['active_agents_mask'].clone()
        return td_observations

    
    def sample_action(self, td: TensorDict)-> TensorDict:
        """
        Compute a random action from avaliable actions to current agent.
        
        Args:
            td(TensorDict): Environment instance tensor.

        Returns:
            td(TensorDict): Environment instance tensor with updated action.
        """
        action = torch.multinomial(self.td_state['cur_agent']["action_mask"].float(), 1).to(self.device)
        td['action'] = action
        return td


    def reset(self, 
              num_agents:int|None=None, 
              num_nodes:int|None=None, 
              profits: str = 'distance',
              instance_name:str|None=None, 
              sample_type:str='random',
              batch_size: Optional[torch.Size] = None,
              n_augment: Optional[int] = None,
              seed:int|None=None,
              max_steps: Optional[int] = None,
              max_flight_dist: Optional[float] = None,
              allow_depot_refuel: Optional[bool] = None,
              forbid_depot_self_loop: Optional[bool] = None,
              horizon: Optional[int] = None,
              done_feasible_check_freq: Optional[int] = None)-> TensorDict:
        """
        Reset the environment.

        Args:
            num_agents(int, optional): Total number of agents. Defaults to None.
            num_nodes(int, optional): Total number of nodes. Defaults to None.
            capacity(int, optional): Total capacity for each agent. Defaults to None.
            service_times(float, optional): Service time in the nodes. Defaults to None.
            instance_name(str, optional): Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(int, optional): Random number generator seed. Defaults to None. 

        Returns:
            TensorDict: Environment information dictionary.
        """

        if seed is not None:
            self._set_seed(seed)

        if max_steps is not None:
            self.max_steps = max_steps
        if max_flight_dist is not None:
            self.max_flight_dist = max_flight_dist
        if allow_depot_refuel is not None:
            self.allow_depot_refuel = allow_depot_refuel
        if forbid_depot_self_loop is not None:
            self.forbid_depot_self_loop = bool(forbid_depot_self_loop)
        if done_feasible_check_freq is not None:
            self.done_feasible_check_freq = int(done_feasible_check_freq)
        # In refuel mode, episodes can otherwise run for an extremely long time (or effectively "hang")
        # because agents never become inactive. Provide a safe default horizon if caller didn't set it.
        if self.allow_depot_refuel and self.max_steps is None:
            self.max_steps = horizon if horizon is not None else (int(num_nodes or 0) + int(num_agents or 0) + 1)
        self._done_by_feasible_cache = None
        self._done_by_feasible_cache_step = -1

        if batch_size is None:
            batch_size = self.batch_size 
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
            self.inst_generator.batch_size = torch.Size(batch_size)

        instance_info = self.inst_generator.sample_instance(num_agents=num_agents, 
                                                            num_nodes=num_nodes, 
                                                            instance_name=instance_name, 
                                                            sample_type=sample_type, 
                                                            batch_size=batch_size,
                                                            n_augment=n_augment,
                                                            seed=seed,
                                                            horizon=horizon,
                                                            profits=profits)

        self.num_nodes = instance_info['num_nodes']
        self.num_agents = instance_info['num_agents']

        if 'n_digits' in instance_info:
            self.n_digits = instance_info['n_digits'] 
        else:
            self.n_digits = None

        self.td_state = instance_info['data']

        self.td_state['done'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['is_last_step'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['depot_loc'] = self.td_state['coords'].gather(1, self.td_state['depot_idx'][:,:,None].expand(-1, -1, 2))

        self.td_state['max_tour_duration'] =  self.td_state['end_time'] - self.td_state['start_time']

        time2depot = torch.pairwise_distance(self.td_state['depot_loc'], 
                                                              self.td_state['coords'], eps=0, keepdim = False)
        if self.n_digits is not None:
            time2depot = torch.floor(self.n_digits * time2depot) / self.n_digits

        self.td_state['time2depot'] = time2depot

        self.td_state['nodes'] = TensorDict(
                                    source={'cur_demands': self.td_state['demands'].clone(),
                                            'cur_profits': self.td_state['profits'].clone(),
                                            'active_nodes_mask': torch.ones((*batch_size, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'served_nodes_mask': torch.zeros((*batch_size, self.num_nodes), dtype=torch.bool, device=self.device)},
                                    batch_size=batch_size, device=self.device)
        
        cust_mask = self.td_state['appear_time'] <= self.td_state['start_time'].unsqueeze(-1)

        self.td_state['agents'] =  TensorDict(
                                    source={'capacity': self.td_state['capacity'],
                                            'cur_load': self.td_state['capacity'].clone() * torch.ones((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_time': self.td_state['start_time'].unsqueeze(1).clone() * torch.ones((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_node': self.td_state['depot_idx'] * torch.ones((*batch_size, self.num_agents), dtype = torch.int64, device=self.device),
                                            'cur_ttime': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cum_ttime': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cum_profit': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'step_profit': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'visited_nodes': torch.zeros((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'feasible_nodes': cust_mask[:,None,:].repeat(1, self.num_agents, 1),
                                            'active_agents_mask': torch.ones((*batch_size, self.num_agents), dtype=torch.bool, device=self.device),
                                            'cur_step': torch.zeros((*batch_size, self.num_agents), dtype=torch.int32, device=self.device)},
                                    batch_size=batch_size, device=self.device)

        self.td_state['cur_agent_idx'] = torch.zeros((*batch_size, 1), dtype = torch.int64, device=self.device)
        self.td_state['cur_node_idx'] = self.td_state['depot_idx'].clone()

        self.td_state['cur_agent'] = TensorDict({
                                'action_mask': self.td_state['agents']['feasible_nodes'].gather(1, self.td_state['cur_agent_idx'][:,:,None].expand(-1, -1, self.num_nodes)).squeeze(1),
                                'cur_load': self.td_state['agents']['cur_load'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_time': self.td_state['agents']['cur_time'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_node': self.td_state['agents']['cur_node'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_ttime': self.td_state['agents']['cur_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cum_ttime': self.td_state['agents']['cum_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cum_profit': self.td_state['agents']['cum_profit'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'step_profit': self.td_state['agents']['step_profit'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_step': self.td_state['agents']['cur_step'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                }, batch_size=batch_size)

        if self.max_flight_dist is not None:
            self.td_state['max_flight_dist'] = self.max_flight_dist * torch.ones((*batch_size, 1), dtype=torch.float, device=self.device)
            rem_dist = self.td_state['max_flight_dist'].clone() * torch.ones((*batch_size, self.num_agents), dtype=torch.float, device=self.device)
            self.td_state['agents']['rem_dist'] = rem_dist
            self.td_state['cur_agent']['rem_dist'] = rem_dist.gather(1, self.td_state['cur_agent_idx']).clone()

        self.td_state['solution'] = TensorDict({}, batch_size=batch_size)

        self.agent_selector.set_env(self)
        self.obs_builder.set_env(self)
        self.reward_evaluator.set_env(self)

        agent_step = self.td_state['cur_agent']['cur_step']
        done = self.td_state['done'].clone()
        reward = torch.zeros_like(done, dtype = torch.float, device=self.device)
        penalty = torch.zeros_like(done, dtype = torch.float, device=self.device)

        td_observations = self.observe(is_reset=True)
        self._update_done_from_empty_actions(td_observations['action_mask'])
        done = self.td_state['done'].clone()
        self.env_nsteps = 0
        return TensorDict(
            {
                "agent_step": agent_step,
                "observations": td_observations,
                "cur_agent_idx":self.td_state['cur_agent_idx'].clone(),
                "cur_node_idx": self.td_state['cur_node_idx'].clone(),
                "reward": reward,
                "penalty":penalty,
                "done": done,
            },
            batch_size=batch_size, device=self.device)


    def _update_feasibility(self):

        """
        Update actions feasibility.
        
        Args:
            n/a.

        Returns:
            None.
        """

        cust_mask = self.td_state['appear_time'] <= self.td_state['cur_agent']['cur_time']
        _mask = self.td_state['nodes']['active_nodes_mask'] & cust_mask

        # time windows constraints
        loc = self.td_state['coords'].gather(1, self.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        ptime = self.td_state['cur_agent']['cur_time']
        diff = self.td_state["coords"] - loc
        time2j = diff.pow(2).sum(dim=-1).sqrt()
        if self.n_digits is not None:
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits
        
        arrivej = ptime + time2j
        waitj = torch.clamp(self.td_state['tw_low']-arrivej, min=0)
        service_startj = arrivej + waitj
        c1 = service_startj <= self.td_state['tw_high']
        c2 = service_startj + self.td_state['service_time'] + self.td_state['time2depot'] <= self.td_state['end_time'].unsqueeze(-1)
        # capacity constraints
        c3 = self.td_state['demands'] <= self.td_state['cur_agent']['cur_load']

        _mask = _mask & c1 & c2 & c3

        if self.max_flight_dist is not None:
            rem_dist = self.td_state['cur_agent']['rem_dist']
            c4 = time2j + self.td_state['time2depot'] <= rem_dist
            _mask = _mask & c4

        # Optional: forbid depot self-loop if there exists any feasible non-depot action.
        # Keep depot feasible if it is the only feasible action (to avoid all-false masks).
        if getattr(self, "forbid_depot_self_loop", False):
            at_depot = self.td_state['cur_agent']['cur_node'].eq(self.td_state['depot_idx']).squeeze(1)  # [B]
            feasible_non_depot = _mask.clone()
            feasible_non_depot[:, 0] = False  # depot is always index 0 in this env
            has_non_depot = feasible_non_depot.any(dim=1)  # [B]
            forbid = at_depot & has_non_depot
            if forbid.any():
                _mask[forbid, 0] = False
        # update state
        self.td_state['cur_agent'].update({'action_mask': _mask}) 
        self.td_state['agents']['feasible_nodes'].scatter_(1, 
                                            self.td_state['cur_agent_idx'][:,:,None].expand(-1,-1,self.num_nodes), _mask.unsqueeze(1))

    def _compute_feasible_mask_all_agents(self):
        """
        Compute feasibility mask for all agents (used for global termination).

        Returns:
            torch.Tensor: feasibility mask [B, A, N].
        """
        locs = self.td_state['coords'].gather(
            1,
            self.td_state['agents']['cur_node'][:, :, None].expand(-1, -1, 2),
        )
        # [B, A, N, 2]
        diff = locs[:, :, None, :] - self.td_state['coords'][:, None, :, :]
        time2j = diff.pow(2).sum(-1).sqrt()
        if self.n_digits is not None:
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits

        ptime = self.td_state['agents']['cur_time'].unsqueeze(-1)
        arrivej = ptime + time2j
        waitj = torch.clamp(self.td_state['tw_low'].unsqueeze(1) - arrivej, min=0)
        service_startj = arrivej + waitj

        cust_mask = self.td_state['appear_time'].unsqueeze(1) <= self.td_state['agents']['cur_time'].unsqueeze(-1)
        active_mask = self.td_state['nodes']['active_nodes_mask'].unsqueeze(1)
        c1 = service_startj <= self.td_state['tw_high'].unsqueeze(1)
        c2 = service_startj + self.td_state['service_time'].unsqueeze(1) + self.td_state['time2depot'].unsqueeze(1) <= self.td_state['end_time'].unsqueeze(-1).unsqueeze(1)
        c3 = self.td_state['demands'].unsqueeze(1) <= self.td_state['agents']['cur_load'].unsqueeze(-1)

        mask = active_mask & cust_mask & c1 & c2 & c3

        if self.max_flight_dist is not None:
            rem_dist = self.td_state['agents']['rem_dist'].unsqueeze(-1)
            c4 = time2j + self.td_state['time2depot'].unsqueeze(1) <= rem_dist
            mask = mask & c4

        return mask

    def _compute_serviceable_pending_mask_all_agents(self):
        """
        Compute whether each agent can still serve each pending non-depot node
        either directly from current state or (when refuel mode is enabled)
        after first returning to depot and refueling.

        Returns:
            torch.Tensor: serviceable mask [B, A, N].
        """
        # [B, N]
        pending_nodes = self.td_state['nodes']['active_nodes_mask'] & (~self.td_state['is_depot'])

        # [B, A, 2]
        agent_locs = self.td_state['coords'].gather(
            1,
            self.td_state['agents']['cur_node'][:, :, None].expand(-1, -1, 2),
        )
        # [B, A, N, 2]
        diff = agent_locs[:, :, None, :] - self.td_state['coords'][:, None, :, :]
        # [B, A, N]
        dist_cur_to_node = diff.pow(2).sum(-1).sqrt()
        if self.n_digits is not None:
            dist_cur_to_node = torch.floor(self.n_digits * dist_cur_to_node) / self.n_digits

        # Shared node-wise tensors -> [B, 1, N]
        tw_low = self.td_state['tw_low'].unsqueeze(1)
        tw_high = self.td_state['tw_high'].unsqueeze(1)
        service_time = self.td_state['service_time'].unsqueeze(1)
        appear_time = self.td_state['appear_time'].unsqueeze(1)
        time2depot = self.td_state['time2depot'].unsqueeze(1)
        end_time = self.td_state['end_time'].unsqueeze(-1).unsqueeze(1)
        demands = self.td_state['demands'].unsqueeze(1)

        # Agent-wise tensors -> [B, A, 1]
        cur_time = self.td_state['agents']['cur_time'].unsqueeze(-1)
        cur_load = self.td_state['agents']['cur_load'].unsqueeze(-1)

        # Direct service (no prior depot return).
        arrive_direct = cur_time + dist_cur_to_node
        service_start_direct = torch.maximum(torch.maximum(arrive_direct, tw_low), appear_time)
        c1_direct = service_start_direct <= tw_high
        c2_direct = service_start_direct + service_time + time2depot <= end_time
        c3_direct = demands <= cur_load
        direct = pending_nodes.unsqueeze(1) & c1_direct & c2_direct & c3_direct

        if self.max_flight_dist is not None:
            rem_dist = self.td_state['agents']['rem_dist'].unsqueeze(-1)
            c4_direct = dist_cur_to_node + time2depot <= rem_dist
            direct = direct & c4_direct

        # Optional one-hop via depot (for refuel/reload mode).
        if self.allow_depot_refuel:
            depot_locs = self.td_state['depot_loc'][:, :1, :]  # [B,1,2]
            dist_cur_to_depot = (agent_locs - depot_locs).pow(2).sum(-1).sqrt().unsqueeze(-1)  # [B,A,1]
            if self.n_digits is not None:
                dist_cur_to_depot = torch.floor(self.n_digits * dist_cur_to_depot) / self.n_digits

            time_at_depot = cur_time + dist_cur_to_depot  # [B,A,1]
            # dist(depot,node) equals time2depot in this env's Euclidean metric.
            arrive_via = time_at_depot + time2depot
            service_start_via = torch.maximum(torch.maximum(arrive_via, tw_low), appear_time)
            c1_via = service_start_via <= tw_high
            c2_via = service_start_via + service_time + time2depot <= end_time

            # After refuel/reload at depot, load resets to capacity.
            cap = self.td_state['agents']['capacity']
            while cap.dim() < 3:
                cap = cap.unsqueeze(-1)
            c3_via = demands <= cap

            via = pending_nodes.unsqueeze(1) & c1_via & c2_via & c3_via

            if self.max_flight_dist is not None:
                rem_dist = self.td_state['agents']['rem_dist'].unsqueeze(-1)
                max_dist = self.td_state['max_flight_dist'].unsqueeze(1)  # [B,1,1]
                can_reach_depot_now = dist_cur_to_depot <= rem_dist
                # After refuel at depot: need depot->node->depot within max flight distance.
                can_roundtrip_from_depot = (2.0 * time2depot) <= max_dist
                via = via & can_reach_depot_now & can_roundtrip_from_depot

            return direct | via

        return direct

    def _update_done_from_empty_actions(self, action_mask):
        """
        Mark a sample done when its current agent has no legal action and no
        agent can service any remaining non-depot node.

        This check is only performed on rows with an empty action mask. Checking
        all agents on every step would add an expensive O(A*N) operation to the
        normal rollout path.
        """
        if not self.allow_depot_refuel:
            return

        empty_actions = ~action_mask.any(dim=-1)
        if not empty_actions.any():
            return

        serviceable_all = self._compute_serviceable_pending_mask_all_agents()
        no_serviceable = ~serviceable_all.any(dim=-1).any(dim=-1)
        newly_done = empty_actions & no_serviceable & (~self.td_state['done'])
        self.td_state['done'] = self.td_state['done'] | newly_done
        self.td_state['is_last_step'] = self.td_state['is_last_step'] | newly_done

    def _compute_current_wait_mask_and_next_time(self, action):
        """
        Detect depot self-loops that should advance time to the next dynamic reveal.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                wait_mask [B,1] and next_time [B,1].
        """
        cur_time = self.td_state['cur_agent']['cur_time']
        is_depot = self.td_state['is_depot']
        active_non_depot = self.td_state['nodes']['active_nodes_mask'] & (~is_depot)
        future_pending = active_non_depot & (self.td_state['appear_time'] > cur_time)

        feasible_non_depot = self.td_state['cur_agent']['action_mask'] & (~is_depot)
        no_non_depot_feasible = ~feasible_non_depot.any(dim=1, keepdim=True)

        at_depot = self.td_state['cur_agent']['cur_node'].eq(self.td_state['depot_idx'])
        depot_action = action.eq(self.td_state['depot_idx'])
        has_future_pending = future_pending.any(dim=1, keepdim=True)

        wait_mask = at_depot & depot_action & no_non_depot_feasible & has_future_pending

        inf_time = torch.full_like(self.td_state['appear_time'], float("inf"))
        next_time = torch.where(
            future_pending,
            self.td_state['appear_time'],
            inf_time,
        ).min(dim=1, keepdim=True).values
        next_time = torch.where(wait_mask, next_time, cur_time)
        return wait_mask, next_time


    def _update_done(self, action):

        """
        Update done state.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns:
            None.
        """

        former_done = self.td_state['done'].clone()

        # update served nodes
        self.td_state['nodes']['active_nodes_mask'].scatter_(1, action, action.eq(self.td_state['depot_idx']))

        if self.allow_depot_refuel:
            # keep all agents active while allowing depot returns
            self.td_state['agents']['active_agents_mask'][:] = True

            done_by_steps = torch.zeros(*self.batch_size, dtype=torch.bool, device=self.device)
            if self.max_steps is not None:
                done_by_steps = done_by_steps | ((self.env_nsteps + 1) >= self.max_steps)

            # done_by_feasible is expensive (O(A*N)). Cache/compute it less frequently to keep sampling fast.
            done_by_feasible = torch.zeros(*self.batch_size, dtype=torch.bool, device=self.device)
            if not done_by_steps.any():
                freq = max(1, int(getattr(self, "done_feasible_check_freq", 10)))
                need_check = (
                    self._done_by_feasible_cache is None
                    or freq <= 1
                    or ((self.env_nsteps + 1) % freq == 0)
                )
                if need_check:
                    # End only if no pending non-depot node is serviceable anymore.
                    # Serviceable includes direct service and (in refuel mode) one-hop via depot.
                    serviceable_all = self._compute_serviceable_pending_mask_all_agents()
                    done_by_feasible = ~serviceable_all.any(dim=-1).any(dim=-1)
                    self._done_by_feasible_cache = done_by_feasible
                    self._done_by_feasible_cache_step = self.env_nsteps + 1
                else:
                    done_by_feasible = self._done_by_feasible_cache

            self.td_state['done'] = done_by_steps | done_by_feasible
            self.td_state['done'][former_done] = True
        else:
            wait_mask, _ = self._compute_current_wait_mask_and_next_time(action)
            # update done agents
            self.td_state['agents']['active_agents_mask'].scatter_(
                1,
                self.td_state['cur_agent_idx'],
                ~(action.eq(self.td_state['depot_idx']) & ~wait_mask),
            )
            self.td_state['done'] = (~self.td_state['agents']['active_agents_mask']).all(dim=-1)
            self.td_state['done'][former_done] = True

        self.td_state['is_last_step'] = self.td_state['done'].eq(~former_done)


    def _update_state(self, action):

        """
        Update environment state.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns:
            None.
        """

        loc = self.td_state['coords'].gather(1, self.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        next_loc = self.td_state['coords'].gather(1, action[:,:,None].expand(-1, -1, 2))

        ptime = self.td_state['cur_agent']['cur_time'].clone()
        time2j = torch.pairwise_distance(loc, next_loc, eps=0, keepdim = False)
        if self.n_digits is not None:
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits
        tw = self.td_state['tw_low'].gather(1, action)
        service_time = self.td_state['service_time'].gather(1, action)

        arrivej = ptime + time2j
        waitj = torch.clip(tw-arrivej, min=0)

        time_update = arrivej + waitj + service_time
        wait_mask, next_time = self._compute_current_wait_mask_and_next_time(action)
        time_update = torch.where(wait_mask, next_time, time_update)
        # update agent cur node
        self.td_state['cur_agent']['cur_node'] = action
        self.td_state['agents']['cur_node'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_node'])
        # update agent cur time
        self.td_state['cur_agent']['cur_time'] = time_update

        # if agent is done set agent time to end_time
        agents_done = ~self.td_state['agents']['active_agents_mask'].gather(1, self.td_state['cur_agent_idx']).clone()

        self.td_state['cur_agent']['cur_time'] = torch.where(agents_done, self.td_state['end_time'].unsqueeze(-1), 
                                                             self.td_state['cur_agent']['cur_time'])
        self.td_state['agents']['cur_time'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_time'])

        # update agent cum traveled time
        self.td_state['cur_agent']['cur_ttime'] = time2j
        self.td_state['cur_agent']['cum_ttime'] += time2j
        self.td_state['agents']['cur_ttime'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_ttime'])
        self.td_state['agents']['cum_ttime'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cum_ttime'])

        if self.max_flight_dist is not None:
            self.td_state['cur_agent']['rem_dist'] -= time2j
            reset_mask = action.eq(self.td_state['depot_idx'])
            self.td_state['cur_agent']['rem_dist'] = torch.where(
                reset_mask,
                self.td_state['max_flight_dist'],
                self.td_state['cur_agent']['rem_dist'],
            )
            self.td_state['agents']['rem_dist'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['rem_dist'])

        # update agent profit
        self.td_state['cur_agent']['cum_profit'] += self.td_state['profits'].gather(1, action)
        self.td_state['agents']['cum_profit'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cum_profit'])
        self.td_state['cur_agent']['step_profit'] = self.td_state['profits'].gather(1, action)
        self.td_state['agents']['step_profit'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['step_profit'])
        self.td_state['nodes']['cur_profits'].scatter_(1, action, torch.zeros_like(action, dtype = torch.float))

        # update agent load and node demands
        self.td_state['cur_agent']['cur_load'] -= self.td_state['demands'].gather(1, action)
        if self.allow_depot_refuel:
            # Refuel/reload at depot.
            self.td_state['cur_agent']['cur_load'] = torch.where(
                action.eq(self.td_state['depot_idx']),
                self.td_state['agents']['capacity'],
                self.td_state['cur_agent']['cur_load'],
            )
        # is agent is done set agent cur_load to 0
        self.td_state['cur_agent']['cur_load'] = torch.where( agents_done, 0., 
                                                             self.td_state['cur_agent']['cur_load'])
        
        self.td_state['nodes']['cur_demands'].scatter_(1, action, torch.zeros_like(action, dtype = torch.float))
        self.td_state['agents']['cur_load'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_load'])
        # update visited nodes
        r = torch.arange(*self.td_state.batch_size, device=self.device)
        self.td_state['agents']['visited_nodes'][r, self.td_state['cur_agent_idx'].squeeze(-1), action.squeeze(-1)] = True

        # update agent step
        self.td_state['cur_agent']['cur_step'] = torch.where(~agents_done, self.td_state['cur_agent']['cur_step']+1, 
                                                             self.td_state['cur_agent']['cur_step'])
        self.td_state['agents']['cur_step'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_step'])
        self.td_state['cur_node_idx'] = action.clone()

        # if all done activate first agent to guarantee batch consistency during agent sampling
        self.td_state['agents']['active_agents_mask'][self.td_state['agents']['active_agents_mask'].sum(1).eq(0), 0] = True

    def _update_cur_agent(self, cur_agent_idx):

        """
        Update current agent.

        Args:
            cur_agent_idx(torch.Tensor): Current agent id.

        Returns:
            None.
        """

        self.td_state['cur_agent_idx'] =  cur_agent_idx
        self.td_state['cur_agent'] = TensorDict({
                                'action_mask': self.td_state['agents']['feasible_nodes'].gather(1, self.td_state['cur_agent_idx'][:,:,None].expand(-1, -1, self.num_nodes)).squeeze(1).clone(),
                                'cur_load': self.td_state['agents']['cur_load'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_time': self.td_state['agents']['cur_time'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_node': self.td_state['agents']['cur_node'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_ttime': self.td_state['agents']['cur_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cum_ttime': self.td_state['agents']['cum_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cum_profit': self.td_state['agents']['cum_profit'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'step_profit': self.td_state['agents']['step_profit'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_step': self.td_state['agents']['cur_step'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                }, batch_size=self.td_state.batch_size, device=self.device)
        if self.max_flight_dist is not None:
            self.td_state['cur_agent']['rem_dist'] = self.td_state['agents']['rem_dist'].gather(1, self.td_state['cur_agent_idx']).clone()
        
    def _update_solution(self, action):

        """
        Update agents and actions in solution.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns: 
            None.
        """

        # update solution dic
        if 'actions' in self.td_state['solution'].keys():
            self.td_state['solution','actions'] = torch.concat( [self.td_state['solution','actions'], action], dim=-1)
        else:
            self.td_state['solution','actions'] = action

        if 'agents' in self.td_state['solution'].keys():
            self.td_state['solution','agents'] = torch.concat( [self.td_state['solution','agents'], self.td_state['cur_agent_idx']], dim=-1)
        else:
            self.td_state['solution','agents'] = self.td_state['cur_agent_idx']

    def step(self, td: TensorDict) -> TensorDict:
        """
        Perform an environment step for active agent.

        Args:
            td(TensorDict): Environment tensor instance.

        Returns:
            td(TensorDict): Updated environment tensor instance.
        """
        action = td["action"]
        if self.use_local_view and self._local_node_idx is not None:
            action = self._local_node_idx.gather(1, action)
            td["action_global"] = action
            td["action"] = action
        if self.validate_actions:
            ok = self.td_state['cur_agent']['action_mask'].gather(1, action).all()
            assert bool(ok.item()), "not feasible action"

        self._update_done(action)
        done = self.td_state['done'].clone()
        is_last_step = self.td_state['is_last_step'].clone()

        # get reward and penalty
        reward, penalty = self.reward_evaluator.get_reward(action)

        # update env state    
        self._update_state(action)

        # update solution dic
        self. _update_solution(action)

        # select and update cur agent
        cur_agent_idx =  self.agent_selector._next_agent()
        self._update_cur_agent(cur_agent_idx)
        agent_step = self.td_state['cur_agent']['cur_step']

        # new observations
        td_observations = self.observe()
        self._update_done_from_empty_actions(td_observations['action_mask'])
        done = self.td_state['done'].clone()
        is_last_step = self.td_state['is_last_step'].clone()
        self.env_nsteps += 1
        td.update(
            {
                "agent_step": agent_step,
                "observations": td_observations,
                "reward": reward,
                "penalty":penalty,  
                "cur_agent_idx":cur_agent_idx,     
                "cur_node_idx": self.td_state['cur_node_idx'].clone(),
                "done": done,
                "is_last_step": is_last_step
            },
        )
        return td

    def check_solution_validity(self):
        """
        Check if solution is valid according to CVRPTW constraints.

        Args:
            N/a.

        Returns:
            None. Raises AssertionError if invalid.
        """

        curr_node = torch.zeros(*self.batch_size, dtype=torch.int64, device=self.device)
        curr_time = torch.zeros(*self.batch_size, dtype=torch.float32, device=self.device)
        visited_nodes = torch.zeros(*self.batch_size, self.num_nodes, dtype=torch.int64, device=self.device)

        sorted_indices = torch.argsort(self.td_state['solution']['agents'], dim=-1, stable=True)
        sorted_data = torch.gather(self.td_state['solution']['actions'], dim=-1, index=sorted_indices)
        demand = self.td_state['demands'].gather(1, sorted_data)

        for ii in range(sorted_data.size(1)):
            next_node = sorted_data[:, ii]

            curr_loc = gather_by_index(self.td_state['coords'], curr_node)
            next_loc = gather_by_index(self.td_state['coords'], next_node)
            dist = torch.pairwise_distance(curr_loc, next_loc, eps=0, keepdim=False)

            # Time window constraints
            arrivej = curr_time + dist
            tw_low = gather_by_index(self.td_state['tw_low'], next_node)
            tw_high = gather_by_index(self.td_state['tw_high'], next_node)
            service_time = gather_by_index(self.td_state['service_time'], next_node)
            time2depot = gather_by_index(self.td_state['time2depot'], next_node)
            end_time = self.td_state['end_time']

            waitj = torch.clip(tw_low - arrivej, min=0)
            service_startj = arrivej + waitj

            # c1: service must start before tw_high_limit
            assert torch.all(service_startj <= tw_high), "Service started after allowed time window."
            # c2: must be able to finish service and return to depot before end_time
            assert torch.all(service_startj + service_time + time2depot <= end_time.unsqueeze(-1)), "Cannot finish service and return to depot in time."

            # Mark node as visited
            fill = visited_nodes.gather(1, next_node.unsqueeze(-1))
            visited_nodes.scatter_(1, next_node.unsqueeze(-1), fill + 1)

            # Update time and load
            curr_time = torch.max(arrivej, tw_low) + service_time
            curr_node = next_node
            curr_time[next_node == 0] = 0.0

        visited_nodes_exc_depot = visited_nodes[:, 1:]
        assert torch.all((visited_nodes_exc_depot == 0) | (visited_nodes_exc_depot == 1)), "Nodes were visited more than once!"
        # c3: capacity constraint
        used_cap = torch.zeros_like(self.td_state['demands'][:, 0])
        for ii in range(sorted_data.size(1)):
            #reset at depot
            used_cap = used_cap * (sorted_data[:, ii] != 0)
            used_cap += demand[:, ii]

            #Loads must no exceed capacity
            assert torch.all(used_cap <= self.td_state['capacity']), "Agent exceeded vehicle capacity."
