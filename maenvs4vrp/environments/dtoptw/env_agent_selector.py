from maenvs4vrp.core.env import AECEnv
from maenvs4vrp.core.env_agent_selector import BaseSelector
import torch


class AgentSelector(BaseSelector):
    """
    DTOPTW agent selector class.
    """
    def __init__(self):
        super().__init__()

        """
        Constructor.

        Args:
            n/a.

        Returns:
            None.
        """

    def set_env(self, env: AECEnv):
        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """
        super().set_env(env)

    def _next_agent(self):
        """
        Return the next agent.

        Args:
            n/a.

        Returns:
            selected_agent(torch.Tensor): Next agent.  
        """
        if getattr(self.env, "allow_depot_refuel", False):
            return (self.env.td_state['cur_agent_idx'] + 1) % self.env.num_agents
        avail = torch.arange(self.env.num_agents, dtype = torch.float).unsqueeze(0).repeat(*self.env.batch_size, 1).to(self.env.device)
        avail[~self.env.td_state['agents']['active_agents_mask']] = float('inf')
        selected_agent = avail.argmin(1, keepdim = True)
        return selected_agent
    

class RandomSelector(BaseSelector):
    """
    DVRPTW random agent selector class.
    """
    def __init__(self):
        super().__init__()

        """
        Constructor.

        Args:
            n/a.

        Returns:
            None.
        """

    def set_env(self, env: AECEnv):
        super().set_env(env)

        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """

    def _next_agent(self):
        """
        Return the next agent.

        Args:
            n/a.

        Returns:
            selected_agent(torch.Tensor): Next agent. 
        """
        selected_agent = torch.multinomial(self.env.td_state['agents']['active_agents_mask'].float(), 1).to(self.env.device)
        return selected_agent
    


class SmallestTimeAgentSelector(BaseSelector):
    """
    DVRPTW smallest time agent selector class.
    """
    def __init__(self):
        super().__init__()
        """
        Constructor.

        Args:
            n/a.

        Returns:
            None.
        """

    def set_env(self, env: AECEnv):
        super().set_env(env)

        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """

    def _next_agent(self):
        """
        Return the next agent.

        Args:
            n/a.

        Returns:
            selected_agent(torch.Tensor): Next agent. 
        """
        avail = self.env.td_state['agents']['cur_time'].clone()
        active = self.env.td_state['agents']['active_agents_mask']
        avail[~active] = float('inf')

        # In refuel mode, many agents can share identical cur_time (often 0.0 initially).
        # A plain argmin will keep picking the same smallest-index agent and the simulation "stalls"
        # (cur_time never advances if the policy keeps choosing depot). Break ties stochastically.
        if getattr(self.env, "allow_depot_refuel", False):
            min_t = avail.min(dim=1, keepdim=True).values
            ties = active & (avail == min_t)
            selected_agent = torch.multinomial(ties.float(), 1).to(self.env.device)
            return selected_agent

        selected_agent = avail.argmin(1, keepdim=True)
        return selected_agent
