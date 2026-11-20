from maenvs4vrp.environments.dtoptw.env import Environment as _BaseEnv


class Environment(_BaseEnv):
    """Same dynamics as dtoptw, but typically paired with dtoptw_abs.observations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_name = "dtoptw_abs"

