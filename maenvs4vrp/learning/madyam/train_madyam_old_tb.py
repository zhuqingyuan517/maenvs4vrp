"""
TensorBoard-enabled entrypoint for the original policy_net_ma_ac_old setup on dtoptw_abs_full.

This is a thin wrapper so you can run a clearly named script for the baseline.
"""

from maenvs4vrp.learning.madyam.train_madyam_obs import main


if __name__ == "__main__":
    main()

