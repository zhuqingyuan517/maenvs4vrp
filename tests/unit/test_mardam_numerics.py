import pytest
import torch

from maenvs4vrp.learning.mardam._learner import AttentionLearner
from maenvs4vrp.learning.mardam.layers import reinforce_mask_loss
from maenvs4vrp.learning.mardam.layers._mha import _MHA_V2
from maenvs4vrp.learning.mardam.numerics import (
    NonFiniteTrainingError,
    assert_finite,
)


def test_all_masked_attention_is_finite_and_zero():
    torch.manual_seed(0)
    attention = _MHA_V2(head_count=2, query_size=4)
    queries = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.ones(2, 3, 3, dtype=torch.bool)

    output = attention(queries, mask=mask)

    assert torch.isfinite(output).all()
    assert torch.allclose(output, torch.zeros_like(output))
    output.sum().backward()
    assert torch.isfinite(queries.grad).all()


def test_all_masked_action_log_probabilities_keep_depot_fallback():
    compat = torch.randn(2, 1, 5, requires_grad=True)
    all_masked = torch.ones(2, 1, 5, dtype=torch.bool)

    logp = AttentionLearner._get_logp(compat, all_masked, debug=True)

    assert torch.isfinite(logp[:, :1]).all()
    assert torch.isneginf(logp[:, 1:]).all()
    assert torch.allclose(logp.exp()[:, 0], torch.ones(2))


def test_reinforce_mask_loss_can_normalize_by_valid_steps():
    rewards = [
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[1.0], [2.0]]),
    ]
    logps = [torch.full((2, 1), -1.0) for _ in range(3)]
    baseline = [torch.zeros(2, 1) for _ in range(3)]
    step_mask = [
        torch.tensor([[True], [True]]),
        torch.tensor([[True], [False]]),
        torch.tensor([[False], [False]]),
    ]

    summed = reinforce_mask_loss(logps, rewards, baseline, step_mask)
    normalized = reinforce_mask_loss(
        logps,
        rewards,
        baseline,
        step_mask,
        normalize_step_loss=True,
    )

    assert torch.isclose(summed, torch.tensor(5.5))
    assert torch.isclose(normalized, torch.tensor(4.25))


def test_assert_finite_reports_non_finite_values():
    with pytest.raises(NonFiniteTrainingError, match="action_probabilities"):
        assert_finite(
            "action_probabilities",
            torch.tensor([0.5, float("nan")]),
        )
