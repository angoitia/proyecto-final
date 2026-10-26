"""Champion-selection policy tests (no MLflow server required)."""

from __future__ import annotations

from mlops.training.registry import decide_promotion


def test_first_candidate_passing_gate_is_promoted():
    decision = decide_promotion(new_rmse=53.2, threshold=60.0, current_champion_rmse=None)

    assert decision.promote is True
    assert decision.gate_passed is True


def test_candidate_above_threshold_is_rejected_even_without_champion():
    decision = decide_promotion(new_rmse=61.0, threshold=60.0, current_champion_rmse=None)

    assert decision.promote is False
    assert decision.gate_passed is False
    assert "Quality gate failed" in decision.reason


def test_better_candidate_replaces_champion():
    decision = decide_promotion(new_rmse=52.0, threshold=60.0, current_champion_rmse=53.0)

    assert decision.promote is True
    assert decision.gate_passed is True


def test_equal_candidate_is_promoted():
    decision = decide_promotion(new_rmse=53.0, threshold=60.0, current_champion_rmse=53.0)

    assert decision.promote is True


def test_worse_candidate_does_not_replace_champion():
    decision = decide_promotion(new_rmse=54.0, threshold=60.0, current_champion_rmse=53.0)

    assert decision.promote is False
    assert decision.gate_passed is True
    assert "current champion is kept" in decision.reason
