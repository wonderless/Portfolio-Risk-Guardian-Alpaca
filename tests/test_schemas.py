"""
Tests de `core/schemas.py` — los modelos compartidos que hacen posible
que el orquestador combine votos de agentes que no se conocen entre sí.
Lo que importa validar acá no es pydantic (eso ya está probado), sino que
las reglas propias del proyecto (confidence entre 0 y 1, reasoning
obligatorio) realmente se cumplen.
"""

import pytest
from pydantic import ValidationError

from core.schemas import AgentVote, ConsensusDecision, Signal


def _vote(signal=Signal.BUY, confidence=0.8, agent_name="technical_agent"):
    return AgentVote(
        agent_name=agent_name,
        symbol="AAPL",
        signal=signal,
        confidence=confidence,
        reasoning="RSI en 25 indica sobreventa.",
    )


def test_agent_vote_accepts_valid_confidence():
    vote = _vote(confidence=0.5)
    assert vote.confidence == 0.5
    assert vote.signal == Signal.BUY


@pytest.mark.parametrize("bad_confidence", [-0.01, 1.01, 2.0, -5.0])
def test_agent_vote_rejects_confidence_out_of_range(bad_confidence):
    with pytest.raises(ValidationError):
        _vote(confidence=bad_confidence)


def test_agent_vote_requires_reasoning():
    with pytest.raises(ValidationError):
        AgentVote(agent_name="technical_agent", symbol="AAPL", signal=Signal.HOLD, confidence=0.5)


def test_agent_vote_raw_metrics_defaults_to_empty_dict():
    vote = _vote()
    assert vote.raw_metrics == {}


def test_signal_accepts_only_buy_sell_hold():
    assert {s.value for s in Signal} == {"buy", "sell", "hold"}


def test_consensus_decision_defaults_not_executed():
    votes = [_vote(agent_name=name) for name in ("technical_agent", "sentiment_agent", "risk_agent")]
    decision = ConsensusDecision(
        symbol="AAPL",
        final_signal=Signal.BUY,
        votes=votes,
        consensus_reached=True,
        explanation="Consenso alcanzado: 3/3 agentes votaron BUY.",
    )
    assert decision.executed is False
    assert decision.order_id is None
    assert len(decision.votes) == 3
