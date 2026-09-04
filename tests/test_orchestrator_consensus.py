"""
Tests de la lógica de consenso del Agente 4 (`agents/orchestrator.py`).

Se prueban directamente `_majority_signal` y `_votes_summary` (funciones
puras, sin tocar LangGraph ni Alpaca) porque son el corazón de la regla
"2 de 3" que documenta el README — y esa regla nunca se había validado
con un test, solo corriendo el sistema completo a mano.
"""

from core.schemas import AgentVote, Signal
from agents.orchestrator import _majority_signal, _votes_summary


def _vote(agent_name: str, signal: Signal, confidence: float = 0.8) -> AgentVote:
    return AgentVote(
        agent_name=agent_name,
        symbol="AAPL",
        signal=signal,
        confidence=confidence,
        reasoning="razón de prueba",
    )


def test_unanimous_buy_reaches_consensus():
    votes = [_vote("technical_agent", Signal.BUY), _vote("sentiment_agent", Signal.BUY),
             _vote("risk_agent", Signal.BUY)]
    signal, count = _majority_signal(votes)
    assert signal == Signal.BUY
    assert count == 3


def test_two_of_three_reaches_consensus():
    votes = [_vote("technical_agent", Signal.SELL), _vote("sentiment_agent", Signal.SELL),
             _vote("risk_agent", Signal.HOLD)]
    signal, count = _majority_signal(votes)
    assert signal == Signal.SELL
    assert count == 2


def test_three_way_split_has_no_majority():
    votes = [_vote("technical_agent", Signal.BUY), _vote("sentiment_agent", Signal.SELL),
             _vote("risk_agent", Signal.HOLD)]
    signal, count = _majority_signal(votes)
    assert signal is None
    assert count == 0


def test_majority_signal_ignores_confidence_only_counts_votes():
    # Dos votos SELL con baja confianza igual ganan sobre un BUY con alta confianza:
    # la regla es "2 de 3" por conteo, no ponderada por confianza (documentado en el README).
    votes = [_vote("technical_agent", Signal.SELL, confidence=0.3),
             _vote("sentiment_agent", Signal.SELL, confidence=0.3),
             _vote("risk_agent", Signal.BUY, confidence=0.95)]
    signal, count = _majority_signal(votes)
    assert signal == Signal.SELL
    assert count == 2


def test_votes_summary_includes_each_agent_and_signal():
    votes = [_vote("technical_agent", Signal.BUY, confidence=0.8),
             _vote("sentiment_agent", Signal.HOLD, confidence=0.5)]
    summary = _votes_summary(votes)
    assert "technical_agent=BUY(80%)" in summary
    assert "sentiment_agent=HOLD(50%)" in summary
