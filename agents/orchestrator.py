"""
Agente 4 — Orquestador / Ejecutor

Responsabilidad: coordinar a los agentes 1, 2 y 3 (que corren en paralelo),
aplicar lógica de consenso sobre sus votos, y — si corresponde — ejecutar
la orden resultante en Alpaca (paper trading).

A diferencia de los agentes 1-3, el orquestador SÍ conoce a los demás
agentes (es su trabajo coordinarlos), pero no duplica su lógica interna:
solo lee el AgentVote estandarizado que cada uno produce.

Lógica de consenso: mayoría simple 2 de 3. Si al menos dos agentes
coinciden en la misma señal (BUY/SELL/HOLD), esa es la decisión final.
Si los tres votan distinto (empate total), se opta por HOLD por cautela
— en un sistema que puede mover dinero (aunque sea simulado), la falta
de consenso claro no debe traducirse en una acción agresiva.
"""

import operator
from collections import Counter
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from core.alpaca_client import AlpacaClient
from core.schemas import AgentVote, ConsensusDecision, Signal
from agents.technical_agent import TechnicalAgent
from agents.sentiment_agent import SentimentAgent
from agents.risk_agent import RiskAgent


class OrchestratorState(TypedDict):
    symbol: str
    votes: Annotated[list[AgentVote], operator.add]
    consensus: ConsensusDecision | None
    auto_execute: bool
    order_qty: float
    use_atr_sizing: bool


def _majority_signal(votes: list[AgentVote]) -> tuple[Signal | None, int]:
    """Devuelve la señal mayoritaria (2 de 3) y cuántos votos tuvo, o (None, 0) si no hay mayoría."""
    counts = Counter(v.signal for v in votes)
    signal, count = counts.most_common(1)[0]
    return (signal, count) if count >= 2 else (None, 0)


def _votes_summary(votes: list[AgentVote]) -> str:
    return " | ".join(f"{v.agent_name}={v.signal.value.upper()}({v.confidence:.0%})" for v in votes)


def build_orchestrator_graph(technical_agent: TechnicalAgent, sentiment_agent: SentimentAgent,
                              risk_agent: RiskAgent, alpaca_client: AlpacaClient):
    """
    Construye el grafo de LangGraph:

        START -> [technical, sentiment, risk] (en paralelo) -> consensus -> execute? -> END
    """

    def technical_node(state: OrchestratorState) -> dict:
        return {"votes": [technical_agent.analyze(state["symbol"])]}

    def sentiment_node(state: OrchestratorState) -> dict:
        return {"votes": [sentiment_agent.analyze(state["symbol"])]}

    def risk_node(state: OrchestratorState) -> dict:
        return {"votes": [risk_agent.analyze(state["symbol"])]}

    def consensus_node(state: OrchestratorState) -> dict:
        votes = state["votes"]
        signal, count = _majority_signal(votes)
        summary = _votes_summary(votes)

        if signal is not None:
            consensus = ConsensusDecision(
                symbol=state["symbol"],
                final_signal=signal,
                votes=votes,
                consensus_reached=True,
                explanation=f"Consenso alcanzado: {count}/3 agentes votaron {signal.value.upper()}. Detalle: {summary}",
            )
        else:
            consensus = ConsensusDecision(
                symbol=state["symbol"],
                final_signal=Signal.HOLD,
                votes=votes,
                consensus_reached=False,
                explanation=f"Sin mayoría clara (3 señales distintas): se opta por HOLD por cautela. Detalle: {summary}",
            )
        return {"consensus": consensus}

    def route_after_consensus(state: OrchestratorState) -> str:
        consensus = state["consensus"]
        should_execute = (
            state.get("auto_execute", False)
            and consensus.consensus_reached
            and consensus.final_signal != Signal.HOLD
        )
        return "execute" if should_execute else "end"

    def execute_node(state: OrchestratorState) -> dict:
        """
        Ejecuta la orden de consenso, sin abrir posiciones cortas: el
        backtest de la Fase 5 mostró que vender en corto ante señal SELL
        empeora el resultado en los tres tickers probados (ver
        core/backtester.py) — un SELL sin posición existente simplemente
        no hace nada, en vez de apostar a la baja.

        Si `use_atr_sizing` está activo, una entrada BUY se dimensiona por
        volatilidad (ATR) en vez de usar `order_qty` fijo — la misma
        fórmula validada en el backtest de Fase 5 (ver
        `TechnicalAgent.calculate_position_size`). Es opt-in explícito,
        no el default: cambiar cómo se dimensiona una posición real debe
        ser una decisión consciente, igual que `AUTO_EXECUTE`.
        """
        consensus = state["consensus"]
        symbol = state["symbol"]

        try:
            if consensus.final_signal == Signal.SELL:
                positions = alpaca_client.get_open_positions()
                held_qty = next((p["qty"] for p in positions if p["symbol"] == symbol), 0.0)
                if held_qty <= 0:
                    consensus.executed = False
                    consensus.explanation += (
                        f" | No se ejecuta: señal SELL pero no hay posición abierta en {symbol} "
                        f"(el sistema no abre posiciones cortas)."
                    )
                    return {"consensus": consensus}
                order_qty = min(state["order_qty"], held_qty)
            elif state.get("use_atr_sizing", False):
                equity = alpaca_client.get_account_info()["equity"]
                order_qty = technical_agent.calculate_position_size(symbol, account_equity=equity)
                if order_qty <= 0:
                    consensus.executed = False
                    consensus.explanation += (
                        f" | No se ejecuta: el sizing por ATR dio 0 acciones para {symbol} "
                        f"(volatilidad insuficiente o datos no disponibles)."
                    )
                    return {"consensus": consensus}
            else:
                order_qty = state["order_qty"]

            order = alpaca_client.submit_market_order(
                symbol=symbol,
                qty=order_qty,
                side=consensus.final_signal.value,
            )
            consensus.executed = True
            consensus.order_id = order["id"]
            consensus.explanation += f" | Orden ejecutada en paper trading: {order}"
        except Exception as e:
            consensus.executed = False
            consensus.explanation += f" | Error al ejecutar la orden: {e}"
        return {"consensus": consensus}

    graph = StateGraph(OrchestratorState)
    graph.add_node("technical", technical_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("risk", risk_node)
    graph.add_node("consensus", consensus_node)
    graph.add_node("execute", execute_node)

    graph.add_edge(START, "technical")
    graph.add_edge(START, "sentiment")
    graph.add_edge(START, "risk")
    graph.add_edge("technical", "consensus")
    graph.add_edge("sentiment", "consensus")
    graph.add_edge("risk", "consensus")
    graph.add_conditional_edges("consensus", route_after_consensus, {"execute": "execute", "end": END})
    graph.add_edge("execute", END)

    return graph.compile()
