"""
Script de prueba de la Fase 3: Orquestador con LangGraph (Agente 4).

Corre los agentes 1, 2 y 3 en paralelo para cada ticker, aplica la lógica
de consenso (mayoría 2 de 3) y — solo si AUTO_EXECUTE=true — ejecuta la
orden resultante en Alpaca (paper trading).

Por defecto AUTO_EXECUTE está desactivado: el script muestra qué haría el
sistema sin mover dinero simulado, para que puedas revisar el
razonamiento antes de dejarlo operar solo.

Si además se activa USE_ATR_SIZING, una entrada BUY no compra ORDER_QTY
acciones fijas: dimensiona la posición por volatilidad (ATR), la misma
fórmula ya validada en el backtest de Fase 5 (ver README, "Exploración de
breadth..." y `TechnicalAgent.calculate_position_size`). Sin esta bandera,
ORDER_QTY sigue siendo fijo, como en toda la Fase 3 original.

Uso:
    python run_phase3.py                                        # solo simula, no ejecuta órdenes
    AUTO_EXECUTE=true python run_phase3.py                      # ejecuta con tamaño fijo (ORDER_QTY)
    AUTO_EXECUTE=true USE_ATR_SIZING=true python run_phase3.py  # ejecuta con sizing por ATR
"""

import os

from core.alpaca_client import AlpacaClient
from agents.technical_agent import TechnicalAgent
from agents.sentiment_agent import SentimentAgent
from agents.risk_agent import RiskAgent
from agents.orchestrator import build_orchestrator_graph

AUTO_EXECUTE = os.getenv("AUTO_EXECUTE", "false").lower() == "true"
USE_ATR_SIZING = os.getenv("USE_ATR_SIZING", "false").lower() == "true"
ORDER_QTY = 1


def main():
    print("=" * 60)
    print("PORTFOLIO RISK GUARDIAN — Fase 3: Orquestador (LangGraph)")
    print("=" * 60)
    print(f"AUTO_EXECUTE = {AUTO_EXECUTE} "
          f"({'ejecutará órdenes reales de paper trading' if AUTO_EXECUTE else 'modo simulación, no ejecuta nada'})")
    print(f"USE_ATR_SIZING = {USE_ATR_SIZING} "
          f"({'tamaño por volatilidad (ATR)' if USE_ATR_SIZING else f'tamaño fijo ({ORDER_QTY} acción)'})")

    client = AlpacaClient()
    account = client.get_account_info()
    print(f"\nOK - Conectado a Alpaca Paper Trading")
    print(f"  Cash disponible: ${account['cash']:,.2f}")
    print(f"  Equity total:    ${account['equity']:,.2f}")

    graph = build_orchestrator_graph(
        technical_agent=TechnicalAgent(alpaca_client=client),
        sentiment_agent=SentimentAgent(),
        risk_agent=RiskAgent(alpaca_client=client),
        alpaca_client=client,
    )

    tickers = ["AAPL", "MSFT", "TSLA"]

    for symbol in tickers:
        print(f"\n{'-' * 60}")
        print(f"Orquestando {symbol}...")

        result = graph.invoke({
            "symbol": symbol,
            "votes": [],
            "consensus": None,
            "auto_execute": AUTO_EXECUTE,
            "order_qty": ORDER_QTY,
            "use_atr_sizing": USE_ATR_SIZING,
        })

        consensus = result["consensus"]
        print(f"\n  Decisión final: {consensus.final_signal.value.upper()}")
        print(f"  Consenso alcanzado: {consensus.consensus_reached}")
        print(f"  Ejecutada: {consensus.executed}"
              + (f" (order_id={consensus.order_id})" if consensus.order_id else ""))
        print(f"  Explicación: {consensus.explanation}")


if __name__ == "__main__":
    main()
