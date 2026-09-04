"""
¿El propio `confidence` del Agente Técnico predice mejores resultados?
(Fase 5 — validación de una regla de decisión alternativa)

El orquestador combina 3 agentes con la regla "2 de 3" (mayoría simple),
pero esa regla nunca se validó contra alternativas porque Sentimiento y
Riesgo no tienen historial real para backtestear (ver README). Lo que sí
se puede probar con datos reales es más chico pero honesto: el Agente
Técnico ya calcula una `confidence` (0-1) por señal, más alta cuando RSI y
tendencia coinciden. Este script compara tamaño de posición FIJO (±1,
Fase 5) contra tamaño ESCALADO por esa confianza, para ver si pesar más
los días de alta confianza mejora el resultado ajustado por riesgo.

Uso:
    python run_confidence_sizing_backtest.py                  # AAPL, MSFT, TSLA
    python run_confidence_sizing_backtest.py NVDA GOOGL
"""

import sys

from core.alpaca_client import AlpacaClient
from agents.technical_agent import TechnicalAgent
from core.backtester import run_technical_backtest, BacktestResult

LOOKBACK_DAYS = 730


def print_result(label: str, result: BacktestResult):
    active_signals = result.signal_counts["buy"] + result.signal_counts["sell"]
    print(f"  [{label:18}] retorno={result.strategy_total_return:+7.1%}  "
          f"Sharpe={result.sharpe_ratio:6.2f}  Sortino={result.sortino_ratio:6.2f}  "
          f"MaxDD={result.strategy_max_drawdown:7.1%}  señales activas={active_signals}")


def main():
    tickers = sys.argv[1:] or ["AAPL", "MSFT", "TSLA"]

    print("=" * 70)
    print("Tamaño fijo vs. tamaño escalado por confianza (sin cortos, ADX>=25)")
    print("=" * 70)

    client = AlpacaClient()
    agent = TechnicalAgent(alpaca_client=client)

    for symbol in tickers:
        print(f"\n{symbol}")
        try:
            bars = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
            fixed = run_technical_backtest(agent, bars, symbol, allow_short=False)
            weighted = run_technical_backtest(agent, bars, symbol, allow_short=False, size_by_confidence=True)
            print_result("tamaño fijo", fixed)
            print_result("por confianza", weighted)
        except Exception as e:
            print(f"  AVISO: no se pudo backtestear {symbol}: {e}")


if __name__ == "__main__":
    main()
