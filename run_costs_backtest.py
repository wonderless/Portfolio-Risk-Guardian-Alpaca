"""
Impacto de costos de transacción (comisión + slippage) en el backtest
(Fase 5 — mejora post-diagnóstico de sobreajuste)

Todos los backtests anteriores del proyecto asumían ejecución perfecta y
gratuita. Alpaca no cobra comisión en acciones, pero el slippage (la
diferencia entre el precio que ves y el precio al que realmente se
ejecuta la orden) es real incluso en brokers "commission-free". Este
script corre la misma estrategia (RSI+SMA+ADX, sin cortos) con y sin un
supuesto conservador de costos, para ver cuánto de la mejora reportada en
la Fase 5 sobrevive a una ejecución realista.

Uso:
    python run_costs_backtest.py                  # AAPL, MSFT, TSLA
    python run_costs_backtest.py NVDA GOOGL
"""

import sys

from core.alpaca_client import AlpacaClient
from agents.technical_agent import TechnicalAgent
from core.backtester import run_technical_backtest, BacktestResult

LOOKBACK_DAYS = 730
COMMISSION_BPS = 0.0   # Alpaca no cobra comisión en acciones
SLIPPAGE_BPS = 5.0     # supuesto conservador para acciones líquidas (~0.05% por trade)


def print_result(label: str, result: BacktestResult):
    active_signals = result.signal_counts["buy"] + result.signal_counts["sell"]
    print(f"  [{label:20}] retorno={result.strategy_total_return:+7.1%}  "
          f"Sharpe={result.sharpe_ratio:6.2f}  MaxDD={result.strategy_max_drawdown:7.1%}  "
          f"señales activas={active_signals}")


def main():
    tickers = sys.argv[1:] or ["AAPL", "MSFT", "TSLA"]

    print("=" * 70)
    print("Impacto de costos de transacción en el backtest (sin cortos, ADX>=25)")
    print("=" * 70)
    print(f"Supuesto: comisión {COMMISSION_BPS:.0f}bps + slippage {SLIPPAGE_BPS:.0f}bps por cambio de posición.")

    client = AlpacaClient()
    agent = TechnicalAgent(alpaca_client=client)

    for symbol in tickers:
        print(f"\n{symbol}")
        try:
            bars = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
            result_no_cost = run_technical_backtest(agent, bars, symbol, allow_short=False)
            result_with_cost = run_technical_backtest(
                agent, bars, symbol, allow_short=False,
                commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS,
            )
            print_result("sin costos", result_no_cost)
            print_result("con costos", result_with_cost)
            drag = result_no_cost.strategy_total_return - result_with_cost.strategy_total_return
            print(f"  Costo total absorbido por el retorno: {drag:+.2%}")
        except Exception as e:
            print(f"  AVISO: no se pudo backtestear {symbol}: {e}")


if __name__ == "__main__":
    main()
