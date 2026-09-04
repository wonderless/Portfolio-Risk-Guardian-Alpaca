"""
Script de la Fase 5: Backtesting del Agente Técnico sobre datos históricos.

Solo el Agente Técnico se backtestea (ver core/backtester.py para la
justificación). Es gratis y rápido: no llama a Claude ni a NewsAPI, solo
usa datos históricos de precio de Alpaca.

Corre dos variantes por ticker para poder comparar:
  - "con cortos": SELL abre posición corta (como haría el sistema real).
  - "sin cortos": SELL simplemente sale a plano, nunca corto.

Uso:
    python run_backtest.py                  # AAPL, MSFT, TSLA, ~2 años
    python run_backtest.py NVDA GOOGL        # tickers específicos
"""

import sys

from core.alpaca_client import AlpacaClient
from agents.technical_agent import TechnicalAgent
from core.backtester import run_technical_backtest, BacktestResult

LOOKBACK_DAYS = 730  # ~2 años de historia diaria


def print_result(label: str, result: BacktestResult):
    print(f"\n  [{label}]")
    print(f"  Señales:              {result.signal_counts}")
    print(f"  Retorno estrategia:   {result.strategy_total_return:+.1%}  (buy & hold: {result.buy_and_hold_return:+.1%})")
    print(f"  Máximo drawdown:      {result.strategy_max_drawdown:.1%}")
    print(f"  Sharpe / Sortino:     {result.sharpe_ratio:.2f} / {result.sortino_ratio:.2f}")
    print(f"  Calmar:               {result.calmar_ratio:.2f}")
    print(f"  Win rate:             {result.win_rate:.1%}")
    print(f"  Profit factor:        {result.profit_factor:.2f}")


def main():
    tickers = sys.argv[1:] or ["AAPL", "MSFT", "TSLA"]

    print("=" * 60)
    print("PORTFOLIO RISK GUARDIAN — Fase 5: Backtesting (Agente Técnico)")
    print("=" * 60)
    print(f"Ventana: últimos {LOOKBACK_DAYS} días.")
    print("Sin comisiones, slippage ni position sizing — compara la calidad de las señales, no simula ejecución real.")

    client = AlpacaClient()
    agent = TechnicalAgent(alpaca_client=client)

    for symbol in tickers:
        print(f"\n{'-' * 60}")
        print(f"{symbol}")
        try:
            bars = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
            print(f"  Período: {bars.index[0].date()} -> {bars.index[-1].date()} ({len(bars)} velas diarias)")

            result_short = run_technical_backtest(agent, bars, symbol, allow_short=True)
            print_result("con cortos", result_short)

            result_long_only = run_technical_backtest(agent, bars, symbol, allow_short=False)
            print_result("sin cortos (solo largo/plano)", result_long_only)
        except Exception as e:
            print(f"\n  AVISO: no se pudo backtestear {symbol}: {e}")


if __name__ == "__main__":
    main()
