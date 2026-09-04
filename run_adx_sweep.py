"""
Barrido de sensibilidad del umbral de ADX (Fase 5 — diagnóstico de overfitting)

Corre el backtest del Agente Técnico variando SOLO el umbral de ADX (15 a 35,
en pasos de 1), manteniendo todo lo demás fijo (RSI 14, SMA 20/50, sin
cortos). El objetivo es distinguir si el umbral ADX>=25 que elegimos capta
una propiedad real del activo (rendimiento estable en una "meseta" de
umbrales cercanos) o si fue una coincidencia con el ruido de este período
específico (un "pico" aislado que colapsa en valores vecinos) — ver la
sección de investigación en README.md.

Uso:
    python run_adx_sweep.py                  # AAPL, MSFT, TSLA
    python run_adx_sweep.py NVDA
"""

import sys

from core.alpaca_client import AlpacaClient
from agents.technical_agent import TechnicalAgent
from core.backtester import run_technical_backtest

LOOKBACK_DAYS = 730
ADX_RANGE = range(15, 36)  # 15..35 inclusive


def main():
    tickers = sys.argv[1:] or ["AAPL", "MSFT", "TSLA"]

    client = AlpacaClient()

    print("=" * 70)
    print("Barrido de sensibilidad del umbral ADX (15-35), sin cortos")
    print("=" * 70)
    print("Meseta estable = el filtro capta algo real. Pico aislado = sobreajuste.")

    for symbol in tickers:
        print(f"\n{symbol}")
        print(f"{'ADX':>5} {'Retorno':>10} {'Sharpe':>8} {'MaxDD':>8} {'#Señales activas':>18}")

        try:
            bars = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
        except Exception as e:
            print(f"  AVISO: no se pudo traer historial de {symbol}: {e}")
            continue

        for threshold in ADX_RANGE:
            agent = TechnicalAgent(alpaca_client=client, adx_threshold=float(threshold))
            try:
                result = run_technical_backtest(agent, bars, symbol, allow_short=False)
                active_signals = result.signal_counts["buy"] + result.signal_counts["sell"]
                print(f"{threshold:>5} {result.strategy_total_return:>+9.1%} {result.sharpe_ratio:>8.2f} "
                      f"{result.strategy_max_drawdown:>8.1%} {active_signals:>18}")
            except Exception as e:
                print(f"{threshold:>5}  ERROR: {e}")


if __name__ == "__main__":
    main()
