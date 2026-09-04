"""
Prueba fuera de universo / multi-sector (Fase 5 — diagnóstico de overfitting)

Corre la estrategia del Agente Técnico (RSI 14 + SMA 20/50 + ADX>=25, sin
cortos) SIN TOCAR NINGÚN PARÁMETRO, sobre una cesta diversificada de
tickers de distintos sectores y un período más largo que el usado para
desarrollarla (AAPL/MSFT/TSLA, ~2 años). El objetivo es ver si el enfoque
generaliza o si solo funcionaba en los 3 tickers sobre los que se ajustó.

Uso:
    python run_universe_test.py
"""

from core.alpaca_client import AlpacaClient
from agents.technical_agent import TechnicalAgent
from core.backtester import run_technical_backtest

LOOKBACK_DAYS = 1825  # ~5 años

# Cesta diversificada por sector (deliberadamente sin TSLA/AAPL/MSFT,
# los tickers sobre los que se desarrolló y ajustó la estrategia)
UNIVERSE = {
    "Consumo defensivo": ["KO", "PG", "WMT"],
    "Energía": ["XOM", "CVX"],
    "Salud": ["JNJ", "UNH"],
    "Industrial": ["CAT", "HON"],
    "Financiero": ["JPM", "BAC"],
    "Utilities": ["NEE"],
    "Comunicaciones": ["DIS"],
}


def main():
    client = AlpacaClient()
    agent = TechnicalAgent(alpaca_client=client)  # parámetros por defecto, sin tocar

    print("=" * 70)
    print("Prueba fuera de universo — mismos parámetros, tickers y período distintos")
    print("=" * 70)
    print(f"Ventana: ~{LOOKBACK_DAYS // 365} años. Estrategia: RSI14 + SMA20/50 + ADX>=25, sin cortos.")

    all_results = []

    for sector, tickers in UNIVERSE.items():
        print(f"\n{sector}")
        for symbol in tickers:
            try:
                bars = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
                result = run_technical_backtest(agent, bars, symbol, allow_short=False)
                active_signals = result.signal_counts["buy"] + result.signal_counts["sell"]
                beat_bh = "SÍ" if result.strategy_total_return > result.buy_and_hold_return else "no"
                print(f"  {symbol:6} retorno={result.strategy_total_return:+7.1%}  "
                      f"buy&hold={result.buy_and_hold_return:+7.1%}  ¿le gana?={beat_bh:3}  "
                      f"Sharpe={result.sharpe_ratio:5.2f}  señales={active_signals}")
                all_results.append(result)
            except Exception as e:
                print(f"  {symbol:6} AVISO: no se pudo backtestear: {e}")

    if not all_results:
        print("\nNo se pudo backtestear ningún ticker.")
        return

    total_trades = sum(r.signal_counts["buy"] + r.signal_counts["sell"] for r in all_results)
    num_beat_bh = sum(1 for r in all_results if r.strategy_total_return > r.buy_and_hold_return)
    avg_sharpe = sum(r.sharpe_ratio for r in all_results) / len(all_results)
    positive_return = sum(1 for r in all_results if r.strategy_total_return > 0)

    print(f"\n{'-' * 70}")
    print("RESUMEN")
    print(f"  Tickers probados:                {len(all_results)}")
    print(f"  Señales activas acumuladas:       {total_trades}  "
          f"(la investigación recomienda >=100-250 para confianza estadística)")
    print(f"  Tickers con retorno positivo:     {positive_return}/{len(all_results)}")
    print(f"  Tickers que le ganan a buy&hold:  {num_beat_bh}/{len(all_results)}")
    print(f"  Sharpe promedio:                  {avg_sharpe:.2f}")


if __name__ == "__main__":
    main()
