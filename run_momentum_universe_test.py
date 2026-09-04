"""
Prueba fuera de universo del momentum cruzado (mismo diagnóstico que
`run_universe_test.py` le aplicó al Agente Técnico en Fase 5).

El momentum cruzado se desarrolló y validó (ventana única, walk-forward,
costos) siempre sobre la misma cesta de 16 tickers
(`run_momentum_backtest.py`). Este script corre la MISMA lógica, SIN TOCAR
NINGÚN PARÁMETRO (lookback=12m, skip=1m, top_n=5, rebalance mensual, sin
cortos), sobre una cesta de 16 tickers DISTINTOS, de sectores similares
pero sin superposición con la original — para ver si el resultado
generaliza o si dependía del universo específico sobre el que se probó.

Uso:
    python run_momentum_universe_test.py
"""

from core.alpaca_client import AlpacaClient
from core.momentum_backtester import run_cross_sectional_momentum_backtest

LOOKBACK_DAYS = 1825  # ~5 años, igual que run_momentum_backtest.py

# Deliberadamente sin ningún ticker de la cesta original de
# run_momentum_backtest.py (AAPL, MSFT, TSLA, KO, PG, WMT, XOM, CVX, JNJ,
# UNH, CAT, HON, JPM, BAC, NEE, DIS), pero cubriendo sectores similares.
UNIVERSE = {
    "Tecnología": ["NVDA", "ORCL", "CSCO"],
    "Consumo discrecional": ["HD", "NKE", "SBUX"],
    "Consumo defensivo": ["PEP", "CL"],
    "Energía": ["COP", "SLB"],
    "Salud": ["PFE", "MRK"],
    "Industrial": ["GE", "MMM"],
    "Financiero": ["WFC", "GS"],
    "Utilities": ["DUK"],
}

# Referencias ya documentadas en el README, sobre el universo original
ORIGINAL_SINGLE_WINDOW_SHARPE = 0.97
ORIGINAL_WALK_FORWARD_SHARPE = 0.69


def main():
    client = AlpacaClient()
    tickers = [t for group in UNIVERSE.values() for t in group]

    print("=" * 70)
    print("Prueba fuera de universo — momentum cruzado, mismos parámetros, tickers distintos")
    print("=" * 70)
    print(f"Cesta: {len(tickers)} tickers en {len(UNIVERSE)} sectores. Ventana: ~{LOOKBACK_DAYS // 365} años.")
    print("Estrategia: formación 12m-1m, top 5 equal-weight, rebalance mensual, sin cortos (sin tocar).")

    bars_by_symbol = {}
    for symbol in tickers:
        try:
            bars_by_symbol[symbol] = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
        except Exception as e:
            print(f"  AVISO: no se pudo traer historial de {symbol}: {e}")

    result = run_cross_sectional_momentum_backtest(bars_by_symbol)

    print(f"\nPeríodo: {result.start_date} -> {result.end_date} ({result.num_days} días, "
          f"{result.num_rebalances} rebalanceos)")
    print(f"\nRetorno de la estrategia: {result.strategy_total_return:+.1%}")
    print(f"Retorno de la cesta (buy & hold, equal-weight): {result.buy_and_hold_return:+.1%}")
    print(f"Máximo drawdown:    {result.strategy_max_drawdown:.1%}")
    print(f"Sharpe / Sortino:   {result.sharpe_ratio:.2f} / {result.sortino_ratio:.2f}")
    print(f"Calmar:             {result.calmar_ratio:.2f}")
    print(f"Win rate:           {result.win_rate:.1%}")
    print(f"Profit factor:      {result.profit_factor:.2f}")

    beats_bh = "SÍ" if result.strategy_total_return > result.buy_and_hold_return else "no"
    print(f"\n¿Le gana a buy & hold de ESTA cesta?  {beats_bh}")
    print(f"Sharpe universo original (ventana única): {ORIGINAL_SINGLE_WINDOW_SHARPE:.2f}")
    print(f"Sharpe universo original (walk-forward OOS): {ORIGINAL_WALK_FORWARD_SHARPE:.2f}")
    print(f"Sharpe universo nuevo (ventana única):    {result.sharpe_ratio:.2f}")

    print("\nÚltimos 3 rebalanceos:")
    for entry in result.rebalance_log[-3:]:
        print(f"  {entry['date']}: {', '.join(entry['selected'])}")


if __name__ == "__main__":
    main()
