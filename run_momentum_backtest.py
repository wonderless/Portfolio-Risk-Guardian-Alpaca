"""
Backtest de momentum cruzado (exploración — ver README, "Por qué el Agente
Técnico no muestra una ventaja generalizable"). Corre sobre el mismo
universo de 16 tickers de `run_universe_test.py`/`run_breadth_backtest.py`
(comparación directa), y compara contra:
  (a) buy & hold de la misma cesta,
  (b) la referencia ya documentada de breadth+ATR (Sharpe 0.49, ver README).

Uso:
    python run_momentum_backtest.py
"""

from core.alpaca_client import AlpacaClient
from core.momentum_backtester import run_cross_sectional_momentum_backtest

LOOKBACK_DAYS = 1825  # ~5 años, igual que run_universe_test.py

UNIVERSE = {
    "Tecnología": ["AAPL", "MSFT"],
    "Consumo discrecional": ["TSLA"],
    "Consumo defensivo": ["KO", "PG", "WMT"],
    "Energía": ["XOM", "CVX"],
    "Salud": ["JNJ", "UNH"],
    "Industrial": ["CAT", "HON"],
    "Financiero": ["JPM", "BAC"],
    "Utilities": ["NEE"],
    "Comunicaciones": ["DIS"],
}

BREADTH_ATR_SHARPE_REFERENCE = 0.49  # ver README, "Exploración de breadth..."


def main():
    client = AlpacaClient()
    tickers = [t for group in UNIVERSE.values() for t in group]

    print("=" * 70)
    print("Backtest de momentum cruzado (cross-sectional)")
    print("=" * 70)
    print(f"Cesta: {len(tickers)} tickers en {len(UNIVERSE)} sectores. Ventana: ~{LOOKBACK_DAYS // 365} años.")
    print("Formación: 12 meses menos el último mes (Jegadeesh-Titman). Top 5, equal-weight, rebalance mensual.")

    bars_by_symbol = {}
    for symbol in tickers:
        try:
            bars_by_symbol[symbol] = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
        except Exception as e:
            print(f"  AVISO: no se pudo traer historial de {symbol}: {e}")

    result = run_cross_sectional_momentum_backtest(bars_by_symbol)

    print(f"\nPeríodo: {result.start_date} -> {result.end_date} ({result.num_days} días, "
          f"{result.num_rebalances} rebalanceos)")
    print(f"\nRetorno de la estrategia (momentum cruzado): {result.strategy_total_return:+.1%}")
    print(f"Retorno de la cesta (buy & hold, equal-weight): {result.buy_and_hold_return:+.1%}")
    print(f"Máximo drawdown:    {result.strategy_max_drawdown:.1%}")
    print(f"Sharpe / Sortino:   {result.sharpe_ratio:.2f} / {result.sortino_ratio:.2f}")
    print(f"Calmar:             {result.calmar_ratio:.2f}")
    print(f"Win rate:           {result.win_rate:.1%}")
    print(f"Profit factor:      {result.profit_factor:.2f}")

    beats_bh = "SÍ" if result.strategy_total_return > result.buy_and_hold_return else "no"
    beats_breadth_atr = "SÍ" if result.sharpe_ratio > BREADTH_ATR_SHARPE_REFERENCE else "no"
    print(f"\n¿Le gana a buy & hold?             {beats_bh}")
    print(f"¿Sharpe > referencia breadth+ATR ({BREADTH_ATR_SHARPE_REFERENCE})? {beats_breadth_atr}")

    print("\nÚltimos 3 rebalanceos:")
    for entry in result.rebalance_log[-3:]:
        print(f"  {entry['date']}: {', '.join(entry['selected'])}")


if __name__ == "__main__":
    main()
