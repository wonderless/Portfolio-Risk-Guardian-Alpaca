"""
Impacto de costos de transacción (comisión + slippage) en el momentum
cruzado — mismo supuesto conservador que `run_costs_backtest.py` aplicó al
Agente Técnico en Fase 6 (Alpaca no cobra comisión en acciones, pero el
slippage es real incluso en brokers "commission-free").

El momentum cruzado rebalancea mensualmente sobre una cesta de 16 tickers
(más turnover que el Agente Técnico, que solo opera cuando ADX confirma
tendencia), así que el efecto de costos podría ser más grande acá. Este
script corre, con y sin costos, tanto el backtest de ventana única
(`run_momentum_backtest.py`) como el walk-forward
(`run_momentum_walk_forward.py`) — el walk-forward es el que más importa
verificar, porque su Sharpe de 0.69 fuera de muestra es el número que este
proyecto reporta como "creíble" para el momentum cruzado.

Uso:
    python run_momentum_costs_backtest.py
"""

from core.alpaca_client import AlpacaClient
from core.momentum_backtester import run_cross_sectional_momentum_backtest, run_momentum_walk_forward_backtest

LOOKBACK_DAYS = 1825   # ~5 años, igual que run_momentum_backtest.py
TRAIN_DAYS = 756
TEST_DAYS = 126
COMMISSION_BPS = 0.0   # Alpaca no cobra comisión en acciones
SLIPPAGE_BPS = 5.0     # mismo supuesto conservador que run_costs_backtest.py (~0.05% por cambio)

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


def main():
    client = AlpacaClient()
    tickers = [t for group in UNIVERSE.values() for t in group]

    print("=" * 70)
    print("Impacto de costos de transacción en el momentum cruzado")
    print("=" * 70)
    print(f"Supuesto: comisión {COMMISSION_BPS:.0f}bps + slippage {SLIPPAGE_BPS:.0f}bps "
          f"sobre el turnover de cada rebalanceo mensual.")

    bars_by_symbol = {}
    for symbol in tickers:
        try:
            bars_by_symbol[symbol] = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
        except Exception as e:
            print(f"  AVISO: no se pudo traer historial de {symbol}: {e}")

    print("\n--- Ventana única (referencia de run_momentum_backtest.py) ---")
    single_no_cost = run_cross_sectional_momentum_backtest(bars_by_symbol)
    single_with_cost = run_cross_sectional_momentum_backtest(
        bars_by_symbol, commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS,
    )
    print(f"  sin costos: retorno={single_no_cost.strategy_total_return:+7.1%}  "
          f"Sharpe={single_no_cost.sharpe_ratio:5.2f}")
    print(f"  con costos: retorno={single_with_cost.strategy_total_return:+7.1%}  "
          f"Sharpe={single_with_cost.sharpe_ratio:5.2f}")
    drag_single = single_no_cost.strategy_total_return - single_with_cost.strategy_total_return
    print(f"  Costo total absorbido por el retorno: {drag_single:+.2%}")

    print("\n--- Walk-forward (referencia de run_momentum_walk_forward.py) ---")
    wf_no_cost = run_momentum_walk_forward_backtest(bars_by_symbol, train_days=TRAIN_DAYS, test_days=TEST_DAYS)
    wf_with_cost = run_momentum_walk_forward_backtest(
        bars_by_symbol, train_days=TRAIN_DAYS, test_days=TEST_DAYS,
        commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS,
    )
    print(f"  sin costos: retorno OOS={wf_no_cost.combined_test_total_return:+7.1%}  "
          f"Sharpe OOS={wf_no_cost.sharpe_ratio:5.2f}")
    print(f"  con costos: retorno OOS={wf_with_cost.combined_test_total_return:+7.1%}  "
          f"Sharpe OOS={wf_with_cost.sharpe_ratio:5.2f}")
    drag_wf = wf_no_cost.combined_test_total_return - wf_with_cost.combined_test_total_return
    print(f"  Costo total absorbido por el retorno OOS: {drag_wf:+.2%}")

    print(f"\n  top_n elegido por fold (sin costos):  {[f.chosen_top_n for f in wf_no_cost.folds]}")
    print(f"  top_n elegido por fold (con costos):  {[f.chosen_top_n for f in wf_with_cost.folds]}")


if __name__ == "__main__":
    main()
