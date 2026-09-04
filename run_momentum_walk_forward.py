"""
Validación walk-forward del momentum cruzado (mismo tipo de rigor que
`run_walk_forward.py` aplicó al umbral de ADX del Agente Técnico).

El resultado de `run_momentum_backtest.py` (Sharpe 0.97 sobre una sola
ventana de ~5 años, ver README) es la señal más fuerte que ha producido
cualquier variante probada en este proyecto — pero se calculó sobre una
sola ventana, igual que el ADX=25 original antes de exponerse como
sobreajuste. Este script reoptimiza `top_n` (cuántos tickers sostener)
cada `test_days` usando SOLO el pasado (`train_days` inmediatamente
anteriores), y evalúa fuera de muestra en el período siguiente que la
optimización nunca vio — repitiendo esto en ventanas rodantes.

El resultado a mirar es `sharpe_ratio` / `combined_test_total_return`: es
la curva que se hubiera obtenido operando en tiempo real, sin usar nunca
datos futuros para elegir `top_n`. Si es mucho peor que el backtest de una
sola ventana, es evidencia de que ese Sharpe de 0.97 también estaba
inflado por mirar todo el período de una vez.

Uso:
    python run_momentum_walk_forward.py
"""

from core.alpaca_client import AlpacaClient
from core.momentum_backtester import run_cross_sectional_momentum_backtest, run_momentum_walk_forward_backtest

LOOKBACK_DAYS = 1825  # ~5 años, igual que run_momentum_backtest.py
TRAIN_DAYS = 756       # ~3 años de entrenamiento (más largo que en ADX: momentum necesita ~273d de calentamiento)
TEST_DAYS = 126        # ~6 meses de test fuera de muestra por fold

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
    print("Walk-forward de momentum cruzado (train 3 años -> test 6 meses, rodante)")
    print("=" * 70)
    print(f"Cesta: {len(tickers)} tickers. Grid de top_n: 2-8. lookback=12m, skip=1m (fijos, convención académica).")

    bars_by_symbol = {}
    for symbol in tickers:
        try:
            bars_by_symbol[symbol] = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
        except Exception as e:
            print(f"  AVISO: no se pudo traer historial de {symbol}: {e}")

    wf = run_momentum_walk_forward_backtest(bars_by_symbol, train_days=TRAIN_DAYS, test_days=TEST_DAYS)

    print(f"\n{'Fold':>4} {'Train':>23} {'top_n*':>7} {'TrainSharpe':>12}   "
          f"{'Test':>23} {'TestSharpe':>11} {'TestRet':>9}")
    for f in wf.folds:
        print(f"{f.fold_number:>4} {f.train_start} -> {f.train_end} {f.chosen_top_n:>7} "
              f"{f.train_sharpe:>12.2f}   {f.test_start} -> {f.test_end} "
              f"{f.test_sharpe:>11.2f} {f.test_return:>+9.1%}")

    fixed = run_cross_sectional_momentum_backtest(bars_by_symbol)

    print(f"\nWalk-forward combinado (fuera de muestra, reoptimizado cada {TEST_DAYS}d):")
    print(f"  Retorno: {wf.combined_test_total_return:+.1%}  "
          f"buy&hold: {wf.combined_buy_and_hold_return:+.1%}  MaxDD: {wf.combined_max_drawdown:.1%}")
    print(f"  Sharpe/Sortino/Calmar: {wf.sharpe_ratio:.2f} / {wf.sortino_ratio:.2f} / {wf.calmar_ratio:.2f}")
    print(f"  Win rate: {wf.win_rate:.1%}  Profit factor: {wf.profit_factor:.2f}")

    print(f"\nComparación — top_n=5 fijo sobre todo el período "
          f"(referencia, resultado de run_momentum_backtest.py):")
    print(f"  Retorno: {fixed.strategy_total_return:+.1%}  Sharpe: {fixed.sharpe_ratio:.2f}")


if __name__ == "__main__":
    main()
