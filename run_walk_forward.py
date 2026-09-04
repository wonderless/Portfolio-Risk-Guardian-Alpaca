"""
Validación walk-forward del umbral de ADX (Fase 5 — pipeline formal de
validación fuera de muestra)

El barrido de sensibilidad (`run_adx_sweep.py`) y la prueba fuera de
universo (`run_universe_test.py`) ya daban indicios de sobreajuste en el
umbral ADX=25 elegido para AAPL. Este script hace la prueba definitiva:
reoptimiza el umbral cada `test_days` días usando SOLO datos pasados
(`train_days` inmediatamente anteriores), y evalúa ese umbral en el
período siguiente que la optimización nunca vio — repitiendo esto en
ventanas rodantes a lo largo de todo el historial.

El resultado a mirar es `combined_test_total_return` / `sharpe_ratio`:
es la curva que se hubiera obtenido operando en tiempo real, jamás usando
datos futuros para elegir el parámetro. Si esos números son mucho peores
que los del backtest de una sola ventana con ADX=25 fijo (Fase 5), es
evidencia adicional (más fuerte que el barrido o la prueba fuera de
universo) de que el resultado original no generaliza.

Uso:
    python run_walk_forward.py                  # AAPL, MSFT, TSLA
    python run_walk_forward.py NVDA GOOGL
"""

import sys

from core.alpaca_client import AlpacaClient
from agents.technical_agent import TechnicalAgent
from core.backtester import run_technical_backtest, run_walk_forward_backtest

LOOKBACK_DAYS = 1825  # ~5 años, para tener varios folds
TRAIN_DAYS = 504      # ~2 años de entrenamiento (mismo largo que el backtest original de Fase 5)
TEST_DAYS = 126       # ~6 meses de test fuera de muestra por fold


def main():
    tickers = sys.argv[1:] or ["AAPL", "MSFT", "TSLA"]

    print("=" * 70)
    print("Walk-forward del umbral de ADX (train 2 años -> test 6 meses, rodante)")
    print("=" * 70)

    client = AlpacaClient()

    for symbol in tickers:
        print(f"\n{'-' * 70}")
        print(symbol)
        try:
            bars = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
            wf = run_walk_forward_backtest(
                bars, symbol, train_days=TRAIN_DAYS, test_days=TEST_DAYS, allow_short=False,
            )

            print(f"  {'Fold':>4} {'Train':>23} {'ADX*':>5} {'TrainSharpe':>12}   "
                  f"{'Test':>23} {'TestSharpe':>11} {'TestRet':>9}")
            for f in wf.folds:
                print(f"  {f.fold_number:>4} {f.train_start} -> {f.train_end} {f.chosen_adx_threshold:>5.0f} "
                      f"{f.train_sharpe:>12.2f}   {f.test_start} -> {f.test_end} "
                      f"{f.test_sharpe:>11.2f} {f.test_return:>+9.1%}")

            default_agent = TechnicalAgent(alpaca_client=client)
            fixed = run_technical_backtest(default_agent, bars, symbol, allow_short=False)

            print(f"\n  Walk-forward combinado (fuera de muestra, reoptimizado cada {TEST_DAYS}d):")
            print(f"    Retorno: {wf.combined_test_total_return:+.1%}  "
                  f"buy&hold: {wf.combined_buy_and_hold_return:+.1%}  "
                  f"MaxDD: {wf.combined_max_drawdown:.1%}")
            print(f"    Sharpe/Sortino/Calmar: {wf.sharpe_ratio:.2f} / {wf.sortino_ratio:.2f} / {wf.calmar_ratio:.2f}")
            print(f"  Comparación — ADX={default_agent.adx_threshold:.0f} fijo sobre todo el período "
                  f"(Fase 5, referencia, con look-ahead de haber elegido el umbral mirando todo el período):")
            print(f"    Retorno: {fixed.strategy_total_return:+.1%}  Sharpe: {fixed.sharpe_ratio:.2f}")
        except Exception as e:
            print(f"  AVISO: no se pudo correr walk-forward para {symbol}: {e}")


if __name__ == "__main__":
    main()
