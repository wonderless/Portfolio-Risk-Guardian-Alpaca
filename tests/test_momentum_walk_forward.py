"""
Tests de `run_momentum_walk_forward_backtest` sobre datos sintéticos. No se
prueba que el walk-forward "confirme o refute" el resultado de la ventana
única (eso se reporta, sea cual sea, en el README vía
`run_momentum_walk_forward.py`) sino que el motor respeta la propiedad que
importa: cada fold de test se evalúa con un `top_n` elegido SOLO con datos
de su ventana de train, nunca con datos del futuro.
"""

import numpy as np
import pandas as pd
import pytest

from core.momentum_backtester import run_momentum_walk_forward_backtest

TRAIN_DAYS = 200
TEST_DAYS = 40
TOP_N_GRID = [1, 2, 3]
LOOKBACK_MONTHS = 3
SKIP_MONTHS = 1


def _monotonic_bars(n: int, daily_return: float, start_date: str = "2020-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start_date, periods=n)
    close = 100 * (1 + daily_return) ** np.arange(n)
    close = pd.Series(close, index=dates)
    return pd.DataFrame({"close": close, "high": close * 1.001, "low": close * 0.999}, index=dates)


def _basket(n: int) -> dict[str, pd.DataFrame]:
    return {
        "A": _monotonic_bars(n, 0.003),
        "B": _monotonic_bars(n, 0.001),
        "C": _monotonic_bars(n, -0.001),
    }


def _run(bars_by_symbol, **kwargs):
    defaults = dict(
        top_n_grid=TOP_N_GRID, lookback_months=LOOKBACK_MONTHS, skip_months=SKIP_MONTHS,
        train_days=TRAIN_DAYS, test_days=TEST_DAYS, min_train_rebalances=1,
    )
    defaults.update(kwargs)
    return run_momentum_walk_forward_backtest(bars_by_symbol, **defaults)


def test_chosen_top_n_always_within_grid():
    bars_by_symbol = _basket(n=500)
    result = _run(bars_by_symbol)

    assert len(result.folds) > 0
    for fold in result.folds:
        assert fold.chosen_top_n in TOP_N_GRID


def test_test_windows_are_sequential_and_non_overlapping():
    bars_by_symbol = _basket(n=500)
    result = _run(bars_by_symbol)

    for prev_fold, fold in zip(result.folds, result.folds[1:]):
        assert pd.Timestamp(fold.train_start) > pd.Timestamp(prev_fold.train_start)
        assert pd.Timestamp(fold.test_start) > pd.Timestamp(prev_fold.test_end)
        # Cada train empieza donde terminaba el train anterior + test_days (rodante, no acumulativo)
        assert pd.Timestamp(fold.test_end) > pd.Timestamp(fold.train_end)


def test_no_lookahead_earlier_folds_unaffected_by_future_data():
    """
    Los primeros folds no deberían cambiar si se agrega más historial al
    final: truncar los datos justo después del segundo fold debe reproducir
    exactamente el primer fold del backtest completo (mismo top_n elegido,
    mismo test_sharpe).
    """
    bars_by_symbol = _basket(n=600)
    full_result = _run(bars_by_symbol)
    assert len(full_result.folds) >= 2

    second_fold_test_end = pd.Timestamp(full_result.folds[1].test_end)
    truncated_by_symbol = {
        sym: bars.loc[:second_fold_test_end + pd.tseries.offsets.BDay(5)]
        for sym, bars in bars_by_symbol.items()
    }
    truncated_result = _run(truncated_by_symbol)

    assert truncated_result.folds[0].chosen_top_n == full_result.folds[0].chosen_top_n
    assert truncated_result.folds[0].test_sharpe == pytest.approx(full_result.folds[0].test_sharpe)
    assert truncated_result.folds[0].test_return == pytest.approx(full_result.folds[0].test_return)


def test_insufficient_history_raises_value_error():
    bars_by_symbol = _basket(n=100)
    with pytest.raises(ValueError):
        _run(bars_by_symbol)


def test_transaction_costs_never_improve_walk_forward_return():
    bars_by_symbol = _basket(n=500)

    no_cost = _run(bars_by_symbol)
    with_cost = _run(bars_by_symbol, commission_bps=5.0, slippage_bps=10.0)

    assert with_cost.combined_test_total_return <= no_cost.combined_test_total_return
