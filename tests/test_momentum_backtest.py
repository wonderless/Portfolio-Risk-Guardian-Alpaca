"""
Tests de `core/momentum_backtester.py` sobre datos sintéticos con
resultado conocido a mano. No se prueba "el momentum cruzado gana dinero"
(eso se reporta, sea cual sea, en el README vía `run_momentum_backtest.py`)
sino que el ranking, el skip-month y la ausencia de look-ahead funcionan
como se documentan.
"""

import numpy as np
import pandas as pd
import pytest

from core.momentum_backtester import _formation_return, run_cross_sectional_momentum_backtest


def _flat_bars(n: int, start_date: str = "2020-01-02") -> pd.Series:
    dates = pd.bdate_range(start_date, periods=n)
    return pd.Series(100.0, index=dates)


def _monotonic_bars(n: int, daily_return: float, start_date: str = "2020-01-02") -> pd.Series:
    dates = pd.bdate_range(start_date, periods=n)
    close = 100 * (1 + daily_return) ** np.arange(n)
    return pd.Series(close, index=dates)


def _to_ohlc(closes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"close": closes, "high": closes * 1.001, "low": closes * 0.999}, index=closes.index)


def test_formation_return_excludes_skip_window():
    """
    Serie plana durante el período de formación, pero con un salto grande
    justo en el mes de skip: el retorno de formación NO debe reflejar ese
    salto (es lo que el skip-month está diseñado a evitar).
    """
    n = 300
    dates = pd.bdate_range("2020-01-02", periods=n)
    closes = pd.Series(100.0, index=dates)
    # Salto del 50% dentro de la ventana de skip (últimos 21 días antes de as_of)
    as_of_idx = 280
    closes.iloc[as_of_idx - 10:as_of_idx + 1] = 150.0

    score = _formation_return(closes, as_of_idx, lookback_days=252, skip_days=21)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_ranking_selects_the_actual_winner():
    """Con 3 tickers de rendimiento claramente distinto durante la ventana
    de formación, el rebalanceo debe seleccionar el/los de mejor retorno."""
    n = 320
    bars_by_symbol = {
        "WINNER": _to_ohlc(_monotonic_bars(n, 0.004)),
        "LOSER": _to_ohlc(_monotonic_bars(n, -0.003)),
        "FLAT": _to_ohlc(_flat_bars(n)),
    }

    result = run_cross_sectional_momentum_backtest(
        bars_by_symbol, lookback_months=12, skip_months=1, top_n=1, rebalance_freq_days=21,
    )

    assert result.num_rebalances > 0
    first_selection = result.rebalance_log[0]["selected"]
    assert first_selection == ["WINNER"]


def test_no_lookahead_in_rebalance_selection():
    """
    La selección del primer rebalanceo no debe cambiar si se le da al
    backtest datos futuros adicionales más allá de esa fecha (truncar el
    historial después del primer rebalanceo debe reproducir la misma
    primera selección).
    """
    n = 400
    bars_by_symbol = {
        "A": _to_ohlc(_monotonic_bars(n, 0.003)),
        "B": _to_ohlc(_monotonic_bars(n, 0.001)),
        "C": _to_ohlc(_monotonic_bars(n, -0.001)),
    }
    full_result = run_cross_sectional_momentum_backtest(
        bars_by_symbol, lookback_months=12, skip_months=1, top_n=2, rebalance_freq_days=21,
    )
    first_rebalance_date = pd.Timestamp(full_result.rebalance_log[0]["date"])

    truncated_by_symbol = {
        sym: bars.loc[:first_rebalance_date + pd.tseries.offsets.BDay(5)]
        for sym, bars in bars_by_symbol.items()
    }
    truncated_result = run_cross_sectional_momentum_backtest(
        truncated_by_symbol, lookback_months=12, skip_months=1, top_n=2, rebalance_freq_days=21,
    )

    assert truncated_result.rebalance_log[0]["selected"] == full_result.rebalance_log[0]["selected"]


def test_transaction_costs_never_improve_return():
    n = 320
    bars_by_symbol = {
        "A": _to_ohlc(_monotonic_bars(n, 0.003)),
        "B": _to_ohlc(_monotonic_bars(n, 0.001)),
        "C": _to_ohlc(_monotonic_bars(n, -0.001)),
    }

    no_cost = run_cross_sectional_momentum_backtest(bars_by_symbol, top_n=2)
    with_cost = run_cross_sectional_momentum_backtest(bars_by_symbol, top_n=2, commission_bps=5.0, slippage_bps=10.0)

    assert with_cost.strategy_total_return <= no_cost.strategy_total_return


def test_insufficient_history_raises_value_error():
    bars_by_symbol = {
        "A": _to_ohlc(_monotonic_bars(30, 0.001)),
        "B": _to_ohlc(_monotonic_bars(30, 0.002)),
    }
    with pytest.raises(ValueError):
        run_cross_sectional_momentum_backtest(bars_by_symbol)
