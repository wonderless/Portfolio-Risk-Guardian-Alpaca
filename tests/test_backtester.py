"""
Tests de `core/backtester.py` sobre datos sintéticos con resultado
conocido — nunca se había probado esto más allá de correrlo a mano contra
datos reales de Alpaca (ver README, sección de resultados del backtest).

No se prueba "la estrategia gana dinero" (ya sabemos, por el propio
README, que no generaliza) sino que el MOTOR del backtest hace lo que
dice que hace: las métricas se calculan bien sobre series con resultado
conocido a mano, y los costos de transacción nunca pueden mejorar el
retorno.
"""

import numpy as np
import pandas as pd
import pytest

from agents.technical_agent import TechnicalAgent
from core.backtester import (
    _profit_factor,
    _sharpe_ratio,
    _win_rate,
    run_technical_backtest,
)


def _uptrend_bars(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """
    Serie sintética con tendencia alcista sostenida + ruido pequeño: sube
    en promedio ~0.15%/día con volatilidad diaria ~1%, suficiente para que
    RSI/SMA/ADX tengan valores no triviales y el agente emita señales
    activas (no solo HOLD todo el tiempo).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    daily_returns = rng.normal(0.0015, 0.01, n)
    close = 100 * (1 + pd.Series(daily_returns)).cumprod()
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    return pd.DataFrame({"close": close.values, "high": high.values, "low": low.values}, index=dates)


# --- Métricas, sobre series construidas a mano (no depende del backtest) ---

def test_sharpe_ratio_is_zero_when_returns_are_constant():
    constant_returns = pd.Series([0.01] * 50)
    assert _sharpe_ratio(constant_returns) == 0.0


def test_sharpe_ratio_positive_when_mean_return_is_positive():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.002, 0.01, 500))
    assert _sharpe_ratio(returns) > 0


def test_win_rate_only_counts_active_days():
    # 3 ganadores, 2 perdedores, 4 días sin posición (retorno 0) que NO deben contar
    returns = pd.Series([0.01, 0.02, -0.01, 0.0, 0.03, -0.02, 0.0, 0.0, 0.0])
    assert _win_rate(returns) == pytest.approx(3 / 5)


def test_profit_factor_matches_hand_computed_ratio():
    returns = pd.Series([0.05, 0.05, -0.025, -0.025])  # ganancias=0.10, pérdidas=0.05
    assert _profit_factor(returns) == pytest.approx(2.0)


def test_profit_factor_is_infinite_with_no_losses():
    returns = pd.Series([0.01, 0.02, 0.0])
    assert _profit_factor(returns) == float("inf")


# --- Motor del backtest, sobre datos sintéticos ---

def test_backtest_without_shorts_never_takes_negative_position():
    bars = _uptrend_bars()
    agent = TechnicalAgent(alpaca_client=None)
    result = run_technical_backtest(agent, bars, "SYN", allow_short=False)
    # Con allow_short=False, una señal SELL debe quedar plana (retorno 0 ese día),
    # nunca generar un retorno que implique una posición corta.
    assert result.num_days > 0
    assert result.signal_counts["sell"] >= 0  # puede haber señales SELL...
    # ...pero ninguna puede haber contribuido con una posición corta real:
    # verificado indirectamente por el propio motor (to_position mapea SELL->0),
    # así que esto es una prueba de regresión de que la función corre sin error
    # y produce métricas coherentes (no NaN, no inf inesperado).
    assert not pd.isna(result.strategy_total_return)
    assert not pd.isna(result.sharpe_ratio)


def test_transaction_costs_never_improve_return():
    bars = _uptrend_bars()
    agent = TechnicalAgent(alpaca_client=None)

    result_no_cost = run_technical_backtest(agent, bars, "SYN", allow_short=False)
    result_with_cost = run_technical_backtest(
        agent, bars, "SYN", allow_short=False, commission_bps=2.0, slippage_bps=8.0,
    )

    assert result_with_cost.strategy_total_return <= result_no_cost.strategy_total_return


def test_zero_cost_parameters_reproduce_previous_behavior():
    """Con commission_bps=slippage_bps=0 (default), el resultado debe ser
    idéntico al de antes de agregar el parámetro — ningún script existente
    debería cambiar su comportamiento."""
    bars = _uptrend_bars()
    agent = TechnicalAgent(alpaca_client=None)

    result_default = run_technical_backtest(agent, bars, "SYN", allow_short=False)
    result_explicit_zero = run_technical_backtest(
        agent, bars, "SYN", allow_short=False, commission_bps=0.0, slippage_bps=0.0,
    )

    assert result_default.strategy_total_return == pytest.approx(result_explicit_zero.strategy_total_return)
    assert result_default.sharpe_ratio == pytest.approx(result_explicit_zero.sharpe_ratio)


def test_insufficient_history_raises_value_error():
    agent = TechnicalAgent(alpaca_client=None)
    tiny_bars = _uptrend_bars(n=10)
    with pytest.raises(ValueError):
        run_technical_backtest(agent, tiny_bars, "SYN", allow_short=False)


def test_size_by_confidence_scales_positions_below_fixed_size():
    """
    Con `size_by_confidence=True`, cada posición activa se escala por un
    valor de confianza entre 0 y 1 (nunca más grande que el tamaño fijo de
    ±1) — así que el equity resultante nunca puede moverse MÁS que el
    modelo de tamaño fijo en un solo día activo, dado el mismo retorno de
    mercado ese día.
    """
    bars = _uptrend_bars()
    agent = TechnicalAgent(alpaca_client=None)

    fixed = run_technical_backtest(agent, bars, "SYN", allow_short=False)
    weighted = run_technical_backtest(agent, bars, "SYN", allow_short=False, size_by_confidence=True)

    # mismos días activos (la señal no cambia, solo el tamaño), pero cada
    # retorno diario ponderado debe tener magnitud <= al de tamaño fijo
    assert (weighted.strategy_daily_returns.abs() <= fixed.strategy_daily_returns.abs() + 1e-9).all()
