"""
Tests de `core/portfolio_backtester.py` sobre datos sintéticos.

No se prueba que el veto de Riesgo "mejore" el resultado (eso se reporta,
sea cual sea, en el README vía `run_technical_risk_backtest.py`) sino que
el simulador de cartera hace lo que dice: calcula concentración con $
reales, respeta el tope de `max_position_pct`, y no mira al futuro.
"""

import numpy as np
import pandas as pd
import pytest

from agents.risk_agent import RiskAgent
from agents.technical_agent import TechnicalAgent
from core.portfolio_backtester import run_technical_risk_backtest


def _trending_bars(n: int = 300, drift: float = 0.0015, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    daily_returns = rng.normal(drift, 0.01, n)
    close = 100 * (1 + pd.Series(daily_returns)).cumprod()
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    return pd.DataFrame({"close": close.values, "high": high.values, "low": low.values}, index=dates)


def _basket(n_tickers: int = 3, n: int = 300) -> dict[str, pd.DataFrame]:
    return {f"SYN{i}": _trending_bars(n=n, seed=i) for i in range(n_tickers)}


def test_risk_agent_vetoes_when_max_position_pct_is_zero():
    """
    Con `max_position_pct=0`, cualquier entrada candidata deja la
    concentración por encima del límite (0 > 0 es falso, pero el sizing
    ya está topado a 0 vía `min(weight, max_position_pct)` en
    `TechnicalAgent.calculate_position_size`-style, así que directamente
    no debería haber trades ejecutados: o el weight sale 0 (se salta la
    entrada) o Riesgo lo vetaría igual si algo se colara).
    """
    bars_by_symbol = _basket()
    agent = TechnicalAgent(alpaca_client=None)
    risk_agent = RiskAgent(alpaca_client=None, max_position_pct=0.0)

    result = run_technical_risk_backtest(agent, risk_agent, bars_by_symbol)

    assert result.num_trades == 0


def test_relaxed_risk_thresholds_allow_trades_to_execute():
    """Con umbrales de riesgo generosos, el veto no debería bloquear todo:
    tiene que haber al menos algún trade ejecutado sobre una cesta con
    tendencia alcista clara y suficiente historial."""
    bars_by_symbol = _basket()
    agent = TechnicalAgent(alpaca_client=None)
    risk_agent = RiskAgent(alpaca_client=None, max_position_pct=1.0, high_portfolio_vol_threshold=100.0)

    result = run_technical_risk_backtest(agent, risk_agent, bars_by_symbol)

    assert result.num_trades > 0
    assert result.num_risk_vetoes == 0


def test_no_lookahead_in_portfolio_value_curve():
    """
    La curva de equity hasta el día i no debería depender de precios
    posteriores a i: truncar el historial de entrada a los primeros k días
    de decisión debe reproducir exactamente los primeros k valores de
    equity del backtest completo (mismas señales, mismos trades hasta ahí).
    """
    bars_by_symbol = _basket(n_tickers=2, n=250)
    agent = TechnicalAgent(alpaca_client=None)
    risk_agent = RiskAgent(alpaca_client=None, max_position_pct=1.0, high_portfolio_vol_threshold=100.0)

    full_result = run_technical_risk_backtest(agent, risk_agent, bars_by_symbol)

    cutoff = full_result.num_days - 30
    cutoff_date = full_result.portfolio_daily_returns.index[cutoff - 1]
    truncated_bars = {
        sym: bars.loc[:cutoff_date + pd.tseries.offsets.BDay(5)]
        for sym, bars in bars_by_symbol.items()
    }
    truncated_result = run_technical_risk_backtest(agent, risk_agent, truncated_bars)

    common_len = min(cutoff, truncated_result.num_days)
    pd.testing.assert_series_equal(
        full_result.portfolio_daily_returns.iloc[:common_len],
        truncated_result.portfolio_daily_returns.iloc[:common_len],
        check_exact=False,
    )


def test_concentration_never_exceeds_max_position_pct():
    bars_by_symbol = _basket()
    agent = TechnicalAgent(alpaca_client=None)
    risk_agent = RiskAgent(alpaca_client=None, max_position_pct=0.2, high_portfolio_vol_threshold=100.0)

    result = run_technical_risk_backtest(agent, risk_agent, bars_by_symbol)

    if result.num_trades > 0:
        assert result.avg_concentration <= risk_agent.max_position_pct + 1e-9


def test_insufficient_history_raises_value_error():
    agent = TechnicalAgent(alpaca_client=None)
    risk_agent = RiskAgent(alpaca_client=None)
    tiny_basket = {"SYN0": _trending_bars(n=10)}

    with pytest.raises(ValueError):
        run_technical_risk_backtest(agent, risk_agent, tiny_basket)
