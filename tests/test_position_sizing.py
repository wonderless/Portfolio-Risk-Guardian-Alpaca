"""
Tests de `core/position_sizing.py` (mejora #4: portar el sizing por ATR
del backtest de Fase 5 al orquestador real) y de
`TechnicalAgent.calculate_position_size`, que lo usa para dimensionar
órdenes reales de paper trading en vez de un tamaño fijo.
"""

import numpy as np
import pandas as pd
import pytest

from agents.technical_agent import TechnicalAgent
from core.position_sizing import calculate_atr, volatility_scaled_weight


class _FakeAlpacaClient:
    """Devuelve barras sintéticas fijas, sin llamar a la API real de Alpaca."""

    def __init__(self, bars: pd.DataFrame):
        self._bars = bars

    def get_recent_bars(self, symbol: str, lookback_days: int = 60) -> pd.DataFrame:
        return self._bars


def _flat_bars(n: int = 60, price: float = 100.0, daily_range_pct: float = 0.02) -> pd.DataFrame:
    """Precio constante con un rango intradía fijo — ATR converge a un valor conocido: price * daily_range_pct."""
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series([price] * n)
    high = close * (1 + daily_range_pct / 2)
    low = close * (1 - daily_range_pct / 2)
    return pd.DataFrame({"close": close.values, "high": high.values, "low": low.values}, index=dates)


def test_calculate_atr_converges_to_known_daily_range():
    bars = _flat_bars(n=60, price=100.0, daily_range_pct=0.02)
    atr = calculate_atr(bars["high"], bars["low"], bars["close"], period=14)
    # con precio y rango constantes, el ATR converge a exactamente el rango diario (2.0)
    assert atr.iloc[-1] == pytest.approx(2.0, rel=1e-6)


def test_calculate_atr_is_nan_before_warmup():
    bars = _flat_bars(n=10)
    atr = calculate_atr(bars["high"], bars["low"], bars["close"], period=14)
    assert atr.isna().all()  # no hay suficientes datos para completar ni un período


@pytest.mark.parametrize("atr_value,price,expected_weight", [
    (1.0, 100.0, 1.0),     # vol_pct=1% == target_daily_vol=1% -> weight=1.0
    (2.0, 100.0, 0.5),     # vol_pct=2% -> la mitad de peso que el caso anterior
    (0.1, 100.0, 3.0),     # vol_pct=0.1% pediría weight=10, pero max_weight=3.0 lo topa
])
def test_volatility_scaled_weight_matches_hand_computed_values(atr_value, price, expected_weight):
    weight = volatility_scaled_weight(atr_value, price, target_daily_vol=0.01, max_weight=3.0)
    assert weight == pytest.approx(expected_weight)


def test_volatility_scaled_weight_is_zero_for_invalid_inputs():
    assert volatility_scaled_weight(0.0, 100.0) == 0.0
    assert volatility_scaled_weight(1.0, 0.0) == 0.0
    assert volatility_scaled_weight(-1.0, 100.0) == 0.0


def test_technical_agent_position_size_scales_with_equity():
    bars = _flat_bars(n=60, price=100.0, daily_range_pct=0.04)  # vol_pct=4% -> weight=0.25 (bajo el cap de concentración)
    agent = TechnicalAgent(alpaca_client=_FakeAlpacaClient(bars))

    qty_small = agent.calculate_position_size("SYN", account_equity=10_000)
    qty_large = agent.calculate_position_size("SYN", account_equity=20_000)

    # notional = weight * equity = 0.25 * equity; qty = notional / price
    assert qty_small == pytest.approx(0.25 * 10_000 / 100.0)
    assert qty_large == pytest.approx(2 * qty_small)


def test_technical_agent_position_size_is_capped_by_max_position_pct():
    # vol_pct=2% -> weight "crudo" de 0.5 (50% de la cuenta), pero el cap de
    # concentración (default 0.25, igual que RiskAgent.max_position_pct) debe ganar.
    bars = _flat_bars(n=60, price=100.0, daily_range_pct=0.02)
    agent = TechnicalAgent(alpaca_client=_FakeAlpacaClient(bars))

    qty = agent.calculate_position_size("SYN", account_equity=10_000)

    assert qty == pytest.approx(0.25 * 10_000 / 100.0)  # topado en 25%, no el 50% "crudo"


def test_technical_agent_position_size_is_zero_when_price_is_flat_and_range_is_zero():
    bars = _flat_bars(n=60, price=100.0, daily_range_pct=0.0)  # ATR=0 -> sin volatilidad medible
    agent = TechnicalAgent(alpaca_client=_FakeAlpacaClient(bars))

    assert agent.calculate_position_size("SYN", account_equity=10_000) == 0.0
