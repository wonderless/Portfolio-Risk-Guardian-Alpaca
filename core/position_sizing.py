"""
Position sizing por volatilidad (ATR) — compartido entre el backtester
(`core/backtester.py::run_technical_backtest_atr`, validado en la Fase 5)
y el orquestador real (`agents/orchestrator.py`), para que el paper
trading en vivo use exactamente la misma fórmula que ya se probó en el
backtest, no una reinventada aparte.

Ver README, sección "Exploración de breadth: cesta diversificada +
gestión de riesgo (ATR)" para el resultado de validación: esta fórmula
mejora el Sharpe y reduce el drawdown de forma medible, pero no convierte
una señal débil en una fuerte — sigue siendo gestión de riesgo, no una
fuente de ventaja por sí sola.
"""

import pandas as pd


def calculate_atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
    """ATR de Wilder — rango verdadero promedio, causal (cada punto solo usa datos pasados)."""
    prev_close = closes.shift(1)
    true_range = pd.concat([
        highs - lows,
        (highs - prev_close).abs(),
        (lows - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def volatility_scaled_weight(atr_value: float, price: float, target_daily_vol: float = 0.01,
                              max_weight: float = 3.0) -> float:
    """
    Fracción del portafolio a asignar a un trade, inversamente
    proporcional a su volatilidad (ATR/precio) — un activo tan volátil
    como TSLA recibe menos peso que uno tranquilo como KO para arriesgar
    aproximadamente lo mismo por trade (`target_daily_vol`). `max_weight`
    evita apalancarse de más en activos anormalmente tranquilos.
    """
    if price <= 0 or atr_value <= 0:
        return 0.0
    vol_pct = atr_value / price
    return min(target_daily_vol / vol_pct, max_weight)
