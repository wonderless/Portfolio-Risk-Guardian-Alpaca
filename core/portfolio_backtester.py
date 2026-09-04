"""
Backtest del consenso Técnico+Riesgo, con una cartera simulada real
($cash, posiciones, valor de mercado día a día) — lo que faltaba para
poder correr `RiskAgent._decide` sobre datos históricos.

`core/backtester.py` (Fase 5-6) trabaja en espacio de retornos/pesos
(`weight` como fracción del portafolio), nunca materializa una cartera con
$ reales, así que nunca pudo calcular `concentration_pct` (posición / valor
total de cartera) — la métrica de la que depende toda la lógica del Agente
de Riesgo (`agents/risk_agent.py`). Este módulo cierra ese hueco.

Reusa, sin duplicar:
- `TechnicalAgent._decide` (vía `_generate_signals` de `core/backtester.py`)
  para las señales día a día.
- `RiskAgent._decide` (ya es una función pura, no toca `self.client`) para
  el veto de riesgo, con los mismos umbrales que usa en real.
- `volatility_scaled_weight` (`core/position_sizing.py`) para el tamaño de
  cada entrada — la misma fórmula validada en la Fase 5 y portada al
  orquestador real.
- Las métricas (`_sharpe_ratio`, `_sortino_ratio`, etc.) de
  `core/backtester.py`.

Regla de consenso con 2 agentes presentes (Sentimiento sigue bloqueado por
el límite de historial de NewsAPI — ver README): `RiskAgent._decide` nunca
emite BUY por diseño (ver `agents/risk_agent.py` — su única señal
afirmativa es "sin problema", un HOLD de baja confianza), así que la regla
de veto es: la entrada se ejecuta salvo que Riesgo marque un problema real
(SELL por sobre-concentración, o HOLD de alta confianza por volatilidad de
cartera alta). Mismo espíritu que "sin mayoría clara, HOLD por cautela" del
orquestador real (`agents/orchestrator.py::consensus_node`), adaptado a que
aquí solo hay 2 votantes con datos reales disponibles y uno de ellos (Riesgo)
estructuralmente nunca puede aportar un voto positivo, solo un freno.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agents.risk_agent import RiskAgent
from agents.technical_agent import TechnicalAgent
from core.backtester import (
    _calmar_ratio,
    _generate_signals,
    _profit_factor,
    _sharpe_ratio,
    _sortino_ratio,
    _win_rate,
)
from core.position_sizing import calculate_atr, volatility_scaled_weight
from core.schemas import Signal

VOL_LOOKBACK_DAYS = 20


@dataclass
class PortfolioBacktestResult:
    tickers: list[str]
    start_date: str
    end_date: str
    num_days: int
    portfolio_total_return: float
    portfolio_max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    num_risk_vetoes: int
    avg_concentration: float
    portfolio_daily_returns: pd.Series = field(repr=False)


def run_technical_risk_backtest(agent: TechnicalAgent, risk_agent: RiskAgent,
                                 bars_by_symbol: dict[str, pd.DataFrame],
                                 initial_capital: float = 100_000.0, atr_period: int = 14,
                                 target_daily_vol: float = 0.01, max_weight: float = 3.0,
                                 commission_bps: float = 0.0, slippage_bps: float = 0.0) -> PortfolioBacktestResult:
    """
    Simula, día por día y con una sola cartera compartida entre todos los
    tickers, lo que haría el sistema real si el Agente de Riesgo pudiera
    vetar entradas usando la cartera simulada hasta ese momento (sin mirar
    al futuro en ningún cálculo: señales, ATR y volatilidad de cartera
    usan solo datos hasta el día de la decisión).

    Cada día, para cada símbolo con señal BUY del Técnico y sin posición
    abierta: se calcula el tamaño candidato (ATR-scaled, igual que
    `TechnicalAgent.calculate_position_size` en real), la concentración
    resultante y la volatilidad de cartera (promedio ponderado por valor de
    mercado de la volatilidad rolling de cada posición ya abierta, causal),
    y se llama a `RiskAgent._decide`. Solo se ejecuta si Riesgo no marcó un
    problema real (SELL por sobre-concentración, o HOLD de alta confianza
    por volatilidad de cartera alta) — ver el módulo para el detalle de por
    qué "ambos votan BUY" no es la regla correcta aquí.

    Señal SELL del Técnico con posición abierta: cierra la posición
    completa (sin cortos, mismo criterio que `execute_node` del
    orquestador real).
    """
    warmup = max(agent.sma_slow, agent.rsi_period, agent.adx_period, atr_period, VOL_LOOKBACK_DAYS) + 1

    per_symbol_signals: dict[str, pd.Series] = {}
    per_symbol_atr: dict[str, pd.Series] = {}
    per_symbol_returns: dict[str, pd.Series] = {}

    for symbol, bars in bars_by_symbol.items():
        closes, highs, lows = bars["close"], bars["high"], bars["low"]
        if len(closes) <= warmup + 1:
            continue  # historial insuficiente para este ticker, se omite (igual que run_basket_backtest)
        decision_dates, signals, _confidences = _generate_signals(agent, closes, highs, lows, warmup)
        per_symbol_signals[symbol] = pd.Series(signals, index=decision_dates)
        per_symbol_atr[symbol] = calculate_atr(highs, lows, closes, atr_period)
        per_symbol_returns[symbol] = closes.pct_change()

    if not per_symbol_signals:
        raise ValueError("Ningún ticker tuvo suficiente historial para el backtest de cartera.")

    common_dates = sorted(set.intersection(*[set(s.index) for s in per_symbol_signals.values()]))
    if not common_dates:
        raise ValueError("Los tickers no comparten fechas de decisión en común.")

    cash = initial_capital
    positions: dict[str, dict] = {}  # symbol -> {"qty": float}
    daily_portfolio_value: list[float] = []
    concentrations: list[float] = []
    num_trades = 0
    num_risk_vetoes = 0

    for date in common_dates:
        # Volatilidad de cartera: promedio ponderado por valor de mercado de
        # la volatilidad rolling (causal) de cada posición ya abierta.
        portfolio_value_before = cash + sum(
            positions[sym]["qty"] * float(bars_by_symbol[sym]["close"].loc[date])
            for sym in positions
        )

        def portfolio_vol_with(candidate_symbol: str | None, candidate_value: float) -> float:
            total = portfolio_value_before + (candidate_value if candidate_symbol else 0.0)
            if total <= 0:
                return 0.0
            weighted = 0.0
            held = dict(positions)
            for sym, pos in held.items():
                value = pos["qty"] * float(bars_by_symbol[sym]["close"].loc[date])
                returns = per_symbol_returns[sym].loc[:date].tail(VOL_LOOKBACK_DAYS)
                vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 and not pd.isna(returns.std()) else 0.0
                weighted += (value / total) * vol
            if candidate_symbol:
                returns = per_symbol_returns[candidate_symbol].loc[:date].tail(VOL_LOOKBACK_DAYS)
                vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 and not pd.isna(returns.std()) else 0.0
                weighted += (candidate_value / total) * vol
            return weighted

        cost_rate = (commission_bps + slippage_bps) / 10000

        # --- Salidas: señal SELL con posición abierta se cierra completa ---
        for symbol in list(positions.keys()):
            signal = per_symbol_signals[symbol].get(date)
            if signal == Signal.SELL:
                price = float(bars_by_symbol[symbol]["close"].loc[date])
                proceeds = positions[symbol]["qty"] * price
                cash += proceeds * (1 - cost_rate)
                del positions[symbol]

        # --- Entradas: señal BUY del Técnico, sin posición, no vetada por Riesgo ---
        for symbol, signals in per_symbol_signals.items():
            signal = signals.get(date)
            if signal != Signal.BUY or symbol in positions:
                continue
            price = float(bars_by_symbol[symbol]["close"].loc[date])
            atr_today = per_symbol_atr[symbol].loc[:date].iloc[-1] if date in per_symbol_atr[symbol].index else np.nan
            atr_today = float(atr_today) if not pd.isna(atr_today) else 0.0
            if atr_today <= 0 or price <= 0:
                continue

            current_value = cash + sum(
                positions[s]["qty"] * float(bars_by_symbol[s]["close"].loc[date]) for s in positions
            )
            weight = volatility_scaled_weight(atr_today, price, target_daily_vol, max_weight)
            weight = min(weight, risk_agent.max_position_pct)
            if weight <= 0:
                continue
            candidate_value = weight * current_value
            if candidate_value > cash:
                continue  # sin efectivo suficiente para esta entrada

            concentration_pct = candidate_value / (current_value + candidate_value) if current_value + candidate_value > 0 else 0.0
            portfolio_vol = portfolio_vol_with(symbol, candidate_value)

            risk_signal, risk_confidence, _reasoning = risk_agent._decide(
                symbol=symbol, concentration_pct=concentration_pct,
                portfolio_vol=portfolio_vol, num_positions=len(positions),
            )
            # RiskAgent._decide nunca emite BUY (ver agents/risk_agent.py): su
            # única señal afirmativa es "sin problema" (HOLD, confidence=0.3).
            # El veto real es cuando SÍ marca un problema: SELL por
            # sobre-concentración, o HOLD de alta confianza por volatilidad
            # alta (las dos ramas "con reasons" de _decide, ambas >=0.6).
            is_risk_flag = risk_signal == Signal.SELL or (risk_signal == Signal.HOLD and risk_confidence >= 0.5)
            if is_risk_flag:
                num_risk_vetoes += 1
                continue

            qty = candidate_value / price
            cash -= candidate_value * (1 + cost_rate)
            positions[symbol] = {"qty": qty}
            num_trades += 1
            concentrations.append(concentration_pct)

        portfolio_value_after = cash + sum(
            positions[sym]["qty"] * float(bars_by_symbol[sym]["close"].loc[date]) for sym in positions
        )
        daily_portfolio_value.append(portfolio_value_after)

    equity = pd.Series(daily_portfolio_value, index=common_dates)
    portfolio_returns = equity.pct_change().fillna(0.0)
    portfolio_returns.iloc[0] = equity.iloc[0] / initial_capital - 1

    total_return = float(equity.iloc[-1] / initial_capital - 1) if len(equity) else 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    return PortfolioBacktestResult(
        tickers=list(per_symbol_signals.keys()),
        start_date=str(common_dates[0].date()),
        end_date=str(common_dates[-1].date()),
        num_days=len(common_dates),
        portfolio_total_return=total_return,
        portfolio_max_drawdown=max_drawdown,
        sharpe_ratio=_sharpe_ratio(portfolio_returns),
        sortino_ratio=_sortino_ratio(portfolio_returns),
        calmar_ratio=_calmar_ratio(portfolio_returns, max_drawdown),
        win_rate=_win_rate(portfolio_returns),
        profit_factor=_profit_factor(portfolio_returns),
        num_trades=num_trades,
        num_risk_vetoes=num_risk_vetoes,
        avg_concentration=float(np.mean(concentrations)) if concentrations else 0.0,
        portfolio_daily_returns=portfolio_returns,
    )
