"""
Backtester del Agente Técnico (Fase 5)

Simula, día por día sobre datos históricos, qué señal habría emitido el
Agente Técnico usando solo la información disponible hasta ese día (sin
mirar al futuro), y compara el retorno de seguir esas señales contra
simplemente comprar y mantener (buy & hold), usando métricas estándar de
la industria (Sharpe, Sortino, Calmar, win rate, profit factor) — no
solo el retorno total, que por sí solo no dice si una estrategia es
buena en términos de riesgo asumido.

Alcance deliberado — solo el Agente Técnico:
- El Agente de Sentimiento depende de noticias actuales (NewsAPI gratis
  no cubre más de ~1 mes hacia atrás) y cada llamada a Claude tiene costo
  real. Backtestear años de "sentimiento histórico" no es viable.
- El Agente de Riesgo depende del estado del portafolio simulado a lo
  largo del tiempo; se puede agregar más adelante si hace falta.

Modelo de estrategia (simplificación deliberada):
- Señal BUY  -> posición larga (+1) para el día siguiente.
- Señal SELL -> posición corta (-1) para el día siguiente, salvo que
  `allow_short=False` (entonces queda plano, útil para comparar el
  efecto de permitir cortos contra un mercado alcista).
- Señal HOLD -> sin posición (0).
No hay position sizing en el modelo simple (sí en la variante ATR) — pero
sí se puede modelar comisión y slippage (`commission_bps`, `slippage_bps`)
para no reportar retornos sistemáticamente optimistas. Por defecto ambos
son 0 para no cambiar el comportamiento de scripts existentes.

Validación fuera de muestra (walk-forward): `run_walk_forward_backtest`
optimiza un parámetro (por defecto, el umbral de ADX) en una ventana de
entrenamiento y evalúa el resultado en la ventana de test INMEDIATAMENTE
siguiente, que la optimización nunca vio — repitiendo esto en ventanas
rodantes a lo largo de todo el historial. El barrido de sensibilidad
(`run_adx_sweep.py`) y la prueba fuera de universo (`run_universe_test.py`)
ya daban una señal de sobreajuste; esto es el pipeline formal que faltaba
para confirmarlo con el mismo tipo de rigor que usa la industria.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agents.technical_agent import TechnicalAgent
from core.position_sizing import calculate_atr, volatility_scaled_weight
from core.schemas import Signal

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BasketBacktestResult:
    num_tickers: int
    tickers: list[str]
    start_date: str
    end_date: str
    num_days: int
    portfolio_total_return: float
    basket_buy_and_hold_return: float
    portfolio_max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    portfolio_daily_returns: pd.Series = field(repr=False)


@dataclass
class BacktestResult:
    symbol: str
    start_date: str
    end_date: str
    num_days: int
    signal_counts: dict[str, int]
    strategy_total_return: float
    buy_and_hold_return: float
    strategy_max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    strategy_daily_returns: pd.Series = field(repr=False)
    num_trades: int = 0


def _annualized_return(daily_returns: pd.Series) -> float:
    equity = (1 + daily_returns.fillna(0)).cumprod()
    total_return = equity.iloc[-1] - 1 if len(equity) else 0.0
    years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if years <= 0 or (1 + total_return) <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def _sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Retorno ajustado por volatilidad total. >1.0 se suele considerar sólido."""
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    std = excess.std()
    if not std or pd.isna(std) or std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Como Sharpe, pero solo penaliza la volatilidad a la baja (pérdidas)."""
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    downside = excess.where(excess < 0, 0.0)
    downside_std = downside.std()
    if not downside_std or pd.isna(downside_std) or downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _calmar_ratio(daily_returns: pd.Series, max_drawdown: float) -> float:
    """Retorno anualizado dividido por el peor drawdown. Mide retorno por unidad de peor pérdida."""
    if max_drawdown == 0:
        return 0.0
    return float(_annualized_return(daily_returns) / abs(max_drawdown))


def _win_rate(daily_returns: pd.Series) -> float:
    """% de días con posición abierta (retorno != 0) que resultaron ganadores."""
    active_days = daily_returns[daily_returns != 0]
    if len(active_days) == 0:
        return 0.0
    return float((active_days > 0).sum() / len(active_days))


def _profit_factor(daily_returns: pd.Series) -> float:
    """Ganancias brutas / pérdidas brutas. >1.0 significa que gana más de lo que pierde."""
    gains = daily_returns[daily_returns > 0].sum()
    losses = -daily_returns[daily_returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _generate_signals(agent: TechnicalAgent, closes: pd.Series, highs: pd.Series,
                       lows: pd.Series, warmup: int) -> tuple[pd.DatetimeIndex, list[Signal], list[float]]:
    """
    Recorre `closes`/`highs`/`lows` día por día reutilizando los métodos de
    cálculo del propio TechnicalAgent (RSI, SMAs, ADX, volatilidad,
    decisión) — no se duplica su lógica, solo se le pasa una ventana
    histórica creciente en vez de datos "recientes" de la API. Compartido
    entre `run_technical_backtest` y `run_technical_backtest_atr`.

    También devuelve la `confidence` (0-1) que el propio agente calcula
    para cada día — la usa `run_technical_backtest(size_by_confidence=True)`
    para escalar el tamaño de la posición en vez de usar tamaño fijo.
    """
    signals = []
    confidences = []
    for i in range(warmup, len(closes) - 1):
        closes_so_far = closes.iloc[:i + 1]
        highs_so_far = highs.iloc[:i + 1]
        lows_so_far = lows.iloc[:i + 1]

        rsi = agent._calculate_rsi(closes_so_far)
        sma_fast, sma_slow = agent._calculate_smas(closes_so_far)
        volatility = agent._calculate_volatility(closes_so_far)
        adx = agent._calculate_adx(highs_so_far, lows_so_far, closes_so_far)
        current_price = float(closes_so_far.iloc[-1])

        signal, confidence, _ = agent._decide(
            rsi=rsi, sma_fast=sma_fast, sma_slow=sma_slow,
            current_price=current_price, volatility=volatility, adx=adx,
        )
        signals.append(signal)
        confidences.append(confidence)

    decision_dates = closes.index[warmup:len(closes) - 1]
    return decision_dates, signals, confidences


def run_technical_backtest(agent: TechnicalAgent, bars: pd.DataFrame, symbol: str,
                            allow_short: bool = True, commission_bps: float = 0.0,
                            slippage_bps: float = 0.0, size_by_confidence: bool = False) -> BacktestResult:
    """
    Modelo de posición diaria fija (BUY=largo, SELL=corto u plano, HOLD=plano)
    por defecto. Ver `run_technical_backtest_atr` para la variante con
    dimensionamiento por volatilidad y stops dinámicos (ATR).

    `commission_bps` y `slippage_bps` (puntos básicos, 1bps = 0.01%) se cobran
    sobre el CAMBIO de posición cada día (`|posición hoy - posición ayer|`),
    no sobre la posición en sí — así un día sin cambio de posición no paga
    costo, y una reversión de largo a corto paga el doble (se cierra una
    posición y se abre la contraria). Con ambos en 0 (default) el resultado
    es idéntico al de antes de agregar este parámetro.

    `size_by_confidence`: en vez de tamaño fijo (±1), escala la posición por
    la `confidence` (0-1) que el propio agente calcula para esa señal (ver
    `TechnicalAgent._decide`) — la idea a probar es si los días donde el
    agente está más seguro (RSI y tendencia coinciden) realmente producen
    mejor resultado que los de baja confianza, y si pesarlos así mejora el
    retorno ajustado por riesgo frente al tamaño fijo actual.
    """
    closes = bars["close"]
    highs = bars["high"]
    lows = bars["low"]
    warmup = max(agent.sma_slow, agent.rsi_period, agent.adx_period) + 1

    if len(closes) <= warmup + 1:
        raise ValueError(
            f"No hay suficiente historial para backtestear {symbol}: se necesitan al menos "
            f"{warmup + 1} días y hay {len(closes)}."
        )

    daily_returns = closes.pct_change()
    decision_dates, signals, confidences = _generate_signals(agent, closes, highs, lows, warmup)

    def to_position(signal: Signal, confidence: float) -> float:
        if signal == Signal.BUY:
            direction = 1
        elif signal == Signal.SELL:
            direction = -1 if allow_short else 0
        else:
            direction = 0
        return direction * confidence if size_by_confidence else float(direction)

    positions = pd.Series(
        [to_position(s, c) for s, c in zip(signals, confidences)], index=decision_dates,
    )

    # La posición tomada el día i se gana (o pierde) con el retorno del día i+1
    next_day_returns = daily_returns.shift(-1).loc[decision_dates]
    gross_daily_returns = (positions * next_day_returns).fillna(0)

    cost_rate = (commission_bps + slippage_bps) / 10000
    turnover = (positions - positions.shift(1).fillna(0.0)).abs()
    trading_costs = turnover * cost_rate
    strategy_daily_returns = gross_daily_returns - trading_costs

    strategy_equity = (1 + strategy_daily_returns).cumprod()
    strategy_total_return = float(strategy_equity.iloc[-1] - 1) if len(strategy_equity) else 0.0

    bh_returns = daily_returns.loc[decision_dates].fillna(0)
    bh_equity = (1 + bh_returns).cumprod()
    buy_and_hold_return = float(bh_equity.iloc[-1] - 1) if len(bh_equity) else 0.0

    running_max = strategy_equity.cummax()
    drawdown = (strategy_equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    signal_counts = {s.value: sum(1 for x in signals if x == s) for s in Signal}

    return BacktestResult(
        symbol=symbol,
        start_date=str(decision_dates[0].date()),
        end_date=str(decision_dates[-1].date()),
        num_days=len(signals),
        signal_counts=signal_counts,
        strategy_total_return=strategy_total_return,
        buy_and_hold_return=buy_and_hold_return,
        strategy_max_drawdown=max_drawdown,
        sharpe_ratio=_sharpe_ratio(strategy_daily_returns),
        sortino_ratio=_sortino_ratio(strategy_daily_returns),
        calmar_ratio=_calmar_ratio(strategy_daily_returns, max_drawdown),
        win_rate=_win_rate(strategy_daily_returns),
        profit_factor=_profit_factor(strategy_daily_returns),
        strategy_daily_returns=strategy_daily_returns,
    )


def run_basket_backtest(agent: TechnicalAgent, bars_by_symbol: dict[str, pd.DataFrame],
                         allow_short: bool = False, vol_window: int = 20,
                         commission_bps: float = 0.0, slippage_bps: float = 0.0) -> BasketBacktestResult:
    """
    Aplica la misma señal del Agente Técnico SIMULTÁNEAMENTE sobre una
    cesta de tickers, y evalúa el resultado a nivel de PORTAFOLIO — no
    ticker por ticker. Esto es lo que en la práctica hacen los fondos
    CTA/trend-following: una señal débil en un solo activo se vuelve una
    ventaja estadísticamente visible al combinar muchas apuestas
    independientes ("breadth" — ver README, sección "Por qué el Agente
    Técnico no muestra una ventaja generalizable").

    Cada ticker se pondera de forma inversamente proporcional a su
    volatilidad reciente (rolling, calculada solo con datos pasados — sin
    mirar al futuro), para que un activo ruidoso como TSLA no domine el
    riesgo del portafolio combinado. Sin este ajuste, "combinar muchos
    activos" no se parecería a "muchas apuestas independientes de tamaño
    similar", que es justamente la premisa de la que depende el beneficio
    de breadth.
    """
    per_symbol_returns = {}
    per_symbol_asset_returns = {}

    for symbol, bars in bars_by_symbol.items():
        try:
            result = run_technical_backtest(
                agent, bars, symbol, allow_short=allow_short,
                commission_bps=commission_bps, slippage_bps=slippage_bps,
            )
        except ValueError:
            continue  # historial insuficiente para este ticker, se omite
        per_symbol_returns[symbol] = result.strategy_daily_returns
        per_symbol_asset_returns[symbol] = bars["close"].pct_change().reindex(result.strategy_daily_returns.index)

    if not per_symbol_returns:
        raise ValueError("Ningún ticker de la cesta tuvo suficiente historial para backtestear.")

    returns_df = pd.DataFrame(per_symbol_returns)
    asset_returns_df = pd.DataFrame(per_symbol_asset_returns)

    # Peso inverso a la volatilidad realizada del ACTIVO (no de la estrategia),
    # con shift(1) para no usar el retorno del propio día en el cálculo de su peso.
    rolling_vol = asset_returns_df.rolling(window=vol_window, min_periods=vol_window).std().shift(1)
    inv_vol = 1 / rolling_vol.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    portfolio_returns = (returns_df * weights).sum(axis=1, min_count=1).dropna()
    portfolio_bh_returns = (asset_returns_df * weights).sum(axis=1, min_count=1).dropna()

    common_index = portfolio_returns.index.intersection(portfolio_bh_returns.index)
    portfolio_returns = portfolio_returns.loc[common_index]
    portfolio_bh_returns = portfolio_bh_returns.loc[common_index]

    equity = (1 + portfolio_returns).cumprod()
    total_return = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    bh_equity = (1 + portfolio_bh_returns).cumprod()
    bh_total_return = float(bh_equity.iloc[-1] - 1) if len(bh_equity) else 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    return BasketBacktestResult(
        num_tickers=len(per_symbol_returns),
        tickers=list(per_symbol_returns.keys()),
        start_date=str(common_index[0].date()) if len(common_index) else "",
        end_date=str(common_index[-1].date()) if len(common_index) else "",
        num_days=len(common_index),
        portfolio_total_return=total_return,
        basket_buy_and_hold_return=bh_total_return,
        portfolio_max_drawdown=max_drawdown,
        sharpe_ratio=_sharpe_ratio(portfolio_returns),
        sortino_ratio=_sortino_ratio(portfolio_returns),
        calmar_ratio=_calmar_ratio(portfolio_returns, max_drawdown),
        win_rate=_win_rate(portfolio_returns),
        profit_factor=_profit_factor(portfolio_returns),
        portfolio_daily_returns=portfolio_returns,
    )


def run_technical_backtest_atr(agent: TechnicalAgent, bars: pd.DataFrame, symbol: str,
                                atr_period: int = 14, atr_stop_mult: float = 2.0,
                                target_daily_vol: float = 0.01, max_weight: float = 3.0,
                                commission_bps: float = 0.0, slippage_bps: float = 0.0) -> BacktestResult:
    """
    Variante de `run_technical_backtest` con gestión de riesgo dinámica
    (recomendación de mayor prioridad de la investigación — ver README):

    - Dimensionamiento por volatilidad: el tamaño de cada posición es
      inversamente proporcional a su distancia de stop en % (ATR/precio),
      apuntando a que cada trade arriesgue aproximadamente lo mismo
      (`target_daily_vol`), sin importar si el activo es tan volátil como
      TSLA o tan tranquilo como KO.
    - Stop dinámico basado en ATR: al entrar en una posición BUY, se fija
      un stop a `atr_stop_mult` veces el ATR(14) por debajo del precio de
      entrada (no es un trailing stop — se fija una vez al entrar, para
      mantener la simulación simple).
    - La salida también ocurre si la señal deja de ser BUY (igual que en
      el modelo simple), lo que ocurra primero.

    Solo posiciones largas (no hay cortos, siguiendo el hallazgo de la
    Fase 5). `commission_bps`/`slippage_bps` se cobran sobre `weight` en el
    día de entrada y otra vez en el día de salida (round-trip completo);
    no modela gaps más allá de asumir que el stop se ejecuta exactamente a
    su precio.
    """
    closes = bars["close"]
    highs = bars["high"]
    lows = bars["low"]
    warmup = max(agent.sma_slow, agent.rsi_period, agent.adx_period, atr_period) + 1

    if len(closes) <= warmup + 1:
        raise ValueError(
            f"No hay suficiente historial para backtestear {symbol}: se necesitan al menos "
            f"{warmup + 1} días y hay {len(closes)}."
        )

    decision_dates, signals, _confidences = _generate_signals(agent, closes, highs, lows, warmup)
    atr = calculate_atr(highs, lows, closes, atr_period)
    daily_returns = closes.pct_change()
    cost_rate = (commission_bps + slippage_bps) / 10000

    daily_pnl = pd.Series(0.0, index=decision_dates)
    in_position = False
    entry_price = 0.0
    stop_price = 0.0
    weight = 0.0
    num_trades = 0

    for pos_i, date in enumerate(decision_dates):
        signal = signals[pos_i]
        idx = closes.index.get_loc(date)
        next_date = closes.index[idx + 1]
        price_today = float(closes.loc[date])
        next_close = float(closes.loc[next_date])
        next_low = float(lows.loc[next_date])

        entered_now = False
        if not in_position:
            if signal != Signal.BUY:
                continue
            atr_today = float(atr.loc[date]) if not pd.isna(atr.loc[date]) else 0.0
            if atr_today <= 0:
                continue
            entry_price = price_today
            stop_price = entry_price - atr_stop_mult * atr_today
            weight = volatility_scaled_weight(atr_today, price_today, target_daily_vol, max_weight)
            if weight <= 0:
                continue
            in_position = True
            entered_now = True
            num_trades += 1

        exited_now = False
        if next_low <= stop_price:
            daily_pnl.loc[date] = weight * (stop_price / price_today - 1)
            in_position = False
            exited_now = True
        elif signal != Signal.BUY:
            daily_pnl.loc[date] = weight * (next_close / price_today - 1)
            in_position = False
            exited_now = True
        else:
            daily_pnl.loc[date] = weight * (next_close / price_today - 1)
            # sigue en posición para el próximo día (mismo entry_price/stop_price/weight)

        trade_cost = 0.0
        if entered_now:
            trade_cost += weight * cost_rate
        if exited_now:
            trade_cost += weight * cost_rate
        daily_pnl.loc[date] -= trade_cost

    strategy_equity = (1 + daily_pnl).cumprod()
    strategy_total_return = float(strategy_equity.iloc[-1] - 1) if len(strategy_equity) else 0.0

    bh_returns = daily_returns.loc[decision_dates].fillna(0)
    bh_equity = (1 + bh_returns).cumprod()
    buy_and_hold_return = float(bh_equity.iloc[-1] - 1) if len(bh_equity) else 0.0

    running_max = strategy_equity.cummax()
    drawdown = (strategy_equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    signal_counts = {s.value: sum(1 for x in signals if x == s) for s in Signal}

    return BacktestResult(
        symbol=symbol,
        start_date=str(decision_dates[0].date()),
        end_date=str(decision_dates[-1].date()),
        num_days=len(signals),
        signal_counts=signal_counts,
        strategy_total_return=strategy_total_return,
        buy_and_hold_return=buy_and_hold_return,
        strategy_max_drawdown=max_drawdown,
        sharpe_ratio=_sharpe_ratio(daily_pnl),
        sortino_ratio=_sortino_ratio(daily_pnl),
        calmar_ratio=_calmar_ratio(daily_pnl, max_drawdown),
        win_rate=_win_rate(daily_pnl),
        profit_factor=_profit_factor(daily_pnl),
        strategy_daily_returns=daily_pnl,
        num_trades=num_trades,
    )


def run_basket_backtest_atr(agent: TechnicalAgent, bars_by_symbol: dict[str, pd.DataFrame],
                             atr_period: int = 14, atr_stop_mult: float = 2.0,
                             target_daily_vol: float = 0.01, max_weight: float = 3.0,
                             commission_bps: float = 0.0, slippage_bps: float = 0.0) -> BasketBacktestResult:
    """
    Combina `run_technical_backtest_atr` sobre una cesta de tickers. A
    diferencia de `run_basket_backtest` (que reponderaba por volatilidad a
    nivel de portafolio), aquí cada ticker YA viene dimensionado por
    volatilidad a nivel de trade individual, así que la combinación es un
    simple promedio entre los tickers activos cada día.
    """
    per_symbol_returns = {}
    per_symbol_asset_returns = {}
    total_trades = 0

    for symbol, bars in bars_by_symbol.items():
        try:
            result = run_technical_backtest_atr(
                agent, bars, symbol, atr_period=atr_period, atr_stop_mult=atr_stop_mult,
                target_daily_vol=target_daily_vol, max_weight=max_weight,
                commission_bps=commission_bps, slippage_bps=slippage_bps,
            )
        except ValueError:
            continue
        per_symbol_returns[symbol] = result.strategy_daily_returns
        per_symbol_asset_returns[symbol] = bars["close"].pct_change().reindex(result.strategy_daily_returns.index)
        total_trades += result.num_trades

    if not per_symbol_returns:
        raise ValueError("Ningún ticker de la cesta tuvo suficiente historial para backtestear.")

    returns_df = pd.DataFrame(per_symbol_returns)
    asset_returns_df = pd.DataFrame(per_symbol_asset_returns)

    portfolio_returns = returns_df.mean(axis=1, skipna=True).dropna()
    portfolio_bh_returns = asset_returns_df.mean(axis=1, skipna=True).dropna()

    common_index = portfolio_returns.index.intersection(portfolio_bh_returns.index)
    portfolio_returns = portfolio_returns.loc[common_index]
    portfolio_bh_returns = portfolio_bh_returns.loc[common_index]

    equity = (1 + portfolio_returns).cumprod()
    total_return = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    bh_equity = (1 + portfolio_bh_returns).cumprod()
    bh_total_return = float(bh_equity.iloc[-1] - 1) if len(bh_equity) else 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    return BasketBacktestResult(
        num_tickers=len(per_symbol_returns),
        tickers=list(per_symbol_returns.keys()),
        start_date=str(common_index[0].date()) if len(common_index) else "",
        end_date=str(common_index[-1].date()) if len(common_index) else "",
        num_days=len(common_index),
        portfolio_total_return=total_return,
        basket_buy_and_hold_return=bh_total_return,
        portfolio_max_drawdown=max_drawdown,
        sharpe_ratio=_sharpe_ratio(portfolio_returns),
        sortino_ratio=_sortino_ratio(portfolio_returns),
        calmar_ratio=_calmar_ratio(portfolio_returns, max_drawdown),
        win_rate=_win_rate(portfolio_returns),
        profit_factor=_profit_factor(portfolio_returns),
        portfolio_daily_returns=portfolio_returns,
    )


@dataclass
class WalkForwardFold:
    fold_number: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    chosen_adx_threshold: float
    train_sharpe: float
    test_sharpe: float
    test_return: float


@dataclass
class WalkForwardResult:
    symbol: str
    folds: list[WalkForwardFold]
    combined_test_total_return: float
    combined_buy_and_hold_return: float
    combined_max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    combined_test_daily_returns: pd.Series = field(repr=False)


def run_walk_forward_backtest(bars: pd.DataFrame, symbol: str,
                               adx_grid: list[float] | None = None,
                               train_days: int = 504, test_days: int = 126,
                               allow_short: bool = False,
                               commission_bps: float = 0.0, slippage_bps: float = 0.0,
                               min_train_signals: int = 20) -> WalkForwardResult:
    """
    Walk-forward "rodante" sobre el umbral de ADX: la única forma real de
    saber si un parámetro elegido mirando el pasado sigue sirviendo hacia
    adelante es no mirar el futuro al elegirlo.

    Por cada ventana:
    1. Se prueban todos los umbrales de `adx_grid` SOLO sobre los retornos
       de la ventana de ENTRENAMIENTO (`train_days` días), y se elige el
       de mayor Sharpe ahí.
    2. Ese umbral (ya fijo, sin volver a tocarlo) se evalúa sobre la
       ventana de TEST (`test_days` días) que viene inmediatamente después
       y que la optimización nunca vio.
    3. La ventana avanza `test_days` y se repite, hasta agotar el
       historial (walk-forward rodante, no ventana de train que crece sin
       límite — así cada fold refleja qué umbral hubiera elegido alguien
       parado en ese momento, con datos recientes, no con todo el futuro).

    El resultado agregado (`combined_test_*`) concatena SOLO los retornos
    de las ventanas de test de todos los folds, en orden cronológico — es
    la curva de resultado que se hubiera obtenido operando en tiempo real,
    reoptimizando el umbral cada `test_days` días sin nunca usar datos
    futuros. Compararlo con el backtest de una sola ventana (`run_technical_
    backtest`) es lo que distingue "funcionó en el pasado que ya vimos" de
    "hubiera funcionado sin haber visto el futuro".

    Nota: usa el modelo de posición simple (no la variante ATR), porque el
    modelo simple no tiene estado que cruce entre folds (cada día es
    independiente dado su señal) — la variante ATR sí lo tiene (posiciones
    abiertas que persisten), lo que complicaría cortar limpiamente en los
    bordes de cada ventana.
    """
    adx_grid = list(adx_grid) if adx_grid is not None else [float(t) for t in range(15, 36)]
    closes = bars["close"]
    n = len(closes)

    folds: list[WalkForwardFold] = []
    test_returns_by_fold: list[pd.Series] = []
    test_bh_by_fold: list[pd.Series] = []

    fold_number = 0
    train_start_idx = 0
    while True:
        train_end_idx = train_start_idx + train_days
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_days
        if test_end_idx > n:
            break

        history_bars = bars.iloc[:test_end_idx]
        train_dates = closes.index[train_start_idx:train_end_idx]
        test_dates = closes.index[test_start_idx:test_end_idx]

        best_threshold = None
        best_train_sharpe = -np.inf
        best_result = None

        for threshold in adx_grid:
            candidate_agent = TechnicalAgent(alpaca_client=None, adx_threshold=float(threshold))
            try:
                result = run_technical_backtest(
                    candidate_agent, history_bars, symbol, allow_short=allow_short,
                    commission_bps=commission_bps, slippage_bps=slippage_bps,
                )
            except ValueError:
                continue

            train_returns = result.strategy_daily_returns.reindex(train_dates).dropna()
            active_train_signals = (train_returns != 0).sum()
            if active_train_signals < min_train_signals:
                continue  # muy pocas señales activas en train para confiar en el Sharpe elegido

            train_sharpe = _sharpe_ratio(train_returns)
            if train_sharpe > best_train_sharpe:
                best_train_sharpe = train_sharpe
                best_threshold = threshold
                best_result = result

        fold_number += 1
        if best_threshold is None:
            # Ningún umbral tuvo señal suficiente en esta ventana de train: se omite el fold
            train_start_idx += test_days
            continue

        test_returns = best_result.strategy_daily_returns.reindex(test_dates).fillna(0)
        test_bh_returns = closes.pct_change().reindex(test_dates).fillna(0)

        folds.append(WalkForwardFold(
            fold_number=fold_number,
            train_start=str(train_dates[0].date()),
            train_end=str(train_dates[-1].date()),
            test_start=str(test_dates[0].date()),
            test_end=str(test_dates[-1].date()),
            chosen_adx_threshold=best_threshold,
            train_sharpe=round(best_train_sharpe, 3),
            test_sharpe=round(_sharpe_ratio(test_returns), 3),
            test_return=float((1 + test_returns).prod() - 1),
        ))
        test_returns_by_fold.append(test_returns)
        test_bh_by_fold.append(test_bh_returns)

        train_start_idx += test_days

    if not folds:
        raise ValueError(
            f"No se pudo correr walk-forward para {symbol}: no hay suficiente historial "
            f"para al menos una ventana de train ({train_days}d) + test ({test_days}d)."
        )

    combined_test_returns = pd.concat(test_returns_by_fold).sort_index()
    combined_bh_returns = pd.concat(test_bh_by_fold).sort_index()

    equity = (1 + combined_test_returns).cumprod()
    total_return = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    bh_equity = (1 + combined_bh_returns).cumprod()
    bh_total_return = float(bh_equity.iloc[-1] - 1) if len(bh_equity) else 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    return WalkForwardResult(
        symbol=symbol,
        folds=folds,
        combined_test_total_return=total_return,
        combined_buy_and_hold_return=bh_total_return,
        combined_max_drawdown=max_drawdown,
        sharpe_ratio=_sharpe_ratio(combined_test_returns),
        sortino_ratio=_sortino_ratio(combined_test_returns),
        calmar_ratio=_calmar_ratio(combined_test_returns, max_drawdown),
        win_rate=_win_rate(combined_test_returns),
        profit_factor=_profit_factor(combined_test_returns),
        combined_test_daily_returns=combined_test_returns,
    )
