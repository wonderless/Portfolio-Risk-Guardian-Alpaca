"""
Backtest de momentum cruzado (cross-sectional) — la única mejora de FORMA
de estrategia con evidencia académica sólida para el Agente Técnico (ver
README, "Por qué el Agente Técnico no muestra una ventaja generalizable"):
Jegadeesh-Titman (1993) encuentra evidencia consistente en comparar muchos
activos ENTRE SÍ y apostar por los relativamente mejores, a diferencia del
momentum de un solo activo contra su propio pasado (RSI+SMA, lo que ya
hace `agents/technical_agent.py`).

No se integra al orquestador real en esta pasada: `AgentVote`
(`core/schemas.py`) es una interfaz por-símbolo, y esta estrategia necesita
ver el universo completo a la vez para poder rankear — igual que "breadth"
(`core/backtester.py::run_basket_backtest`) empezó como exploración de
backtest antes de convertirse en position sizing real.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.backtester import (
    _calmar_ratio,
    _profit_factor,
    _sharpe_ratio,
    _sortino_ratio,
    _win_rate,
)

TRADING_DAYS_PER_MONTH = 21


@dataclass
class CrossSectionalMomentumResult:
    tickers: list[str]
    start_date: str
    end_date: str
    num_days: int
    num_rebalances: int
    strategy_total_return: float
    buy_and_hold_return: float
    strategy_max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    rebalance_log: list[dict]
    strategy_daily_returns: pd.Series = field(repr=False)


def _formation_return(closes: pd.Series, as_of_idx: int, lookback_days: int, skip_days: int) -> float | None:
    """
    Retorno de formación de Jegadeesh-Titman: retorno de los últimos
    `lookback_days` EXCLUYENDO el último `skip_days` (evita el efecto de
    reversión de muy corto plazo que contamina el momentum "puro" de
    mediano plazo). Usa solo precios hasta `as_of_idx` (causal).
    """
    end_idx = as_of_idx - skip_days
    start_idx = end_idx - lookback_days
    if start_idx < 0 or end_idx <= start_idx:
        return None
    start_price = float(closes.iloc[start_idx])
    end_price = float(closes.iloc[end_idx])
    if start_price <= 0:
        return None
    return end_price / start_price - 1


def run_cross_sectional_momentum_backtest(bars_by_symbol: dict[str, pd.DataFrame],
                                           lookback_months: int = 12, skip_months: int = 1,
                                           top_n: int = 5, rebalance_freq_days: int = 21,
                                           commission_bps: float = 0.0, slippage_bps: float = 0.0
                                           ) -> CrossSectionalMomentumResult:
    """
    Cada `rebalance_freq_days`, rankea todos los tickers del universo por su
    retorno de formación (`_formation_return`, usando solo datos hasta esa
    fecha) y selecciona los `top_n` con mejor retorno, equal-weight, sin
    cortos (mismo criterio "sin cortos" del resto del proyecto — ver
    README). La cesta seleccionada se mantiene hasta el próximo
    rebalanceo, cuando se vuelve a rankear desde cero.

    `commission_bps`/`slippage_bps` se cobran sobre el turnover (fracción
    de la cesta que cambia) en cada rebalanceo.
    """
    lookback_days = lookback_months * TRADING_DAYS_PER_MONTH
    skip_days = skip_months * TRADING_DAYS_PER_MONTH

    closes_by_symbol = {sym: bars["close"] for sym, bars in bars_by_symbol.items()}
    common_dates = sorted(set.intersection(*[set(c.index) for c in closes_by_symbol.values()]))
    warmup = lookback_days + skip_days
    if len(common_dates) <= warmup:
        raise ValueError(
            "No hay suficiente historial común entre los tickers para al menos un "
            "período de formación (lookback + skip)."
        )

    closes_df = pd.DataFrame({sym: c.reindex(common_dates) for sym, c in closes_by_symbol.items()})
    daily_returns_df = closes_df.pct_change()

    rebalance_positions = list(range(warmup, len(common_dates), rebalance_freq_days))

    cost_rate = (commission_bps + slippage_bps) / 10000
    daily_strategy_returns = pd.Series(0.0, index=common_dates)
    rebalance_log: list[dict] = []
    prev_selection: set[str] = set()

    for i, start_pos in enumerate(rebalance_positions):
        end_pos = rebalance_positions[i + 1] if i + 1 < len(rebalance_positions) else len(common_dates)
        as_of_date = common_dates[start_pos]

        scores = {}
        for symbol, closes in closes_by_symbol.items():
            if as_of_date not in closes.index:
                continue
            as_of_idx = closes.index.get_loc(as_of_date)
            score = _formation_return(closes, as_of_idx, lookback_days, skip_days)
            if score is not None:
                scores[symbol] = score

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = [sym for sym, _score in ranked[:top_n]]
        rebalance_log.append({"date": str(as_of_date.date()), "selected": selected})

        if selected:
            period_dates = common_dates[start_pos:end_pos]
            period_returns = daily_returns_df.loc[period_dates, selected].mean(axis=1, skipna=True)
            daily_strategy_returns.loc[period_dates] = period_returns.fillna(0.0)

        turnover = len(set(selected).symmetric_difference(prev_selection)) / max(top_n, 1)
        daily_strategy_returns.loc[as_of_date] -= turnover * cost_rate
        prev_selection = set(selected)

    strategy_daily_returns = daily_strategy_returns.iloc[warmup:]
    strategy_equity = (1 + strategy_daily_returns).cumprod()
    strategy_total_return = float(strategy_equity.iloc[-1] - 1) if len(strategy_equity) else 0.0

    bh_returns = daily_returns_df.iloc[warmup:].mean(axis=1, skipna=True).fillna(0.0)
    bh_equity = (1 + bh_returns).cumprod()
    buy_and_hold_return = float(bh_equity.iloc[-1] - 1) if len(bh_equity) else 0.0

    running_max = strategy_equity.cummax()
    drawdown = (strategy_equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    return CrossSectionalMomentumResult(
        tickers=list(closes_by_symbol.keys()),
        start_date=str(strategy_daily_returns.index[0].date()) if len(strategy_daily_returns) else "",
        end_date=str(strategy_daily_returns.index[-1].date()) if len(strategy_daily_returns) else "",
        num_days=len(strategy_daily_returns),
        num_rebalances=len(rebalance_positions),
        strategy_total_return=strategy_total_return,
        buy_and_hold_return=buy_and_hold_return,
        strategy_max_drawdown=max_drawdown,
        sharpe_ratio=_sharpe_ratio(strategy_daily_returns),
        sortino_ratio=_sortino_ratio(strategy_daily_returns),
        calmar_ratio=_calmar_ratio(strategy_daily_returns, max_drawdown),
        win_rate=_win_rate(strategy_daily_returns),
        profit_factor=_profit_factor(strategy_daily_returns),
        rebalance_log=rebalance_log,
        strategy_daily_returns=strategy_daily_returns,
    )


@dataclass
class MomentumWalkForwardFold:
    fold_number: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    chosen_top_n: int
    train_sharpe: float
    test_sharpe: float
    test_return: float


@dataclass
class MomentumWalkForwardResult:
    tickers: list[str]
    folds: list[MomentumWalkForwardFold]
    combined_test_total_return: float
    combined_buy_and_hold_return: float
    combined_max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    combined_test_daily_returns: pd.Series = field(repr=False)


def run_momentum_walk_forward_backtest(bars_by_symbol: dict[str, pd.DataFrame],
                                        top_n_grid: list[int] | None = None,
                                        lookback_months: int = 12, skip_months: int = 1,
                                        rebalance_freq_days: int = 21,
                                        train_days: int = 756, test_days: int = 126,
                                        commission_bps: float = 0.0, slippage_bps: float = 0.0,
                                        min_train_rebalances: int = 3) -> MomentumWalkForwardResult:
    """
    Walk-forward rodante sobre `top_n` (cuántos tickers sostener), el
    único parámetro "libre" del momentum cruzado — `lookback_months=12` y
    `skip_months=1` son la convención académica estándar de
    Jegadeesh-Titman, no un valor ajustado a este universo, así que no
    tiene sentido barrerlos aquí (barrer un parámetro que nunca se eligió
    mirando los datos no diagnostica sobreajuste, solo lo introduciría).

    Mismo diseño que `run_walk_forward_backtest` (ADX) en `core/backtester.py`:
    por cada ventana, se prueban todos los valores de `top_n_grid` SOLO
    sobre la ventana de entrenamiento (`train_days`), se elige el de mayor
    Sharpe ahí, y ese valor (ya fijo) se evalúa sobre la ventana de test
    (`test_days`) inmediatamente siguiente, que la optimización nunca vio.
    La ventana avanza `test_days` y se repite (rodante, no train creciente).

    `train_days=756` (~3 años, más largo que los 504 de la versión ADX)
    porque el momentum cruzado necesita `lookback_months+skip_months`
    (~273 días hábiles) solo de calentamiento antes del primer rebalanceo
    — con train_days=504 quedaría muy poco margen para varios rebalanceos
    dentro de cada ventana de entrenamiento.
    """
    top_n_grid = list(top_n_grid) if top_n_grid is not None else list(range(2, 9))
    closes_by_symbol = {sym: bars["close"] for sym, bars in bars_by_symbol.items()}
    common_dates = sorted(set.intersection(*[set(c.index) for c in closes_by_symbol.values()]))
    n = len(common_dates)

    folds: list[MomentumWalkForwardFold] = []
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

        window_dates = set(common_dates[train_start_idx:test_end_idx])
        history_bars = {
            sym: bars.loc[bars.index.isin(window_dates)] for sym, bars in bars_by_symbol.items()
        }
        train_dates = common_dates[train_start_idx:train_end_idx]
        test_dates = common_dates[test_start_idx:test_end_idx]
        train_date_strings = {str(d.date()) for d in train_dates}

        best_top_n = None
        best_train_sharpe = -np.inf
        best_result = None

        for top_n in top_n_grid:
            try:
                result = run_cross_sectional_momentum_backtest(
                    history_bars, lookback_months=lookback_months, skip_months=skip_months,
                    top_n=top_n, rebalance_freq_days=rebalance_freq_days,
                    commission_bps=commission_bps, slippage_bps=slippage_bps,
                )
            except ValueError:
                continue

            train_rebalances = sum(1 for e in result.rebalance_log if e["date"] in train_date_strings)
            if train_rebalances < min_train_rebalances:
                continue  # muy pocos rebalanceos en train para confiar en el Sharpe elegido

            train_returns = result.strategy_daily_returns.reindex(train_dates).dropna()
            train_sharpe = _sharpe_ratio(train_returns)
            if train_sharpe > best_train_sharpe:
                best_train_sharpe = train_sharpe
                best_top_n = top_n
                best_result = result

        fold_number += 1
        if best_top_n is None:
            # Ningún top_n tuvo suficientes rebalanceos en esta ventana de train: se omite el fold
            train_start_idx += test_days
            continue

        test_returns = best_result.strategy_daily_returns.reindex(test_dates).fillna(0.0)
        test_closes_df = pd.DataFrame({sym: c.reindex(test_dates) for sym, c in closes_by_symbol.items()})
        test_bh_returns = test_closes_df.pct_change().mean(axis=1, skipna=True).fillna(0.0)

        folds.append(MomentumWalkForwardFold(
            fold_number=fold_number,
            train_start=str(train_dates[0].date()),
            train_end=str(train_dates[-1].date()),
            test_start=str(test_dates[0].date()),
            test_end=str(test_dates[-1].date()),
            chosen_top_n=best_top_n,
            train_sharpe=round(best_train_sharpe, 3),
            test_sharpe=round(_sharpe_ratio(test_returns), 3),
            test_return=float((1 + test_returns).prod() - 1),
        ))
        test_returns_by_fold.append(test_returns)
        test_bh_by_fold.append(test_bh_returns)

        train_start_idx += test_days

    if not folds:
        raise ValueError(
            "No se pudo correr walk-forward de momentum cruzado: no hay suficiente historial "
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

    return MomentumWalkForwardResult(
        tickers=list(closes_by_symbol.keys()),
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
