"""
Agente 1 — Analista Técnico

Responsabilidad única: mirar el precio histórico de un ticker y emitir una
señal (buy/sell/hold) basada en indicadores técnicos clásicos:
- RSI (Relative Strength Index): mide sobrecompra/sobreventa
- Cruce de medias móviles (SMA rápida vs SMA lenta): detecta tendencia
- ADX (Average Directional Index): mide qué tan fuerte es esa tendencia,
  para no operar el cruce de medias en mercados laterales sin dirección
  clara (la causa principal de "whipsaw" — señales que cambian todo el
  tiempo y se comen la rentabilidad, confirmado en el backtest de la
  Fase 5 antes de agregar este filtro)
- Volatilidad reciente: para matizar la confianza de la señal

Este agente NO sabe nada de noticias, riesgo de portafolio, ni ejecución.
Esa separación de responsabilidades es intencional: así cada agente se
puede probar, mejorar y reemplazar de forma independiente.
"""

import numpy as np
import pandas as pd

from core.alpaca_client import AlpacaClient
from core.position_sizing import calculate_atr, volatility_scaled_weight
from core.schemas import AgentVote, Signal


class TechnicalAgent:
    AGENT_NAME = "technical_agent"

    def __init__(self, alpaca_client: AlpacaClient, rsi_period: int = 14,
                 sma_fast: int = 20, sma_slow: int = 50,
                 adx_period: int = 14, adx_threshold: float = 25.0):
        self.client = alpaca_client
        self.rsi_period = rsi_period
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def _calculate_rsi(self, closes: pd.Series) -> float:
        """RSI clásico de Wilder sobre la serie de precios de cierre."""
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    def _calculate_smas(self, closes: pd.Series) -> tuple[float, float]:
        """Media móvil rápida y lenta, para detectar cruces de tendencia."""
        sma_fast = closes.rolling(window=self.sma_fast).mean().iloc[-1]
        sma_slow = closes.rolling(window=self.sma_slow).mean().iloc[-1]
        return float(sma_fast), float(sma_slow)

    def _calculate_volatility(self, closes: pd.Series, window: int = 20) -> float:
        """Volatilidad anualizada aproximada, a partir de retornos diarios."""
        returns = closes.pct_change().dropna()
        daily_vol = returns.tail(window).std()
        annualized_vol = float(daily_vol * np.sqrt(252)) if not pd.isna(daily_vol) else 0.0
        return annualized_vol

    def _calculate_adx(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> float:
        """
        ADX de Wilder: mide la FUERZA de una tendencia (no su dirección) en
        una escala de 0-100. Por debajo de ~25 se considera mercado lateral
        sin tendencia clara — ahí un cruce de medias móviles suele ser
        ruido, no señal.
        """
        period = self.adx_period

        up_move = highs.diff()
        down_move = -lows.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        prev_close = closes.shift(1)
        true_range = pd.concat([
            highs - lows,
            (highs - prev_close).abs(),
            (lows - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Suavizado de Wilder == media móvil exponencial con alpha=1/period
        atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        last = adx.iloc[-1]
        return float(last) if not pd.isna(last) else 0.0

    def analyze(self, symbol: str) -> AgentVote:
        """
        Punto de entrada principal: trae datos de Alpaca, calcula indicadores,
        y devuelve un AgentVote con la señal, confianza y razonamiento.
        """
        # Necesitamos suficiente historial para que la SMA lenta tenga sentido
        lookback_days = max(self.sma_slow, self.rsi_period, self.adx_period) + 30
        bars = self.client.get_recent_bars(symbol, lookback_days=lookback_days)
        closes = bars["close"]

        rsi = self._calculate_rsi(closes)
        sma_fast, sma_slow = self._calculate_smas(closes)
        volatility = self._calculate_volatility(closes)
        adx = self._calculate_adx(bars["high"], bars["low"], closes)
        current_price = float(closes.iloc[-1])

        signal, confidence, reasoning = self._decide(
            rsi=rsi, sma_fast=sma_fast, sma_slow=sma_slow,
            current_price=current_price, volatility=volatility, adx=adx,
        )

        return AgentVote(
            agent_name=self.AGENT_NAME,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            raw_metrics={
                "rsi": round(rsi, 2),
                "sma_fast": round(sma_fast, 2),
                "sma_slow": round(sma_slow, 2),
                "adx": round(adx, 2),
                "current_price": round(current_price, 2),
                "annualized_volatility": round(volatility, 4),
            },
        )

    def calculate_position_size(self, symbol: str, account_equity: float, atr_period: int = 14,
                                 target_daily_vol: float = 0.01, max_weight: float = 3.0,
                                 max_position_pct: float = 0.25) -> float:
        """
        Cantidad de acciones a comprar dimensionada por volatilidad (ATR),
        en vez de un tamaño fijo — la misma fórmula validada en el
        backtest de Fase 5 (`core/backtester.py::run_technical_backtest_atr`,
        ver README, "Exploración de breadth..."), aplicada ahora al paper
        trading real a través de `core/position_sizing.py` (fuente única,
        no una versión reinventada aparte).

        En el backtest, `max_weight=3.0` era seguro porque `weight` se
        promediaba entre una cesta de 16 tickers (`run_basket_backtest_atr`)
        — ningún ticker individual llegaba a dominar el libro. Acá se
        dimensiona UNA sola orden para UN solo ticker, así que sin un tope
        aparte, `weight` solo (0.3-0.5 típico en acciones líquidas) puede
        poner 30-50% de la cuenta en un solo nombre. `max_position_pct`
        (mismo default que `RiskAgent.max_position_pct`, para no
        contradecir el propio criterio de concentración del sistema) topa
        eso explícitamente.

        Solo tiene sentido para entradas BUY: el orquestador nunca abre
        posiciones cortas, así que una señal SELL simplemente cierra lo
        que ya se tenga (ver `agents/orchestrator.py::execute_node`).
        """
        lookback_days = atr_period + 30
        bars = self.client.get_recent_bars(symbol, lookback_days=lookback_days)
        atr = calculate_atr(bars["high"], bars["low"], bars["close"], atr_period).iloc[-1]
        price = float(bars["close"].iloc[-1])

        weight = volatility_scaled_weight(float(atr), price, target_daily_vol, max_weight)
        weight = min(weight, max_position_pct)
        if weight <= 0:
            return 0.0
        return (weight * account_equity) / price

    def _decide(self, rsi: float, sma_fast: float, sma_slow: float,
                current_price: float, volatility: float, adx: float) -> tuple[Signal, float, str]:
        """
        Lógica de decisión: combina RSI + cruce de medias, con el cruce de
        medias filtrado por ADX (solo cuenta como señal de tendencia si el
        ADX confirma que hay una tendencia real, no un mercado lateral).

        Regla simple e interpretable a propósito (esto es clave para el
        dashboard de transparencia — un jurado o usuario debe poder entender
        por qué el agente decidió lo que decidió, sin caja negra).
        """
        trend_bullish = sma_fast > sma_slow
        strong_trend = adx >= self.adx_threshold
        reasons = []

        # Señales de sobrecompra/sobreventa
        if rsi >= 70:
            oversold_signal = Signal.SELL
            reasons.append(f"RSI en {rsi:.1f} indica sobrecompra (umbral: 70)")
        elif rsi <= 30:
            oversold_signal = Signal.BUY
            reasons.append(f"RSI en {rsi:.1f} indica sobreventa (umbral: 30)")
        else:
            oversold_signal = Signal.HOLD
            reasons.append(f"RSI en {rsi:.1f} está en rango neutral (30-70)")

        # Señal de tendencia por cruce de medias, solo si el ADX confirma
        # que hay una tendencia real (evita operar el ruido de un mercado lateral)
        if not strong_trend:
            trend_signal = Signal.HOLD
            reasons.append(
                f"ADX en {adx:.1f} está por debajo del umbral ({self.adx_threshold:.0f}): "
                f"sin tendencia clara, se ignora el cruce de medias"
            )
        elif trend_bullish:
            trend_signal = Signal.BUY
            reasons.append(
                f"SMA rápida (${sma_fast:.2f}) por encima de SMA lenta (${sma_slow:.2f}): "
                f"tendencia alcista confirmada por ADX en {adx:.1f}"
            )
        else:
            trend_signal = Signal.SELL
            reasons.append(
                f"SMA rápida (${sma_fast:.2f}) por debajo de SMA lenta (${sma_slow:.2f}): "
                f"tendencia bajista confirmada por ADX en {adx:.1f}"
            )

        # Combinar: si ambas coinciden, alta confianza; si no, HOLD conservador
        if oversold_signal == trend_signal:
            final_signal = oversold_signal
            confidence = 0.8
        elif oversold_signal == Signal.HOLD:
            final_signal = trend_signal
            confidence = 0.5
        else:
            final_signal = Signal.HOLD
            confidence = 0.3
            reasons.append("RSI y tendencia se contradicen: se opta por cautela (HOLD)")

        # La volatilidad alta reduce la confianza (mercado más impredecible)
        if volatility > 0.5:
            confidence = max(0.2, confidence - 0.2)
            reasons.append(f"Volatilidad anualizada alta ({volatility:.1%}): confianza reducida")

        reasoning = f"Precio actual ${current_price:.2f}.\n" + "\n".join(f"- {r}" for r in reasons)
        return final_signal, round(confidence, 2), reasoning
