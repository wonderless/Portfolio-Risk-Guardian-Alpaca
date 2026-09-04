"""
Agente 3 — Analista de Riesgo

Responsabilidad única: mirar la exposición actual del portafolio (paper
trading) y opinar si tiene sentido, desde el punto de vista de riesgo,
aumentar, mantener o reducir la posición en un ticker.

Este agente es intencionalmente conservador: su trabajo no es predecir
hacia dónde va el precio (eso lo hacen los agentes técnico y de
sentimiento), sino frenar al sistema cuando el portafolio está demasiado
concentrado o demasiado volátil. Por eso rara vez emite BUY.

No sabe nada de indicadores técnicos ni de noticias. Solo emite un
AgentVote estandarizado.
"""

import numpy as np

from core.alpaca_client import AlpacaClient
from core.schemas import AgentVote, Signal


class RiskAgent:
    AGENT_NAME = "risk_agent"

    def __init__(self, alpaca_client: AlpacaClient,
                 max_position_pct: float = 0.25,
                 high_portfolio_vol_threshold: float = 0.40):
        """
        max_position_pct: % máximo recomendado del portafolio en un solo
            activo antes de considerarlo "sobre-concentrado".
        high_portfolio_vol_threshold: volatilidad anualizada del portafolio
            por encima de la cual se considera "alta".
        """
        self.client = alpaca_client
        self.max_position_pct = max_position_pct
        self.high_portfolio_vol_threshold = high_portfolio_vol_threshold

    def _portfolio_volatility(self, positions: list[dict], portfolio_value: float) -> float:
        """
        Aproxima la volatilidad del portafolio como el promedio de las
        volatilidades individuales de cada posición, ponderado por su peso
        en el portafolio. Es una simplificación (ignora correlaciones entre
        activos), pero es simple e interpretable, que es lo que buscamos.
        """
        if not positions or portfolio_value <= 0:
            return 0.0

        weighted_vol = 0.0
        for position in positions:
            weight = position["market_value"] / portfolio_value
            try:
                bars = self.client.get_recent_bars(position["symbol"], lookback_days=60)
                returns = bars["close"].pct_change().dropna()
                vol = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0
            except Exception:
                vol = 0.0
            weighted_vol += weight * vol

        return weighted_vol

    def analyze(self, symbol: str) -> AgentVote:
        """
        Punto de entrada principal: calcula concentración y volatilidad del
        portafolio, y devuelve un AgentVote con la señal, confianza y
        razonamiento desde la óptica de gestión de riesgo.
        """
        account = self.client.get_account_info()
        positions = self.client.get_open_positions()
        portfolio_value = account["portfolio_value"]

        current_position = next((p for p in positions if p["symbol"] == symbol), None)
        position_value = current_position["market_value"] if current_position else 0.0
        concentration_pct = position_value / portfolio_value if portfolio_value > 0 else 0.0

        portfolio_vol = self._portfolio_volatility(positions, portfolio_value)

        signal, confidence, reasoning = self._decide(
            symbol=symbol,
            concentration_pct=concentration_pct,
            portfolio_vol=portfolio_vol,
            num_positions=len(positions),
        )

        return AgentVote(
            agent_name=self.AGENT_NAME,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            raw_metrics={
                "concentration_pct": round(concentration_pct, 4),
                "portfolio_annualized_volatility": round(portfolio_vol, 4),
                "open_positions": len(positions),
                "portfolio_value": round(portfolio_value, 2),
            },
        )

    def _decide(self, symbol: str, concentration_pct: float, portfolio_vol: float,
                num_positions: int) -> tuple[Signal, float, str]:
        reasons = []

        over_concentrated = concentration_pct > self.max_position_pct
        high_vol = portfolio_vol > self.high_portfolio_vol_threshold

        if over_concentrated:
            reasons.append(
                f"{symbol} representa {concentration_pct:.1%} del portafolio, por encima "
                f"del máximo recomendado ({self.max_position_pct:.0%}). Sugerencia de hedging: "
                f"reducir el tamaño de la posición o diversificar hacia activos poco "
                f"correlacionados antes de aumentar la exposición."
            )
            signal = Signal.SELL
            confidence = 0.75
        elif high_vol:
            reasons.append(
                f"La volatilidad anualizada estimada del portafolio ({portfolio_vol:.1%}) "
                f"está por encima del umbral de riesgo ({self.high_portfolio_vol_threshold:.0%}). "
                f"Sugerencia de hedging: evitar aumentar posiciones hasta que la volatilidad "
                f"baje, o considerar activos defensivos que reduzcan el riesgo total."
            )
            signal = Signal.HOLD
            confidence = 0.6
        else:
            reasons.append(
                f"{symbol} representa {concentration_pct:.1%} del portafolio (dentro del límite "
                f"de {self.max_position_pct:.0%}) y la volatilidad estimada del portafolio "
                f"({portfolio_vol:.1%}) está en rango aceptable. No hay señal de riesgo que "
                f"frene la operación."
            )
            signal = Signal.HOLD
            confidence = 0.3

        reasons.append(f"Posiciones abiertas actuales: {num_positions}.")
        reasoning = "\n".join(reasons)
        return signal, confidence, reasoning
