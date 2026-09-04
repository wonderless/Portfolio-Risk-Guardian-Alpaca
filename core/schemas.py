"""
Modelos de datos compartidos entre agentes.

Cada agente del sistema (técnico, sentimiento, riesgo, ejecutor) se comunica
usando estos esquemas comunes. Esto es lo que permite que el Agente 4
(orquestador) pueda comparar "votos" de distintos agentes de forma
consistente, sin importar la lógica interna de cada uno.
"""

from enum import Enum
from pydantic import BaseModel, Field


class Signal(str, Enum):
    """Los tres posibles 'votos' que puede emitir cualquier agente analista."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class AgentVote(BaseModel):
    """
    Resultado estandarizado que produce cada agente analista (1, 2 y 3).
    El campo `reasoning` es intencional: es lo que alimenta el dashboard
    de transparencia ("qué está pensando cada agente y por qué").
    """
    agent_name: str
    symbol: str
    signal: Signal
    confidence: float = Field(ge=0.0, le=1.0, description="Qué tan seguro está el agente, de 0 a 1")
    reasoning: str = Field(description="Explicación en lenguaje natural de por qué llegó a esta señal")
    raw_metrics: dict = Field(default_factory=dict, description="Métricas numéricas de soporte (RSI, sentiment score, etc.)")


class ConsensusDecision(BaseModel):
    """
    Resultado final que produce el Agente 4 (orquestador) después de aplicar
    la lógica de consenso sobre los votos de los agentes 1, 2 y 3.
    """
    symbol: str
    final_signal: Signal
    votes: list[AgentVote]
    consensus_reached: bool
    explanation: str
    executed: bool = False
    order_id: str | None = None
